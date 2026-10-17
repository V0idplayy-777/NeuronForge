import assert from "node:assert/strict";
import test from "node:test";
import * as tf from "@tensorflow/tfjs";
import { TensorFlowOps } from "../src/engine/gpu/tensorflow";
import { acquireBackend, releaseBackend } from "../src/engine/gpu/backend";
import { verifyDevice } from "../src/engine/gpu/verify";
import { Accelerator } from "../src/engine/gpu/executor";
import { SequentialModel } from "../src/engine/model";
import { makeLayerDef } from "../src/engine/layers";
import { Tensor } from "../src/engine/tensor";
import { computeLoss } from "../src/engine/losses";
import { DEFAULT_OPTIMIZER, createOptimizerState, optimizerStep } from "../src/engine/optimizers";
import { DEFAULT_TRAINING, Trainer } from "../src/engine/train";
import type { Architecture } from "../src/engine/types";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tolerance = 3e-5) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]) && Number.isFinite(b[i]) && Math.abs(a[i] - b[i]) < tolerance,
      `index ${i}: ${a[i]} vs ${b[i]}`);
  }
}

const architecture: Architecture = {
  inputShape: [2], inputName: "x", seed: 9, initMethod: "xavier_uniform",
  layers: [
    makeLayerDef("dense", { config: { units: 5, useBias: true, activation: "relu" } }),
    makeLayerDef("residual_dense", { config: { units: 5, activation: "gelu", projection: "auto", scale: 1 } }),
    makeLayerDef("dense", { config: { units: 2, useBias: true, activation: "softmax" } }),
  ],
};
const x = new Tensor([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
const y = new Tensor([4, 2], [1, 0, 0, 1, 0, 1, 1, 0]);

test("TensorFlow.js integration", async t => {
  const ops = await TensorFlowOps.create();
  const baseline = tf.memory().numTensors;
  try {
    await t.test("all 64 device cases agree with CPU, without leaking tensors", async () => {
      for (let i = 0; i < 2; i++) {
        const checks = await verifyDevice(ops);
        assert.equal(checks.length, 64);
        assert.deepEqual(checks.filter(c => !c.passed), []);
        assert.equal(tf.memory().numTensors, baseline);
      }
    });
    await t.test("host uploads are snapshots; reshape views share ownership", async () => {
      const data = Float32Array.from([1, 2, 3, 4]);
      const b = await ops.fromHost(data, [2, 2]);
      data.fill(99);
      const view = await ops.reshapeView(b, [4]);
      ops.dispose(view);
      close(await ops.toHost(b), [1, 2, 3, 4]);
      const out = await ops.unary("scale", view, 2, 0);
      close(await ops.toHost(out), [2, 4, 6, 8]);
      ops.dispose(out); ops.dispose(b); ops.dispose(b);
      assert.equal(tf.memory().numTensors, baseline);
    });
    await t.test("embedding ids use engine rounding and clipping; duplicates sum", async () => {
      const w = await ops.fromHost(Float32Array.from([10, 20, 30]), [3, 1]);
      const ids = await ops.fromHost(Float32Array.from([-1, 0.5, 1.5, 9, 1]), [5]);
      const gathered = await ops.gatherRows(w, ids, 1);
      close(await ops.toHost(gathered), [10, 20, 30, 30, 20]);
      const grad = await ops.fromHost(Float32Array.from([1, 2, 3, 4, 5]), [5, 1]);
      const scattered = await ops.scatterRows(grad, ids, 1, 3);
      close(await ops.toHost(scattered), [1, 7, 7]);
      for (const b of [w, ids, gathered, grad, scattered]) ops.dispose(b);
    });
    await t.test("max pooling ties pick the first strict maximum", async () => {
      const input = await ops.fromHost(Float32Array.from([1, 1, 1, 1]), [1, 2, 2, 1]);
      const dy = await ops.fromHost(Float32Array.from([1]), [1, 1, 1, 1]);
      const dx = await ops.pool2dBackward("max", input, dy, {
        B: 1, H: 2, W: 2, C: 1, outH: 1, outW: 1, pool: 2, stride: 2,
      });
      close(await ops.toHost(dx), [1, 0, 0, 0]);
      for (const b of [input, dy, dx]) ops.dispose(b);
    });
    await t.test("forward, backward and Adam updates agree with the custom engine", async () => {
      const plain = new SequentialModel(architecture);
      const model = new SequentialModel(architecture);
      const acc = new Accelerator(model, ops);
      const opt = { ...DEFAULT_OPTIMIZER, learningRate: 0.01 };
      const stateA = createOptimizerState(), stateB = createOptimizerState();
      try {
        for (let step = 0; step < 5; step++) {
          plain.zeroGrads(); model.zeroGrads();
          const expected = plain.forward(x, true, false).output;
          plain.backward(computeLoss("cce", expected, y).grad);
          await acc.syncParams();
          const actual = await acc.runForward(x, true);
          close(expected.data, actual.data);
          await acc.runBackward(computeLoss("cce", actual, y).grad);
          for (let i = 0; i < plain.layers.length; i++) {
            for (const key of Object.keys(plain.layers[i].grads)) {
              close(plain.layers[i].grads[key].data, model.layers[i].grads[key].data);
            }
          }
          optimizerStep(plain, opt, stateA); optimizerStep(model, opt, stateB);
          for (let i = 0; i < plain.layers.length; i++) {
            for (const key of Object.keys(plain.layers[i].params)) {
              close(plain.layers[i].params[key].data, model.layers[i].params[key].data);
            }
          }
        }
        // Repeated validation must not accumulate the forward caches.
        await acc.runForward(x, false);
        const resident = tf.memory().numTensors;
        for (let i = 0; i < 5; i++) await acc.runForward(x, false);
        assert.equal(tf.memory().numTensors, resident);
      } finally { acc.destroy(); }
      assert.equal(tf.memory().numTensors, baseline);
    });
    await t.test("Trainer uses TFJS across epochs and validation", async () => {
      const model = new SequentialModel(architecture);
      const acc = new Accelerator(model, ops);
      const trainer = new Trainer(model, {
        ...DEFAULT_TRAINING, epochs: 3, batchSize: 2, loss: "cce",
      });
      trainer.accelerator = acc;
      try {
        const progress = await trainer.run({ xs: x, ys: y }, { xs: x, ys: y });
        assert.equal(progress.status, "completed", progress.message);
        assert.equal(progress.step, 6);
        assert.equal(progress.history.length, 3);
        assert.ok(Number.isFinite(progress.valLoss));
        assert.equal(trainer.accelerator, acc, "must not silently fall back to custom CPU");
      } finally { acc.destroy(); }
      assert.equal(tf.memory().numTensors, baseline);
    });
  } finally { ops.destroy(); }
  assert.equal(tf.memory().numTensors, baseline);

  await t.test("backend selection is cached, labeled and released", async () => {
    try {
      const a = await acquireBackend("tensorflow");
      const b = await acquireBackend("tensorflow");
      assert.equal(a.status.kind, "tensorflow");
      assert.match(a.status.label, /TensorFlow.js.*cpu/);
      assert.equal(a.ops, b.ops);
      await a.ops.zeros([2]);
      releaseBackend();
      assert.equal(tf.memory().numTensors, baseline);
      const cpu = await acquireBackend("cpu");
      assert.equal(cpu.status.kind, "cpu");
    } finally { releaseBackend(); }
  });
});
