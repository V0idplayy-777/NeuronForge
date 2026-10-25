import { Tensor, matmul, transpose2d, add, mul, softmaxLast, mean, sum } from "./tensor";
import { relu, sigmoid, gelu } from "./activations";
import { mse, bce, cce } from "./losses";
import { RNG } from "./rng";
import { SequentialModel } from "./model";
import { makeLayerDef } from "./layers";
import { optimizerStep, createOptimizerState, DEFAULT_OPTIMIZER } from "./optimizers";
import { computeLoss } from "./losses";
import { exportModel, importModel, verifyRoundTrip } from "./serialize";
import { classificationFromPred, regressionFromPred } from "./metrics";
import { generateBuiltin, DEFAULT_SYNTH, SEQ_WINDOW } from "../data/datasets";
import { CsvParser, parseCsv, parseCsvFile, DEFAULT_CSV_CONFIG, type FileLike } from "../data/csv";
import { materializeCsv } from "../data/preprocess";
import { runLayerCoverage, buildStandaloneBundle, verifyStandalone, loadRuntime, COVERAGE_CASES } from "../export/standalone";
import { compileExpr } from "../data/expr";
import { trainBpe, bpeEncode, bpeEncodeWith, bpeDecode, bpeStats, encodeTable } from "../data/bpe";
import { buildCustom } from "../data/custom";
import { GENERATORS, generatorShape } from "../data/generators";
import {
  DEFAULT_IMAGE_CONFIG,
  augmentPixels,
  buildImageDataset,
  labelFromName,
  resizePixels,
  toGrayscale,
  type ImageRow,
} from "../data/images";
import { PRESETS, alignToVocab } from "./presets";
import { compileLayer, makeLayerDef as mkDef } from "./layers";
import { CpuOps } from "./gpu/cpu";
import { Accelerator } from "./gpu/executor";
import { verifyDevice, summarizeChecks } from "./gpu/verify";
import { broadcastPlan, convGeomFor } from "./gpu/ops";
import {
  activationCode,
  backwardCode,
  binaryCode,
  binaryStep,
  broadcastPlanesStep,
  col2imStep,
  gatherStep,
  im2colStep,
  matmulStep,
  poolBwdStep,
  poolFwdStep,
  reduceOuterStep,
  scatterStep,
  softmaxBwdStep,
  softmaxStep,
  unaryBwdStep,
  unaryStep,
} from "./webgpu/plan";
import {
  simBinary,
  simBroadcastPlanes,
  simCol2im,
  simGather,
  simIm2col,
  simMatmul,
  simPoolBwd,
  simPoolFwd,
  simReduceOuter,
  simScatter,
  simSoftmax,
  simSoftmaxBwd,
  simUnary,
  simUnaryBwd,
} from "./webgpu/sim";
import { TEXT_CORPUS, TEXT_CHARS, LM_SEQ_LEN } from "../data/corpus";
import { buildMassiveCorpus, buildWordTokenizer, encodeWordsToIds, decodeTokenIds, tokenizeWords, LM2_SEQ_LEN } from "../data/corpusLarge";
import { buildNovaCorpus } from "../data/corpusChat";
import { buildPythonCorpus } from "../data/corpusCode";
import { buildReasonCorpus, REASON_SEQ_LEN } from "../data/corpusReason";
import { sampleFromProbs } from "./sampling";
import { tokenizeForModel, detokenize, generateTokens } from "./generate";
import type { Architecture } from "./types";

export interface TestResult {
  name: string;
  passed: boolean;
  message: string;
  ms: number;
}

function approx(a: number, b: number, tol = 1e-4): boolean {
  return Math.abs(a - b) <= tol;
}

function run(name: string, fn: () => void): TestResult {
  const t0 = performance.now();
  try {
    fn();
    return { name, passed: true, message: "ok", ms: performance.now() - t0 };
  } catch (e) {
    return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
  }
}

function expect(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

export function runSelfTests(): TestResult[] {
  const tests: TestResult[] = [];

  tests.push(
    run("matmul 2×3 @ 3×2", () => {
      const a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
      const b = new Tensor([3, 2], [7, 8, 9, 10, 11, 12]);
      const c = matmul(a, b);
      expect(c.shape[0] === 2 && c.shape[1] === 2, "shape");
      expect(approx(c.data[0]!, 58) && approx(c.data[1]!, 64), `got ${c.data[0]}, ${c.data[1]} expected 58, 64`);
      expect(approx(c.data[2]!, 139) && approx(c.data[3]!, 154), `got ${c.data[2]}, ${c.data[3]}`);
    }),
  );

  tests.push(
    run("transpose2d", () => {
      const a = new Tensor([2, 3], [1, 2, 3, 4, 5, 6]);
      const t = transpose2d(a);
      expect(t.shape[0] === 3 && t.shape[1] === 2, "shape");
      expect(t.data[0] === 1 && t.data[1] === 4 && t.data[2] === 2, "values");
    }),
  );

  tests.push(
    run("relu + sigmoid", () => {
      const x = new Tensor([4], [-2, 0, 0.5, 3]);
      const r = relu(x);
      expect(r.data[0] === 0 && r.data[3] === 3, "relu");
      const s = sigmoid(new Tensor([1], [0]));
      expect(approx(s.data[0]!, 0.5), "sigmoid(0)");
    }),
  );

  tests.push(
    run("softmax sums to 1", () => {
      const x = new Tensor([2, 3], [1, 2, 3, 0, 0, 0]);
      const p = softmaxLast(x);
      expect(approx(p.data[0]! + p.data[1]! + p.data[2]!, 1, 1e-5), "row0");
      expect(approx(p.data[3]! + p.data[4]! + p.data[5]!, 1, 1e-5), "row1");
      expect(p.data[2]! > p.data[1]! && p.data[1]! > p.data[0]!, "order");
    }),
  );

  tests.push(
    run("mse gradient", () => {
      const pred = new Tensor([2], [1, 3]);
      const tgt = new Tensor([2], [0, 1]);
      const L = mse(pred, tgt);
      expect(approx(L.value, (1 + 4) / 2), `loss ${L.value}`);
      expect(approx(L.grad.data[0]!, 1) && approx(L.grad.data[1]!, 2), "grad");
    }),
  );

  tests.push(
    run("bce at p=y=0.5 is ln 2", () => {
      const pred = new Tensor([1], [0.5]);
      const tgt = new Tensor([1], [0.5]);
      const L = bce(pred, tgt);
      expect(approx(L.value, Math.LN2, 1e-5), `got ${L.value}`);
    }),
  );

  tests.push(
    run("cce one-hot", () => {
      const pred = new Tensor([1, 3], [0.1, 0.2, 0.7]);
      const tgt = new Tensor([1, 3], [0, 0, 1]);
      const L = cce(pred, tgt);
      expect(approx(L.value, -Math.log(0.7), 1e-5), `got ${L.value}`);
    }),
  );

  tests.push(
    run("dense numerical gradient", () => {
      const arch: Architecture = {
        inputShape: [2],
        inputName: "x",
        seed: 1,
        initMethod: "random_normal",
        layers: [
          makeLayerDef("dense", { id: "d1", name: "d1", config: { units: 1, useBias: true, activation: "none" } }),
        ],
      };
      const model = new SequentialModel(arch);
      const x = new Tensor([1, 2], [0.4, -0.2]);
      const y = new Tensor([1, 1], [0.5]);
      const W = model.layers[0]!.params.W!;
      const pred = model.forward(x, false).output;
      const L = computeLoss("mse", pred, y);
      model.backward(L.grad);
      const analytic = model.layers[0]!.grads.W!.data[0]!;
      const eps = 1e-3;
      const orig = W.data[0]!;
      W.data[0] = orig + eps;
      const lp = computeLoss("mse", model.forward(x, false).output, y).value;
      W.data[0] = orig - eps;
      const lm = computeLoss("mse", model.forward(x, false).output, y).value;
      W.data[0] = orig;
      const numeric = (lp - lm) / (2 * eps);
      expect(approx(analytic, numeric, 5e-2), `analytic ${analytic} numeric ${numeric}`);
    }),
  );

  tests.push(
    run("SGD updates parameters", () => {
      const arch: Architecture = {
        inputShape: [1],
        inputName: "x",
        seed: 2,
        initMethod: "random_normal",
        layers: [makeLayerDef("dense", { id: "d", name: "d", config: { units: 1, useBias: true, activation: "none" } })],
      };
      const model = new SequentialModel(arch);
      const before = model.layers[0]!.params.W!.data[0]!;
      const x = new Tensor([1, 1], [1]);
      const y = new Tensor([1, 1], [0]);
      const pred = model.forward(x, true).output;
      const L = computeLoss("mse", pred, y);
      model.backward(L.grad);
      const st = createOptimizerState();
      optimizerStep(model, { ...DEFAULT_OPTIMIZER, name: "sgd", learningRate: 0.1, weightDecay: 0, gradClip: 0, momentum: 0, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, st);
      const after = model.layers[0]!.params.W!.data[0]!;
      expect(before !== after, `weight did not change (${before})`);
    }),
  );

  tests.push(
    run("XOR training reduces loss", () => {
      const arch: Architecture = {
        inputShape: [2],
        inputName: "x",
        seed: 3,
        initMethod: "he_normal",
        layers: [
          makeLayerDef("dense", { id: "h", name: "h", config: { units: 8, useBias: true, activation: "none" } }),
          makeLayerDef("relu", { id: "r", name: "r" }),
          makeLayerDef("dense", { id: "o", name: "o", config: { units: 1, useBias: true, activation: "none" } }),
          makeLayerDef("sigmoid", { id: "s", name: "s" }),
        ],
      };
      const model = new SequentialModel(arch);
      const xs = new Tensor([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
      const ys = new Tensor([4, 1], [0, 1, 1, 0]);
      const lossBefore = computeLoss("bce", model.forward(xs, false).output, ys).value;
      const st = createOptimizerState();
      const opt = { ...DEFAULT_OPTIMIZER, name: "adam" as const, learningRate: 0.05, weightDecay: 0, gradClip: 0 };
      for (let i = 0; i < 200; i++) {
        model.zeroGrads();
        const pred = model.forward(xs, true).output;
        const L = computeLoss("bce", pred, ys);
        model.backward(L.grad);
        optimizerStep(model, opt, st);
      }
      const lossAfter = computeLoss("bce", model.forward(xs, false).output, ys).value;
      expect(lossAfter < lossBefore * 0.5, `loss ${lossBefore.toFixed(3)} → ${lossAfter.toFixed(3)} did not drop enough`);
      expect(lossAfter < 0.4, `final loss ${lossAfter} still high`);
    }),
  );

  tests.push(
    run("Adam maintains moment state", () => {
      const arch: Architecture = {
        inputShape: [1],
        inputName: "x",
        seed: 4,
        initMethod: "random_normal",
        layers: [makeLayerDef("dense", { id: "d", name: "d", config: { units: 1, useBias: false, activation: "none" } })],
      };
      const model = new SequentialModel(arch);
      const st = createOptimizerState();
      const x = new Tensor([1, 1], [1]);
      const y = new Tensor([1, 1], [0]);
      model.zeroGrads();
      const L = computeLoss("mse", model.forward(x, true).output, y);
      model.backward(L.grad);
      optimizerStep(model, { ...DEFAULT_OPTIMIZER, name: "adam" }, st);
      expect(st.t === 1, "timestep");
      const slot = Object.values(st.slots)[0];
      expect(!!slot?.m && !!slot?.v, "moments allocated");
      expect(slot!.m!.some((v) => v !== 0) || slot!.v!.some((v) => v !== 0), "moments updated");
    }),
  );

  tests.push(
    run("serialize round-trip inference", () => {
      const arch: Architecture = {
        inputShape: [2],
        inputName: "x",
        seed: 5,
        initMethod: "he_normal",
        layers: [
          makeLayerDef("dense", { id: "h", name: "h", config: { units: 4, useBias: true, activation: "none" } }),
          makeLayerDef("tanh", { id: "t", name: "t" }),
          makeLayerDef("dense", { id: "o", name: "o", config: { units: 1, useBias: true, activation: "none" } }),
        ],
      };
      const model = new SequentialModel(arch);
      const x = new Tensor([3, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
      const r = verifyRoundTrip(model, x);
      expect(r.ok, r.message);
    }),
  );

  tests.push(
    run("export / import preserves weights", () => {
      const arch: Architecture = {
        inputShape: [2],
        inputName: "x",
        seed: 6,
        initMethod: "he_normal",
        layers: [makeLayerDef("dense", { id: "d", name: "d", config: { units: 3, useBias: true, activation: "none" } })],
      };
      const model = new SequentialModel(arch);
      const exp = exportModel(model);
      const { model: m2, warnings } = importModel(exp);
      expect(warnings.length === 0, warnings.join(", "));
      const a = model.layers[0]!.params.W!.data;
      const b = m2.layers[0]!.params.W!.data;
      for (let i = 0; i < a.length; i++) expect(a[i] === b[i], `weight ${i}`);
    }),
  );

  tests.push(
    run("shape inference dense stack", () => {
      const arch: Architecture = {
        inputShape: [5],
        inputName: "x",
        seed: 1,
        initMethod: "he_normal",
        layers: [
          makeLayerDef("dense", { config: { units: 7, useBias: true, activation: "none" } }),
          makeLayerDef("relu"),
          makeLayerDef("dense", { config: { units: 2, useBias: true, activation: "none" } }),
        ],
      };
      const m = new SequentialModel(arch);
      expect(m.valid, m.issues[0]?.message ?? "invalid");
      expect(m.outputShape[0] === 2, `out ${m.outputShape}`);
      const pc = m.paramCount();
      expect(pc.trainable === 5 * 7 + 7 + 7 * 2 + 2, `params ${pc.trainable}`);
    }),
  );

  tests.push(
    run("incompatible flatten/dense reported", () => {
      const arch: Architecture = {
        inputShape: [4, 4, 1],
        inputName: "x",
        seed: 1,
        initMethod: "he_normal",
        layers: [makeLayerDef("dense", { config: { units: 3, useBias: true, activation: "none" } })],
      };
      const m = new SequentialModel(arch);
      expect(!m.valid, "should be invalid");
      expect(/Flatten/i.test(m.issues[0]?.message ?? ""), m.issues[0]?.message ?? "no issue");
    }),
  );

  tests.push(
    run("classification metrics", () => {
      const pred = new Tensor([4, 2], [0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6]);
      const tgt = new Tensor([4, 2], [1, 0, 0, 1, 1, 0, 1, 0]);
      const r = classificationFromPred(pred, tgt, false);
      expect(r.accuracy === 0.75, `acc ${r.accuracy}`);
      expect(r.confusion.length === 2, "confusion");
    }),
  );

  tests.push(
    run("regression metrics", () => {
      const pred = new Tensor([2], [1, 3]);
      const tgt = new Tensor([2], [1, 1]);
      const r = regressionFromPred(pred, tgt);
      expect(approx(r.mae, 1), `mae ${r.mae}`);
      expect(approx(r.mse, 2), `mse ${r.mse}`);
    }),
  );

  tests.push(
    run("CSV quoted fields", () => {
      const { headers, rows } = parseCsv('a,b\n"hello, world",3\n"x ""y""",4\n');
      expect(headers[0] === "a" && headers[1] === "b", "headers");
      expect(rows[0]![0] === "hello, world" && rows[0]![1] === "3", `row0 ${rows[0]}`);
      expect(rows[1]![0] === 'x "y"', `escaped quotes ${rows[1]![0]}`);
    }),
  );

  tests.push(
    run("dataset split sizes", () => {
      const ds = generateBuiltin("circles", { ...DEFAULT_SYNTH, n: 100, trainFrac: 0.7, valFrac: 0.15, testFrac: 0.15, seed: 1, noise: 0.1, classes: 2 });
      const n = ds.train.xs.shape[0]! + ds.val.xs.shape[0]! + ds.test.xs.shape[0]!;
      expect(n === 100, `split sum ${n}`);
    }),
  );

  tests.push(
    run("wave_seq dataset: windows are real consecutive slices", () => {
      const ds = generateBuiltin("wave_seq", { ...DEFAULT_SYNTH, n: 200, noise: 0, seed: 3 });
      expect(ds.meta.inputShape[0] === SEQ_WINDOW && ds.meta.inputShape[1] === 1, `shape ${ds.meta.inputShape}`);
      expect(ds.meta.task === "regression", `task ${ds.meta.task}`);
      const B = ds.train.xs.shape[0]!;
      expect(B > 50, `only ${B} training windows`);
      // With zero noise, each window's last value should feed smoothly into the target
      // (the series is continuous), so consecutive window points must differ from random.
      const row0 = Array.from(ds.train.xs.data.subarray(0, SEQ_WINDOW));
      expect(row0.some((v, i) => i === 0 || v !== row0[0]), "window looks constant, not a real sequence");
    }),
  );

  tests.push(
    run("new presets compile and forward-pass finite (one per category incl. Sequence)", () => {
      for (const id of ["noise-robust-clf", "layernorm-reg", "cnn-batchnorm", "ae-layernorm", "char-rnn-seq", "seq-rnn", "seq-rnn-deep", "seq-mlp-baseline", "novachat-transformer", "novachat-gru", "python-code-transformer", "python-code-lstm", "deep-reasoner-40k"]) {
        const p = PRESETS.find((x) => x.id === id);
        expect(!!p, `missing preset ${id}`);
        const m = new SequentialModel(p!.arch);
        expect(m.valid, `${id}: ${m.issues[0]?.message ?? "invalid"}`);
        const inSize = p!.arch.inputShape.reduce((a, b) => a * b, 1);
        const x = new Tensor([2, ...p!.arch.inputShape], Float32Array.from({ length: 2 * inSize }, (_, i) => ((i % 7) - 3) / 5));
        const y = m.forward(x, false).output;
        for (let i = 0; i < y.size; i++) expect(Number.isFinite(y.data[i]!), `${id}: output ${i} is ${y.data[i]}`);
      }
      const categories = new Set(PRESETS.map((p) => p.category));
      expect(categories.has("Sequence"), "Sequence category missing from presets");
      expect(PRESETS.filter((p) => p.category === "Sequence").length >= 3, "Sequence category needs at least 3 presets");
    }),
  );

  tests.push(
    run("gelu(0) = 0", () => {
      const y = gelu(new Tensor([1], [0]));
      expect(approx(y.data[0]!, 0, 1e-6), `${y.data[0]}`);
    }),
  );

  tests.push(
    run("broadcast add", () => {
      const a = new Tensor([2, 3], [1, 1, 1, 2, 2, 2]);
      const b = new Tensor([3], [1, 2, 3]);
      const c = add(a, b);
      expect(c.data[0] === 2 && c.data[5] === 5, `${Array.from(c.data)}`);
    }),
  );

  tests.push(
    run("RNG is deterministic", () => {
      const a = new RNG(99);
      const b = new RNG(99);
      for (let i = 0; i < 20; i++) expect(a.next() === b.next(), "stream");
    }),
  );

  tests.push(
    run("mean / sum", () => {
      const t = new Tensor([3], [1, 2, 3]);
      expect(sum(t) === 6 && approx(mean(t), 2), "stats");
    }),
  );

  tests.push(
    run("mul elementwise", () => {
      const a = new Tensor([3], [2, 3, 4]);
      const b = new Tensor([3], [5, 6, 7]);
      const c = mul(a, b);
      expect(c.data[0] === 10 && c.data[2] === 28, "mul");
    }),
  );

  tests.push(
    run("expression parser: sin(x)+2*x^2", () => {
      const { fn, vars } = compileExpr("sin(x) + 2*x^2");
      expect(vars.length === 1 && vars[0] === "x", `vars ${vars}`);
      const v = fn({ x: 0.5 });
      expect(approx(v, Math.sin(0.5) + 2 * 0.25, 1e-9), `got ${v}`);
    }),
  );

  tests.push(
    run("expression parser: precedence and unary minus", () => {
      const { fn } = compileExpr("-2^2 + 3*4");
      expect(approx(fn({}), -4 + 12, 1e-9), `got ${fn({})}`);
      const { fn: f2 } = compileExpr("max(1, 2, 3) + pi");
      expect(approx(f2({}), 3 + Math.PI, 1e-9), "max/pi");
    }),
  );

  tests.push(
    run("expression parser rejects garbage", () => {
      let threw = false;
      try {
        compileExpr("x + $");
      } catch {
        threw = true;
      }
      expect(threw, "should throw on invalid char");
      threw = false;
      try {
        compileExpr("foo(x)");
      } catch {
        threw = true;
      }
      expect(threw, "should throw on unknown function");
    }),
  );

  tests.push(
    run("custom formula dataset is real data", () => {
      const ds = buildCustom({
        kind: "formula",
        name: "t",
        mode: "regression",
        expr: "2*x",
        n: 50,
        noise: 0,
        seed: 3,
        xMin: -1,
        xMax: 1,
        yMin: -1,
        yMax: 1,
        trainFrac: 0.7,
        valFrac: 0.15,
      });
      const n = ds.train.xs.shape[0]!;
      for (let i = 0; i < n; i++) {
        expect(approx(ds.train.ys.data[i]!, 2 * ds.train.xs.data[i]!, 1e-5), `row ${i}: y != 2x`);
      }
    }),
  );

  tests.push(
    run("custom points dataset builds and splits", () => {
      const pts = Array.from({ length: 20 }, (_, i) => ({ x: (i % 5) / 2 - 1, y: (i % 4) / 2 - 0.5, cls: i % 2 }));
      const ds = buildCustom({ kind: "points", name: "drawn", points: pts, classes: 2, trainFrac: 0.7, valFrac: 0.15, seed: 1 });
      const total = ds.train.xs.shape[0]! + ds.val.xs.shape[0]! + ds.test.xs.shape[0]!;
      expect(total === 20, `split total ${total}`);
      expect(ds.meta.recommendedLoss === "bce", "loss");
    }),
  );

  tests.push(
    run("all presets compile with valid shapes", () => {
      for (const p of PRESETS) {
        const m = new SequentialModel(p.arch);
        expect(m.valid, `${p.name}: ${m.issues[0]?.message ?? "invalid"}`);
        expect(m.paramCount().trainable > 0, `${p.name}: no trainable params`);
      }
    }),
  );

  tests.push(
    run("RNN BPTT matches numerical gradient", () => {
      const layer = compileLayer(
        mkDef("rnn", { id: "r", name: "r", config: { units: 3, returnSequences: false } }),
        [4, 2],
        new RNG(11),
        "xavier_uniform",
      );
      const x = new Tensor([1, 4, 2], [0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.05, 0.2]);
      const w = [0.5, -1, 2];
      const scalar = (l: typeof layer) => {
        const y = l.forward(x, false);
        return y.data[0]! * w[0]! + y.data[1]! * w[1]! + y.data[2]! * w[2]!;
      };
      layer.forward(x, false);
      layer.backward(new Tensor([1, 3], w));
      const analytic = layer.grads.Whh!.data[4]!; // center of 3x3 recurrent matrix
      const Whh = layer.params.Whh!;
      const orig = Whh.data[4]!;
      const eps = 1e-3;
      Whh.data[4] = orig + eps;
      const lp = scalar(layer);
      Whh.data[4] = orig - eps;
      const lm = scalar(layer);
      Whh.data[4] = orig;
      const numeric = (lp - lm) / (2 * eps);
      expect(approx(analytic, numeric, 5e-2), `analytic ${analytic} numeric ${numeric}`);
    }),
  );

  tests.push(
    run("language corpus and tokenizer agree", () => {
      expect(TEXT_CORPUS.length > 4000, `corpus only ${TEXT_CORPUS.length} chars`);
      const ds = generateBuiltin("tiny_text", { ...DEFAULT_SYNTH, seed: 1 });
      expect(ds.meta.inputShape[0] === LM_SEQ_LEN, `context ${ds.meta.inputShape[0]} != ${LM_SEQ_LEN}`);
      expect(ds.meta.outputUnits === TEXT_CHARS.length, `vocab ${ds.meta.outputUnits} != ${TEXT_CHARS.length}`);
      expect(ds.meta.tokenizer?.tokens.length === TEXT_CHARS.length, "tokenizer not attached");
      let maxId = -1;
      for (let i = 0; i < ds.train.ys.size; i++) maxId = Math.max(maxId, ds.train.ys.data[i]!);
      expect(maxId < TEXT_CHARS.length, `target id ${maxId} outside vocab`);
      let maxX = -1;
      for (let i = 0; i < ds.train.xs.size; i++) maxX = Math.max(maxX, ds.train.xs.data[i]!);
      expect(maxX < TEXT_CHARS.length, `input id ${maxX} outside vocab`);
    }),
  );

  tests.push(
    run("alignToVocab fixes a mismatched LM architecture", () => {
      const bad: Architecture = {
        inputShape: [8],
        inputName: "x",
        seed: 1,
        initMethod: "he_normal",
        layers: [
          mkDef("embedding", { id: "e", name: "e", config: { vocabSize: 64, embeddingDim: 8 } }),
          mkDef("flatten", { id: "f", name: "f" }),
          mkDef("dense", { id: "d", name: "d", config: { units: 64, useBias: true, activation: "none" } }),
          mkDef("softmax", { id: "s", name: "s" }),
        ],
      };
      const fixed = alignToVocab(bad, TEXT_CHARS.length, LM_SEQ_LEN);
      const m = new SequentialModel(fixed);
      expect(m.valid, m.issues[0]?.message ?? "invalid after align");
      expect(m.outputShape[m.outputShape.length - 1] === TEXT_CHARS.length, "output classes");
      const emb = fixed.layers.find((l) => l.type === "embedding")!;
      expect(Number(emb.config.vocabSize) === TEXT_CHARS.length, "embedding vocab");
    }),
  );

  tests.push(
    run("massive corpus is genuinely large and tokenizer is consistent", () => {
      const { text, stats } = buildMassiveCorpus(1234);
      expect(stats.chars > 150000, `corpus only ${stats.chars} chars`);
      expect(stats.tokens > 25000, `only ${stats.tokens} word tokens`);
      const tok = buildWordTokenizer(text, 3000, LM2_SEQ_LEN);
      expect(tok.vocabSize <= 3000, `vocab ${tok.vocabSize}`);
      expect(tok.tokens[0] === "<unk>", `token 0 is ${tok.tokens[0]}`);
      expect(tok.oovRate >= 0 && tok.oovRate < 0.5, `oov ${tok.oovRate}`);
      // every token id in the encoded corpus must be inside the vocabulary
      const ids = tokenizeWords(text).map((w) => tok.tokenToId[w] ?? 0);
      let maxId = 0;
      for (const id of ids) if (id > maxId) maxId = id;
      expect(maxId < tok.vocabSize, `max id ${maxId} >= vocab ${tok.vocabSize}`);
      // Round trip on a phrase built from tokens that really are in this
      // vocabulary (the capped vocabulary keeps the most frequent words, so
      // hand-picked words may legitimately map to <unk>).
      expect(tok.tokens.includes("the"), `common word "the" missing from the vocabulary`);
      // Punctuation tokens are re-joined without a preceding space by the
      // decoder, so build the phrase from plain words.
      const phrase = tok.tokens.filter((t) => /^[a-z0-9]+$/.test(t)).slice(0, 6).join(" ");
      expect(phrase.length > 3, `could not build an in-vocabulary phrase (got "${phrase}")`);
      const rt = decodeTokenIds(encodeWordsToIds(phrase, tok.tokens), tok.tokens, "word");
      expect(rt === phrase, `round trip gave "${rt}" for "${phrase}"`);
    }),
  );

  tests.push(
    run("tiny_text_2 dataset shapes agree with its tokenizer", () => {
      const ds = generateBuiltin("tiny_text_2", { ...DEFAULT_SYNTH, seed: 1 });
      expect(ds.meta.inputShape[0] === LM2_SEQ_LEN, `context ${ds.meta.inputShape[0]}`);
      expect(ds.meta.outputUnits === ds.meta.tokenizer?.tokens.length, "output units != vocab");
      let maxX = 0;
      for (let i = 0; i < ds.train.xs.size; i++) maxX = Math.max(maxX, ds.train.xs.data[i]!);
      let maxY = 0;
      for (let i = 0; i < ds.train.ys.size; i++) maxY = Math.max(maxY, ds.train.ys.data[i]!);
      const vocab = ds.meta.tokenizer!.tokens.length;
      expect(maxX < vocab, `input id ${maxX} >= vocab ${vocab}`);
      expect(maxY < vocab, `target id ${maxY} >= vocab ${vocab}`);
      expect(ds.train.xs.shape[0]! > 2000, `only ${ds.train.xs.shape[0]} training windows`);
    }),
  );

  tests.push(
    run("temperature 0 is greedy argmax", () => {
      const p = [0.1, 0.6, 0.05, 0.25];
      for (let i = 0; i < 20; i++) {
        const r = sampleFromProbs(p, 0, 50, new RNG(i));
        expect(r.index === 1, `greedy picked ${r.index}`);
      }
    }),
  );

  tests.push(
    run("top-k=1 always returns the argmax", () => {
      const p = [0.05, 0.5, 0.2, 0.25];
      for (let i = 0; i < 50; i++) {
        const r = sampleFromProbs(p, 1.5, 1, new RNG(i));
        expect(r.index === 1, `k=1 picked ${r.index}`);
      }
    }),
  );

  tests.push(
    run("sampling respects the model distribution at T=1, wide k", () => {
      const p = [0.1, 0.6, 0.05, 0.25];
      const counts = [0, 0, 0, 0];
      const n = 4000;
      for (let i = 0; i < n; i++) counts[sampleFromProbs(p, 1, 4, new RNG(i + 1)).index]!++;
      for (let i = 0; i < 4; i++) {
        const got = counts[i]! / n;
        expect(Math.abs(got - p[i]!) < 0.04, `token ${i}: got ${got.toFixed(3)} want ${p[i]}`);
      }
    }),
  );

  tests.push(
    run("low temperature concentrates on the argmax", () => {
      const p = [0.3, 0.4, 0.2, 0.1];
      let argmax = 0;
      for (let i = 0; i < 500; i++) if (sampleFromProbs(p, 0.15, 4, new RNG(i + 1)).index === 1) argmax++;
      expect(argmax / 500 > 0.8, `argmax share ${(argmax / 500).toFixed(2)} at T=0.15`);
    }),
  );

  tests.push(
    run("top-k excludes tokens outside the k largest", () => {
      const p = [0.5, 0.2, 0.2, 0.06, 0.02, 0.02];
      let worst = 0;
      for (let i = 0; i < 300; i++) {
        const idx = sampleFromProbs(p, 1, 3, new RNG(i + 1)).index;
        worst = Math.max(worst, idx);
      }
      expect(worst <= 2, `sampled id ${worst} outside the top 3`);
    }),
  );

  tests.push(
    run("preset forward pass produces finite outputs", () => {
      for (const p of PRESETS) {
        const m = new SequentialModel(p.arch);
        const inSize = p.arch.inputShape.reduce((a, b) => a * b, 1);
        const x = new Tensor([2, ...p.arch.inputShape], Float32Array.from({ length: 2 * inSize }, (_, i) => ((i % 7) - 3) / 5));
        const y = m.forward(x, false).output;
        for (let i = 0; i < y.size; i++) {
          expect(Number.isFinite(y.data[i]!), `${p.name}: output ${i} is ${y.data[i]}`);
        }
      }
    }),
  );

  tests.push(
    run("standalone runtime: every layer + activation matches the engine", () => {
      const res = runLayerCoverage();
      const bad = res.filter((r) => !r.ok);
      expect(bad.length === 0, bad.map((b) => `${b.name}: ${b.error ?? b.maxDiff}`).join("; "));
      expect(res.length >= 30, `only ${res.length} coverage cases`);
    }),
  );

  tests.push(
    run("standalone export: XOR classifier verifies end-to-end", () => {
      const ds = generateBuiltin("xor", { ...DEFAULT_SYNTH, n: 4 });
      const model = new SequentialModel(PRESETS.find((p) => p.id === "xor")!.arch);
      const { bundle } = buildStandaloneBundle(model, ds, { name: "xor", project: "t", loss: "bce", trained: false });
      expect(bundle.task === "binary", `task ${bundle.task}`);
      const r = verifyStandalone(bundle, model, ds);
      const failed = r.checks.filter((c) => c.status === "fail");
      expect(r.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
      const pred = loadRuntime().createModel(bundle).predict([1, 0]) as { predicted?: number; probability?: number };
      expect(pred.predicted === 0 || pred.predicted === 1, "predict() did not return a class");
      expect(typeof pred.probability === "number", "sigmoid output should be reported as a probability");
    }),
  );

  tests.push(
    run("standalone export: character LM tokenizer + sampled generation parity", () => {
      const ds = generateBuiltin("tiny_text", { ...DEFAULT_SYNTH, seed: 1 });
      const tok = ds.meta.tokenizer!;
      const model = new SequentialModel(alignToVocab(PRESETS.find((p) => p.id === "charlm")!.arch, tok.tokens.length, tok.seqLen));
      const { bundle } = buildStandaloneBundle(model, ds, { name: "lm", project: "t", loss: "sparse_cce", trained: false });
      expect(bundle.task === "language", `task ${bundle.task}`);
      expect(bundle.tokenizer?.tokens.length === tok.tokens.length, "tokenizer not embedded");
      expect(bundle.tests.generation.length >= 2, "no generation vectors embedded");
      const r = verifyStandalone(bundle, model, ds);
      const failed = r.checks.filter((c) => c.status === "fail");
      expect(r.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
    }),
  );

  tests.push(
    run("BPE tokenizer: deterministic, lossless, and it compresses", () => {
      const model = trainBpe(TEXT_CORPUS, 260);
      expect(model.merges.length > 100, `only ${model.merges.length} merges learned`);
      const ids = bpeEncode(TEXT_CORPUS, model.tokens, model.merges);
      expect(bpeDecode(ids, model.tokens) === TEXT_CORPUS.toLowerCase(), "round trip changed the text");
      expect(ids.length < TEXT_CORPUS.length * 0.8, `${ids.length} tokens for ${TEXT_CORPUS.length} characters`);
      const again = trainBpe(TEXT_CORPUS, 260);
      expect(JSON.stringify(again.merges) === JSON.stringify(model.merges), "training is not deterministic");
      const stats = bpeStats(model, TEXT_CORPUS);
      expect(stats.unknown === 0, `${stats.unknown} unknown tokens on the training corpus`);
      // Whitespace survives as its own tokens, so decoding never invents spaces.
      const spaced = bpeEncode("hello   there\nagain", model.tokens, model.merges);
      expect(bpeDecode(spaced, model.tokens) === "hello   there\nagain", `whitespace lost: ${JSON.stringify(bpeDecode(spaced, model.tokens))}`);
      // An unseen character is the only source of <unk>, and it is visible.
      const unknown = bpeEncode("caf\u00e9", model.tokens, model.merges);
      expect(unknown.includes(0), "unseen character should map to <unk>");
    }),
  );

  tests.push(
    run("BPE dataset (tiny_text_bpe) shapes agree with its tokenizer", () => {
      const ds = generateBuiltin("tiny_text_bpe", { ...DEFAULT_SYNTH, seed: 3 });
      const tok = ds.meta.tokenizer!;
      expect(tok.level === "bpe", `level ${tok.level}`);
      expect((tok.merges?.length ?? 0) > 0, "merge rules not stored with the dataset");
      expect(ds.meta.outputUnits === tok.tokens.length, `output units ${ds.meta.outputUnits} != vocab ${tok.tokens.length}`);
      expect(ds.meta.inputShape[0] === tok.seqLen, "context length mismatch");
      let maxX = -1;
      let maxY = -1;
      for (let i = 0; i < ds.train.xs.size; i++) maxX = Math.max(maxX, ds.train.xs.data[i]!);
      for (let i = 0; i < ds.train.ys.size; i++) maxY = Math.max(maxY, ds.train.ys.data[i]!);
      expect(maxX < tok.tokens.length && maxY < tok.tokens.length, `token ids outside the vocabulary (x≤${maxX}, y≤${maxY})`);
      // A training window must decode to real text, not mojibake.
      const window: number[] = [];
      for (let i = 0; i < tok.seqLen; i++) window.push(ds.train.xs.data[i]!);
      const text = bpeDecode(window, tok.tokens);
      expect(text.length >= tok.seqLen && /[a-z]/.test(text), `window decoded to ${JSON.stringify(text.slice(0, 40))}`);
      // The tokenizer used for training must be the one stored in the dataset.
      expect(JSON.stringify(bpeEncode(TEXT_CORPUS.slice(0, 200), tok.tokens, tok.merges!)).length > 2, "stored merge rules do not encode");
    }),
  );

  tests.push(
    run("BPE vocabulary target is honoured and ids stay in range", () => {
      const small = generateBuiltin("tiny_text_bpe", { ...DEFAULT_SYNTH, seed: 5, bpeVocab: 96 });
      const big = generateBuiltin("tiny_text_bpe", { ...DEFAULT_SYNTH, seed: 5, bpeVocab: 512 });
      const tSmall = small.meta.tokenizer!;
      const tBig = big.meta.tokenizer!;
      expect(tSmall.tokens.length < tBig.tokens.length, `vocab did not grow: ${tSmall.tokens.length} vs ${tBig.tokens.length}`);
      expect(tSmall.tokens.length >= 64 && tBig.tokens.length <= 2048, "vocabulary escaped the supported band");
      expect((tBig.merges?.length ?? 0) === tBig.tokens.length - 2 - 33 - 1 || (tBig.merges?.length ?? 0) > 0, "merge rules missing");
      for (const ds of [small, big]) {
        const vocab = ds.meta.tokenizer!.tokens.length;
        let worst = 0;
        for (let i = 0; i < ds.train.xs.size; i++) worst = Math.max(worst, ds.train.xs.data[i]!);
        for (let i = 0; i < ds.train.ys.size; i++) worst = Math.max(worst, ds.train.ys.data[i]!);
        expect(worst < vocab, `id ${worst} outside a ${vocab}-token vocabulary`);
        // Neither vocabulary should drop characters the corpus actually uses.
        const table = encodeTable(ds.meta.tokenizer!.tokens, ds.meta.tokenizer!.merges ?? []);
        const encoded = bpeEncodeWith(table, TEXT_CORPUS);
        expect(encoded.filter((id) => id === 0).length === 0, `unknown tokens with vocab ${vocab}`);
        expect(bpeDecode(encoded, ds.meta.tokenizer!.tokens) === TEXT_CORPUS.toLowerCase(), `round trip broke at vocab ${vocab}`);
      }
    }),
  );

  tests.push(
    run("NovaChat corpus is large, conversational, and self-consistent", () => {
      const { text, stats } = buildNovaCorpus(1234 + 777);
      expect(stats.chars > 1_800_000, `corpus is only ${stats.chars.toLocaleString()} chars`);
      expect(stats.lines > 18_000, `only ${stats.lines} lines`);
      // The persona must be stated, consistently, many times.
      const personaHits = (text.match(/my name is nova/g) ?? []).length;
      expect(personaHits >= 8, `persona name stated only ${personaHits} times`);
      // Turn format is the backbone of the dataset.
      const turns = (text.match(/q: /g) ?? []).length;
      expect(turns > 15_000, `only ${turns} question turns`);
      expect((text.match(/a: /g) ?? []).length > 15_000, "answer turns missing");
      // Multi-turn dialogues must stay adjacent (block shuffle, not line shuffle):
      // this exact 2-turn exchange exists in the hand written bank.
      const adjacent = "q: okay a: you feel the ground with your feet and step downhill. small steps, many times\nq: and the dark part a:";
      expect(text.includes(adjacent), "dialogue turns were split apart by the shuffle");
      // Computed arithmetic is present and correct (the generator computed it).
      expect(text.includes("q: what is two plus three a: two plus three is five"), "simple sum missing");
      expect(text.includes("q: what is seven times eight a: seven times eight is fifty six"), "times table missing");
    }),
  );

  tests.push(
    run("nova_chat dataset: windows, tokenizer and id ranges agree", () => {
      const ds = generateBuiltin("nova_chat", { ...DEFAULT_SYNTH, seed: 11 });
      const tok = ds.meta.tokenizer!;
      expect(tok.level === "bpe", `level ${tok.level}`);
      expect(ds.meta.task === "language", "task is not language");
      expect(ds.meta.outputUnits === tok.tokens.length, `output units ${ds.meta.outputUnits} != vocab ${tok.tokens.length}`);
      expect(ds.meta.inputShape[0] === tok.seqLen, "context length mismatch");
      const nTrain = ds.train.xs.shape[0]!;
      expect(nTrain > 40_000, `only ${nTrain} training windows — this is supposed to be the big one`);
      let worst = 0;
      for (let i = 0; i < ds.train.xs.size; i++) worst = Math.max(worst, ds.train.xs.data[i]!);
      for (let i = 0; i < ds.train.ys.size; i++) worst = Math.max(worst, ds.train.ys.data[i]!);
      expect(worst < tok.tokens.length, `token id ${worst} outside the ${tok.tokens.length}-token vocabulary`);
      // A training window must decode to readable text.
      const window: number[] = [];
      for (let i = 0; i < tok.seqLen; i++) window.push(ds.train.xs.data[i]!);
      const decoded = bpeDecode(window, tok.tokens);
      expect(/[a-z]/.test(decoded) && decoded.length >= tok.seqLen, `window decoded to ${JSON.stringify(decoded.slice(0, 40))}`);
    }),
  );

  tests.push(
    run("python_code dataset: case-preserving BPE keeps real python intact", () => {
      // Unit level: the flag changes lowercasing behaviour, default stays lowercase.
      const sample = "def f():\n    return None  # True";
      const ci = trainBpe(sample + "\ndef g():\n    return True\n", 64);
      const ciIds = bpeEncode(sample, ci.tokens, ci.merges);
      expect(bpeDecode(ciIds, ci.tokens) === sample.toLowerCase(), "default BPE must still lowercase");
      const cs = trainBpe(sample + "\ndef g():\n    return True\n", 64, { caseSensitive: true });
      const csIds = bpeEncode(sample, cs.tokens, cs.merges, true);
      expect(bpeDecode(csIds, cs.tokens) === sample, "case-sensitive round trip lost case");

      // Dataset level: python corpus keeps None/True/CamelCase, ids stay in range.
      const { text: pyText } = buildPythonCorpus(99 + 31337);
      expect(pyText.includes("return None"), "python corpus lost None");
      expect(pyText.includes("ZeroDivisionError"), "python corpus lost builtin exception names");
      const ds = generateBuiltin("python_code", { ...DEFAULT_SYNTH, seed: 12 });
      const tok = ds.meta.tokenizer!;
      expect(tok.level === "bpe" && tok.caseSensitive === true, "python tokenizer is not case-sensitive bpe");
      expect(tok.tokens.some((t) => /[A-Z]/.test(t)), "no uppercase tokens learned");
      expect(ds.meta.outputUnits === tok.tokens.length, "output units != vocabulary");
      let worst = 0;
      for (let i = 0; i < ds.train.xs.size; i++) worst = Math.max(worst, ds.train.xs.data[i]!);
      for (let i = 0; i < ds.train.ys.size; i++) worst = Math.max(worst, ds.train.ys.data[i]!);
      expect(worst < tok.tokens.length, `token id ${worst} outside the vocabulary`);
      const window: number[] = [];
      for (let i = 0; i < tok.seqLen; i++) window.push(ds.train.xs.data[i]!);
      expect(/[a-z]/.test(bpeDecode(window, tok.tokens)), "code window decoded to garbage");
    }),
  );

  tests.push(
    run("standalone export: case-sensitive BPE tokenizer parity in the runtime", () => {
      const ds = generateBuiltin("python_code", { ...DEFAULT_SYNTH, seed: 12 });
      const tok = ds.meta.tokenizer!;
      const arch = alignToVocab(
        {
          inputShape: [tok.seqLen],
          inputName: "input",
          seed: 42,
          initMethod: "xavier_normal",
          layers: [
            makeLayerDef("embedding", { name: "embed", config: { vocabSize: tok.tokens.length, embeddingDim: 16 } }),
            makeLayerDef("flatten", { name: "flatten" }),
            makeLayerDef("dense", { name: "logits", config: { units: tok.tokens.length, useBias: true, activation: "none" } }),
            makeLayerDef("softmax", { name: "softmax" }),
          ],
        },
        tok.tokens.length,
        tok.seqLen,
      );
      const model = new SequentialModel(arch);
      const { bundle } = buildStandaloneBundle(model, ds, { name: "py-lm", project: "t", loss: "sparse_cce", trained: false });
      expect(bundle.tokenizer?.caseSensitive === true, "case-sensitive flag not embedded in the bundle");
      const r = verifyStandalone(bundle, model, ds);
      const failed = r.checks.filter((c) => c.status === "fail");
      expect(r.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
      // The runtime must NOT lowercase for this tokenizer: mixed-case text
      // encodes differently than its lowercase form.
      const rt = loadRuntime().createModel(bundle);
      const upper = rt.tokenize("None True ValueError");
      const lower = rt.tokenize("none true valueerror");
      expect(JSON.stringify(upper) !== JSON.stringify(lower), "runtime lowercased a case-sensitive tokenizer");
    }),
  );

  tests.push(
    run("synthetic generators produce valid, reproducible datasets", () => {
      const cfg = { n: 120, noise: 0.08, seed: 13, trainFrac: 0.7, valFrac: 0.15, testFrac: 0.15 };
      for (const def of GENERATORS) {
        const values = { ...def.defaults };
        const ds = def.build({ values, ...cfg });
        const again = def.build({ values, ...cfg });
        const shape = generatorShape(def, values);
        expect(JSON.stringify(ds.meta.inputShape) === JSON.stringify(shape.shape), `${def.id}: shape ${ds.meta.inputShape} != ${shape.shape}`);
        expect(ds.meta.outputUnits === shape.out, `${def.id}: output units ${ds.meta.outputUnits} != ${shape.out}`);
        const feat = ds.meta.inputShape.reduce((a, b) => a * b, 1);
        expect(ds.train.xs.shape[1] === feat || (ds.meta.inputShape.length > 1 && ds.train.xs.shape.length === ds.meta.inputShape.length + 1), `${def.id}: row width ${ds.train.xs.shape} for shape ${ds.meta.inputShape}`);
        expect(ds.train.xs.size % feat === 0, `${def.id}: ragged features`);
        expect(ds.train.xs.shape[0]! + ds.val.xs.shape[0]! + ds.test.xs.shape[0]! === cfg.n, `${def.id}: split sizes do not add up`);
        // Reproducible: same seed, same tensors.
        expect(JSON.stringify(Array.from(again.train.xs.data)) === JSON.stringify(Array.from(ds.train.xs.data)), `${def.id}: not deterministic`);
        // Targets agree with the declared task.
        const yDim = ds.train.ys.size / ds.train.ys.shape[0]!;
        expect(yDim === ds.meta.outputUnits, `${def.id}: target width ${yDim} != outputUnits ${ds.meta.outputUnits}`);
        if (ds.meta.task === "multiclass") {
          for (let i = 0; i < Math.min(20, ds.train.ys.shape[0]!); i++) {
            let sum = 0;
            for (let c = 0; c < yDim; c++) sum += ds.train.ys.data[i * yDim + c]!;
            expect(approx(sum, 1, 1e-5), `${def.id}: row ${i} is not one-hot (sum ${sum})`);
          }
          expect(ds.meta.classNames?.length === ds.meta.outputUnits, `${def.id}: class names missing`);
        }
        expect(ds.meta.source === "custom", `${def.id}: wrong source`);
      }
    }),
  );

  tests.push(
    run("image resize, grayscale and augmentation behave", () => {
      // Nearest-neighbour upsampling of a 2×2 image must map whole quadrants.
      const quad = new Float32Array([0, 0.25, 0.5, 0.75]);
      const up = resizePixels(quad, 2, 2, 1, 4, 4, "nearest");
      // Row 0: [0, 0, 0.25, 0.25]; row 1 the same; rows 2-3 use the second source row.
      expect(up[0] === 0 && up[1] === 0, `top-left quadrant wrong: ${up[0]}, ${up[1]}`);
      expect(up[2] === 0.25 && up[3] === 0.25, `top-right quadrant wrong: ${up[2]}, ${up[3]}`);
      expect(up[8] === 0.5 && up[9] === 0.5, `bottom-left quadrant wrong: ${up[8]}, ${up[9]}`);
      expect(up[3 * 4 + 3] === 0.75, "bottom-right quadrant wrong");
      // Bilinear of a constant image stays constant, and midpoints interpolate.
      const flat = new Float32Array(64).fill(0.4);
      const resizedFlat = resizePixels(flat, 8, 8, 1, 5, 3, "bilinear");
      let worst = 0;
      for (const v of resizedFlat) worst = Math.max(worst, Math.abs(v - 0.4));
      expect(worst < 1e-6, `constant image changed: ${worst}`);
      const ramp = new Float32Array([0, 1]);
      const half = resizePixels(ramp, 2, 1, 1, 3, 1, "bilinear");
      expect(approx(half[1]!, 0.5, 1e-6), `midpoint should be 0.5, got ${half[1]}`);
      expect(resizePixels(ramp, 2, 1, 1, 2, 1, "bilinear")[1] === 1, "same-size resize must be exact");
      // Grayscale uses luma weights: pure green is the brightest primary.
      const rgb = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      const gray = toGrayscale(rgb, 3);
      expect(approx(gray[0]!, 0.299, 1e-6) && approx(gray[1]!, 0.587, 1e-6) && approx(gray[2]!, 0.114, 1e-6), `luma weights wrong: ${Array.from(gray)}`);
      // Augmentation: flips mirror, 90° rotation is a rotation, everything stays in range.
      const img = new Float32Array(9).map((_, i) => i / 8);
      const mirrored = augmentPixels(img, 3, 3, 1, { flipH: true }, new RNG(1));
      const isMirror = (out: Float32Array): boolean => {
        for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) if (Math.abs(out[y * 3 + x]! - img[y * 3 + (2 - x)]!) > 1e-6) return false;
        return true;
      };
      const isOriginal = (out: Float32Array): boolean => {
        for (let i = 0; i < 9; i++) if (Math.abs(out[i]! - img[i]!) > 1e-6) return false;
        return true;
      };
      expect(isMirror(mirrored) || isOriginal(mirrored), "flip augmentation neither flipped nor passed through");
      const rotated = augmentPixels(img, 3, 3, 1, { rotate90: true }, new RNG(7));
      const rotatedTwice = augmentPixels(rotated, 3, 3, 1, { rotate90: true }, new RNG(7));
      expect(JSON.stringify(Array.from(rotated)) === JSON.stringify(Array.from(rotatedTwice)), "90° rotation is not deterministic");
      const noisy = augmentPixels(img, 3, 3, 1, { brightness: 0.4, contrast: 0.4, noise: 0.2, shift: 1, rotate: 15 }, new RNG(3));
      for (const v of noisy) expect(v >= 0 && v <= 1, `augmented pixel escaped [0, 1]: ${v}`);
      // Labels: prefix, folder and single-class modes.
      expect(labelFromName("cats_001.png", "prefix") === "cats", `prefix label ${labelFromName("cats_001.png", "prefix")}`);
      expect(labelFromName("dogs-07.jpg", "prefix") === "dogs", "dash separator not handled");
      expect(labelFromName("a.png", "folder", "vehicles") === "vehicles", "folder label wrong");
      expect(labelFromName("a.png", "all") === "all", "single-class label wrong");
    }),
  );

  tests.push(
    run("image dataset assembly: augmentation touches the train split only", () => {
      const rows: ImageRow[] = [];
      const mk = (label: string, offset: number): ImageRow => ({
        label,
        pixels: Float32Array.from({ length: 16 }, (_, i) => ((i + offset) % 7) / 7),
      });
      for (let i = 0; i < 10; i++) rows.push(mk(i % 2 === 0 ? "cat" : "dog", i));
      const cfg = { ...DEFAULT_IMAGE_CONFIG, width: 4, height: 4, channels: 1 as const, copies: 2, rotate: 5, shift: 1, noise: 0.02, seed: 11 };
      const ds = buildImageDataset(rows, cfg);
      const nTrain = Math.max(1, Math.floor(rows.length * cfg.trainFrac));
      expect(ds.train.xs.shape[0] === nTrain * (1 + cfg.copies), `train rows ${ds.train.xs.shape[0]} != ${nTrain * (1 + cfg.copies)}`);
      expect(ds.val.xs.shape[0] + ds.test.xs.shape[0] === rows.length - nTrain, "val/test sizes changed by augmentation");
      expect(JSON.stringify(ds.meta.inputShape) === JSON.stringify([4, 4, 1]), `input shape ${ds.meta.inputShape}`);
      expect(ds.meta.image?.range[0] === 0 && ds.meta.image?.range[1] === 1, "image spec range missing");
      expect(ds.meta.classNames?.length === 2 && ds.meta.task === "binary", `classes ${ds.meta.classNames} task ${ds.meta.task}`);
      // Every value is a normalised pixel, original or augmented.
      for (let i = 0; i < ds.train.xs.size; i++) {
        const v = ds.train.xs.data[i]!;
        expect(v >= -1.01 && v <= 1.01, `pixel ${v} outside the normalised range`);
      }
      // Augmented copies differ from the clean row (otherwise augmentation did nothing).
      let changed = 0;
      for (let i = 0; i < 16; i++) if (Math.abs(ds.train.xs.data[16 + i]! - ds.train.xs.data[i]!) > 1e-6) changed++;
      expect(changed > 0, "augmented copy is identical to the original");
      // Single-class mode becomes a reconstruction dataset.
      const auto = buildImageDataset(rows.map((r) => ({ ...r, label: "all" })), { ...cfg, copies: 0, labelMode: "all" });
      expect(auto.meta.reconstruction === true && auto.meta.task === "regression", "single class should build an autoencoder dataset");
      expect(auto.meta.outputUnits === 16, `autoencoder output ${auto.meta.outputUnits}`);
      // [-1, 1] scaling is applied to the stored tensors.
      const pm = buildImageDataset(rows, { ...cfg, copies: 0, scale: "pm1" });
      let sawNegative = false;
      for (let i = 0; i < pm.train.xs.size; i++) if (pm.train.xs.data[i]! < 0) sawNegative = true;
      expect(sawNegative, "[-1, 1] scaling did not produce negative values");
    }),
  );

  tests.push(
    run("CSV: streaming parser matches the string parser at every chunk boundary", () => {
      const text =
        "\uFEFFname,age,note\r\n" +
        '"Smith, John",42,"line one\nline two"\r\n' +
        'plain,7,"he said ""hi"""\n' +
        "\n" +
        'trailing,,last\r';
      const whole = parseCsv(text);
      expect(whole.headers.length === 3 && whole.rows.length === 3, `parsed ${whole.headers.length} headers / ${whole.rows.length} rows`);
      expect(whole.rows[0]![0] === "Smith, John", `quoted comma: ${whole.rows[0]![0]}`);
      expect(whole.rows[0]![2] === "line one\nline two", `embedded newline: ${JSON.stringify(whole.rows[0]![2])}`);
      expect(whole.rows[1]![2] === 'he said "hi"', `escaped quotes: ${whole.rows[1]![2]}`);
      for (let cut = 0; cut <= text.length; cut++) {
        const parser = new CsvParser();
        parser.push(text.slice(0, cut));
        parser.push(text.slice(cut));
        const split = parser.finish();
        expect(JSON.stringify(split) === JSON.stringify(whole), `chunk boundary ${cut} changed the result`);
      }
      // A carriage return that lands at the very end of one chunk is not a row terminator yet.
      const parser = new CsvParser();
      parser.push("a,b\r");
      parser.push("\nc,d\n");
      const crlf = parser.finish();
      expect(crlf.rows.length === 1 && crlf.rows[0]![0] === "c", `split CRLF produced ${JSON.stringify(crlf.rows)}`);
      // Row cap.
      const capped = parseCsv("h\n" + Array.from({ length: 50 }, (_, i) => String(i)).join("\n"), 10);
      expect(capped.rows.length === 10, `cap kept ${capped.rows.length} rows`);
    }),
  );

  tests.push(
    run("standalone export: BPE tokenizer + sampled generation parity", () => {
      const ds = generateBuiltin("tiny_text_bpe", { ...DEFAULT_SYNTH, seed: 2 });
      const tok = ds.meta.tokenizer!;
      const model = new SequentialModel(alignToVocab(PRESETS.find((p) => p.id === "bpe-lstm")!.arch, tok.tokens.length, tok.seqLen));
      const { bundle } = buildStandaloneBundle(model, ds, { name: "bpe-lm", project: "t", loss: "sparse_cce", trained: false });
      expect(bundle.tokenizer?.level === "bpe", `bundle tokenizer level ${bundle.tokenizer?.level}`);
      expect((bundle.tokenizer?.merges?.length ?? 0) === (tok.merges?.length ?? 0), "merge rules missing from the bundle");
      const rt = loadRuntime().createModel(bundle);
      const texts = ["hello there, how are you today?", "the quick brown fox!", "zyxqv neuronforge 42!", "MiXeD Case  and\nnewlines"];
      for (const text of texts) {
        const a = tokenizeForModel(tok, text);
        const b = rt.tokenize(text);
        expect(JSON.stringify(a) === JSON.stringify(b), `tokenizer mismatch on ${JSON.stringify(text)}: ${a.length} vs ${b.length}`);
        expect(detokenize(tok, a) === rt.detokenize(a), `detokenizer mismatch on ${JSON.stringify(text)}`);
      }
      const settings = { maxTokens: 24, temperature: 0.7, topK: 20, seed: 5 };
      const engine = generateTokens(model, tok, "hello there", settings);
      const runtime = rt.generateDetailed("hello there", settings);
      expect(JSON.stringify(engine.ids) === JSON.stringify(runtime.ids), "sampled ids differ between the engine and the exported runtime");
      expect(engine.text === runtime.text, "detokenized generation differs");
      const report = verifyStandalone(bundle, model, ds);
      const failed = report.checks.filter((c) => c.status === "fail");
      expect(report.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
    }),
  );

  tests.push(
    run("standalone export: CSV preprocessing (categorical + missing) reproduced", () => {
      const csv =
        "age,city,income,label\n25,paris,30000,0\n32,berlin,,1\n47,paris,52000,1\n51,rome,61000,1\n23,berlin,28000,0\n" +
        "38,rome,45000,0\n29,paris,33000,0\n44,berlin,58000,1\n36,rome,,1\n27,paris,31000,0\n41,berlin,50000,1\n22,rome,26000,0\n";
      const { headers, rows } = parseCsv(csv);
      const { dataset } = materializeCsv(headers, rows, {
        ...DEFAULT_CSV_CONFIG,
        featureIdx: [0, 1, 2],
        targetIdx: [3],
        missing: "mean",
        categorical: "onehot",
        normalize: "zscore",
        seed: 3,
      });
      expect(!!dataset.meta.preprocessing && dataset.meta.preprocessing.samples.length > 0, "preprocessing not recorded");
      const arch: Architecture = {
        inputShape: dataset.meta.inputShape,
        inputName: "x",
        seed: 5,
        initMethod: "he_normal",
        layers: [
          makeLayerDef("dense", { id: "h", name: "h", config: { units: 4, activation: "relu" } }),
          makeLayerDef("dense", { id: "o", name: "o", config: { units: 1 } }),
          makeLayerDef("sigmoid", { id: "s", name: "s" }),
        ],
      };
      const model = new SequentialModel(arch);
      const { bundle } = buildStandaloneBundle(model, dataset, { name: "csv", project: "t", loss: "bce", trained: false });
      expect(!!bundle.preprocessing, "preprocessing not exported");
      const r = verifyStandalone(bundle, model, dataset);
      const failed = r.checks.filter((c) => c.status === "fail");
      expect(r.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
      // A raw row with a missing value and an unseen category must still encode.
      const rt = loadRuntime().createModel(bundle);
      const enc = rt.encode({ age: "30", city: "tokyo", income: "" });
      expect(enc.length === dataset.meta.inputShape[0], `encoded ${enc.length} features`);
      expect(enc.every((v) => Number.isFinite(v)), "encoded row contains non-finite values");
    }),
  );

  tests.push(
    run("LSTM BPTT matches numerical gradient", () => {
      const layer = compileLayer(
        mkDef("lstm", { id: "l", name: "l", config: { units: 3, returnSequences: false, forgetBias: 1 } }),
        [4, 2],
        new RNG(21),
        "xavier_uniform",
      );
      const x = new Tensor([1, 4, 2], [0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.05, 0.2]);
      const w = [0.5, -1, 2];
      const scalar = () => {
        const y = layer.forward(x, false);
        return y.data[0]! * w[0]! + y.data[1]! * w[1]! + y.data[2]! * w[2]!;
      };
      layer.forward(x, false);
      layer.backward(new Tensor([1, 3], w));
      const eps = 1e-3;
      // One weight in the input projection and one in the recurrent projection
      // (the recurrent one exercises the cell-state gradient path).
      for (const [name, index] of [
        ["Wx", 6],
        ["Wh", 4],
      ] as const) {
        const p = layer.params[name]!;
        const analytic = layer.grads[name]!.data[index]!;
        const orig = p.data[index]!;
        p.data[index] = orig + eps;
        const lp = scalar();
        p.data[index] = orig - eps;
        const lm = scalar();
        p.data[index] = orig;
        const numeric = (lp - lm) / (2 * eps);
        expect(approx(analytic, numeric, 5e-2), `${name}[${index}]: analytic ${analytic} numeric ${numeric}`);
      }
    }),
  );

  tests.push(
    run("GRU BPTT matches numerical gradient", () => {
      const layer = compileLayer(
        mkDef("gru", { id: "g", name: "g", config: { units: 3, returnSequences: true } }),
        [4, 2],
        new RNG(31),
        "xavier_uniform",
      );
      const x = new Tensor([1, 4, 2], [0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.05, 0.2]);
      const w = new Float32Array(12).map((_, i) => (i % 3) - 1);
      const scalar = () => {
        const y = layer.forward(x, false);
        let acc = 0;
        for (let i = 0; i < y.size; i++) acc += y.data[i]! * w[i]!;
        return acc;
      };
      layer.forward(x, false);
      layer.backward(new Tensor([1, 4, 3], w));
      const eps = 1e-3;
      for (const [name, index] of [
        ["Wxz", 4],
        ["Whn", 2],
      ] as const) {
        const p = layer.params[name]!;
        const analytic = layer.grads[name]!.data[index]!;
        const orig = p.data[index]!;
        p.data[index] = orig + eps;
        const lp = scalar();
        p.data[index] = orig - eps;
        const lm = scalar();
        p.data[index] = orig;
        const numeric = (lp - lm) / (2 * eps);
        expect(approx(analytic, numeric, 5e-2), `${name}[${index}]: analytic ${analytic} numeric ${numeric}`);
      }
    }),
  );

  tests.push(
    run("residual dense block reduces to the identity when its branch is zero", () => {
      const layer = compileLayer(
        mkDef("residual_dense", { id: "rd", name: "rd", config: { units: 4, activation: "relu", useBias: true, projection: "auto", scale: 1, finalActivation: "none" } }),
        [4],
        new RNG(41),
        "xavier_uniform",
      );
      for (const p of Object.values(layer.params)) p.data.fill(0);
      const x = new Tensor([2, 4], [0.5, -1, 2, 0.25, -0.5, 0.75, 1.5, -2]);
      const y = layer.forward(x, false);
      let worst = 0;
      for (let i = 0; i < x.size; i++) worst = Math.max(worst, Math.abs(y.data[i]! - x.data[i]!));
      expect(worst < 1e-7, `identity shortcut broke: worst |y-x| = ${worst}`);
      // Gradient of the shortcut path: every input must receive at least the
      // shortcut's contribution.
      layer.backward(new Tensor([2, 4], [1, 0, 0, 0, 0, 1, 0, 0]));
      expect(approx(layer.grads.W1!.data[0]!, 0, 1e-6) && approx(layer.grads.W2!.data[0]!, 0, 1e-6), "zero branch should have zero weight gradients");
    }),
  );

  tests.push(
    run("residual conv identity shortcut preserves the input", () => {
      const layer = compileLayer(
        mkDef("residual_conv", {
          id: "rc",
          name: "rc",
          config: { filters: 2, kernelSize: 3, stride: 1, padding: "same", activation: "relu", useBias: true, projection: "auto", scale: 1, finalActivation: "none" },
        }),
        [5, 5, 2],
        new RNG(51),
        "xavier_uniform",
      );
      for (const p of Object.values(layer.params)) p.data.fill(0);
      expect(layer.params.Kp === undefined, "a same-shape stride-1 block must use the identity shortcut (no projection kernel)");
      const data = new Float32Array(2 * 5 * 5 * 2);
      for (let i = 0; i < data.length; i++) data[i] = Math.sin(i) * 0.5;
      const x = new Tensor([2, 5, 5, 2], data);
      const y = layer.forward(x, false);
      let worst = 0;
      for (let i = 0; i < x.size; i++) worst = Math.max(worst, Math.abs(y.data[i]! - x.data[i]!));
      expect(worst < 1e-7, `identity shortcut broke: worst |y-x| = ${worst}`);
    }),
  );

  tests.push(
    run("attention: causal mask really hides the future", () => {
      const layer = compileLayer(
        mkDef("attention", { id: "a", name: "a", config: { heads: 2, dModel: 4, causal: true, pool: "none" } }),
        [5, 4],
        new RNG(3),
        "xavier_uniform",
      );
      const base = new Float32Array(5 * 4);
      for (let i = 0; i < base.length; i++) base[i] = Math.sin(i * 1.7) * 0.6;
      const y1 = layer.forward(new Tensor([1, 5, 4], new Float32Array(base)), false);
      // Change only the last timestep: earlier positions must be untouched.
      const perturbed = new Float32Array(base);
      for (let j = 0; j < 4; j++) perturbed[4 * 4 + j] = perturbed[4 * 4 + j]! + 3.5;
      layer.resetCache();
      const y2 = layer.forward(new Tensor([1, 5, 4], perturbed), false);
      let worstPast = 0;
      for (let i = 0; i < 4 * 4; i++) worstPast = Math.max(worstPast, Math.abs(y1.data[i]! - y2.data[i]!));
      expect(worstPast < 1e-6, `causal attention leaked the future: worst |Δ| at a past position = ${worstPast}`);
      let lastChanged = 0;
      for (let j = 0; j < 4; j++) lastChanged = Math.max(lastChanged, Math.abs(y1.data[16 + j]! - y2.data[16 + j]!));
      expect(lastChanged > 1e-6, "the final position should react to its own (visible) change");

      const bi = compileLayer(
        mkDef("attention", { id: "b", name: "b", config: { heads: 2, dModel: 4, causal: false, pool: "none" } }),
        [5, 4],
        new RNG(3),
        "xavier_uniform",
      );
      bi.forward(new Tensor([1, 5, 4], new Float32Array(base)), false);
      const b1 = bi.forward(new Tensor([1, 5, 4], new Float32Array(base)), false);
      bi.resetCache();
      const b2 = bi.forward(new Tensor([1, 5, 4], perturbed), false);
      let changedPast = 0;
      for (let i = 0; i < 4 * 4; i++) changedPast = Math.max(changedPast, Math.abs(b1.data[i]! - b2.data[i]!));
      expect(changedPast > 1e-6, "bidirectional attention should let the future influence earlier positions");
    }),
  );

  tests.push(
    run("attention gradients match numerical gradients", () => {
      const layer = compileLayer(
        mkDef("attention", { id: "ag", name: "ag", config: { heads: 2, dModel: 4, causal: true, pool: "none" } }),
        [4, 4],
        new RNG(21),
        "xavier_uniform",
      );
      const x = new Tensor([2, 4, 4], Float32Array.from([0.2, -0.4, 0.7, 0.1, -0.3, 0.5, 0.15, -0.6, 0.9, 0.35, -0.25, 0.05, 0.45, -0.8, 0.6, 0.3,
                                                            -0.15, 0.55, 0.25, -0.35, 0.65, 0.1, -0.5, 0.4, 0.3, -0.2, 0.75, 0.05, -0.45, 0.35, 0.2, -0.1]));
      const dOut = new Tensor([2, 4, 4], Float32Array.from({ length: 32 }, (_, i) => Math.cos(i * 1.3)));
      const scalar = () => {
        const y = layer.forward(x, false);
        let s = 0;
        for (let i = 0; i < y.size; i++) s += y.data[i]! * dOut.data[i]!;
        return s;
      };
      layer.forward(x, false);
      layer.backward(dOut);
      const eps = 1e-3;
      const checkParam = (name: string, index: number) => {
        const t = layer.params[name]!;
        const analytic = layer.grads[name]!.data[index]!;
        const orig = t.data[index]!;
        t.data[index] = orig + eps;
        const lp = scalar();
        t.data[index] = orig - eps;
        const lm = scalar();
        t.data[index] = orig;
        const numeric = (lp - lm) / (2 * eps);
        expect(approx(analytic, numeric, 2e-2), `${name}[${index}]: analytic ${analytic} vs numeric ${numeric}`);
      };
      checkParam("Wq", 5);
      checkParam("Wk", 12);
      checkParam("Wv", 3);
      checkParam("Wo", 9);
      checkParam("bq", 2);
      checkParam("bo", 1);
    }),
  );

  tests.push(
    run("transformer block gradients match numerical gradients", () => {
      const layer = compileLayer(
        mkDef("transformer_block", { id: "tg", name: "tg", config: { heads: 2, dModel: 4, ffDim: 6, causal: true, pool: "last", activation: "gelu" } }),
        [3, 4],
        new RNG(31),
        "xavier_uniform",
      );
      const x = new Tensor([1, 3, 4], Float32Array.from([0.25, -0.5, 0.75, 0.1, -0.35, 0.45, 0.2, -0.6, 0.8, 0.3, -0.15, 0.05]));
      const dOut = new Tensor([1, 4], Float32Array.from([0.4, -0.7, 0.2, 0.9]));
      const scalar = () => {
        const y = layer.forward(x, false);
        let s = 0;
        for (let i = 0; i < y.size; i++) s += y.data[i]! * dOut.data[i]!;
        return s;
      };
      layer.forward(x, false);
      layer.backward(dOut);
      const eps = 1e-3;
      const names = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "b1", "b2"];
      for (const name of names) {
        const t = layer.params[name]!;
        const index = Math.min(t.size - 1, 3);
        const analytic = layer.grads[name]!.data[index]!;
        const orig = t.data[index]!;
        t.data[index] = orig + eps;
        const lp = scalar();
        t.data[index] = orig - eps;
        const lm = scalar();
        t.data[index] = orig;
        const numeric = (lp - lm) / (2 * eps);
        expect(approx(analytic, numeric, 3e-2), `${name}[${index}]: analytic ${analytic} vs numeric ${numeric}`);
      }
    }),
  );

  tests.push(
    run("attention: head count must divide the model width", () => {
      let threw = false;
      try {
        compileLayer(mkDef("attention", { id: "z", name: "z", config: { heads: 5, dModel: 6 } }), [4, 6], new RNG(1), "xavier_uniform");
      } catch (e) {
        threw = /divisible/.test(e instanceof Error ? e.message : String(e));
      }
      expect(threw, "6-wide attention with 5 heads must be rejected");
    }),
  );

  tests.push(
    run("transformer LM training step lowers the loss", () => {
      const chars = 6;
      const seq = 4;
      const arch: Architecture = {
        inputShape: [seq],
        inputName: "tokens",
        seed: 5,
        initMethod: "xavier_uniform",
        layers: [
          mkDef("embedding", { id: "e", name: "e", config: { vocabSize: chars, embeddingDim: 8 } }),
          mkDef("transformer_block", { id: "t", name: "t", config: { heads: 2, dModel: 8, ffDim: 16, causal: true, pool: "last", activation: "gelu" } }),
          mkDef("dense", { id: "d", name: "d", config: { units: chars } }),
          mkDef("softmax", { id: "s", name: "s" }),
        ],
      };
      const model = new SequentialModel(arch);
      expect(model.valid, model.issues[0]?.message ?? "invalid");
      // A toy next-token task: predict the last token of a repeating cycle.
      const xs: number[][] = [];
      const ys: number[][] = [];
      for (let i = 0; i < 24; i++) {
        const start = i % chars;
        const row = Array.from({ length: seq }, (_, t) => (start + t) % chars);
        xs.push(row);
        ys.push([row[seq - 1]!]);
      }
      const X = new Tensor([xs.length, seq], Float32Array.from(xs.flat()));
      const Y = new Tensor([ys.length, 1], Float32Array.from(ys.flat()));
      const opt = { ...DEFAULT_OPTIMIZER, name: "adam" as const, learningRate: 0.02 };
      const state = createOptimizerState();
      const lossOf = () => computeLoss("sparse_cce", model.forward(X, false).output, Y).value;
      const before = lossOf();
      for (let step = 0; step < 8; step++) {
        const { value, grad } = computeLoss("sparse_cce", model.forward(X, true).output, Y);
        expect(Number.isFinite(value), `loss became ${value}`);
        model.backward(grad);
        optimizerStep(model, opt, state);
      }
      const after = lossOf();
      expect(after < before * 0.9, `loss did not fall: ${before.toFixed(4)} → ${after.toFixed(4)}`);
    }),
  );

  tests.push(
    run("deep reasoning gradients match numerical gradients", () => {
      const layer = compileLayer(
        mkDef("reasoning_block", {
          id: "rg",
          name: "rg",
          config: { steps: 4, hidden: 6, adaptive: false, activation: "gelu", stepCode: true, stepCodeScale: 1, epsilon: 1e-5 },
        }),
        [4],
        new RNG(71),
        "xavier_uniform",
      );
      const x = new Tensor([2, 4], Float32Array.from([0.2, -0.4, 0.7, 0.1, -0.3, 0.5, 0.15, -0.6]));
      const dOut = new Tensor([2, 4], Float32Array.from([0.4, -0.7, 0.2, 0.9, -0.2, 0.3, 0.5, -0.1]));
      const scalar = () => {
        const y = layer.forward(x, false);
        let s = 0;
        for (let i = 0; i < y.size; i++) s += y.data[i]! * dOut.data[i]!;
        return s;
      };
      layer.forward(x, false);
      layer.backward(dOut);
      const eps = 1e-3;
      for (const name of ["W1", "W2", "Wgate", "ln_gamma", "ln_beta", "b1", "b2", "bgate"]) {
        const t = layer.params[name]!;
        const index = Math.min(t.size - 1, Math.floor(t.size / 2));
        const analytic = layer.grads[name]!.data[index]!;
        const orig = t.data[index]!;
        t.data[index] = orig + eps;
        const lp = scalar();
        t.data[index] = orig - eps;
        const lm = scalar();
        t.data[index] = orig;
        const numeric = (lp - lm) / (2 * eps);
        expect(approx(analytic, numeric, 3e-2), `${name}[${index}]: analytic ${analytic} vs numeric ${numeric}`);
      }
    }),
  );

  tests.push(
    run("deep reasoning: depth changes the output and adaptive halting really stops", () => {
      const short = compileLayer(
        mkDef("reasoning_block", { id: "rs", name: "rs", config: { steps: 3, hidden: 6, adaptive: false, stepCode: true } }),
        [5],
        new RNG(17),
        "xavier_uniform",
      );
      const long = compileLayer(
        mkDef("reasoning_block", { id: "rl", name: "rl", config: { steps: 7, hidden: 6, adaptive: false, stepCode: true } }),
        [5],
        new RNG(17),
        "xavier_uniform",
      );
      for (const k of Object.keys(short.params)) long.params[k]!.data.set(short.params[k]!.data);
      const x = new Tensor([1, 5], Float32Array.from([0.2, -0.5, 0.9, 0.1, -0.3]));
      const a = short.forward(x, false);
      const b = long.forward(x, false);
      let diff = 0;
      for (let i = 0; i < a.size; i++) diff = Math.max(diff, Math.abs(a.data[i]! - b.data[i]!));
      expect(diff > 1e-3, `extra reasoning depth did not change the output (max |Δ| ${diff})`);
      expect(short.reasoning?.trace?.stepsUsed === 3, `short trace ${short.reasoning?.trace?.stepsUsed}`);
      expect(long.reasoning?.trace?.stepsUsed === 7, `long trace ${long.reasoning?.trace?.stepsUsed}`);

      const adaptive = compileLayer(
        mkDef("reasoning_block", { id: "ra", name: "ra", config: { steps: 8, hidden: 6, adaptive: true, haltThreshold: 0.9, minSteps: 2, stepCode: true } }),
        [5],
        new RNG(19),
        "xavier_uniform",
      );
      adaptive.forward(x, false);
      const used = adaptive.reasoning?.trace?.stepsUsed ?? 0;
      expect(used >= 2 && used < 8, `adaptive halting should stop before the full depth, used ${used}`);
      expect(adaptive.reasoning?.trace?.halted === true, "adaptive trace did not report halting");
      adaptive.forward(x, true);
      expect(adaptive.reasoning?.trace?.stepsUsed === 8, "training forwards must run the full depth");
    }),
  );

  tests.push(
    run("deep reasoner preset is ~40k parameters and really reasons", () => {
      const p = PRESETS.find((x) => x.id === "deep-reasoner-40k");
      expect(!!p, "deep-reasoner-40k preset missing");
      const m = new SequentialModel(p!.arch);
      expect(m.valid, m.issues[0]?.message ?? "invalid");
      const total = m.paramCount().trainable + m.paramCount().nonTrainable;
      expect(total >= 38_000 && total <= 42_000, `expected ~40k parameters, got ${total}`);
      expect(m.reasoningLayers().length === 1, "preset must contain one Deep Reasoning block");
      const inSize = p!.arch.inputShape.reduce((a, b) => a * b, 1);
      const x = new Tensor([1, ...p!.arch.inputShape], Float32Array.from({ length: inSize }, (_, i) => i % 256));
      const y = m.forward(x, false).output;
      for (let i = 0; i < y.size; i++) expect(Number.isFinite(y.data[i]!), `output ${i} is ${y.data[i]}`);
      const trace = m.reasoningLayers()[0]!.trace;
      expect(!!trace && trace.stepsUsed > 1, "preset forward did not record a reasoning trace");
    }),
  );

  tests.push(
    run("reasoning corpus has explicit chain-of-thought and a closed tokenizer", () => {
      const { text, stats } = buildReasonCorpus(1);
      expect(text.includes("think:") && text.includes("a:"), "reasoning corpus lost its thought/answer format");
      expect(stats.lines > 800, `only ${stats.lines} reasoning lines`);
      const line = text.split("\n").find((l) => l.includes("think:"))!;
      const thinkAt = line.indexOf("think:");
      const answerAt = line.indexOf("a:");
      expect(thinkAt >= 0 && answerAt > thinkAt, "answer must follow the thought trace");
      const tok = buildWordTokenizer(text, 256, REASON_SEQ_LEN);
      expect(tok.vocabSize >= 100 && tok.vocabSize < 256, `word vocab ${tok.vocabSize}`);
      expect(tok.oovRate === 0, `reasoning corpus has ${(tok.oovRate * 100).toFixed(1)}% unknown words`);
      const ids = tokenizeWords(text.slice(0, 200)).map((w) => tok.tokenToId[w] ?? 0);
      expect(ids.every((id) => id >= 0 && id < tok.vocabSize), "token ids outside the vocabulary");
      const longest = Math.max(...text.split("\n").map((l) => tokenizeWords(l).length));
      expect(longest <= REASON_SEQ_LEN, `longest trace is ${longest} words, context is ${REASON_SEQ_LEN}`);
    }),
  );

  tests.push(
    run("native export round-trip preserves the deep reasoning plan and weights", () => {
      const p = PRESETS.find((x) => x.id === "deep-reasoner-40k")!;
      const model = new SequentialModel(p.arch);
      const x = new Tensor([1, ...p.arch.inputShape], Float32Array.from({ length: p.arch.inputShape.reduce((a, b) => a * b, 1) }, (_, i) => i % 148));
      const before = model.forward(x, false).output;
      const exported = exportModel(model, { kind: "model", metadata: { reasoning: model.reasoningLayers().map((l) => l.info.plan) } });
      const { model: restored, warnings } = importModel(exported);
      expect(warnings.length === 0, warnings.join("; "));
      expect(restored.reasoningLayers().length === 1, "reasoning layer missing after import");
      const after = restored.forward(x, false).output;
      let worst = 0;
      for (let i = 0; i < before.size; i++) worst = Math.max(worst, Math.abs(before.data[i]! - after.data[i]!));
      expect(worst < 1e-6, `round-trip output differs by ${worst}`);
      expect((restored.reasoningLayers()[0]!.trace?.stepsUsed ?? 0) === 8, "imported model did not execute its 8 reasoning steps");
    }),
  );

  tests.push(
    run("deep reasoning survives standalone export with an identical trace", () => {
      const arch: Architecture = {
        inputShape: [6],
        inputName: "x",
        seed: 23,
        initMethod: "xavier_uniform",
        layers: [
          mkDef("dense", { id: "d1", name: "d1", config: { units: 6 } }),
          mkDef("reasoning_block", { id: "r1", name: "r1", config: { steps: 5, hidden: 8, adaptive: true, haltThreshold: 0.95, minSteps: 2, stepCode: true } }),
          mkDef("dense", { id: "d2", name: "d2", config: { units: 3 } }),
        ],
      };
      const model = new SequentialModel(arch);
      const { bundle } = buildStandaloneBundle(model, null, { name: "reason", project: "t", loss: "mse", trained: false });
      expect(!!bundle.reasoning && bundle.reasoning.layers.length === 1, "bundle does not carry reasoning metadata");
      expect(bundle.tests.reasoning.length > 0, "no reasoning test vectors");
      const rep = verifyStandalone(bundle, model, null);
      const failed = rep.checks.filter((c) => c.status === "fail");
      expect(rep.ok, failed.map((c) => `${c.label}: ${c.detail}`).join(" | "));
      const rc = rep.checks.find((c) => c.id === "reasoning");
      expect(rc?.status === "pass", `reasoning check: ${rc?.status} ${rc?.detail}`);
      const rt = loadRuntime().createModel(bundle);
      const tc = bundle.tests.reasoning[0]!;
      const got = rt.reason(tc.input);
      expect(got.traces.length === 1 && got.traces[0]!.stepsUsed === tc.traces[0]!.stepsUsed, "exported reason() depth differs");
      let worst = 0;
      for (let i = 0; i < got.data.length; i++) worst = Math.max(worst, Math.abs(got.data[i]! - tc.output[i]!));
      expect(worst < 1e-6, `exported reasoning output differs by ${worst}`);
    }),
  );

  tests.push(
    run("attention: export coverage includes the new layers", () => {
      const names = COVERAGE_CASES.map((c) => c.name).join(" | ");
      expect(/Attention/.test(names) && /Transformer block/.test(names) && /Deep Reasoning/.test(names), `coverage cases missing attention/reasoning: ${names.slice(0, 160)}`);
    }),
  );

  tests.push(
    run("residual blocks reject impossible shortcuts", () => {
      let threw = false;
      try {
        compileLayer(mkDef("residual_dense", { id: "x", name: "x", config: { units: 3, projection: "never" } }), [4], new RNG(1), "xavier_uniform");
      } catch (e) {
        threw = /identity shortcut/.test(e instanceof Error ? e.message : String(e));
      }
      expect(threw, "residual dense with units != input width and projection never should throw");
      threw = false;
      try {
        compileLayer(
          mkDef("residual_conv", { id: "y", name: "y", config: { filters: 4, kernelSize: 3, stride: 2, padding: "valid", projection: "auto" } }),
          [5, 5, 2],
          new RNG(1),
          "xavier_uniform",
        );
      } catch (e) {
        threw = /padding/.test(e instanceof Error ? e.message : String(e));
      }
      expect(threw, "residual conv with valid padding + stride 2 cannot build a matching shortcut");
    }),
  );

  return tests;
}

/**
 * Checks that need to await the accelerator. Kept separate from the synchronous
 * suite so the panel can run the fast tests first and stream these in after.
 */
export async function runAsyncSelfTests(): Promise<TestResult[]> {
  const tests: TestResult[] = [];
  const ops = new CpuOps();

  tests.push(
    await (async () => {
      const name = "CSV: file streaming reads progress and stops at the cap";
      const t0 = performance.now();
      try {
        const text = "a,b\n" + Array.from({ length: 4000 }, (_, i) => `${i},${i * 2}`).join("\n") + "\n";
        const bytes = new TextEncoder().encode(text);
        // Each call gets a fresh stream, so the cap can be tested on the same content.
        const file: FileLike = {
          size: bytes.length,
          stream: () => {
            let sent = 0;
            return new ReadableStream<Uint8Array>({
              pull(controller) {
                if (sent >= bytes.length) {
                  controller.close();
                  return;
                }
                const chunk = bytes.subarray(sent, sent + 977); // deliberately not a line boundary
                sent += chunk.length;
                controller.enqueue(chunk);
              },
            });
          },
          text: async () => text,
        };
        const progress: number[] = [];
        const parsed = await parseCsvFile(file, { onProgress: (stats) => progress.push(stats.rows) });
        expect(parsed.headers.length === 2 && !parsed.truncated, `streamed ${parsed.headers.length} headers, truncated=${parsed.truncated}`);
        expect(parsed.rows.length === 4000, `streamed ${parsed.rows.length} rows`);
        expect(parsed.rows[3999]![1] === "7998", `last row ${JSON.stringify(parsed.rows[3999])}`);
        expect(progress.length >= 2 && progress[progress.length - 1] === 4000, `progress callbacks: ${progress.slice(0, 5)}`);
        // Cap: reading stops early instead of buffering the whole file.
        const limited = await parseCsvFile(file, { maxRows: 100 });
        expect(limited.rows.length === 100 && limited.truncated, `cap not honoured: ${limited.rows.length} rows, truncated=${limited.truncated}`);
        expect(limited.bytes < bytes.length, `cap should stop reading early (read ${limited.bytes} of ${bytes.length} bytes)`);
        return { name, passed: true, message: `streamed 4,000 rows in 977-byte chunks, cap stopped after ${limited.rows.length} rows`, ms: performance.now() - t0 };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      try {
        const checks = await verifyDevice(ops);
        const summary = summarizeChecks(checks);
        expect(summary.failed === 0, `${summary.failed} of ${checks.length} device ops differ from the reference`);
        return { name: `device op surface (${checks.length} ops)`, passed: true, message: `worst |diff| ${summary.worst}`, ms: performance.now() - t0 };
      } catch (e) {
        return { name: "device op surface", passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      const name = "accelerator forward/backward matches the CPU engine";
      try {
        const arch: Architecture = {
          inputShape: [16],
          inputName: "x",
          seed: 7,
          initMethod: "he_normal",
          layers: [
            makeLayerDef("reshape", { id: "rs", name: "rs", config: { targetShape: [4, 4, 1] } }),
            makeLayerDef("conv2d", { id: "c1", name: "c1", config: { filters: 3, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "relu" } }),
            makeLayerDef("residual_conv", { id: "rc", name: "rc", config: { filters: 4, kernelSize: 3, stride: 2, padding: "same", activation: "relu", projection: "auto", scale: 0.7 } }),
            makeLayerDef("maxpool2d", { id: "mp", name: "mp", config: { poolSize: 2, stride: 2 } }),
            makeLayerDef("gap2d", { id: "gap", name: "gap" }),
            makeLayerDef("dense", { id: "d1", name: "d1", config: { units: 3, useBias: true, activation: "softmax" } }),
          ],
        };
        const plain = new SequentialModel(arch);
        expect(plain.valid, plain.issues[0]?.message ?? "model invalid");
        // Two independent models with the same seed: `clone()` shares the
        // parameter tensors with its source, which would hide differences.
        const accelerated = new SequentialModel(arch);
        const acc = new Accelerator(accelerated, ops);
        const rng = new RNG(3);
        const bx = new Tensor([2, 16], Float32Array.from({ length: 32 }, () => rng.normal()));
        const dOut = new Tensor([2, 3], [0.5, -0.2, 0.1, -0.3, 0.7, 0.05]);
        const want = plain.forward(bx, false).output;
        const got = await acc.runForward(bx, false);
        let worstFwd = 0;
        for (let i = 0; i < want.size; i++) worstFwd = Math.max(worstFwd, Math.abs(want.data[i]! - got.data[i]!));
        expect(worstFwd < 1e-6, `forward differs by ${worstFwd}`);
        plain.zeroGrads();
        plain.backward(dOut);
        accelerated.zeroGrads();
        await acc.runForward(bx, true);
        await acc.runBackward(dOut);
        let worstGrad = 0;
        for (const layer of plain.layers) {
          const other = accelerated.layers.find((l) => l.def.id === layer.def.id)!;
          for (const key of Object.keys(layer.grads)) {
            const a = layer.grads[key]!.data;
            const b = other.grads[key]!.data;
            for (let i = 0; i < a.length; i++) worstGrad = Math.max(worstGrad, Math.abs(a[i]! - b[i]!));
          }
        }
        acc.destroy();
        expect(worstGrad < 1e-6, `gradients differ by ${worstGrad}`);
        return { name, passed: true, message: `forward and gradients agree (≤ ${Math.max(worstFwd, worstGrad).toExponential(1)})`, ms: performance.now() - t0 };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      const name = "accelerator: attention layers fall back to the CPU and stay exact";
      try {
        const arch: Architecture = {
          inputShape: [4],
          inputName: "tokens",
          seed: 19,
          initMethod: "xavier_uniform",
          layers: [
            makeLayerDef("embedding", { id: "e", name: "e", config: { vocabSize: 5, embeddingDim: 6 } }),
            makeLayerDef("transformer_block", { id: "t", name: "t", config: { heads: 2, dModel: 6, ffDim: 8, causal: true, pool: "last", activation: "gelu" } }),
            makeLayerDef("dense", { id: "d", name: "d", config: { units: 5, activation: "softmax" } }),
          ],
        };
        const plain = new SequentialModel(arch);
        expect(plain.valid, plain.issues[0]?.message ?? "model invalid");
        const accelerated = new SequentialModel(arch);
        const acc = new Accelerator(accelerated, ops);
        const bx = new Tensor([2, 4], Float32Array.from([1, 2, 3, 4, 4, 3, 2, 0]));
        const dOut = new Tensor([2, 5], [0.2, -0.4, 0.1, 0.05, -0.15, -0.3, 0.6, -0.1, 0.25, 0.05]);
        const want = plain.forward(bx, false).output;
        const got = await acc.runForward(bx, false);
        let worstFwd = 0;
        for (let i = 0; i < want.size; i++) worstFwd = Math.max(worstFwd, Math.abs(want.data[i]! - got.data[i]!));
        expect(worstFwd < 1e-6, `forward differs by ${worstFwd}`);
        plain.zeroGrads();
        plain.backward(dOut);
        accelerated.zeroGrads();
        await acc.runForward(bx, true);
        await acc.runBackward(dOut);
        let worstGrad = 0;
        for (const layer of plain.layers) {
          const other = accelerated.layers.find((l) => l.def.id === layer.def.id)!;
          for (const key of Object.keys(layer.grads)) {
            const a = layer.grads[key]!.data;
            const b = other.grads[key]!.data;
            for (let i = 0; i < a.length; i++) worstGrad = Math.max(worstGrad, Math.abs(a[i]! - b[i]!));
          }
        }
        const placements = acc.placements;
        const cpuLayers = placements.filter((pl) => pl.mode === "cpu").map((pl) => pl.type);
        acc.destroy();
        expect(cpuLayers.includes("transformer_block"), `expected the block on the CPU path, placements: ${placements.map((pl) => pl.type + ":" + pl.mode).join(", ")}`);
        expect(worstGrad < 1e-6, `gradients differ by ${worstGrad}`);
        return { name, passed: true, message: `forward and gradients agree (≤ ${Math.max(worstFwd, worstGrad).toExponential(1)}); ${cpuLayers.join(", ")} ran on the CPU`, ms: performance.now() - t0 };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      const name = "accelerator: deep reasoning falls back to the CPU and stays exact";
      try {
        const arch: Architecture = {
          inputShape: [6],
          inputName: "x",
          seed: 29,
          initMethod: "xavier_uniform",
          layers: [
            makeLayerDef("dense", { id: "d1", name: "d1", config: { units: 6 } }),
            makeLayerDef("reasoning_block", { id: "r1", name: "r1", config: { steps: 4, hidden: 7, adaptive: false, stepCode: true } }),
            makeLayerDef("dense", { id: "d2", name: "d2", config: { units: 3 } }),
          ],
        };
        const plain = new SequentialModel(arch);
        expect(plain.valid, plain.issues[0]?.message ?? "model invalid");
        const accelerated = new SequentialModel(arch);
        const acc = new Accelerator(accelerated, ops);
        const bx = new Tensor([2, 6], Float32Array.from([0.2, -0.4, 0.7, 0.1, -0.3, 0.5, 0.15, -0.6, 0.9, 0.35, -0.25, 0.05]));
        const dOut = new Tensor([2, 3], [0.2, -0.4, 0.1, -0.3, 0.6, 0.05]);
        const want = plain.forward(bx, false).output;
        const got = await acc.runForward(bx, false);
        let worstFwd = 0;
        for (let i = 0; i < want.size; i++) worstFwd = Math.max(worstFwd, Math.abs(want.data[i]! - got.data[i]!));
        expect(worstFwd < 1e-6, `forward differs by ${worstFwd}`);
        plain.zeroGrads();
        plain.backward(dOut);
        accelerated.zeroGrads();
        await acc.runForward(bx, true);
        await acc.runBackward(dOut);
        let worstGrad = 0;
        for (const layer of plain.layers) {
          const other = accelerated.layers.find((l) => l.def.id === layer.def.id)!;
          for (const key of Object.keys(layer.grads)) {
            const a = layer.grads[key]!.data;
            const b = other.grads[key]!.data;
            for (let i = 0; i < a.length; i++) worstGrad = Math.max(worstGrad, Math.abs(a[i]! - b[i]!));
          }
        }
        const placements = acc.placements;
        const cpuLayers = placements.filter((pl) => pl.mode === "cpu").map((pl) => pl.type);
        acc.destroy();
        expect(cpuLayers.includes("reasoning_block"), `expected reasoning on the CPU path, placements: ${placements.map((pl) => pl.type + ":" + pl.mode).join(", ")}`);
        expect(worstGrad < 1e-6, `gradients differ by ${worstGrad}`);
        return { name, passed: true, message: `forward and gradients agree (≤ ${Math.max(worstFwd, worstGrad).toExponential(1)}); ${cpuLayers.join(", ")} ran on the CPU`, ms: performance.now() - t0 };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      const name = "accelerator training step matches a pure CPU step";
      try {
        const arch: Architecture = {
          inputShape: [6],
          inputName: "x",
          seed: 9,
          initMethod: "xavier_uniform",
          layers: [
            makeLayerDef("dense", { id: "d", name: "d", config: { units: 5, useBias: true, activation: "relu" } }),
            makeLayerDef("residual_dense", { id: "rd", name: "rd", config: { units: 5, activation: "gelu", projection: "auto", scale: 1 } }),
            makeLayerDef("dense", { id: "out", name: "out", config: { units: 2, useBias: true, activation: "softmax" } }),
          ],
        };
        const plain = new SequentialModel(arch);
        const acceleratedModel = new SequentialModel(arch);
        const acc = new Accelerator(acceleratedModel, ops);
        const rng = new RNG(13);
        const bx = new Tensor([4, 6], Float32Array.from({ length: 24 }, () => rng.normal()));
        const by = new Tensor([4, 2], [1, 0, 0, 1, 1, 0, 0, 1]);
        const opt = { ...DEFAULT_OPTIMIZER, learningRate: 0.05 };
        const stateA = createOptimizerState();
        const stateB = createOptimizerState();
        for (let step = 0; step < 3; step++) {
          plain.zeroGrads();
          const outA = plain.forward(bx, true, false).output;
          const lossA = computeLoss("cce", outA, by);
          plain.backward(lossA.grad);
          optimizerStep(plain, opt, stateA);

          acceleratedModel.zeroGrads();
          await acc.syncParams();
          const outB = await acc.runForward(bx, true);
          const lossB = computeLoss("cce", outB, by);
          await acc.runBackward(lossB.grad);
          optimizerStep(acceleratedModel, opt, stateB);
          expect(approx(lossA.value, lossB.value, 1e-6), `step ${step}: loss ${lossA.value} vs ${lossB.value}`);
        }
        let worst = 0;
        for (const layer of plain.layers) {
          const other = acceleratedModel.layers.find((l) => l.def.id === layer.def.id)!;
          for (const key of Object.keys(layer.params)) {
            const a = layer.params[key]!.data;
            const b = other.params[key]!.data;
            for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
          }
        }
        acc.destroy();
        expect(worst < 1e-6, `weights diverged by ${worst} after 3 steps`);
        return { name, passed: true, message: `3 optimiser steps agree (≤ ${worst.toExponential(1)})`, ms: performance.now() - t0 };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  tests.push(
    await (async () => {
      const t0 = performance.now();
      const name = "WGSL kernel arithmetic matches the CPU reference";
      try {
        const ops = new CpuOps();
        const rng = new RNG(0xc0ffee);
        const fill = (n: number, scale = 1): Float32Array => {
          const out = new Float32Array(n);
          for (let i = 0; i < n; i++) out[i] = rng.normal() * scale;
          return out;
        };
        const host = async (data: Float32Array, shape: number[]) => ops.fromHost(data, shape);
        const worstOf = (a: Float32Array, b: Float32Array): number => {
          let worst = 0;
          for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
          return worst;
        };

        // matmul, including a size that straddles the 16×16 tile
        {
          const M = 17, N = 19, K = 23;
          const a = fill(M * K);
          const b = fill(K * N);
          const plan = matmulStep(M, N, K, false, false);
          const got = simMatmul(plan.params.toU32Array(), a, b);
          const want = await ops.toHost(await ops.matmul(await host(a, [M, K]), await host(b, [K, N]), false, false));
          expect(worstOf(want, got) < 1e-4, `matmul worst |diff| ${worstOf(want, got)}`);
          expect(
            plan.grid[0] * 16 >= N && plan.grid[1] * 16 >= M,
            `matmul grid ${plan.grid[0]}×${plan.grid[1]} does not cover ${M}×${N}`,
          );
        }

        // unary ops and every activation, forward and backward
        {
          const x = fill(37, 2);
          for (const [op, extra] of [
            ["neg", 0],
            ["abs", 0],
            ["square", 0],
            ["scale", 1.7],
            ["clip", 0],
          ] as const) {
            const step = unaryStep(x.length, op === "neg" ? 1 : op === "abs" ? 2 : op === "square" ? 3 : op === "scale" ? 4 : 5, extra, 0.9);
            const got = simUnary(step.params.toU32Array(), x);
            const want = await ops.toHost(await ops.unary(op, await host(x, [37]), extra, 0.9));
            expect(worstOf(want, got) < 1e-6, `unary ${op} worst |diff| ${worstOf(want, got)}`);
          }
          const dx = await host(x, [37]);
          for (const act of ["relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh"]) {
            const fwd = await ops.activationFwd(act, dx, 0.1);
            const wantY = await ops.toHost(fwd);
            const gotY = simUnary(unaryStep(x.length, activationCode(act), 0.1, 0).params.toU32Array(), x);
            expect(worstOf(wantY, gotY) < 1e-6, `activation ${act} worst |diff| ${worstOf(wantY, gotY)}`);

            const g = fill(37);
            const wantDx = await ops.toHost(await ops.activationBwd(act, dx, fwd, await host(g, [37]), 0.1));
            const gotDx = simUnaryBwd(unaryBwdStep(x.length, backwardCode(act), 0.1).params.toU32Array(), x, wantY, g);
            expect(worstOf(wantDx, gotDx) < 1e-5, `activation bwd ${act} worst |diff| ${worstOf(wantDx, gotDx)}`);
          }
        }

        // broadcasting (the four-axis stride plan) and softmax rows
        {
          const a = fill(12);
          const b = fill(4);
          for (const op of ["add", "sub", "mul", "div"] as const) {
            const plan = broadcastPlan([3, 4], [4]);
            const params = binaryStep(12, binaryCode(op), plan.out4, plan.a4, plan.b4).params.toU32Array();
            const got = simBinary(params, a, b);
            const want = await ops.toHost(await ops.binary(op, await host(a, [3, 4]), await host(b, [4])));
            expect(worstOf(want, got) < 1e-6, `binary ${op} worst |diff| ${worstOf(want, got)}`);
          }
          const x = fill(390, 2);
          const params = softmaxStep(130, 3).params.toU32Array();
          const got = simSoftmax(params, x);
          const want = await ops.toHost(await ops.softmaxLast(await host(x, [3, 130])));
          expect(worstOf(want, got) < 1e-6, `softmax worst |diff| ${worstOf(want, got)}`);
          const g = fill(390);
          const back = simSoftmaxBwd(softmaxBwdStep(130, 3).params.toU32Array(), want, g);
          const wantBack = await ops.toHost(await ops.softmaxBackward(await host(want, [3, 130]), await host(g, [3, 130])));
          expect(worstOf(wantBack, back) < 1e-6, `softmax backward worst |diff| ${worstOf(wantBack, back)}`);
        }

        // reducers and the global-average-pooling inverse
        {
          const x = fill(72);
          for (const mode of ["sum", "mean"] as const) {
            // [2,3,3,4] with group = 2: inner 4, outer 72/4/2 = 9.
            const got = simReduceOuter(reduceOuterStep(9, 4, 2, mode).params.toU32Array(), x);
            const want = await ops.toHost(await ops.reduceOuter(mode, await host(x, [2, 3, 3, 4]), 2));
            expect(worstOf(want, got) < 1e-5, `reduceOuter ${mode} worst |diff| ${worstOf(want, got)}`);
          }
          const plane = fill(8);
          const geom = { B: 2, H: 3, W: 3, C: 4 };
          const got = simBroadcastPlanes(broadcastPlanesStep(72, 36, 4, 0.25).params.toU32Array(), plane);
          const want = await ops.toHost(await ops.broadcastPlanes(await host(plane, [2, 4]), geom, 0.25));
          expect(worstOf(want, got) < 1e-6, `broadcastPlanes worst |diff| ${worstOf(want, got)}`);
        }

        // convolution: im2col + matmul (forward) and matmul + col2im (input gradient)
        {
          const g = convGeomFor(2, 6, 6, 3, 4, { kernelSize: 3, stride: 1, padding: "same" });
          const x = fill(2 * 6 * 6 * 3);
          const k = fill(3 * 3 * 3 * 4);
          const col = simIm2col(im2colStep(g, 4 * 3 * 3 * 3).params.toU32Array(), x);
          const got = simMatmul(matmulStep(2 * g.outH * g.outW, 4, 27, false, false).params.toU32Array(), col, k);
          const want = await ops.toHost(await ops.conv2d(await host(x, [2, 6, 6, 3]), await host(k, [3, 3, 3, 4]), null, g));
          expect(worstOf(want, got) < 1e-4, `conv im2col+matmul worst |diff| ${worstOf(want, got)}`);

          const dy = fill(2 * g.outH * g.outW * 4);
          const dcol = simMatmul(matmulStep(2 * g.outH * g.outW, 27, 4, false, true).params.toU32Array(), dy, k);
          const gotDx = simCol2im(col2imStep(g, 2 * 6 * 6 * 3).params.toU32Array(), dcol);
          const wantDx = await ops.toHost(await ops.conv2dInputGrad(await host(dy, [2, g.outH, g.outW, 4]), await host(k, [3, 3, 3, 4]), g));
          expect(worstOf(wantDx, gotDx) < 1e-4, `conv matmul+col2im worst |diff| ${worstOf(wantDx, gotDx)}`);
        }

        // pooling forward/backward, including the overlapping-window case
        {
          const g = { B: 2, H: 5, W: 5, C: 2, pool: 2, stride: 1, outH: 4, outW: 4 };
          const x = fill(2 * 5 * 5 * 2, 2);
          for (const mode of ["max", "avg"] as const) {
            const got = simPoolFwd(poolFwdStep(mode, g, 64).params.toU32Array(), x);
            const want = await ops.toHost(await ops.pool2d(mode, await host(x, [2, 5, 5, 2]), g));
            expect(worstOf(want, got) < 1e-6, `pool ${mode} fwd worst |diff| ${worstOf(want, got)}`);

            const dy = fill(64);
            const gotB = simPoolBwd(poolBwdStep(mode, g, 50).params.toU32Array(), x, dy);
            const wantB = await ops.toHost(await ops.pool2dBackward(mode, await host(x, [2, 5, 5, 2]), await host(dy, [2, 4, 4, 2]), g));
            expect(worstOf(wantB, gotB) < 1e-6, `pool ${mode} bwd worst |diff| ${worstOf(wantB, gotB)}`);
          }
        }

        // embedding gather/scatter with duplicate and out-of-range ids
        {
          const w = fill(36);
          const ids = Float32Array.from([-3, 8, 3, 3, 99, 1]);
          const got = simGather(gatherStep(24, 4, 9).params.toU32Array(), w, ids);
          const want = await ops.toHost(await ops.gatherRows(await host(w, [9, 4]), await host(ids, [6]), 4));
          expect(worstOf(want, got) < 1e-6, `gather worst |diff| ${worstOf(want, got)}`);

          const grad = fill(24);
          const dup = Float32Array.from([2, 2, 0, 8, 5, 2]);
          const gotS = simScatter(scatterStep(36, 4, 9, 6).params.toU32Array(), grad, dup);
          const wantS = await ops.toHost(await ops.scatterRows(await host(grad, [6, 4]), await host(dup, [6]), 4, 9));
          expect(worstOf(wantS, gotS) < 1e-6, `scatter worst |diff| ${worstOf(wantS, gotS)}`);
        }

        return {
          name,
          passed: true,
          message: "every kernel reproduces the CPU reference on the parameters the device sends",
          ms: performance.now() - t0,
        };
      } catch (e) {
        return { name, passed: false, message: e instanceof Error ? e.message : String(e), ms: performance.now() - t0 };
      }
    })(),
  );

  return tests;
}
