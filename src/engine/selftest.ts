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
import { parseCsv, DEFAULT_CSV_CONFIG } from "../data/csv";
import { materializeCsv } from "../data/preprocess";
import { runLayerCoverage, buildStandaloneBundle, verifyStandalone, loadRuntime } from "../export/standalone";
import { compileExpr } from "../data/expr";
import { buildCustom } from "../data/custom";
import { PRESETS, alignToVocab } from "./presets";
import { compileLayer, makeLayerDef as mkDef } from "./layers";
import { TEXT_CORPUS, TEXT_CHARS, LM_SEQ_LEN } from "../data/corpus";
import { buildMassiveCorpus, buildWordTokenizer, encodeWordsToIds, decodeTokenIds, tokenizeWords, LM2_SEQ_LEN } from "../data/corpusLarge";
import { sampleFromProbs } from "./sampling";
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
      for (const id of ["noise-robust-clf", "layernorm-reg", "cnn-batchnorm", "ae-layernorm", "char-rnn-seq", "seq-rnn", "seq-rnn-deep", "seq-mlp-baseline"]) {
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
      // round trip on an in-vocabulary phrase
      const rt = decodeTokenIds(encodeWordsToIds("the quick brown fox", tok.tokens), tok.tokens, "word");
      expect(rt === "the quick brown fox", `round trip gave "${rt}"`);
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

  return tests;
}
