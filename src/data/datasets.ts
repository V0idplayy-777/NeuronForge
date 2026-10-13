import { Tensor } from "../engine/tensor";
import { RNG } from "../engine/rng";
import type { LossName, TaskType } from "../engine/types";
import { TEXT_CORPUS, TEXT_CHARS, LM_SEQ_LEN } from "./corpus";
import {
  buildMassiveCorpus,
  buildWordTokenizer,
  tokenizeWords,
  LM2_SEQ_LEN,
  LM2_VOCAB_CAP,
  LM2_MAX_WINDOWS,
  UNK_TOKEN,
  type CorpusStats,
} from "./corpusLarge";

export interface ColumnInfo {
  name: string;
  kind: "numeric" | "categorical" | "target";
}

export interface DatasetMeta {
  id: string;
  name: string;
  description: string;
  source: "builtin" | "synthetic" | "csv" | "custom";
  task: TaskType;
  featureNames: string[];
  targetNames: string[];
  classNames?: string[];
  recommendedLoss: LossName;
  inputShape: number[];
  outputUnits: number;
  attribution?: string;
  /** For language datasets: the token<->id mapping and context length. Saved with exported models. */
  tokenizer?: TokenizerSpec;
  /** For imported tables: exactly how raw rows were turned into feature vectors. */
  preprocessing?: TabularPreprocessing;
  /** For image datasets: pixel layout and value range the model was trained on. */
  image?: ImageSpec;
  /** True when targets are the inputs themselves (autoencoder). */
  reconstruction?: boolean;
}

export type TabularColumn =
  | { name: string; type: "numeric" }
  | { name: string; type: "categorical"; encoding: "onehot" | "label"; categories: string[] };

export interface TabularPreprocessing {
  kind: "tabular";
  /** Raw input columns, in the order their encoded features appear. */
  columns: TabularColumn[];
  /** Names of the encoded feature dimensions the model actually sees. */
  featureNames: string[];
  normalize: "none" | "minmax" | "zscore";
  /** Per encoded feature dimension, fit on the training split only. */
  mean: number[];
  std: number[];
  min: number[];
  max: number[];
  missing: "drop" | "zero" | "mean";
  /** Real (raw row -> encoded vector) pairs produced by the training pipeline, for verification. */
  samples: { raw: Record<string, string>; encoded: number[] }[];
}

export interface ImageSpec {
  height: number;
  width: number;
  channels: number;
  layout: "HWC";
  /** Pixel values the model saw during training, e.g. [0, 1]. */
  range: [number, number];
  description: string;
}

export interface TokenizerSpec {
  /** "char" = one token per character, "word" = one token per word. */
  level: "char" | "word";
  /** id -> token string. Index 0 is the unknown token for word level. */
  tokens: string[];
  seqLen: number;
  /** Token used to left-pad a prompt shorter than seqLen. */
  padToken: string;
  /** Share of corpus tokens outside the vocabulary cap (word level only). */
  oovRate?: number;
}

export interface SplitSet {
  xs: Tensor;
  ys: Tensor;
}

export interface PreparedDataset {
  meta: DatasetMeta;
  train: SplitSet;
  val: SplitSet;
  test: SplitSet;
  allX: Tensor;
  allY: Tensor;
  xMin?: number[];
  xMax?: number[];
  xMean?: number[];
  xStd?: number[];
  rawPreview?: { headers: string[]; rows: string[][] };
}

export type BuiltinId =
  | "xor"
  | "and"
  | "or"
  | "circles"
  | "moons"
  | "spiral"
  | "blobs"
  | "linear"
  | "sine"
  | "poly"
  | "gauss"
  | "xor_noise"
  | "patterns"
  | "wave_seq"
  | "tiny_text"
  | "tiny_text_2";

/** Context window length for the sequence-forecasting dataset. */
export const SEQ_WINDOW = 12;

export interface SynthParams {
  n: number;
  noise: number;
  seed: number;
  classes: number;
  trainFrac: number;
  valFrac: number;
  testFrac: number;
}

export const DEFAULT_SYNTH: SynthParams = {
  n: 400,
  noise: 0.08,
  seed: 7,
  classes: 2,
  trainFrac: 0.7,
  valFrac: 0.15,
  testFrac: 0.15,
};

export const BUILTIN_INFO: Record<BuiltinId, { name: string; description: string; task: TaskType; loss: LossName; shape: number[]; out: number; classes?: string[] }> = {
  xor: { name: "XOR", description: "Classic 4-point exclusive-or. A linear model cannot solve this.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["0", "1"] },
  and: { name: "AND", description: "Binary AND on two bits.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["0", "1"] },
  or: { name: "OR", description: "Binary OR on two bits.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["0", "1"] },
  xor_noise: { name: "Noisy XOR", description: "XOR clusters with Gaussian noise. Tests robustness.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["0", "1"] },
  circles: { name: "Concentric circles", description: "Two nested rings. Nonlinear decision boundary.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["inner", "outer"] },
  moons: { name: "Two moons", description: "Interleaving crescent classes.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["moon A", "moon B"] },
  spiral: { name: "Spiral", description: "Multiclass interlocking spirals.", task: "multiclass", loss: "cce", shape: [2], out: 3, classes: ["A", "B", "C"] },
  blobs: { name: "Gaussian blobs", description: "Isotropic Gaussian clusters.", task: "multiclass", loss: "cce", shape: [2], out: 3, classes: ["A", "B", "C"] },
  gauss: { name: "Overlapping Gaussians", description: "Two overlapping 2D Gaussians — probabilistic boundary.", task: "binary", loss: "bce", shape: [2], out: 1, classes: ["A", "B"] },
  linear: { name: "Linear regression", description: "y = 1.7x − 0.3 plus noise.", task: "regression", loss: "mse", shape: [1], out: 1 },
  sine: { name: "Noisy sine", description: "y = sin(3x) plus noise. Tests function approximation.", task: "regression", loss: "mse", shape: [1], out: 1 },
  poly: { name: "Polynomial", description: "y = 0.5x³ − x² + 0.2x plus noise.", task: "regression", loss: "mse", shape: [1], out: 1 },
  patterns: { name: "8×8 patterns", description: "Tiny images: bar, plus, diag, blob. For Conv2D / Flatten nets.", task: "multiclass", loss: "cce", shape: [8, 8, 1], out: 4, classes: ["bar", "plus", "diag", "blob"] },
  wave_seq: { name: "Wave sequence forecasting", description: `Each example is a window of ${SEQ_WINDOW} consecutive points from a noisy two-frequency wave; the target is the value that comes right after. For RNN or sequence models — not a language dataset.`, task: "regression", loss: "mse", shape: [SEQ_WINDOW, 1], out: 1 },
  tiny_text: { name: "Tiny character LM", description: `Next-character prediction over ${LM_SEQ_LEN} characters of context on a built-in corpus of greetings, conversations, facts, and advice. Character-level tokens.`, task: "language", loss: "sparse_cce", shape: [LM_SEQ_LEN], out: 1 },
  tiny_text_2: { name: "Tiny character LM 2", description: `The massive word-level corpus: ${LM2_SEQ_LEN} words of context, a frequency-capped vocabulary, and a generated text corpus of greetings, dialogue, narrative, science, computing, history, food, music, sport, health and nature.`, task: "language", loss: "sparse_cce", shape: [LM2_SEQ_LEN], out: 1 },
};

export function generateBuiltin(id: BuiltinId, params: SynthParams = DEFAULT_SYNTH): PreparedDataset {
  const info = BUILTIN_INFO[id];
  const rng = new RNG(params.seed);
  let xs: number[][] = [];
  let ys: number[][] = [];
  let featureNames = info.shape.length === 1 && info.shape[0] === 2 ? ["x", "y"] : info.shape.length === 1 && info.shape[0] === 1 ? ["x"] : featureNamesFromShape(info.shape);
  let targetNames = info.task === "regression" ? ["y"] : info.classes ? info.classes : ["label"];
  let classNames = info.classes;
  let outputUnits = info.out;
  let inputShape = info.shape.slice();
  let lmTokenizer: DatasetMeta["tokenizer"];
  let corpusStats: CorpusStats | undefined;

  switch (id) {
    case "xor": {
      xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
      ys = [[0], [1], [1], [0]];
      break;
    }
    case "and": {
      xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
      ys = [[0], [0], [0], [1]];
      break;
    }
    case "or": {
      xs = [[0, 0], [0, 1], [1, 0], [1, 1]];
      ys = [[0], [1], [1], [1]];
      break;
    }
    case "xor_noise": {
      for (let i = 0; i < params.n; i++) {
        const a = i % 2;
        const b = Math.floor(i / 2) % 2;
        xs.push([a + rng.normal() * params.noise, b + rng.normal() * params.noise]);
        ys.push([a ^ b]);
      }
      break;
    }
    case "circles": {
      for (let i = 0; i < params.n; i++) {
        const cls = i < params.n / 2 ? 0 : 1;
        const ang = rng.uniform(0, Math.PI * 2);
        const r = cls === 0 ? 0.4 + rng.normal() * params.noise : 1.0 + rng.normal() * params.noise;
        xs.push([r * Math.cos(ang), r * Math.sin(ang)]);
        ys.push([cls]);
      }
      break;
    }
    case "moons": {
      for (let i = 0; i < params.n; i++) {
        const cls = i < params.n / 2 ? 0 : 1;
        const t = rng.uniform(0, Math.PI);
        if (cls === 0) xs.push([Math.cos(t) + rng.normal() * params.noise, Math.sin(t) + rng.normal() * params.noise]);
        else xs.push([1 - Math.cos(t) + rng.normal() * params.noise, 0.5 - Math.sin(t) + rng.normal() * params.noise]);
        ys.push([cls]);
      }
      break;
    }
    case "spiral": {
      const k = Math.max(2, Math.min(5, params.classes));
      outputUnits = k;
      classNames = Array.from({ length: k }, (_, i) => String.fromCharCode(65 + i));
      targetNames = classNames;
      for (let c = 0; c < k; c++) {
        const m = Math.floor(params.n / k);
        for (let i = 0; i < m; i++) {
          const r = (i / m) * 1.6 + 0.05;
          const t = (c * 2 * Math.PI) / k + (i / m) * 2.8 + rng.normal() * params.noise;
          xs.push([r * Math.cos(t), r * Math.sin(t)]);
          ys.push(oneHot(c, k));
        }
      }
      break;
    }
    case "blobs": {
      const k = Math.max(2, Math.min(5, params.classes));
      outputUnits = k;
      classNames = Array.from({ length: k }, (_, i) => String.fromCharCode(65 + i));
      targetNames = classNames;
      const centers = Array.from({ length: k }, () => [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)]);
      for (let i = 0; i < params.n; i++) {
        const c = i % k;
        xs.push([centers[c]![0]! + rng.normal() * (0.25 + params.noise), centers[c]![1]! + rng.normal() * (0.25 + params.noise)]);
        ys.push(oneHot(c, k));
      }
      break;
    }
    case "gauss": {
      for (let i = 0; i < params.n; i++) {
        const cls = i < params.n / 2 ? 0 : 1;
        const mx = cls === 0 ? -0.5 : 0.5;
        xs.push([mx + rng.normal() * (0.45 + params.noise), rng.normal() * (0.45 + params.noise)]);
        ys.push([cls]);
      }
      break;
    }
    case "linear": {
      for (let i = 0; i < params.n; i++) {
        const x = rng.uniform(-1.5, 1.5);
        xs.push([x]);
        ys.push([1.7 * x - 0.3 + rng.normal() * params.noise]);
      }
      break;
    }
    case "sine": {
      for (let i = 0; i < params.n; i++) {
        const x = rng.uniform(-Math.PI, Math.PI);
        xs.push([x]);
        ys.push([Math.sin(3 * x) + rng.normal() * params.noise]);
      }
      break;
    }
    case "poly": {
      for (let i = 0; i < params.n; i++) {
        const x = rng.uniform(-1.6, 1.6);
        xs.push([x]);
        ys.push([0.5 * x * x * x - x * x + 0.2 * x + rng.normal() * params.noise]);
      }
      break;
    }
    case "patterns": {
      const labels = 4;
      outputUnits = 4;
      for (let i = 0; i < params.n; i++) {
        const c = i % labels;
        xs.push(makePattern(c, rng, params.noise));
        ys.push(oneHot(c, labels));
      }
      featureNames = Array.from({ length: 64 }, (_, i) => `p${i}`);
      break;
    }
    case "wave_seq": {
      const W = SEQ_WINDOW;
      inputShape = [W, 1];
      const total = Math.max(64, params.n) + W + 2;
      const f1 = 0.28 + rng.uniform(-0.03, 0.03);
      const f2 = 0.07 + rng.uniform(-0.01, 0.01);
      const amp2 = 0.5;
      const phase1 = rng.uniform(0, Math.PI * 2);
      const phase2 = rng.uniform(0, Math.PI * 2);
      const series: number[] = [];
      for (let t = 0; t < total; t++) {
        series.push(Math.sin(t * f1 + phase1) + amp2 * Math.sin(t * f2 + phase2) + rng.normal() * params.noise);
      }
      for (let t = W; t < series.length; t++) {
        xs.push(series.slice(t - W, t));
        ys.push([series[t]!]);
      }
      featureNames = Array.from({ length: W }, (_, i) => `t-${W - i}`);
      targetNames = ["next_value"];
      break;
    }
    case "tiny_text": {
      const seq = LM_SEQ_LEN;
      inputShape = [seq];
      const chars = TEXT_CHARS;
      const stoi = new Map(chars.map((ch, i) => [ch, i]));
      const ids = TEXT_CORPUS.split("").map((ch) => stoi.get(ch)!);
      // Stride 2 keeps the window count moderate while covering the whole corpus.
      for (let i = 0; i + seq < ids.length; i += 2) {
        xs.push(ids.slice(i, i + seq));
        ys.push([ids[i + seq]!]);
      }
      featureNames = Array.from({ length: seq }, (_, i) => `t-${seq - i}`);
      targetNames = ["next_char"];
      classNames = chars;
      outputUnits = chars.length;
      lmTokenizer = { level: "char", tokens: chars.slice(), seqLen: seq, padToken: " " };
      break;
    }
    case "tiny_text_2": {
      const seq = LM2_SEQ_LEN;
      inputShape = [seq];
      const { text, stats } = buildMassiveCorpus(params.seed + 4242);
      corpusStats = stats;
      const tok = buildWordTokenizer(text, LM2_VOCAB_CAP, seq);
      const ids = tokenizeWords(text).map((w) => tok.tokenToId[w] ?? 0);
      // Step so the window count never exceeds the cap while still covering the corpus.
      const span = Math.max(1, ids.length - seq);
      const stride = Math.max(1, Math.ceil(span / LM2_MAX_WINDOWS));
      for (let i = 0; i + seq < ids.length; i += stride) {
        xs.push(ids.slice(i, i + seq));
        ys.push([ids[i + seq]!]);
      }
      featureNames = Array.from({ length: seq }, (_, i) => `w-${seq - i}`);
      targetNames = ["next_word"];
      classNames = tok.tokens;
      outputUnits = tok.vocabSize;
      lmTokenizer = {
        level: "word",
        tokens: tok.tokens.slice(),
        seqLen: seq,
        padToken: tok.tokenToId["."] !== undefined ? "." : UNK_TOKEN,
        oovRate: tok.oovRate,
      };
      break;
    }
  }

  const split =
    xs.length <= 8
      ? { train: [xs, ys] as [number[][], number[][]], val: [xs, ys] as [number[][], number[][]], test: [xs, ys] as [number[][], number[][]] }
      : splitRows(xs, ys, params, new RNG(params.seed + 99));
  const meta: DatasetMeta = {
    id,
    name: info.name,
    description: info.description,
    source: "builtin",
    task: info.task,
    featureNames,
    targetNames,
    classNames,
    recommendedLoss: id === "tiny_text" ? "sparse_cce" : info.loss,
    inputShape,
    outputUnits,
    attribution: attributionFor(id, corpusStats, lmTokenizer),
    tokenizer: lmTokenizer,
    image:
      id === "patterns"
        ? { height: 8, width: 8, channels: 1, layout: "HWC", range: [0, 1], description: "8×8 grayscale, row-major HWC, bright strokes (1) on dark background (0)." }
        : undefined,
  };

  return pack(meta, split);
}

function attributionFor(
  id: BuiltinId,
  stats: CorpusStats | undefined,
  tok: DatasetMeta["tokenizer"],
): string {
  if (id === "tiny_text") {
    return `Built-in character corpus (${TEXT_CORPUS.length} chars, ${TEXT_CHARS.length} unique tokens, ${LM_SEQ_LEN}-char context). Tokenizer mapping is saved with exported models.`;
  }
  if (id === "tiny_text_2" && stats && tok) {
    return (
      `Massive generated corpus: ${stats.chars.toLocaleString()} characters, ${stats.lines.toLocaleString()} sentences ` +
      `(${stats.handwrittenLines.toLocaleString()} hand-written + ${stats.templateLines.toLocaleString()} templated), ` +
      `${stats.tokens.toLocaleString()} word tokens, ${stats.uniqueTokens.toLocaleString()} unique word types. ` +
      `Vocabulary capped at ${tok.tokens.length.toLocaleString()} by frequency (${((tok.oovRate ?? 0) * 100).toFixed(1)}% of tokens map to ${tok.tokens[0]}). ` +
      `Context is ${tok.seqLen} words. Tokenizer mapping is saved with exported models.`
    );
  }
  return "Generated locally in the browser. No external download.";
}

function featureNamesFromShape(shape: number[]): string[] {
  const n = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: n }, (_, i) => `f${i}`);
}

function oneHot(i: number, k: number): number[] {
  const y = new Array(k).fill(0);
  y[i] = 1;
  return y;
}

function makePattern(cls: number, rng: RNG, noise: number): number[] {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  if (cls === 0) {
    const row = 2 + rng.int(4);
    for (let x = 0; x < 8; x++) g[row]![x] = 1;
  } else if (cls === 1) {
    for (let i = 0; i < 8; i++) {
      g[3]![i] = 1;
      g[i]![3] = 1;
    }
  } else if (cls === 2) {
    for (let i = 0; i < 8; i++) g[i]![i] = 1;
  } else {
    for (let y = 2; y < 6; y++) for (let x = 2; x < 6; x++) g[y]![x] = 1;
  }
  const out: number[] = [];
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      out.push(Math.min(1, Math.max(0, g[y]![x]! + rng.normal() * noise)));
    }
  }
  return out;
}

export function splitRows(
  xs: number[][],
  ys: number[][],
  params: SynthParams,
  rng: RNG,
): { train: [number[][], number[][]]; val: [number[][], number[][]]; test: [number[][], number[][]] } {
  const n = xs.length;
  const idx = rng.shuffle(Array.from({ length: n }, (_, i) => i));
  const nTrain = Math.max(1, Math.floor(n * params.trainFrac));
  const nVal = Math.max(0, Math.floor(n * params.valFrac));
  const trainX: number[][] = [];
  const trainY: number[][] = [];
  const valX: number[][] = [];
  const valY: number[][] = [];
  const testX: number[][] = [];
  const testY: number[][] = [];
  for (let i = 0; i < n; i++) {
    const j = idx[i]!;
    if (i < nTrain) {
      trainX.push(xs[j]!);
      trainY.push(ys[j]!);
    } else if (i < nTrain + nVal) {
      valX.push(xs[j]!);
      valY.push(ys[j]!);
    } else {
      testX.push(xs[j]!);
      testY.push(ys[j]!);
    }
  }
  if (valX.length === 0 && trainX.length > 1) {
    valX.push(trainX[trainX.length - 1]!);
    valY.push(trainY[trainY.length - 1]!);
  }
  if (testX.length === 0 && trainX.length > 1) {
    testX.push(trainX[0]!);
    testY.push(trainY[0]!);
  }
  return { train: [trainX, trainY], val: [valX, valY], test: [testX, testY] };
}

export function pack(
  meta: DatasetMeta,
  split: { train: [number[][], number[][]]; val: [number[][], number[][]]; test: [number[][], number[][]] },
): PreparedDataset {
  const train = toSplit(split.train[0], split.train[1], meta.inputShape);
  const val = toSplit(split.val[0], split.val[1], meta.inputShape);
  const test = toSplit(split.test[0], split.test[1], meta.inputShape);
  const allX = concatSplit([train.xs, val.xs, test.xs]);
  const allY = concatSplit([train.ys, val.ys, test.ys]);
  return { meta, train, val, test, allX, allY };
}

function toSplit(xs: number[][], ys: number[][], inputShape: number[]): SplitSet {
  const n = xs.length;
  if (n === 0) {
    return {
      xs: Tensor.zeros([1, ...inputShape]),
      ys: Tensor.zeros([1, ys[0]?.length ?? 1]),
    };
  }
  const feat = xs[0]!.length;
  const xd = new Float32Array(n * feat);
  for (let i = 0; i < n; i++) xd.set(xs[i]!, i * feat);
  const tout = ys[0]!.length;
  const yd = new Float32Array(n * tout);
  for (let i = 0; i < n; i++) yd.set(ys[i]!, i * tout);
  return {
    xs: new Tensor([n, ...inputShape], xd),
    ys: new Tensor(tout === 1 ? [n, 1] : [n, tout], yd),
  };
}

function concatSplit(parts: Tensor[]): Tensor {
  const nonempty = parts.filter((t) => t.size > 0);
  if (nonempty.length === 0) return Tensor.zeros([0]);
  const rest = nonempty[0]!.shape.slice(1) as number[];
  const B = nonempty.reduce((s, t) => s + t.shape[0]!, 0);
  const row = nonempty[0]!.size / nonempty[0]!.shape[0]!;
  const data = new Float32Array(B * row);
  let off = 0;
  for (const t of nonempty) {
    data.set(t.data, off);
    off += t.size;
  }
  return new Tensor([B, ...rest], data);
}

export function datasetStats(ds: PreparedDataset): {
  nTrain: number;
  nVal: number;
  nTest: number;
  features: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
} {
  return {
    nTrain: ds.train.xs.shape[0]!,
    nVal: ds.val.xs.shape[0]!,
    nTest: ds.test.xs.shape[0]!,
    features: ds.meta.featureNames.length,
    xMin: minArr(ds.allX.data),
    xMax: maxArr(ds.allX.data),
    yMin: minArr(ds.allY.data),
    yMax: maxArr(ds.allY.data),
  };
}

function minArr(a: Float32Array): number {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) if (a[i]! < m) m = a[i]!;
  return m;
}
function maxArr(a: Float32Array): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i]! > m) m = a[i]!;
  return m;
}

export { TEXT_CORPUS, TEXT_CHARS, LM_SEQ_LEN } from "./corpus";
