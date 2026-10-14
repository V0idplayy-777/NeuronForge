/**
 * Synthetic dataset generators.
 *
 * Everything here is produced locally by a seeded RNG — no downloads, no
 * hidden fixtures. Each generator is a pure function of its parameters plus
 * the split fractions, so regenerating with the same seed rebuilds exactly the
 * same tensors (which is what makes the exported test vectors meaningful).
 */
import type { LossName, TaskType } from "../engine/types";
import { RNG } from "../engine/rng";
import { pack, splitRows, type DatasetMeta, type PreparedDataset, type SynthParams } from "./datasets";

export type GeneratorValue = number | string;

export interface GeneratorParam {
  key: string;
  label: string;
  kind: "number" | "select";
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  hint?: string;
}

export interface GenerateConfig {
  values: Record<string, GeneratorValue>;
  n: number;
  noise: number;
  seed: number;
  trainFrac: number;
  valFrac: number;
  testFrac: number;
}

export interface GeneratorDef {
  id: string;
  name: string;
  /** Short line for the picker. */
  description: string;
  /** Longer explanation of what makes the dataset interesting. */
  detail: string;
  task: TaskType;
  loss: LossName;
  inputShape: number[];
  outputUnits: number;
  classNames?: string[];
  params: GeneratorParam[];
  defaults: Record<string, GeneratorValue>;
  build: (cfg: GenerateConfig) => PreparedDataset;
}

/* ------------------------------------------------------------------ helpers */

const clamp = (v: unknown, lo: number, hi: number, d: number): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : d;
};

const clampInt = (v: unknown, lo: number, hi: number, d: number): number => Math.round(clamp(v, lo, hi, d));

const str = (v: unknown, d: string): string => (typeof v === "string" && v ? v : d);

/** Two classes are binary (one sigmoid unit); three or more are one-hot/softmax. */
function classOverrides(names: string[] | undefined, k: number): FinishOverrides {
  return k > 2
    ? { task: "multiclass", loss: "cce", outputUnits: k, classNames: names?.slice(0, k) }
    : { task: "binary", loss: "bce", outputUnits: 1, classNames: names?.slice(0, 2) };
}

function oneHot(i: number, k: number): number[] {
  const y = new Array(k).fill(0);
  y[i] = 1;
  return y;
}

function synthOf(cfg: GenerateConfig, classes = 2): SynthParams {
  return {
    n: cfg.n,
    noise: cfg.noise,
    seed: cfg.seed,
    classes,
    trainFrac: cfg.trainFrac,
    valFrac: cfg.valFrac,
    testFrac: cfg.testFrac,
  };
}

interface FinishOverrides {
  task?: TaskType;
  loss?: LossName;
  outputUnits?: number;
  classNames?: string[];
  inputShape?: number[];
}

/** Split, pack and stamp a generator's rows with a standard meta record. */
function finish(
  cfg: GenerateConfig,
  def: Omit<GeneratorDef, "build" | "params" | "defaults">,
  xs: number[][],
  ys: number[][],
  extra: Partial<DatasetMeta> = {},
  overrides: FinishOverrides = {},
): PreparedDataset {
  const task = overrides.task ?? def.task;
  const classNames = overrides.classNames ?? def.classNames;
  const outputUnits = overrides.outputUnits ?? def.outputUnits;
  const inputShape = overrides.inputShape ?? def.inputShape;
  const meta: DatasetMeta = {
    id: `gen_${def.id}_${cfg.seed}_${xs.length}`,
    name: def.name,
    description: def.detail,
    source: "custom",
    task,
    featureNames: inputShape.map((_, i) => `f${i}`),
    targetNames: classNames ?? Array.from({ length: outputUnits }, (_, i) => `y${i}`),
    classNames,
    recommendedLoss: overrides.loss ?? def.loss,
    inputShape,
    outputUnits,
    attribution: "Synthetic generator. Seeded RNG, reproducible, no external data.",
    ...extra,
  };
  return pack(meta, splitRows(xs, ys, synthOf(cfg, classNames?.length ?? 2), new RNG(cfg.seed + 4242)));
}

function windowNames(w: number): string[] {
  return Array.from({ length: w }, (_, i) => `t-${w - i}`);
}

/** Roll a series into (window -> next value) rows. */
function windowRows(series: number[], w: number): { xs: number[][]; ys: number[][] } {
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i + w < series.length; i++) {
    xs.push(series.slice(i, i + w));
    ys.push([series[i + w]!]);
  }
  return { xs, ys };
}

function standardize(series: number[]): number[] {
  let mean = 0;
  for (const v of series) mean += v;
  mean /= series.length;
  let varSum = 0;
  for (const v of series) varSum += (v - mean) * (v - mean);
  const std = Math.sqrt(varSum / series.length) || 1;
  return series.map((v) => (v - mean) / std);
}

/* --------------------------------------------------------------- generators */

function buildClusters(cfg: GenerateConfig): PreparedDataset {
  const k = clampInt(cfg.values.classes, 2, 5, 3);
  const spread = clamp(cfg.values.spread, 0.05, 1.2, 0.22);
  const separation = clamp(cfg.values.separation, 0.4, 3, 1.4);
  const rng = new RNG(cfg.seed + 11);
  const centers = Array.from({ length: k }, (_, c) => {
    const a = (2 * Math.PI * c) / k + 0.4;
    return [Math.cos(a) * separation, Math.sin(a) * separation];
  });
  const xs: number[][] = [];
  const ys: number[][] = [];
  const multi = k > 2;
  for (let i = 0; i < cfg.n; i++) {
    const c = i % k;
    const center = centers[c]!;
    xs.push([center[0]! + rng.normal() * spread, center[1]! + rng.normal() * spread]);
    ys.push(multi ? oneHot(c, k) : [c]);
  }
  return finish(cfg, CLUSTERS, xs, ys, {}, classOverrides(CLUSTERS.classNames, k));
}

function buildRings(cfg: GenerateConfig): PreparedDataset {
  const k = clampInt(cfg.values.classes, 2, 5, 3);
  const gap = clamp(cfg.values.gap, 0.3, 1.5, 0.6);
  const jitter = clamp(cfg.values.jitter, 0, 0.5, 0.12);
  const rng = new RNG(cfg.seed + 23);
  const xs: number[][] = [];
  const ys: number[][] = [];
  const per = Math.max(1, Math.floor(cfg.n / k));
  const multi = k > 2;
  for (let c = 0; c < k; c++) {
    const radius = 0.45 + c * gap;
    for (let i = 0; i < per; i++) {
      const ang = rng.uniform(0, Math.PI * 2);
      const r = radius + rng.normal() * (cfg.noise * 0.5 + jitter * 0.2);
      xs.push([r * Math.cos(ang), r * Math.sin(ang)]);
      ys.push(multi ? oneHot(c, k) : [c]);
    }
  }
  return finish(cfg, RINGS, xs, ys, {}, classOverrides(RINGS.classNames, k));
}

function buildChecker(cfg: GenerateConfig): PreparedDataset {
  const cells = clampInt(cfg.values.cells, 2, 10, 4);
  const rng = new RNG(cfg.seed + 37);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    const x = rng.uniform(-1, 1);
    const y = rng.uniform(-1, 1);
    const jitX = x + rng.normal() * cfg.noise * 0.15;
    const jitY = y + rng.normal() * cfg.noise * 0.15;
    const cx = Math.min(cells - 1, Math.max(0, Math.floor(((jitX + 1) / 2) * cells)));
    const cy = Math.min(cells - 1, Math.max(0, Math.floor(((jitY + 1) / 2) * cells)));
    xs.push([jitX, jitY]);
    ys.push([(cx + cy) % 2]);
  }
  return finish(cfg, CHECKER, xs, ys);
}

function buildParity(cfg: GenerateConfig): PreparedDataset {
  const bits = clampInt(cfg.values.bits, 3, 10, 4);
  const rng = new RNG(cfg.seed + 53);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    const row: number[] = [];
    let ones = 0;
    for (let b = 0; b < bits; b++) {
      const bit = rng.uniform(0, 1) < 0.5 ? 0 : 1;
      row.push(bit);
      ones += bit;
    }
    xs.push(row);
    ys.push([ones % 2]);
  }
  return finish(cfg, PARITY, xs, ys, {}, { inputShape: [bits] });
}

function buildSurface(cfg: GenerateConfig): PreparedDataset {
  const fx = clamp(cfg.values.freqX, 0.25, 3, 1);
  const fy = clamp(cfg.values.freqY, 0.25, 3, 1.5);
  const tilt = clamp(cfg.values.tilt, 0, 1.5, 0.4);
  const rng = new RNG(cfg.seed + 67);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    const x = rng.uniform(-1, 1);
    const y = rng.uniform(-1, 1);
    const v = Math.sin(Math.PI * fx * x) * Math.cos(Math.PI * fy * y) + tilt * x * y;
    xs.push([x, y]);
    ys.push([v + rng.normal() * cfg.noise]);
  }
  return finish(cfg, SURFACE, xs, ys, { featureNames: ["x", "y"], targetNames: ["f(x, y)"] });
}

function buildWave(cfg: GenerateConfig): PreparedDataset {
  const kind = str(cfg.values.kind, "sine");
  const period = clamp(cfg.values.period, 3, 64, 12);
  const amp = clamp(cfg.values.amp, 0.1, 3, 1);
  const w = clampInt(cfg.values.window, 4, 32, 12);
  const rng = new RNG(cfg.seed + 71);
  const len = cfg.n + w;
  const series: number[] = [];
  for (let i = 0; i < len; i++) {
    const t = i;
    let v: number;
    if (kind === "damped") v = amp * Math.exp(-t / (period * 4)) * Math.sin((2 * Math.PI * t) / period);
    else if (kind === "square") v = (Math.sin((2 * Math.PI * t) / period) >= 0 ? 1 : -1) * amp;
    else if (kind === "saw") v = amp * (2 * (((t / period) % 1 + 1) % 1) - 1);
    else if (kind === "chirp") v = amp * Math.sin(((2 * Math.PI) / period) * t * (1 + t / (len * 1.5)));
    else v = amp * Math.sin((2 * Math.PI * t) / period);
    series.push(v + rng.normal() * cfg.noise * amp);
  }
  const { xs, ys } = windowRows(series, w);
  return finish(cfg, WAVE, xs, ys, { featureNames: windowNames(w), targetNames: ["next"] }, { inputShape: [w] });
}

function buildWalk(cfg: GenerateConfig): PreparedDataset {
  const drift = clamp(cfg.values.drift, -0.2, 0.2, 0.01);
  const vol = clamp(cfg.values.volatility, 0.05, 2, 0.4);
  const w = clampInt(cfg.values.window, 4, 32, 12);
  const rng = new RNG(cfg.seed + 89);
  const raw: number[] = [0];
  for (let i = 0; i < cfg.n + w - 1; i++) raw.push(raw[i]! + drift + rng.normal() * vol);
  const series = standardize(raw);
  const { xs, ys } = windowRows(series, w);
  return finish(cfg, WALK, xs, ys, { featureNames: windowNames(w), targetNames: ["next"] }, { inputShape: [w] });
}

function buildManifold(cfg: GenerateConfig): PreparedDataset {
  const kind = str(cfg.values.kind, "scurve");
  const rng = new RNG(cfg.seed + 97);
  const xs: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    if (kind === "circle") {
      const a = rng.uniform(0, Math.PI * 2);
      xs.push([Math.cos(a) + rng.normal() * cfg.noise, Math.sin(a) + rng.normal() * cfg.noise]);
    } else if (kind === "spiral") {
      const t = rng.uniform(0, 1);
      const a = 6 * Math.PI * t;
      const r = 0.15 + 0.85 * t;
      xs.push([r * Math.cos(a) + rng.normal() * cfg.noise, r * Math.sin(a) + rng.normal() * cfg.noise]);
    } else if (kind === "line") {
      const t = rng.uniform(-1, 1);
      xs.push([t + rng.normal() * cfg.noise, 0.25 * t + rng.normal() * cfg.noise]);
    } else {
      const t = rng.uniform(-1, 1);
      xs.push([t, Math.sin(1.5 * Math.PI * t) + rng.normal() * cfg.noise]);
    }
  }
  // Targets are the inputs: this trains a bottleneck autoencoder.
  return finish(cfg, MANIFOLD, xs, xs.map((r) => r.slice()), {
    featureNames: ["x", "y"],
    targetNames: ["x̂", "ŷ"],
    reconstruction: true,
  });
}

function buildImbalance(cfg: GenerateConfig): PreparedDataset {
  const majority = clamp(cfg.values.majority, 0.5, 0.98, 0.9);
  const separation = clamp(cfg.values.separation, 0.2, 3, 0.9);
  const rng = new RNG(cfg.seed + 101);
  const nMajor = Math.round(cfg.n * majority);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    const minority = i >= nMajor;
    const spread = minority ? 0.35 : 0.55;
    const cx = minority ? separation / 2 : -separation / 2;
    xs.push([cx + rng.normal() * spread, rng.normal() * spread]);
    ys.push([minority ? 1 : 0]);
  }
  const actual = nMajor / Math.max(1, cfg.n);
  return finish(cfg, IMBALANCE, xs, ys, {
    targetNames: ["label"],
    description: `${IMBALANCE.detail} This seed is ${(actual * 100).toFixed(1)}% majority / ${((1 - actual) * 100).toFixed(1)}% minority, so a model that always answers "majority" scores ${(actual * 100).toFixed(1)}% accuracy.`,
  });
}

function buildWaveClasses(cfg: GenerateConfig): PreparedDataset {
  const k = clampInt(cfg.values.classes, 2, 4, 3);
  const w = clampInt(cfg.values.window, 8, 32, 16);
  const period = clamp(cfg.values.period, 3, 48, 10);
  const shapes = ["sine", "square", "triangle", "saw"];
  const classNames = shapes.slice(0, k);
  const rng = new RNG(cfg.seed + 113);
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < cfg.n; i++) {
    const c = i % k;
    const p = period * (0.9 + rng.uniform(0, 0.2));
    const phase = rng.uniform(0, p);
    const row: number[] = [];
    for (let t = 0; t < w; t++) {
      const u = ((t + phase) / p) % 1;
      let v: number;
      if (c === 1) v = u < 0.5 ? 1 : -1;
      else if (c === 2) v = 4 * Math.abs(u - 0.5) - 1;
      else if (c === 3) v = 2 * u - 1;
      else v = Math.sin(2 * Math.PI * u);
      row.push(v + rng.normal() * cfg.noise);
    }
    xs.push(row);
    ys.push(k > 2 ? oneHot(c, k) : [c]);
  }
  return finish(cfg, WAVE_CLASSES, xs, ys, {}, { ...classOverrides(classNames, k), inputShape: [w] });
}

/* ------------------------------------------------------------------ registry */

const CLUSTERS: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "clusters",
  name: "Gaussian clusters",
  description: "k separable blobs with tunable spread.",
  detail:
    "k isotropic Gaussian clusters on a circle, one per class. Spread controls the within-class variance and separation the distance between class centres: slide separation down until the classes overlap and watch accuracy fall.",
  task: "multiclass",
  loss: "cce",
  inputShape: [2],
  outputUnits: 3,
  classNames: ["A", "B", "C", "D", "E"],
};

const RINGS: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "rings",
  name: "Concentric rings",
  description: "Nested circles — needs radial features.",
  detail:
    "One ring per class, nested at growing radii. The classes are not linearly separable in x and y, so a plain perceptron stalls and the hidden layers have to learn a radius-like feature. Jitter moves each ring's radius from sample to sample.",
  task: "multiclass",
  loss: "cce",
  inputShape: [2],
  outputUnits: 3,
  classNames: ["A", "B", "C", "D", "E"],
};

const CHECKER: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "checker",
  name: "Checkerboard",
  description: "High-frequency 2D parity on a grid.",
  detail:
    "A classic hard case: the label alternates every cell, so the decision boundary is a comb that a shallow network cannot represent. More cells means higher frequency and a deeper network is needed. Jitter blurs the cell edges.",
  task: "binary",
  loss: "bce",
  inputShape: [2],
  outputUnits: 1,
  classNames: ["dark", "light"],
};

const PARITY: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "parity",
  name: "Bit parity",
  description: "XOR generalised to n bits.",
  detail:
    "The target is the parity (XOR) of n binary inputs. Parity is the canonical function that needs depth: each extra bit roughly doubles the number of regions the network must carve out. Inputs are exact 0/1 values.",
  task: "binary",
  loss: "bce",
  inputShape: [4],
  outputUnits: 1,
  classNames: ["even", "odd"],
};

const SURFACE: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "surface",
  name: "Wave surface (regression)",
  description: "Two-feature sinusoidal surface with a tilt term.",
  detail:
    "Continuous regression on two inputs: a product of sines plus a bilinear tilt. Because the target is smooth, a small MLP approximates it well — use the frequencies to make the surface harder to fit.",
  task: "regression",
  loss: "mse",
  inputShape: [2],
  outputUnits: 1,
};

const WAVE: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "wave",
  name: "Waveform forecasting",
  description: "Rolling windows of a periodic signal.",
  detail:
    "Each row is a window of consecutive samples and the target is the next one, so the network learns the dynamics rather than a static mapping. Switch the waveform to change what has to be predicted: a sine is linear-predictable, a sawtooth is not.",
  task: "regression",
  loss: "mse",
  inputShape: [12],
  outputUnits: 1,
};

const WALK: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "walk",
  name: "Random walk forecasting",
  description: "Drifted random walk, standardised.",
  detail:
    "A random walk with drift and volatility, standardised to zero mean and unit variance. The optimal predictor is close to persistence (repeat the last value), which makes it a good sanity check: any model that does much better than that is fitting noise.",
  task: "regression",
  loss: "mse",
  inputShape: [12],
  outputUnits: 1,
};

const MANIFOLD: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "manifold",
  name: "2D manifold (autoencoder)",
  description: "Points on a curve; targets are the inputs.",
  detail:
    "Samples from a 1-D curve embedded in the plane, with targets equal to the inputs. Train a bottleneck network on this to see how few hidden units can still reconstruct the shape — the reconstruction error is reported in the Evaluation view.",
  task: "regression",
  loss: "mse",
  inputShape: [2],
  outputUnits: 2,
};

const IMBALANCE: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "imbalance",
  name: "Imbalanced classes",
  description: "Skewed binary problem with overlapping clouds.",
  detail:
    "Two Gaussian clouds with a configurable majority share. Accuracy is misleading here: a model that always predicts the majority class scores the majority share. Compare accuracy against the class counts shown in the inspector.",
  task: "binary",
  loss: "bce",
  inputShape: [2],
  outputUnits: 1,
  classNames: ["majority", "minority"],
};

const WAVE_CLASSES: Omit<GeneratorDef, "build" | "params" | "defaults"> = {
  id: "wave_classes",
  name: "Waveform classification",
  description: "Classify a window by its shape.",
  detail:
    "Sequence classification: every window belongs to one waveform family (sine, square, triangle, sawtooth), with random phase and period jitter. The network has to recognise the shape of the whole window, not just its last few samples.",
  task: "multiclass",
  loss: "cce",
  inputShape: [16],
  outputUnits: 3,
  classNames: ["sine", "square", "triangle"],
};

export const GENERATORS: GeneratorDef[] = [
  {
    ...CLUSTERS,
    params: [
      { key: "classes", label: "Classes", kind: "number", min: 2, max: 5, step: 1 },
      { key: "spread", label: "Spread", kind: "number", min: 0.05, max: 1.2, step: 0.01 },
      { key: "separation", label: "Separation", kind: "number", min: 0.4, max: 3, step: 0.1 },
    ],
    defaults: { classes: 3, spread: 0.22, separation: 1.4 },
    build: buildClusters,
  },
  {
    ...RINGS,
    params: [
      { key: "classes", label: "Rings", kind: "number", min: 2, max: 5, step: 1 },
      { key: "gap", label: "Ring gap", kind: "number", min: 0.3, max: 1.5, step: 0.05 },
      { key: "jitter", label: "Radius jitter", kind: "number", min: 0, max: 0.5, step: 0.01 },
    ],
    defaults: { classes: 3, gap: 0.6, jitter: 0.12 },
    build: buildRings,
  },
  {
    ...CHECKER,
    params: [{ key: "cells", label: "Cells per axis", kind: "number", min: 2, max: 10, step: 1 }],
    defaults: { cells: 4 },
    build: buildChecker,
  },
  {
    ...PARITY,
    params: [{ key: "bits", label: "Bits", kind: "number", min: 3, max: 10, step: 1 }],
    defaults: { bits: 4 },
    build: buildParity,
  },
  {
    ...SURFACE,
    params: [
      { key: "freqX", label: "Frequency x", kind: "number", min: 0.25, max: 3, step: 0.25 },
      { key: "freqY", label: "Frequency y", kind: "number", min: 0.25, max: 3, step: 0.25 },
      { key: "tilt", label: "Tilt x·y", kind: "number", min: 0, max: 1.5, step: 0.1 },
    ],
    defaults: { freqX: 1, freqY: 1.5, tilt: 0.4 },
    build: buildSurface,
  },
  {
    ...WAVE,
    params: [
      {
        key: "kind",
        label: "Waveform",
        kind: "select",
        options: [
          { value: "sine", label: "sine" },
          { value: "damped", label: "damped sine" },
          { value: "square", label: "square" },
          { value: "saw", label: "sawtooth" },
          { value: "chirp", label: "chirp" },
        ],
      },
      { key: "period", label: "Period (samples)", kind: "number", min: 3, max: 64, step: 1 },
      { key: "amp", label: "Amplitude", kind: "number", min: 0.1, max: 3, step: 0.1 },
      { key: "window", label: "Window", kind: "number", min: 4, max: 32, step: 1 },
    ],
    defaults: { kind: "sine", period: 12, amp: 1, window: 12 },
    build: buildWave,
  },
  {
    ...WALK,
    params: [
      { key: "drift", label: "Drift", kind: "number", min: -0.2, max: 0.2, step: 0.005 },
      { key: "volatility", label: "Volatility", kind: "number", min: 0.05, max: 2, step: 0.05 },
      { key: "window", label: "Window", kind: "number", min: 4, max: 32, step: 1 },
    ],
    defaults: { drift: 0.01, volatility: 0.4, window: 12 },
    build: buildWalk,
  },
  {
    ...MANIFOLD,
    params: [
      {
        key: "kind",
        label: "Shape",
        kind: "select",
        options: [
          { value: "scurve", label: "S-curve" },
          { value: "circle", label: "circle" },
          { value: "spiral", label: "spiral" },
          { value: "line", label: "line" },
        ],
      },
    ],
    defaults: { kind: "scurve" },
    build: buildManifold,
  },
  {
    ...IMBALANCE,
    params: [
      { key: "majority", label: "Majority share", kind: "number", min: 0.5, max: 0.98, step: 0.01 },
      { key: "separation", label: "Separation", kind: "number", min: 0.2, max: 3, step: 0.1 },
    ],
    defaults: { majority: 0.9, separation: 0.9 },
    build: buildImbalance,
  },
  {
    ...WAVE_CLASSES,
    params: [
      { key: "classes", label: "Classes", kind: "number", min: 2, max: 4, step: 1 },
      { key: "window", label: "Window", kind: "number", min: 8, max: 32, step: 1 },
      { key: "period", label: "Period (samples)", kind: "number", min: 3, max: 48, step: 1 },
    ],
    defaults: { classes: 3, window: 16, period: 10 },
    build: buildWaveClasses,
  },
];

export function generatorById(id: string): GeneratorDef | undefined {
  return GENERATORS.find((g) => g.id === id);
}

/** Input shape and class count depend on the parameters — resolve them for the UI/architecture. */
export function generatorShape(def: GeneratorDef, values: Record<string, GeneratorValue>): { shape: number[]; out: number; classes?: string[] } {
  const v = { ...def.defaults, ...values };
  const k = clampInt(v.classes, 2, 5, 3);
  switch (def.id) {
    case "clusters":
    case "rings":
      return { shape: [2], out: k > 2 ? k : 1, classes: def.classNames?.slice(0, k) };
    case "imbalance":
      return { shape: [2], out: 1, classes: def.classNames };
    case "parity":
      return { shape: [clampInt(v.bits, 3, 10, 4)], out: 1, classes: def.classNames };
    case "wave":
    case "walk":
      return { shape: [clampInt(v.window, 4, 32, 12)], out: 1 };
    case "wave_classes":
      return { shape: [clampInt(v.window, 8, 32, 16)], out: k > 2 ? k : 2, classes: def.classNames?.slice(0, k) };
    default:
      return { shape: def.inputShape, out: def.outputUnits, classes: def.classNames };
  }
}
