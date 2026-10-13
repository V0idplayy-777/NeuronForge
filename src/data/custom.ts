import { RNG } from "../engine/rng";
import { compileExpr } from "./expr";
import { pack, splitRows, type DatasetMeta, type PreparedDataset, type SynthParams } from "./datasets";

/** A drawn 2D point dataset: user clicks points onto a canvas per class. */
export interface PointsSpec {
  kind: "points";
  name: string;
  points: { x: number; y: number; cls: number }[];
  classes: number;
  trainFrac: number;
  valFrac: number;
  seed: number;
}

/** Formula dataset: y = f(x) regression, or binary classification by sign(f(x, y)). */
export interface FormulaSpec {
  kind: "formula";
  name: string;
  mode: "regression" | "classify";
  expr: string;
  n: number;
  noise: number;
  seed: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  trainFrac: number;
  valFrac: number;
}

export type CustomSpec = PointsSpec | FormulaSpec;

export function buildCustom(spec: CustomSpec): PreparedDataset {
  if (spec.kind === "points") return buildPoints(spec);
  return buildFormula(spec);
}

function splitParams(trainFrac: number, valFrac: number, seed: number): SynthParams {
  return {
    n: 0,
    noise: 0,
    seed,
    classes: 2,
    trainFrac: clamp01(trainFrac, 0.7),
    valFrac: clamp01(valFrac, 0.15),
    testFrac: Math.max(0, 1 - clamp01(trainFrac, 0.7) - clamp01(valFrac, 0.15)),
  };
}

function clamp01(v: number, d: number): number {
  return Number.isFinite(v) && v > 0 && v < 1 ? v : d;
}

function buildPoints(spec: PointsSpec): PreparedDataset {
  if (spec.points.length < 6) throw new Error(`Need at least 6 points to build a dataset (have ${spec.points.length}).`);
  const present = new Set(spec.points.map((p) => p.cls));
  if (present.size < 2) throw new Error("Place points from at least 2 different classes.");
  const k = Math.max(2, Math.min(5, spec.classes));
  const multi = k > 2;

  const xs: number[][] = [];
  const ys: number[][] = [];
  for (const p of spec.points) {
    xs.push([p.x, p.y]);
    if (multi) {
      const oh = new Array(k).fill(0);
      oh[Math.min(k - 1, Math.max(0, p.cls))] = 1;
      ys.push(oh);
    } else {
      ys.push([p.cls >= 1 ? 1 : 0]);
    }
  }

  const params = splitParams(spec.trainFrac, spec.valFrac, spec.seed);
  const split = splitRows(xs, ys, params, new RNG(spec.seed + 101));
  const classNames = Array.from({ length: k }, (_, i) => `class ${i}`);
  const meta: DatasetMeta = {
    id: `custom_points_${spec.seed}_${spec.points.length}`,
    name: spec.name || "Drawn dataset",
    description: `${spec.points.length} hand-placed 2D points across ${present.size} classes. Drawn directly in the Dataset Studio.`,
    source: "custom",
    task: multi ? "multiclass" : "binary",
    featureNames: ["x", "y"],
    targetNames: multi ? classNames : ["label"],
    classNames,
    recommendedLoss: multi ? "cce" : "bce",
    inputShape: [2],
    outputUnits: multi ? k : 1,
    attribution: "User-drawn. Stored with the project.",
  };
  return pack(meta, split);
}

function buildFormula(spec: FormulaSpec): PreparedDataset {
  const { fn, vars } = compileExpr(spec.expr);
  const rng = new RNG(spec.seed);
  const n = Math.max(8, Math.min(5000, Math.floor(spec.n)));
  const xs: number[][] = [];
  const ys: number[][] = [];

  if (spec.mode === "regression") {
    if (vars.some((v) => v !== "x")) {
      throw new Error(`Regression mode uses only the variable x. Found: ${vars.join(", ")}`);
    }
    for (let i = 0; i < n; i++) {
      const x = rng.uniform(spec.xMin, spec.xMax);
      const y = fn({ x });
      if (!Number.isFinite(y)) throw new Error(`f(${x.toFixed(3)}) is not finite. Check the expression's domain.`);
      xs.push([x]);
      ys.push([y + rng.normal() * spec.noise]);
    }
  } else {
    if (vars.some((v) => v !== "x" && v !== "y")) {
      throw new Error(`Classify mode uses variables x and y. Found: ${vars.join(", ")}`);
    }
    for (let i = 0; i < n; i++) {
      const x = rng.uniform(spec.xMin, spec.xMax);
      const y = rng.uniform(spec.yMin, spec.yMax);
      const v = fn({ x, y }) + rng.normal() * spec.noise;
      if (!Number.isFinite(v)) throw new Error(`f(${x.toFixed(2)}, ${y.toFixed(2)}) is not finite. Check the expression's domain.`);
      xs.push([x, y]);
      ys.push([v > 0 ? 1 : 0]);
    }
  }

  const params = splitParams(spec.trainFrac, spec.valFrac, spec.seed);
  const split = splitRows(xs, ys, params, new RNG(spec.seed + 77));
  const reg = spec.mode === "regression";
  const meta: DatasetMeta = {
    id: `custom_formula_${spec.seed}_${n}`,
    name: spec.name || (reg ? `y = ${spec.expr}` : `sign(${spec.expr})`),
    description: reg
      ? `${n} samples of y = ${spec.expr} + noise·𝒩(0,1), x ∈ [${spec.xMin}, ${spec.xMax}].`
      : `${n} points in [${spec.xMin}, ${spec.xMax}]×[${spec.yMin}, ${spec.yMax}], labeled by sign(${spec.expr}) with noise before thresholding.`,
    source: "custom",
    task: reg ? "regression" : "binary",
    featureNames: reg ? ["x"] : ["x", "y"],
    targetNames: reg ? ["y"] : ["label"],
    classNames: reg ? undefined : ["negative", "positive"],
    recommendedLoss: reg ? "mse" : "bce",
    inputShape: [reg ? 1 : 2],
    outputUnits: 1,
    attribution: "Generated locally from a user formula (parsed, never eval()'d).",
  };
  return pack(meta, split);
}

/* ---------- Dataset serialization (persist custom/CSV datasets with projects) ---------- */

export interface SerializedSplit {
  x: number[];
  y: number[];
  xShape: number[];
  yShape: number[];
}

export interface SerializedDataset {
  meta: DatasetMeta;
  train: SerializedSplit;
  val: SerializedSplit;
  test: SerializedSplit;
}

const MAX_PERSIST_VALUES = 400_000;

export function serializePrepared(ds: PreparedDataset): SerializedDataset | null {
  const total = ds.train.xs.size + ds.train.ys.size + ds.val.xs.size + ds.val.ys.size + ds.test.xs.size + ds.test.ys.size;
  if (total > MAX_PERSIST_VALUES) return null;
  const pick = (s: { xs: { data: Float32Array; shape: readonly number[] }; ys: { data: Float32Array; shape: readonly number[] } }): SerializedSplit => ({
    x: Array.from(s.xs.data),
    y: Array.from(s.ys.data),
    xShape: s.xs.shape as number[],
    yShape: s.ys.shape as number[],
  });
  return { meta: ds.meta, train: pick(ds.train), val: pick(ds.val), test: pick(ds.test) };
}

export function deserializePrepared(s: SerializedDataset): PreparedDataset {
  const mk = (sp: SerializedSplit) => ({
    xs: new Tensor(sp.xShape, Float32Array.from(sp.x)),
    ys: new Tensor(sp.yShape, Float32Array.from(sp.y)),
  });
  const train = mk(s.train);
  const val = mk(s.val);
  const test = mk(s.test);
  const B = train.xs.shape[0]! + val.xs.shape[0]! + test.xs.shape[0]!;
  const allX = new Tensor([B, ...(train.xs.shape as number[]).slice(1)], concat3(train.xs.data, val.xs.data, test.xs.data));
  const yRest = train.ys.size / train.ys.shape[0]!;
  const allY = new Tensor([B, yRest], concat3(train.ys.data, val.ys.data, test.ys.data));
  return { meta: s.meta, train, val, test, allX, allY };
}

function concat3(a: Float32Array, b: Float32Array, c: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length + c.length);
  out.set(a, 0);
  out.set(b, a.length);
  out.set(c, a.length + b.length);
  return out;
}

import { Tensor } from "../engine/tensor";
