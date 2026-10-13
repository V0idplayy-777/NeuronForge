import type { LossName } from "./types";
import { Tensor, mean } from "./tensor";

const EPS = 1e-7;

export interface LossResult {
  value: number;
  grad: Tensor;
}

export function computeLoss(name: LossName, pred: Tensor, target: Tensor): LossResult {
  switch (name) {
    case "mse":
      return mse(pred, target);
    case "mae":
      return mae(pred, target);
    case "bce":
      return bce(pred, target);
    case "cce":
      return cce(pred, target);
    case "sparse_cce":
      return sparseCce(pred, target);
  }
}

export function mse(pred: Tensor, target: Tensor): LossResult {
  assertSame(pred, target, "mse");
  const n = pred.size || 1;
  let acc = 0;
  const grad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = pred.data[i]! - target.data[i]!;
    acc += d * d;
    grad[i] = (2 * d) / n;
  }
  return { value: acc / n, grad: new Tensor(pred.shape as number[], grad) };
}

export function mae(pred: Tensor, target: Tensor): LossResult {
  assertSame(pred, target, "mae");
  const n = pred.size || 1;
  let acc = 0;
  const grad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = pred.data[i]! - target.data[i]!;
    acc += Math.abs(d);
    grad[i] = (d >= 0 ? 1 : -1) / n;
  }
  return { value: acc / n, grad: new Tensor(pred.shape as number[], grad) };
}

/** Binary cross-entropy on probabilities in (0, 1). Numerically clipped. */
export function bce(pred: Tensor, target: Tensor): LossResult {
  assertSame(pred, target, "bce");
  const n = pred.size || 1;
  let acc = 0;
  const grad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = clamp(pred.data[i]!, EPS, 1 - EPS);
    const y = target.data[i]!;
    acc += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    grad[i] = -(y / p - (1 - y) / (1 - p)) / n;
  }
  return { value: acc / n, grad: new Tensor(pred.shape as number[], grad) };
}

/** Categorical cross-entropy. pred is a probability distribution on the last axis; target is one-hot. */
export function cce(pred: Tensor, target: Tensor): LossResult {
  assertSame(pred, target, "cce");
  const classes = pred.shape[pred.rank - 1]!;
  const rows = pred.size / classes;
  let acc = 0;
  const grad = new Float32Array(pred.size);
  for (let r = 0; r < rows; r++) {
    const off = r * classes;
    for (let c = 0; c < classes; c++) {
      const p = clamp(pred.data[off + c]!, EPS, 1);
      const y = target.data[off + c]!;
      acc += -y * Math.log(p);
      grad[off + c] = -y / p / rows;
    }
  }
  return { value: acc / rows, grad: new Tensor(pred.shape as number[], grad) };
}

/** Sparse categorical cross-entropy. target is [B] integer class indices (stored as floats). */
export function sparseCce(pred: Tensor, target: Tensor): LossResult {
  const classes = pred.shape[pred.rank - 1]!;
  const rows = pred.size / classes;
  if (target.size !== rows) {
    throw new Error(`sparse_cce: target size ${target.size} != batch ${rows}`);
  }
  let acc = 0;
  const grad = new Float32Array(pred.size);
  for (let r = 0; r < rows; r++) {
    const cls = Math.round(target.data[r]!);
    const p = clamp(pred.data[r * classes + cls]!, EPS, 1);
    acc += -Math.log(p);
    grad[r * classes + cls] = -1 / p / rows;
  }
  return { value: acc / rows, grad: new Tensor(pred.shape as number[], grad) };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function assertSame(a: Tensor, b: Tensor, name: string): void {
  if (a.size !== b.size) {
    throw new Error(`${name}: prediction size ${a.size} != target size ${b.size}`);
  }
}

export function lossIsClassification(name: LossName): boolean {
  return name === "bce" || name === "cce" || name === "sparse_cce";
}

export const LOSS_DOCS: Record<LossName, string> = {
  mse: "Mean squared error. Default for regression. Penalizes large errors more than MAE.",
  mae: "Mean absolute error. Robust to outliers; gradient is constant in magnitude.",
  bce: "Binary cross-entropy. Pair with a sigmoid output and targets in {0, 1}.",
  cce: "Categorical cross-entropy. Pair with softmax output and one-hot targets.",
  sparse_cce: "Sparse categorical cross-entropy. Pair with softmax output and integer class indices.",
};

export { mean };
