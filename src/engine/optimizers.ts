import type { OptimizerName } from "./types";
import type { SequentialModel } from "./model";

export interface OptimizerConfig {
  name: OptimizerName;
  learningRate: number;
  momentum: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  weightDecay: number;
  gradClip: number;
}

export const DEFAULT_OPTIMIZER: OptimizerConfig = {
  name: "adam",
  learningRate: 0.01,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  weightDecay: 0,
  gradClip: 0,
};

export interface OptimizerState {
  t: number;
  slots: Record<string, { m?: Float32Array; v?: Float32Array }>;
}

export function createOptimizerState(): OptimizerState {
  return { t: 0, slots: {} };
}

function key(layerId: string, name: string): string {
  return `${layerId}.${name}`;
}

export function clipGrads(model: SequentialModel, maxNorm: number): number {
  if (maxNorm <= 0) return 0;
  let sumSq = 0;
  const params = model.trainableParams();
  for (const p of params) {
    for (let i = 0; i < p.grad.size; i++) {
      const g = p.grad.data[i]!;
      sumSq += g * g;
    }
  }
  const norm = Math.sqrt(sumSq);
  if (norm > maxNorm && norm > 0) {
    const s = maxNorm / norm;
    for (const p of params) {
      for (let i = 0; i < p.grad.size; i++) p.grad.data[i]! *= s;
    }
  }
  return norm;
}

export function optimizerStep(model: SequentialModel, cfg: OptimizerConfig, state: OptimizerState): void {
  const params = model.trainableParams();
  if (cfg.gradClip > 0) clipGrads(model, cfg.gradClip);
  state.t += 1;
  const t = state.t;
  const lr = cfg.learningRate;

  for (const p of params) {
    const k = key(p.layer.def.id, p.name);
    const n = p.data.size;
    const w = p.data.data;
    const g = p.grad.data;

    if (cfg.name === "sgd") {
      const wd = cfg.weightDecay;
      for (let i = 0; i < n; i++) {
        w[i]! -= lr * (g[i]! + wd * w[i]!);
      }
      continue;
    }

    if (cfg.name === "momentum") {
      let slot = state.slots[k];
      if (!slot || !slot.v || slot.v.length !== n) {
        slot = { v: new Float32Array(n) };
        state.slots[k] = slot;
      }
      const v = slot.v!;
      const mu = cfg.momentum;
      const wd = cfg.weightDecay;
      for (let i = 0; i < n; i++) {
        v[i] = mu * v[i]! + (g[i]! + wd * w[i]!);
        w[i]! -= lr * v[i]!;
      }
      continue;
    }

    // Adam / AdamW
    let slot = state.slots[k];
    if (!slot || !slot.m || !slot.v || slot.m.length !== n) {
      slot = { m: new Float32Array(n), v: new Float32Array(n) };
      state.slots[k] = slot;
    }
    const m = slot.m!;
    const v = slot.v!;
    const b1 = cfg.beta1;
    const b2 = cfg.beta2;
    const eps = cfg.epsilon;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);

    if (cfg.name === "adamw") {
      const wd = cfg.weightDecay;
      for (let i = 0; i < n; i++) {
        const gi = g[i]!;
        m[i] = b1 * m[i]! + (1 - b1) * gi;
        v[i] = b2 * v[i]! + (1 - b2) * gi * gi;
        const mhat = m[i]! / bc1;
        const vhat = v[i]! / bc2;
        w[i]! -= lr * (mhat / (Math.sqrt(vhat) + eps) + wd * w[i]!);
      }
    } else {
      const wd = cfg.weightDecay;
      for (let i = 0; i < n; i++) {
        const gi = g[i]! + wd * w[i]!;
        m[i] = b1 * m[i]! + (1 - b1) * gi;
        v[i] = b2 * v[i]! + (1 - b2) * gi * gi;
        const mhat = m[i]! / bc1;
        const vhat = v[i]! / bc2;
        w[i]! -= lr * (mhat / (Math.sqrt(vhat) + eps));
      }
    }
  }
}

export function serializeOptimizerState(state: OptimizerState): {
  t: number;
  slots: Record<string, { m?: number[]; v?: number[] }>;
} {
  const slots: Record<string, { m?: number[]; v?: number[] }> = {};
  for (const [k, s] of Object.entries(state.slots)) {
    slots[k] = {
      m: s.m ? Array.from(s.m) : undefined,
      v: s.v ? Array.from(s.v) : undefined,
    };
  }
  return { t: state.t, slots };
}

export function loadOptimizerState(raw: { t: number; slots: Record<string, { m?: ArrayLike<number>; v?: ArrayLike<number> }> }): OptimizerState {
  const slots: OptimizerState["slots"] = {};
  for (const [k, s] of Object.entries(raw.slots ?? {})) {
    slots[k] = {
      m: s.m ? Float32Array.from(s.m as ArrayLike<number>) : undefined,
      v: s.v ? Float32Array.from(s.v as ArrayLike<number>) : undefined,
    };
  }
  return { t: raw.t ?? 0, slots };
}

export const OPTIMIZER_DOCS: Record<OptimizerName, string> = {
  sgd: "Stochastic gradient descent: θ ← θ − η ∇L. Simple, requires a well-tuned learning rate.",
  momentum: "SGD with velocity: v ← μv + ∇L, θ ← θ − ηv. Accelerates consistent directions.",
  adam: "Adaptive moments. Maintains biased first/second moment estimates with bias correction. Robust default.",
  adamw: "Adam with decoupled weight decay. Prefer this over Adam when using L2 regularization.",
};
