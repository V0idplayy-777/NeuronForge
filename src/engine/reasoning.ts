/**
 * Deep Reasoning Block — adaptive-depth latent deliberation.
 *
 * This is the one layer in NeuronForge whose *depth is a runtime parameter*.
 * A conventional stack spends more parameters to reason more deeply; this block
 * spends more *compute* on the same weights, iterating a shared thought cell
 * over a latent state before handing the result to the next layer:
 *
 *   h₀ = x
 *   for t = 0 … steps−1:
 *     uₜ = LayerNorm(hₜ) + stepCode(t)          # recurrent depth, sinusoidal step signal
 *     pₜ = act(uₜ·W₁ + b₁)                      # thought hidden state
 *     Δₜ = pₜ·W₂ + b₂                            # proposed state update
 *     [wₜ, qₜ] = sigmoid(uₜ·W_gate + b_gate)    # write gate, halting probability
 *     hₜ₊₁ = hₜ + wₜ·(1−qₜ)·Δₜ                  # gated residual deliberation
 *   y = h_T
 *
 * The weights are shared across all steps, so the parameter count is the size of
 * *one* thought cell while the computation can be arbitrarily deep. The halting
 * head learns when further thought is unnecessary; at inference `adaptive`
 * mode stops once cumulative halting confidence crosses `haltThreshold`
 * (training always runs every step so gradients see the full depth).
 *
 * Everything here is written as raw Float32Array loops, for the same reason the
 * attention and recurrent layers are: the backward derivation is explicit, and
 * the exported JavaScript / NumPy runtimes mirror the forward pass operation for
 * operation. The exported runtime keeps the loop, so the reasoning ability
 * survives the export — `model.reason()` exposes the per-step trace.
 */
import type { CompiledLayer } from "./layers";
import type { ActivationName, InitMethod, LayerDef } from "./types";
import { assert, numel, shapeStr } from "./types";
import { Tensor } from "./tensor";
import { applyActivation, activationBackward } from "./activations";
import { initTensor } from "./init";
import type { RNG } from "./rng";
import {
  colSum,
  layerNormBackward,
  layerNormForward,
  mmAtInto,
  mmBtInto,
  mmInto,
  type NormCache,
} from "./attention";

/* ---------------------------------------------------------------- config */

export type ReasoningActivation = ActivationName | "none";

export interface ReasoningPlan {
  /** Width of the latent state (must equal the incoming feature count). */
  width: number;
  /** Hidden width of the shared thought cell. */
  hidden: number;
  /** Deep-reasoning depth: how many deliberation steps a normal forward pass runs. */
  steps: number;
  /** Stop early at inference once cumulative halting confidence is high enough. */
  adaptive: boolean;
  haltThreshold: number;
  /** Never stop before this many steps, even if the halting head is confident. */
  minSteps: number;
  activation: ReasoningActivation;
  /** Add a parameter-free sinusoidal code that distinguishes step t from t−1. */
  stepCode: boolean;
  stepCodeScale: number;
  epsilon: number;
}

/** Per-step diagnostics. Values are batch means, which keeps traces comparable across batch sizes. */
export interface ReasoningStepTrace {
  step: number;
  /** Mean halting probability emitted by this step. */
  halt: number;
  /** Cumulative halting confidence after this step (1 − Π(1 − halt)). */
  cumulativeHalt: number;
  /** Mean L2 norm of the proposed update Δₜ. */
  deltaNorm: number;
  /** Mean L2 norm of the state h after this step. */
  stateNorm: number;
  /** Mean gated change actually written into the state. */
  changeNorm: number;
}

export interface ReasoningTrace {
  /** Configured depth. */
  steps: number;
  /** Steps actually executed (smaller than `steps` when adaptive halting fired). */
  stepsUsed: number;
  adaptive: boolean;
  halted: boolean;
  halt: number[];
  cumulativeHalt: number[];
  deltaNorm: number[];
  stateNorm: number[];
  changeNorm: number[];
  perStep: ReasoningStepTrace[];
}

export interface ReasoningLayerInfo {
  type: "reasoning_block";
  plan: ReasoningPlan;
  /** Trace of the most recent forward pass, or null before the first one. */
  trace: ReasoningTrace | null;
}

export const REASONING_MAX_STEPS = 64;

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function bool(v: unknown, d: boolean): boolean {
  if (typeof v === "boolean") return v;
  if (v === 1 || v === 0) return v === 1;
  return d;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

/** Validate + normalise a reasoning config against a declared `[features]` input. */
export function reasoningPlan(config: Record<string, unknown>, inputShape: number[]): ReasoningPlan {
  if (inputShape.length !== 1) {
    throw new Error(
      `Deep Reasoning expects a flat vector [features], got ${shapeStr(inputShape)}. ` +
        "Pool a sequence first (Attention / Transformer Block with output 'last') or add a Flatten layer.",
    );
  }
  const width = inputShape[0]!;
  if (!Number.isFinite(width) || width <= 0) {
    throw new Error(`Deep Reasoning needs a positive feature width, got ${width}.`);
  }
  const rawHidden = Math.round(num(config.hidden, 0));
  const hidden = rawHidden > 0 ? rawHidden : Math.max(1, width * 2);
  const steps = clampInt(num(config.steps, 8), 1, REASONING_MAX_STEPS);
  const minSteps = clampInt(num(config.minSteps, 2), 1, steps);
  const haltThreshold = Math.max(0.5, Math.min(0.9999, num(config.haltThreshold, 0.99)));
  const act = String(config.activation ?? "gelu") as ReasoningActivation;
  return {
    width,
    hidden,
    steps,
    adaptive: bool(config.adaptive, false),
    haltThreshold,
    minSteps,
    activation: act === "none" ? "linear" : act,
    stepCode: bool(config.stepCode, true),
    stepCodeScale: num(config.stepCodeScale, 1),
    epsilon: Math.max(1e-6, Math.min(1e-2, num(config.epsilon, 1e-5))),
  };
}

/** One-line summary for the Builder / Inspector. Never throws. */
export function reasoningSummary(config: Record<string, unknown>): string {
  const steps = clampInt(num(config.steps, 8), 1, REASONING_MAX_STEPS);
  const hidden = Math.round(num(config.hidden, 0));
  const adaptive = bool(config.adaptive, true);
  return `${steps} shared-weight thought steps · hidden ${hidden > 0 ? hidden : "2×width"} · ${adaptive ? "adaptive halt" : "fixed depth"}`;
}

/**
 * Parameter-free sinusoidal step signal. It makes step t a different function
 * from step t−1 without adding weights, so the same cell can keep refining the
 * state instead of reaching a fixed point immediately.
 *
 * The exported runtimes compute this formula literally; do not change it
 * without bumping the standalone runtime version.
 */
export function reasoningStepCode(step: number, width: number, scale: number): Float32Array {
  const code = new Float32Array(width);
  const denom = Math.max(1, width);
  for (let j = 0; j < width; j++) {
    const freq = Math.exp(-(j / denom) * Math.log(10000));
    const angle = (step + 1) * freq;
    code[j] = (j % 2 === 0 ? Math.sin(angle) : Math.cos(angle)) * scale;
  }
  return code;
}

/* ------------------------------------------------------------- the layer */

interface StepCache {
  /** State before this step. */
  hBefore: Float32Array;
  /** LayerNorm(hBefore) + step code — the input to every projection in this step. */
  u: Float32Array;
  lnCache: NormCache;
  pre: Float32Array;
  act: Float32Array;
  delta: Float32Array;
  write: Float32Array;
  halt: Float32Array;
  gate: Float32Array;
}

function emptyGrads(params: Record<string, Tensor>): Record<string, Tensor> {
  const g: Record<string, Tensor> = {};
  for (const k of Object.keys(params)) g[k] = Tensor.zeros(params[k]!.shape as number[]);
  return g;
}

function maybeReuse(existing: { params?: Record<string, Tensor> } | undefined, name: string, shape: number[]): Tensor | null {
  const t = existing?.params?.[name];
  if (!t) return null;
  if (t.size !== numel(shape)) return null;
  if (t.shape.length !== shape.length || t.shape.some((d, i) => d !== shape[i])) return t.reshape(shape);
  return t;
}

function sigmoid1(v: number): number {
  return v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
}

function copyTrace(t: ReasoningTrace | null): ReasoningTrace | null {
  if (!t) return null;
  return {
    ...t,
    halt: t.halt.slice(),
    cumulativeHalt: t.cumulativeHalt.slice(),
    deltaNorm: t.deltaNorm.slice(),
    stateNorm: t.stateNorm.slice(),
    changeNorm: t.changeNorm.slice(),
    perStep: t.perStep.map((p) => ({ ...p })),
  };
}

export function compileReasoningBlock(
  def: LayerDef,
  inputShape: number[],
  rng: RNG,
  defaultInit: InitMethod,
  existing: { params: Record<string, Tensor>; nonTrainable: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const plan = reasoningPlan(def.config, inputShape);
  const D = plan.width;
  const H = plan.hidden;
  const act = plan.activation as ActivationName;
  const init = (def.config.init as InitMethod) || defaultInit;

  const params: Record<string, Tensor> = {
    W1: maybeReuse(existing, "W1", [D, H]) ?? initTensor([D, H], init, rng, D, H),
    W2: maybeReuse(existing, "W2", [H, D]) ?? initTensor([H, D], init, rng, H, D),
    Wgate: maybeReuse(existing, "Wgate", [D, 2]) ?? Tensor.zeros([D, 2]),
    ln_gamma: maybeReuse(existing, "ln_gamma", [D]) ?? Tensor.ones([D]),
    ln_beta: maybeReuse(existing, "ln_beta", [D]) ?? Tensor.zeros([D]),
  };
  params.b1 = maybeReuse(existing, "b1", [H]) ?? Tensor.zeros([H]);
  params.b2 = maybeReuse(existing, "b2", [D]) ?? Tensor.zeros([D]);
  params.bgate = maybeReuse(existing, "bgate", [2]) ?? Tensor.zeros([2]);

  const grads = emptyGrads(params);
  let cache: StepCache[] | null = null;
  let trace: ReasoningTrace | null = null;

  return {
    def,
    inputShape,
    outputShape: [D],
    params,
    grads,
    nonTrainable: {},
    frozen,
    reasoning: {
      type: "reasoning_block",
      plan,
      get trace() {
        return copyTrace(trace);
      },
    },
    forward(x, training) {
      assert(x.rank === 2 && x.shape[1] === D, `Deep Reasoning ${def.name}: expected [B, ${D}], got ${shapeStr(x.shape)}`);
      const B = x.shape[0]!;
      const gamma = params.ln_gamma!.data;
      const beta = params.ln_beta!.data;
      const W1 = params.W1!.data;
      const W2 = params.W2!.data;
      const Wgate = params.Wgate!.data;
      const h = new Float32Array(x.data);
      const steps: StepCache[] = [];
      const haltTrace: number[] = [];
      const cumulativeTrace: number[] = [];
      const deltaNormTrace: number[] = [];
      const stateNormTrace: number[] = [];
      const changeNormTrace: number[] = [];
      const perStep: ReasoningStepTrace[] = [];
      let remaining = 1;
      let halted = false;

      for (let t = 0; t < plan.steps; t++) {
        const hBefore = new Float32Array(h);
        const ln = layerNormForward(hBefore, B, D, gamma, beta, plan.epsilon);
        const u = new Float32Array(ln.out);
        if (plan.stepCode) {
          const code = reasoningStepCode(t, D, plan.stepCodeScale);
          for (let b = 0; b < B; b++) {
            const off = b * D;
            for (let d = 0; d < D; d++) u[off + d]! += code[d]!;
          }
        }
        const pre = new Float32Array(B * H);
        mmInto(u, W1, B, D, H, pre);
        for (let b = 0; b < B; b++) {
          const off = b * H;
          for (let j = 0; j < H; j++) pre[off + j]! += params.b1!.data[j]!;
        }
        const actOut = act === "linear"
          ? pre
          : new Float32Array(applyActivation(act, new Tensor([B, H], pre)).data);
        const delta = new Float32Array(B * D);
        mmInto(actOut, W2, B, H, D, delta);
        for (let b = 0; b < B; b++) {
          const off = b * D;
          for (let d = 0; d < D; d++) delta[off + d]! += params.b2!.data[d]!;
        }
        const logits = new Float32Array(B * 2);
        mmInto(u, Wgate, B, D, 2, logits);
        const write = new Float32Array(B);
        const halt = new Float32Array(B);
        const gate = new Float32Array(B);
        for (let b = 0; b < B; b++) {
          write[b] = sigmoid1(logits[b * 2]! + params.bgate!.data[0]!);
          halt[b] = sigmoid1(logits[b * 2 + 1]! + params.bgate!.data[1]!);
          gate[b] = write[b]! * (1 - halt[b]!);
        }

        let meanHalt = 0;
        let meanDelta = 0;
        let meanState = 0;
        let meanChange = 0;
        for (let b = 0; b < B; b++) {
          const off = b * D;
          let d2 = 0;
          let s2 = 0;
          let c2 = 0;
          const g = gate[b]!;
          for (let d = 0; d < D; d++) {
            const dv = delta[off + d]!;
            const change = g * dv;
            h[off + d]! += change;
            d2 += dv * dv;
            c2 += change * change;
            const hv = h[off + d]!;
            s2 += hv * hv;
          }
          meanHalt += halt[b]!;
          meanDelta += Math.sqrt(d2);
          meanState += Math.sqrt(s2);
          meanChange += Math.sqrt(c2);
        }
        meanHalt /= B;
        meanDelta /= B;
        meanState /= B;
        meanChange /= B;

        steps.push({ hBefore, u, lnCache: ln.cache, pre, act: actOut, delta, write, halt, gate });
        remaining *= 1 - meanHalt;
        const cumulative = 1 - remaining;
        haltTrace.push(meanHalt);
        cumulativeTrace.push(cumulative);
        deltaNormTrace.push(meanDelta);
        stateNormTrace.push(meanState);
        changeNormTrace.push(meanChange);
        perStep.push({
          step: t,
          halt: meanHalt,
          cumulativeHalt: cumulative,
          deltaNorm: meanDelta,
          stateNorm: meanState,
          changeNorm: meanChange,
        });

        if (!training && plan.adaptive && t + 1 >= plan.minSteps && cumulative >= plan.haltThreshold) {
          halted = true;
          break;
        }
      }

      cache = steps;
      trace = {
        steps: plan.steps,
        stepsUsed: steps.length,
        adaptive: plan.adaptive,
        halted,
        halt: haltTrace,
        cumulativeHalt: cumulativeTrace,
        deltaNorm: deltaNormTrace,
        stateNorm: stateNormTrace,
        changeNorm: changeNormTrace,
        perStep,
      };
      return new Tensor([B, D], h);
    },
    backward(dy) {
      const steps = cache;
      assert(!!steps && steps.length > 0, `Deep Reasoning ${def.name}: backward called before forward`);
      const B = dy.shape[0]!;
      const D2 = dy.shape[1]!;
      assert(D2 === D, `Deep Reasoning ${def.name}: expected gradient [B, ${D}], got ${shapeStr(dy.shape)}`);
      let dh = new Float32Array(dy.data);

      const dW1 = new Float32Array(D * H);
      const db1 = new Float32Array(H);
      const dW2 = new Float32Array(H * D);
      const db2 = new Float32Array(D);
      const dWgate = new Float32Array(D * 2);
      const dbgate = new Float32Array(2);
      const dGamma = new Float32Array(D);
      const dBeta = new Float32Array(D);
      const W1 = params.W1!.data;
      const W2 = params.W2!.data;
      const Wgate = params.Wgate!.data;

      for (let t = steps.length - 1; t >= 0; t--) {
        const c = steps[t]!;
        const dHnext = dh;
        // h_next = h_before + gate ⊙ Δ: the residual path passes the gradient straight through.
        const dx = new Float32Array(dHnext);
        const dDelta = new Float32Array(B * D);
        const dGate = new Float32Array(B);
        for (let b = 0; b < B; b++) {
          const off = b * D;
          const g = c.gate[b]!;
          let acc = 0;
          for (let d = 0; d < D; d++) {
            const gv = dHnext[off + d]!;
            dDelta[off + d] = gv * g;
            acc += gv * c.delta[off + d]!;
          }
          dGate[b] = acc;
        }

        // Δ = act·W₂ + b₂
        mmAtInto(c.act, dDelta, B, H, D, dW2);
        colSum(dDelta, B, D, db2);
        const dAct = new Float32Array(B * H);
        mmBtInto(dDelta, W2, B, D, H, dAct);
        let dPre: Float32Array;
        if (act === "linear") {
          dPre = dAct;
        } else {
          dPre = new Float32Array(
            activationBackward(act, new Tensor([B, H], c.pre), new Tensor([B, H], c.act), new Tensor([B, H], dAct)).data,
          );
        }
        // pre = u·W₁ + b₁
        mmAtInto(c.u, dPre, B, D, H, dW1);
        colSum(dPre, B, H, db1);
        const dU = new Float32Array(B * D);
        mmBtInto(dPre, W1, B, H, D, dU);

        // gate = write·(1 − halt), both sigmoids of u·W_gate + b_gate
        for (let b = 0; b < B; b++) {
          const w = c.write[b]!;
          const q = c.halt[b]!;
          const dWrite = dGate[b]! * (1 - q);
          const dHalt = -dGate[b]! * w;
          const dLogitW = dWrite * w * (1 - w);
          const dLogitH = dHalt * q * (1 - q);
          dbgate[0]! += dLogitW;
          dbgate[1]! += dLogitH;
          const off = b * D;
          for (let d = 0; d < D; d++) {
            const uv = c.u[off + d]!;
            dWgate[d * 2]! += uv * dLogitW;
            dWgate[d * 2 + 1]! += uv * dLogitH;
            dU[off + d]! += dLogitW * Wgate[d * 2]! + dLogitH * Wgate[d * 2 + 1]!;
          }
        }

        // u = LayerNorm(h_before) + step code
        const lnGrad = layerNormBackward(dU, B, D, params.ln_gamma!.data, c.lnCache);
        for (let i = 0; i < dx.length; i++) dx[i]! += lnGrad.dx[i]!;
        for (let d = 0; d < D; d++) {
          dGamma[d]! += lnGrad.dGamma[d]!;
          dBeta[d]! += lnGrad.dBeta[d]!;
        }
        dh = dx;
      }

      grads.W1 = new Tensor([D, H], dW1);
      grads.b1 = new Tensor([H], db1);
      grads.W2 = new Tensor([H, D], dW2);
      grads.b2 = new Tensor([D], db2);
      grads.Wgate = new Tensor([D, 2], dWgate);
      grads.bgate = new Tensor([2], dbgate);
      grads.ln_gamma = new Tensor([D], dGamma);
      grads.ln_beta = new Tensor([D], dBeta);
      return new Tensor([B, D], dh);
    },
    resetCache() {
      cache = null;
      trace = null;
    },
  };
}
