/**
 * Multi-head self-attention and a full pre-LayerNorm transformer block.
 *
 * Both layers consume a batched sequence `[B, T, D]` (the declared shape is
 * `[T, D]`, exactly like the recurrent layers) and return either the whole
 * sequence `[B, T, M]` or a single pooled vector `[B, M]`.
 *
 * All of the numerics live here as raw Float32Array loops, for the same reason
 * the recurrent layers are written that way: the gradient derivation is explicit
 * and audit-able line by line, and the exported standalone runtime can mirror the
 * forward pass operation for operation.
 *
 * Shapes
 * -----
 *   B  batch,  T  timesteps,  D  input features,  M  model width (dModel)
 *   H  heads,  dh = M / H  (M must be divisible by H),  F  feed-forward width
 *
 * Attention (scaled dot product, optional causal mask)
 * -----
 *   X = input, projected to width M when D ≠ M
 *   Q = X·Wq, K = X·Wk, V = X·Wv                                  [B, T, M]
 *   S_h = Q_h · K_hᵀ · scale                                      [T, T]
 *   A_h = softmax(S_h + mask)      mask = −∞ above the diagonal when causal
 *   O_h = A_h · V_h                                               [T, dh]
 *   Y   = concat(O_0…O_{H−1})·Wo                                  [B, T, M]
 *
 * Optional dropout is the standard attention dropout: inverted scaling on the
 * probabilities, without renormalising the rows (this is what PyTorch's
 * multi_head_attention_forward does). Dropout is off at inference, so exported
 * models are unaffected by it.
 *
 * The transformer block wraps attention in the two classic residual sub-layers,
 * pre-normalised (the GPT ordering):
 *
 *   a = LN₁(x);  h = x + MHA(a);  y = h + act(LN₂(h)·W₁ + b₁)·W₂ + b₂
 */
import type { CompiledLayer } from "./layers";
import type { ActivationName, LayerDef } from "./types";
import { assert, numel, shapeStr } from "./types";
import { Tensor } from "./tensor";
import { initTensor } from "./init";
import { applyActivation, activationBackward } from "./activations";
import type { RNG } from "./rng";

/* ------------------------------------------------------------------ config */

export type PoolMode = "none" | "last" | "mean";

export interface AttentionPlan {
  heads: number;
  dModel: number;
  causal: boolean;
  useBias: boolean;
  dropout: number;
  scale: number;
  pool: PoolMode;
}

export interface BlockPlan extends AttentionPlan {
  ffDim: number;
  activation: string;
  epsilon: number;
}

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function bool(v: unknown, d: boolean): boolean {
  if (typeof v === "boolean") return v;
  if (v === 1 || v === 0) return v === 1;
  return d;
}

function poolMode(v: unknown): PoolMode {
  const s = String(v ?? "none");
  return s === "last" || s === "mean" ? s : "none";
}

/** Validate + normalise an attention config against a declared `[T, D]` input. */
export function attentionPlan(config: Record<string, unknown>, inputShape: number[]): AttentionPlan {
  if (inputShape.length !== 2) {
    throw new Error(
      `Attention expects a sequence [seqLen, features] (from an Embedding, or a [T, F] input shape), got ${shapeStr(inputShape)}.`,
    );
  }
  const rawWidth = Math.round(num(config.dModel, 0));
  const M = rawWidth > 0 ? rawWidth : inputShape[1]!;
  const heads = Math.max(1, Math.round(num(config.heads, 4)));
  if (M <= 0) throw new Error("Attention needs dModel > 0.");
  if (M % heads !== 0) {
    throw new Error(
      `Attention: dModel ${M} is not divisible by ${heads} heads. Pick a head count that divides the model width (e.g. 32 with 4 heads).`,
    );
  }
  return {
    heads,
    dModel: M,
    causal: bool(config.causal, true),
    useBias: bool(config.useBias, true),
    dropout: Math.min(0.9, Math.max(0, num(config.dropout, 0))),
    scale: num(config.scale, 1),
    pool: poolMode(config.pool),
  };
}

/** Validate + normalise a transformer-block config against a declared `[T, D]` input. */
export function blockPlan(config: Record<string, unknown>, inputShape: number[]): BlockPlan {
  const base = attentionPlan(config, inputShape);
  const rawFf = Math.round(num(config.ffDim, 0));
  return {
    ...base,
    ffDim: rawFf > 0 ? rawFf : 4 * base.dModel,
    activation: String(config.activation ?? "gelu"),
    epsilon: num(config.epsilon, 1e-5),
  };
}

/** Declared output shape (`[T, M]`, or `[M]` when pooled) for input shape `[T, D]`. */
export function attentionOutputShape(plan: { dModel: number; pool: PoolMode }, seqLen: number): number[] {
  return plan.pool === "none" ? [seqLen, plan.dModel] : [plan.dModel];
}

/** One-line summary for the Inspector / Builder. Never throws. */
export function attentionSummary(type: "attention" | "transformer_block", config: Record<string, unknown>): string {
  const heads = Math.max(1, Math.round(num(config.heads, 4)));
  const dModel = Math.max(1, Math.round(num(config.dModel, 0))) || 0;
  const causal = bool(config.causal, true);
  const pool = poolMode(config.pool);
  const width = dModel > 0 ? `d=${dModel}` : "d=input";
  const head = `${heads} head${heads > 1 ? "s" : ""} · ${width} · ${causal ? "causal" : "bidirectional"}`;
  const poolText = pool === "none" ? "sequence out" : `${pool}-pooled`;
  if (type === "attention") return `${head} · ${poolText}`;
  const rawFf = Math.round(num(config.ffDim, 0));
  const ff = rawFf > 0 ? rawFf : dModel > 0 ? 4 * dModel : 0;
  const act = String(config.activation ?? "gelu");
  return `${head} · ff=${ff > 0 ? ff : "4×d"} · ${act} · ${poolText}`;
}

/* ------------------------------------------------------- Float32 primitives */

/** C[M,N] += alpha · A[M,K] · B[K,N]. */
function mmInto(A: Float32Array, B: Float32Array, M: number, K: number, N: number, C: Float32Array, alpha = 1): void {
  for (let i = 0; i < M; i++) {
    const aOff = i * K;
    const cOff = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[aOff + k]! * alpha;
      if (a === 0) continue;
      const bOff = k * N;
      for (let j = 0; j < N; j++) C[cOff + j]! += a * B[bOff + j]!;
    }
  }
}

/** C[M,N] += alpha · Aᵀ · B with A [K,M] and B [K,N]. */
function mmAtInto(A: Float32Array, B: Float32Array, K: number, M: number, N: number, C: Float32Array, alpha = 1): void {
  for (let k = 0; k < K; k++) {
    const aOff = k * M;
    const bOff = k * N;
    for (let i = 0; i < M; i++) {
      const a = A[aOff + i]! * alpha;
      if (a === 0) continue;
      const cOff = i * N;
      for (let j = 0; j < N; j++) C[cOff + j]! += a * B[bOff + j]!;
    }
  }
}

/** C[M,N] += alpha · A · Bᵀ with A [M,K] and B [N,K]. */
function mmBtInto(A: Float32Array, B: Float32Array, M: number, K: number, N: number, C: Float32Array, alpha = 1): void {
  for (let i = 0; i < M; i++) {
    const aOff = i * K;
    const cOff = i * N;
    for (let j = 0; j < N; j++) {
      const bOff = j * K;
      let acc = 0;
      for (let k = 0; k < K; k++) acc += A[aOff + k]! * B[bOff + k]!;
      C[cOff + j]! += alpha * acc;
    }
  }
}

/** Column sums of a [rows, cols] matrix (used for bias gradients). */
function colSum(X: Float32Array, rows: number, cols: number, out: Float32Array): void {
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let j = 0; j < cols; j++) out[j]! += X[off + j]!;
  }
}

/** y[rows, cols] += bias (broadcast over rows). */
function addBias(Y: Float32Array, rows: number, cols: number, b: Float32Array): void {
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let j = 0; j < cols; j++) Y[off + j]! += b[j]!;
  }
}

/** Row-wise softmax in place over `cols`-wide rows starting at `offset`. */
function softmaxRows(S: Float32Array, offset: number, rows: number, cols: number): void {
  for (let r = 0; r < rows; r++) {
    const off = offset + r * cols;
    let max = -Infinity;
    for (let i = 0; i < cols; i++) {
      const v = S[off + i]!;
      if (v > max) max = v;
    }
    if (!Number.isFinite(max)) {
      const u = 1 / cols;
      for (let i = 0; i < cols; i++) S[off + i] = u;
      continue;
    }
    let sum = 0;
    for (let i = 0; i < cols; i++) {
      const e = Math.exp(S[off + i]! - max);
      S[off + i] = e;
      sum += e;
    }
    const inv = 1 / (sum || 1e-12);
    for (let i = 0; i < cols; i++) S[off + i]! *= inv;
  }
}

/** Softmax Jacobian-vector product, row-wise: out = A ⊙ (dA − ⟨A, dA⟩). */
function softmaxRowsBackward(
  A: Float32Array,
  aOff: number,
  dA: Float32Array,
  dOff: number,
  rows: number,
  cols: number,
  out: Float32Array,
  oOff: number,
): void {
  for (let r = 0; r < rows; r++) {
    const a = aOff + r * cols;
    const d = dOff + r * cols;
    const o = oOff + r * cols;
    let dot = 0;
    for (let i = 0; i < cols; i++) dot += A[a + i]! * dA[d + i]!;
    for (let i = 0; i < cols; i++) out[o + i] = A[a + i]! * (dA[d + i]! - dot);
  }
}

/* -------------------------------------------------------------- layer norm */

interface NormCache {
  xhat: Float32Array;
  invStd: Float32Array;
}

function layerNormForward(
  X: Float32Array,
  rows: number,
  cols: number,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number,
): { out: Float32Array; cache: NormCache } {
  const out = new Float32Array(X.length);
  const xhat = new Float32Array(X.length);
  const invStd = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let m = 0;
    for (let i = 0; i < cols; i++) m += X[off + i]!;
    m /= cols;
    let v = 0;
    for (let i = 0; i < cols; i++) {
      const d = X[off + i]! - m;
      v += d * d;
    }
    v /= cols;
    const inv = 1 / Math.sqrt(v + eps);
    invStd[r] = inv;
    for (let i = 0; i < cols; i++) {
      const nh = (X[off + i]! - m) * inv;
      xhat[off + i] = nh;
      out[off + i] = gamma[i]! * nh + beta[i]!;
    }
  }
  return { out, cache: { xhat, invStd } };
}

function layerNormBackward(
  dy: Float32Array,
  rows: number,
  cols: number,
  gamma: Float32Array,
  cache: NormCache,
): { dx: Float32Array; dGamma: Float32Array; dBeta: Float32Array } {
  const dx = new Float32Array(dy.length);
  const dGamma = new Float32Array(cols);
  const dBeta = new Float32Array(cols);
  const { xhat, invStd } = cache;
  for (let i = 0; i < dy.length; i++) {
    const c = i % cols;
    dBeta[c]! += dy[i]!;
    dGamma[c]! += dy[i]! * xhat[i]!;
  }
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    let sumDy = 0;
    let sumDyXhat = 0;
    for (let c = 0; c < cols; c++) {
      sumDy += dy[off + c]!;
      sumDyXhat += dy[off + c]! * xhat[off + c]!;
    }
    const inv = invStd[r]!;
    for (let c = 0; c < cols; c++) {
      dx[off + c] = ((gamma[c]! * inv) / cols) * (cols * dy[off + c]! - sumDy - xhat[off + c]! * sumDyXhat);
    }
  }
  return { dx, dGamma, dBeta };
}

/* ------------------------------------------------------- multi-head engine */

export interface MhaParams {
  Wq: Tensor;
  Wk: Tensor;
  Wv: Tensor;
  Wo: Tensor;
  bq?: Tensor;
  bk?: Tensor;
  bv?: Tensor;
  bo?: Tensor;
}

interface MhaDims {
  B: number;
  T: number;
  M: number;
  H: number;
  dh: number;
  causal: boolean;
  /** 1/√dh, already multiplied by the config's scale. */
  scale: number;
}

interface MhaCache {
  xin: Float32Array; // [B·T, M] the MHA input (after any input projection)
  Q: Float32Array;
  K: Float32Array;
  V: Float32Array;
  O: Float32Array; // concatenated head outputs, before Wo
  A: Float32Array; // [B·H, T·T] attention probabilities (post-softmax, pre-dropout)
  dropMask: Float32Array | null; // [B·H, T·T] inverted-dropout mask, or null
  dims: MhaDims;
}

function causalMasked(causal: boolean, i: number, j: number): boolean {
  return causal && j > i;
}

export function mhaForward(
  xin: Float32Array,
  dims: MhaDims,
  P: MhaParams,
  training: boolean,
  rng: RNG,
  dropout: number,
): { Y: Float32Array; cache: MhaCache } {
  const { B, T, M, H, dh, causal, scale } = dims;
  const rows = B * T;
  const Q = new Float32Array(rows * M);
  const K = new Float32Array(rows * M);
  const V = new Float32Array(rows * M);
  mmInto(xin, P.Wq.data, rows, M, M, Q);
  mmInto(xin, P.Wk.data, rows, M, M, K);
  mmInto(xin, P.Wv.data, rows, M, M, V);
  if (P.bq) addBias(Q, rows, M, P.bq.data);
  if (P.bk) addBias(K, rows, M, P.bk.data);
  if (P.bv) addBias(V, rows, M, P.bv.data);

  const A = new Float32Array(B * H * T * T);
  const useDropout = dropout > 0 && training;
  const dropMask = useDropout ? new Float32Array(B * H * T * T) : null;
  const O = new Float32Array(rows * M);
  const S = new Float32Array(T * T);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      S.fill(-Infinity);
      for (let i = 0; i < T; i++) {
        const qOff = (b * T + i) * M + h * dh;
        const sOff = i * T;
        for (let j = 0; j < T; j++) {
          if (causalMasked(causal, i, j)) continue;
          const kOff = (b * T + j) * M + h * dh;
          let acc = 0;
          for (let k = 0; k < dh; k++) acc += Q[qOff + k]! * K[kOff + k]!;
          S[sOff + j] = acc;
        }
      }
      if (scale !== 1) {
        for (let i = 0; i < T; i++) {
          const sOff = i * T;
          for (let j = 0; j < T; j++) {
            if (causalMasked(causal, i, j)) continue;
            S[sOff + j]! *= scale;
          }
        }
      }
      softmaxRows(S, 0, T, T);
      const aOff = (b * H + h) * T * T;
      A.set(S, aOff);
      if (dropMask) {
        const inv = 1 / (1 - dropout);
        for (let i = 0; i < T * T; i++) {
          const m = rng.next() >= dropout ? inv : 0;
          dropMask[aOff + i] = m;
          A[aOff + i]! *= m;
        }
      }
      // O_h = A_h · V_h  (with the dropout scaling folded into A when training)
      for (let i = 0; i < T; i++) {
        const sOff = aOff + i * T;
        const oOff = (b * T + i) * M + h * dh;
        for (let k = 0; k < dh; k++) {
          let acc = 0;
          for (let j = 0; j < T; j++) acc += A[sOff + j]! * V[(b * T + j) * M + h * dh + k]!;
          O[oOff + k] = acc;
        }
      }
    }
  }

  const Y = new Float32Array(rows * M);
  mmInto(O, P.Wo.data, rows, M, M, Y);
  if (P.bo) addBias(Y, rows, M, P.bo.data);
  return { Y, cache: { xin, Q, K, V, O, A, dropMask, dims } };
}

export interface MhaGrads {
  dXin: Float32Array;
  dWq: Float32Array;
  dWk: Float32Array;
  dWv: Float32Array;
  dWo: Float32Array;
  dbq: Float32Array;
  dbk: Float32Array;
  dbv: Float32Array;
  dbo: Float32Array;
}

export function mhaBackward(dY: Float32Array, cache: MhaCache, P: MhaParams): MhaGrads {
  const { xin, Q, K, V, O, A, dropMask, dims } = cache;
  const { B, T, M, H, dh, causal, scale } = dims;
  const rows = B * T;

  // dO = dY·Woᵀ ; dWo = Oᵀ·dY ; dbo = ΣdY
  const dO = new Float32Array(rows * M);
  const dWo = new Float32Array(M * M);
  const dbo = new Float32Array(M);
  mmBtInto(dY, P.Wo.data, rows, M, M, dO);
  mmAtInto(O, dY, rows, M, M, dWo);
  colSum(dY, rows, M, dbo);

  const dQ = new Float32Array(rows * M);
  const dK = new Float32Array(rows * M);
  const dV = new Float32Array(rows * M);
  const dS = new Float32Array(T * T);
  const dA = new Float32Array(T * T);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const aOff = (b * H + h) * T * T;
      // dA = dO_h · V_hᵀ
      for (let i = 0; i < T; i++) {
        const oOff = (b * T + i) * M + h * dh;
        for (let j = 0; j < T; j++) {
          const vOff = (b * T + j) * M + h * dh;
          let acc = 0;
          for (let k = 0; k < dh; k++) acc += dO[oOff + k]! * V[vOff + k]!;
          dA[i * T + j] = acc;
        }
      }
      // dV_h = A_hᵀ · dO_h
      for (let j = 0; j < T; j++) {
        const vOff = (b * T + j) * M + h * dh;
        for (let k = 0; k < dh; k++) {
          let acc = 0;
          for (let i = 0; i < T; i++) acc += A[aOff + i * T + j]! * dO[(b * T + i) * M + h * dh + k]!;
          dV[vOff + k]! += acc;
        }
      }
      // Undo the inverted-dropout scaling, then push through the softmax.
      if (dropMask) for (let i = 0; i < T * T; i++) dA[i]! *= dropMask[aOff + i]!;
      softmaxRowsBackward(A, aOff, dA, 0, T, T, dS, 0);
      for (let i = 0; i < T; i++) {
        const sOff = i * T;
        for (let j = 0; j < T; j++) if (causalMasked(causal, i, j)) dS[sOff + j] = 0;
      }
      if (scale !== 1) for (let i = 0; i < T * T; i++) dS[i]! *= scale;
      // dQ_h = dS·K_h ; dK_h = dSᵀ·Q_h
      for (let i = 0; i < T; i++) {
        const qOff = (b * T + i) * M + h * dh;
        const sOff = i * T;
        for (let k = 0; k < dh; k++) {
          let acc = 0;
          for (let j = 0; j < T; j++) acc += dS[sOff + j]! * K[(b * T + j) * M + h * dh + k]!;
          dQ[qOff + k]! += acc;
        }
      }
      for (let j = 0; j < T; j++) {
        const kOff = (b * T + j) * M + h * dh;
        for (let k = 0; k < dh; k++) {
          let acc = 0;
          for (let i = 0; i < T; i++) acc += dS[i * T + j]! * Q[(b * T + i) * M + h * dh + k]!;
          dK[kOff + k]! += acc;
        }
      }
    }
  }

  const dWq = new Float32Array(M * M);
  const dWk = new Float32Array(M * M);
  const dWv = new Float32Array(M * M);
  mmAtInto(xin, dQ, rows, M, M, dWq);
  mmAtInto(xin, dK, rows, M, M, dWk);
  mmAtInto(xin, dV, rows, M, M, dWv);
  const dbq = new Float32Array(M);
  const dbk = new Float32Array(M);
  const dbv = new Float32Array(M);
  colSum(dQ, rows, M, dbq);
  colSum(dK, rows, M, dbk);
  colSum(dV, rows, M, dbv);

  const dXin = new Float32Array(rows * M);
  mmBtInto(dQ, P.Wq.data, rows, M, M, dXin);
  mmBtInto(dK, P.Wk.data, rows, M, M, dXin);
  mmBtInto(dV, P.Wv.data, rows, M, M, dXin);

  return { dXin, dWq, dWk, dWv, dWo, dbq, dbk, dbv, dbo };
}

/* ----------------------------------------------------------- param plumbing */

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

function xavier(
  rng: RNG,
  existing: { params?: Record<string, Tensor> } | undefined,
  name: string,
  shape: number[],
  fanIn: number,
  fanOut: number,
): Tensor {
  return maybeReuse(existing, name, shape) ?? initTensor(shape, "xavier_uniform", rng, fanIn, fanOut);
}

/* ---------------------------------------------------------------- attention */

export function compileAttention(
  def: LayerDef,
  inputShape: number[],
  rng: RNG,
  existing: { params: Record<string, Tensor>; nonTrainable: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const plan = attentionPlan(def.config, inputShape);
  const D = inputShape[1]!;
  const M = plan.dModel;
  const projNeeded = D !== M;
  const dh = M / plan.heads;

  const params: Record<string, Tensor> = {
    Wq: xavier(rng, existing, "Wq", [M, M], M, M),
    Wk: xavier(rng, existing, "Wk", [M, M], M, M),
    Wv: xavier(rng, existing, "Wv", [M, M], M, M),
    Wo: xavier(rng, existing, "Wo", [M, M], M, M),
  };
  if (plan.useBias) {
    params.bq = maybeReuse(existing, "bq", [M]) ?? Tensor.zeros([M]);
    params.bk = maybeReuse(existing, "bk", [M]) ?? Tensor.zeros([M]);
    params.bv = maybeReuse(existing, "bv", [M]) ?? Tensor.zeros([M]);
    params.bo = maybeReuse(existing, "bo", [M]) ?? Tensor.zeros([M]);
  }
  if (projNeeded) {
    params.Win = xavier(rng, existing, "Win", [D, M], D, M);
    params.bin_b = maybeReuse(existing, "bin_b", [M]) ?? Tensor.zeros([M]);
  }

  const grads = emptyGrads(params);
  interface Cache {
    xInput: Float32Array;
    xShape: number[];
    B: number;
    T: number;
    mha: MhaCache;
  }
  let cache: Cache | null = null;

  return {
    def,
    inputShape,
    outputShape: attentionOutputShape(plan, inputShape[0]!),
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x, training) {
      assert(x.rank === 3, `Attention ${def.name}: expected a batched sequence [B, T, ${D}], got ${shapeStr(x.shape)}`);
      const B = x.shape[0]!;
      const T = x.shape[1]!;
      const rows = B * T;
      const xInput = x.data as Float32Array;
      let xin: Float32Array = xInput;
      if (projNeeded) {
        xin = new Float32Array(rows * M);
        mmInto(xInput, params.Win!.data, rows, D, M, xin);
        addBias(xin, rows, M, params.bin_b!.data);
      }
      const dims: MhaDims = { B, T, M, H: plan.heads, dh, causal: plan.causal, scale: plan.scale / Math.sqrt(dh) };
      const { Y, cache: mha } = mhaForward(xin, dims, params as unknown as MhaParams, training, rng, plan.dropout);
      cache = { xInput, xShape: x.shape as number[], B, T, mha };
      return poolOutput(Y, B, T, M, plan.pool);
    },
    backward(dy) {
      const c = cache!;
      const { B, T } = c;
      const dY = expandPooled(dy.data, B, T, M, plan.pool);
      const g = mhaBackward(dY, c.mha, params as unknown as MhaParams);
      grads.Wq = new Tensor([M, M], g.dWq);
      grads.Wk = new Tensor([M, M], g.dWk);
      grads.Wv = new Tensor([M, M], g.dWv);
      grads.Wo = new Tensor([M, M], g.dWo);
      if (plan.useBias) {
        grads.bq = new Tensor([M], g.dbq);
        grads.bk = new Tensor([M], g.dbk);
        grads.bv = new Tensor([M], g.dbv);
        grads.bo = new Tensor([M], g.dbo);
      }
      if (!projNeeded) return new Tensor(c.xShape, g.dXin);
      const rows = B * T;
      const dWin = new Float32Array(D * M);
      const dbin = new Float32Array(M);
      mmAtInto(c.xInput, g.dXin, rows, D, M, dWin);
      colSum(g.dXin, rows, M, dbin);
      grads.Win = new Tensor([D, M], dWin);
      grads.bin_b = new Tensor([M], dbin);
      const dx = new Float32Array(rows * D);
      mmBtInto(g.dXin, params.Win!.data, rows, M, D, dx);
      return new Tensor(c.xShape, dx);
    },
    resetCache() {
      cache = null;
    },
  };
}

/** Reduce a `[B, T, M]` block output to `[B, T, M]`, `[B, M]` (last) or `[B, M]` (mean). */
function poolOutput(Y: Float32Array, B: number, T: number, M: number, pool: PoolMode): Tensor {
  if (pool === "none") return new Tensor([B, T, M], Y);
  const pooled = new Float32Array(B * M);
  if (pool === "last") {
    for (let b = 0; b < B; b++) {
      const off = (b * T + T - 1) * M;
      pooled.set(Y.subarray(off, off + M), b * M);
    }
  } else {
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const off = (b * T + t) * M;
        for (let j = 0; j < M; j++) pooled[b * M + j]! += Y[off + j]! / T;
      }
    }
  }
  return new Tensor([B, M], pooled);
}

/** Inverse of `poolOutput`'s reduction: broadcast a pooled gradient back over T. */
function expandPooled(dy: Float32Array, B: number, T: number, M: number, pool: PoolMode): Float32Array {
  if (pool === "none") return new Float32Array(dy);
  const out = new Float32Array(B * T * M);
  if (pool === "last") {
    for (let b = 0; b < B; b++) out.set(dy.subarray(b * M, (b + 1) * M), b * T * M + (T - 1) * M);
  } else {
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const off = (b * T + t) * M;
        for (let j = 0; j < M; j++) out[off + j] = dy[b * M + j]! / T;
      }
    }
  }
  return out;
}

/* -------------------------------------------------------- transformer block */

export function compileTransformerBlock(
  def: LayerDef,
  inputShape: number[],
  rng: RNG,
  existing: { params: Record<string, Tensor>; nonTrainable: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const plan = blockPlan(def.config, inputShape);
  const D = inputShape[1]!;
  const M = plan.dModel;
  const F = plan.ffDim;
  const projNeeded = D !== M;
  const eps = plan.epsilon;
  const dh = M / plan.heads;
  const act = plan.activation as ActivationName | "none";
  const useBias = plan.useBias;

  const params: Record<string, Tensor> = {
    Wq: xavier(rng, existing, "Wq", [M, M], M, M),
    Wk: xavier(rng, existing, "Wk", [M, M], M, M),
    Wv: xavier(rng, existing, "Wv", [M, M], M, M),
    Wo: xavier(rng, existing, "Wo", [M, M], M, M),
    ln1_gamma: maybeReuse(existing, "ln1_gamma", [M]) ?? Tensor.ones([M]),
    ln1_beta: maybeReuse(existing, "ln1_beta", [M]) ?? Tensor.zeros([M]),
    ln2_gamma: maybeReuse(existing, "ln2_gamma", [M]) ?? Tensor.ones([M]),
    ln2_beta: maybeReuse(existing, "ln2_beta", [M]) ?? Tensor.zeros([M]),
    W1: xavier(rng, existing, "W1", [M, F], M, F),
    W2: xavier(rng, existing, "W2", [F, M], F, M),
  };
  if (useBias) {
    params.bq = maybeReuse(existing, "bq", [M]) ?? Tensor.zeros([M]);
    params.bk = maybeReuse(existing, "bk", [M]) ?? Tensor.zeros([M]);
    params.bv = maybeReuse(existing, "bv", [M]) ?? Tensor.zeros([M]);
    params.bo = maybeReuse(existing, "bo", [M]) ?? Tensor.zeros([M]);
    params.b1 = maybeReuse(existing, "b1", [F]) ?? Tensor.zeros([F]);
    params.b2 = maybeReuse(existing, "b2", [M]) ?? Tensor.zeros([M]);
  }
  if (projNeeded) {
    params.Win = xavier(rng, existing, "Win", [D, M], D, M);
    params.bin_b = maybeReuse(existing, "bin_b", [M]) ?? Tensor.zeros([M]);
  }

  const grads = emptyGrads(params);

  interface Cache {
    xInput: Float32Array;
    xShape: number[];
    B: number;
    T: number;
    ln1Out: Float32Array;
    ln1Cache: NormCache;
    mha: MhaCache;
    ln2Out: Float32Array;
    ln2Cache: NormCache;
    ffPre: Float32Array | null;
    ffAct: Float32Array | null;
  }
  let cache: Cache | null = null;

  return {
    def,
    inputShape,
    outputShape: attentionOutputShape(plan, inputShape[0]!),
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x, training) {
      assert(x.rank === 3, `Transformer block ${def.name}: expected a batched sequence [B, T, ${D}], got ${shapeStr(x.shape)}`);
      const B = x.shape[0]!;
      const T = x.shape[1]!;
      const rows = B * T;
      const xInput = x.data as Float32Array;
      let xin: Float32Array = xInput;
      if (projNeeded) {
        xin = new Float32Array(rows * M);
        mmInto(xInput, params.Win!.data, rows, D, M, xin);
        addBias(xin, rows, M, params.bin_b!.data);
      }

      const ln1 = layerNormForward(xin, rows, M, params.ln1_gamma.data, params.ln1_beta.data, eps);
      const dims: MhaDims = { B, T, M, H: plan.heads, dh, causal: plan.causal, scale: plan.scale / Math.sqrt(dh) };
      const { Y: attn, cache: mha } = mhaForward(ln1.out, dims, params as unknown as MhaParams, training, rng, plan.dropout);
      const h = new Float32Array(rows * M);
      for (let i = 0; i < h.length; i++) h[i] = xin[i]! + attn[i]!;

      const ln2 = layerNormForward(h, rows, M, params.ln2_gamma.data, params.ln2_beta.data, eps);
      let ffPre: Float32Array | null = null;
      let ffAct: Float32Array | null = null;
      const ffHidden = new Float32Array(rows * F);
      mmInto(ln2.out, params.W1!.data, rows, M, F, ffHidden);
      if (params.b1) addBias(ffHidden, rows, F, params.b1.data);
      if (act === "none" || act === "linear") {
        ffAct = ffHidden;
      } else {
        ffPre = ffHidden;
        ffAct = new Float32Array(applyActivation(act, new Tensor([rows, F], ffHidden)).data);
      }
      const ffOut = new Float32Array(rows * M);
      mmInto(ffAct, params.W2!.data, rows, F, M, ffOut);
      if (params.b2) addBias(ffOut, rows, M, params.b2.data);

      const Y = new Float32Array(rows * M);
      for (let i = 0; i < Y.length; i++) Y[i] = h[i]! + ffOut[i]!;

      cache = { xInput, xShape: x.shape as number[], B, T, ln1Out: ln1.out, ln1Cache: ln1.cache, mha, ln2Out: ln2.out, ln2Cache: ln2.cache, ffPre, ffAct };
      return poolOutput(Y, B, T, M, plan.pool);
    },
    backward(dy) {
      const c = cache!;
      const { B, T } = c;
      const rows = B * T;
      const dY = expandPooled(dy.data, B, T, M, plan.pool);

      // Second residual: y = h + ffOut, so dH accumulates dY and the FFN's share.
      const dH = new Float32Array(dY);

      // --- feed-forward branch: ffOut = ffAct·W2 + b2
      const dW1 = new Float32Array(M * F);
      const dW2 = new Float32Array(F * M);
      const db1 = new Float32Array(F);
      const db2 = new Float32Array(M);
      colSum(dY, rows, M, db2);
      mmAtInto(c.ffAct!, dY, rows, F, M, dW2);
      const dffAct = new Float32Array(rows * F);
      mmBtInto(dY, params.W2!.data, rows, M, F, dffAct);
      let dffPre = dffAct;
      if (c.ffPre && c.ffAct && act !== "none" && act !== "linear") {
        const preT = new Tensor([rows, F], c.ffPre);
        const outT = new Tensor([rows, F], c.ffAct);
        dffPre = new Float32Array(activationBackward(act, preT, outT, new Tensor([rows, F], dffAct)).data);
      }
      colSum(dffPre, rows, F, db1);
      mmAtInto(c.ln2Out, dffPre, rows, M, F, dW1);
      const dln2 = new Float32Array(rows * M);
      mmBtInto(dffPre, params.W1!.data, rows, F, M, dln2);
      const ln2Grad = layerNormBackward(dln2, rows, M, params.ln2_gamma.data, c.ln2Cache);
      for (let i = 0; i < dH.length; i++) dH[i]! += ln2Grad.dx[i]!;

      // --- attention branch: h = xin + MHA(ln1(xin)), so dAttn = dH
      const g = mhaBackward(dH, c.mha, params as unknown as MhaParams);
      const ln1Grad = layerNormBackward(g.dXin, rows, M, params.ln1_gamma.data, c.ln1Cache);
      const dXin = new Float32Array(rows * M);
      for (let i = 0; i < dXin.length; i++) dXin[i] = dH[i]! + ln1Grad.dx[i]!;

      grads.Wq = new Tensor([M, M], g.dWq);
      grads.Wk = new Tensor([M, M], g.dWk);
      grads.Wv = new Tensor([M, M], g.dWv);
      grads.Wo = new Tensor([M, M], g.dWo);
      grads.W1 = new Tensor([M, F], dW1);
      grads.W2 = new Tensor([F, M], dW2);
      grads.ln1_gamma = new Tensor([M], ln1Grad.dGamma);
      grads.ln1_beta = new Tensor([M], ln1Grad.dBeta);
      grads.ln2_gamma = new Tensor([M], ln2Grad.dGamma);
      grads.ln2_beta = new Tensor([M], ln2Grad.dBeta);
      if (useBias) {
        grads.bq = new Tensor([M], g.dbq);
        grads.bk = new Tensor([M], g.dbk);
        grads.bv = new Tensor([M], g.dbv);
        grads.bo = new Tensor([M], g.dbo);
        grads.b1 = new Tensor([F], db1);
        grads.b2 = new Tensor([M], db2);
      }
      if (!projNeeded) return new Tensor(c.xShape, dXin);
      const dWin = new Float32Array(D * M);
      const dbin = new Float32Array(M);
      mmAtInto(c.xInput, dXin, rows, D, M, dWin);
      colSum(dXin, rows, M, dbin);
      grads.Win = new Tensor([D, M], dWin);
      grads.bin_b = new Tensor([M], dbin);
      const dx = new Float32Array(rows * D);
      mmBtInto(dXin, params.Win!.data, rows, M, D, dx);
      return new Tensor(c.xShape, dx);
    },
    resetCache() {
      cache = null;
    },
  };
}
