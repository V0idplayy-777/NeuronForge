import type { ActivationName, InitMethod, LayerDef, LayerType } from "./types";
import { assert, numel, shapeStr, uid } from "./types";
import { Tensor, matmul, transpose2d, addBias, sumAxis0, softmaxLast, softmaxBackward } from "./tensor";
import {
  applyActivation,
  activationBackward,
  relu,
  reluBackward,
  leakyRelu,
  leakyReluBackward,
  gelu,
  geluBackward,
  silu,
  siluBackward,
  sigmoid,
  sigmoidBackward,
  tanhAct,
  tanhBackward,
} from "./activations";
import { initTensor } from "./init";
import type { RNG } from "./rng";

export interface CompiledLayer {
  def: LayerDef;
  inputShape: number[];
  outputShape: number[];
  params: Record<string, Tensor>;
  grads: Record<string, Tensor>;
  nonTrainable: Record<string, Tensor>;
  frozen: boolean;
  forward(x: Tensor, training: boolean): Tensor;
  backward(dy: Tensor): Tensor;
  resetCache(): void;
}

export function defaultConfig(type: LayerType): Record<string, unknown> {
  switch (type) {
    case "dense":
      return { units: 8, useBias: true, activation: "none" };
    case "dropout":
      return { rate: 0.2 };
    case "batchnorm":
      return { epsilon: 1e-5, momentum: 0.9 };
    case "layernorm":
      return { epsilon: 1e-5 };
    case "conv2d":
      return { filters: 8, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" };
    case "maxpool2d":
    case "avgpool2d":
      return { poolSize: 2, stride: 2 };
    case "gap2d":
      return {};
    case "flatten":
    case "identity":
      return {};
    case "reshape":
      return { targetShape: [1] };
    case "embedding":
      return { vocabSize: 32, embeddingDim: 8 };
    case "leakyrelu":
      return { alpha: 0.01 };
    default:
      return {};
  }
}

export function layerDisplayName(type: LayerType): string {
  const map: Record<LayerType, string> = {
    dense: "Dense",
    flatten: "Flatten",
    reshape: "Reshape",
    dropout: "Dropout",
    batchnorm: "Batch Norm",
    layernorm: "Layer Norm",
    conv2d: "Conv2D",
    maxpool2d: "MaxPool2D",
    avgpool2d: "AvgPool2D",
    gap2d: "GlobalAvgPool",
    embedding: "Embedding",
    rnn: "RNN",
    identity: "Identity",
    relu: "ReLU",
    leakyrelu: "Leaky ReLU",
    gelu: "GELU",
    silu: "SiLU",
    sigmoid: "Sigmoid",
    tanh: "Tanh",
    softmax: "Softmax",
  };
  return map[type];
}

export function makeLayerDef(type: LayerType, overrides?: Partial<LayerDef>): LayerDef {
  const id = overrides?.id ?? uid(type);
  return {
    id,
    name: overrides?.name ?? `${type}_${id.slice(-4)}`,
    type,
    config: { ...defaultConfig(type), ...(overrides?.config ?? {}) },
    frozen: overrides?.frozen ?? false,
  };
}

export function inferOutputShape(type: LayerType, config: Record<string, unknown>, inputShape: number[]): number[] {
  switch (type) {
    case "dense": {
      if (inputShape.length !== 1) {
        throw new Error(`Dense expects rank-1 input, got ${shapeStr(inputShape)}. Add a Flatten layer first.`);
      }
      return [num(config.units, 8)];
    }
    case "flatten":
      return [numel(inputShape)];
    case "reshape": {
      const t = (config.targetShape as number[]) ?? [1];
      if (numel(t) !== numel(inputShape)) {
        throw new Error(`Reshape size mismatch: ${shapeStr(inputShape)} (${numel(inputShape)}) → ${shapeStr(t)} (${numel(t)})`);
      }
      return t.slice();
    }
    case "dropout":
    case "batchnorm":
    case "layernorm":
    case "identity":
    case "relu":
    case "leakyrelu":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
    case "softmax":
      return inputShape.slice();
    case "conv2d": {
      if (inputShape.length !== 3) {
        throw new Error(`Conv2D expects [H, W, C], got ${shapeStr(inputShape)}`);
      }
      const [H, W] = inputShape;
      const k = kernelWH(config.kernelSize);
      const stride = num(config.stride, 1);
      const pad = String(config.padding ?? "same");
      const { outH, outW } = convOut(H!, W!, k[0], k[1], stride, pad);
      return [outH, outW, num(config.filters, 8)];
    }
    case "maxpool2d":
    case "avgpool2d": {
      if (inputShape.length !== 3) throw new Error(`${type} expects [H, W, C], got ${shapeStr(inputShape)}`);
      const [H, W, C] = inputShape;
      const p = num(config.poolSize, 2);
      const stride = num(config.stride, p);
      const outH = Math.floor((H! - p) / stride) + 1;
      const outW = Math.floor((W! - p) / stride) + 1;
      if (outH <= 0 || outW <= 0) throw new Error(`${type}: pooling ${p}×${p} does not fit input ${shapeStr(inputShape)}`);
      return [outH, outW, C!];
    }
    case "gap2d": {
      if (inputShape.length !== 3) throw new Error(`GlobalAvgPool expects [H, W, C], got ${shapeStr(inputShape)}`);
      return [inputShape[2]!];
    }
    case "embedding": {
      if (inputShape.length !== 1) throw new Error(`Embedding expects [seqLen], got ${shapeStr(inputShape)}`);
      return [inputShape[0]!, num(config.embeddingDim, 8)];
    }
    case "rnn": {
      if (inputShape.length !== 2) {
        throw new Error(`RNN expects a sequence [seqLen, features] (e.g. from an Embedding), got ${shapeStr(inputShape)}`);
      }
      const units = num(config.units, 32);
      return bool(config.returnSequences, false) ? [inputShape[0]!, units] : [units];
    }
  }
}

export function compileLayer(
  def: LayerDef,
  inputShape: number[],
  rng: RNG,
  defaultInit: InitMethod,
  existing?: { params: Record<string, Tensor>; nonTrainable: Record<string, Tensor> },
): CompiledLayer {
  const outputShape = inferOutputShape(def.type, def.config, inputShape);
  const frozen = def.frozen;
  switch (def.type) {
    case "dense":
      return compileDense(def, inputShape, outputShape, rng, defaultInit, existing, frozen);
    case "flatten":
      return compileFlatten(def, inputShape, outputShape, frozen);
    case "reshape":
      return compileReshape(def, inputShape, outputShape, frozen);
    case "dropout":
      return compileDropout(def, inputShape, outputShape, rng, frozen);
    case "batchnorm":
      return compileBatchNorm(def, inputShape, outputShape, existing, frozen);
    case "layernorm":
      return compileLayerNorm(def, inputShape, outputShape, existing, frozen);
    case "conv2d":
      return compileConv2d(def, inputShape, outputShape, rng, defaultInit, existing, frozen);
    case "maxpool2d":
      return compileMaxPool(def, inputShape, outputShape, frozen);
    case "avgpool2d":
      return compileAvgPool(def, inputShape, outputShape, frozen);
    case "gap2d":
      return compileGap(def, inputShape, outputShape, frozen);
    case "embedding":
      return compileEmbedding(def, inputShape, outputShape, rng, defaultInit, existing, frozen);
    case "rnn":
      return compileRnn(def, inputShape, outputShape, rng, existing, frozen);
    case "identity":
      return compileIdentity(def, inputShape, outputShape, frozen);
    case "relu":
    case "leakyrelu":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
    case "softmax":
      return compileActivation(def, inputShape, outputShape, frozen);
  }
}

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function bool(v: unknown, d: boolean): boolean {
  if (typeof v === "boolean") return v;
  return d;
}

function kernelWH(v: unknown): [number, number] {
  if (Array.isArray(v) && v.length >= 2) return [Number(v[0]), Number(v[1])];
  const n = num(v, 3);
  return [n, n];
}

function convOut(H: number, W: number, kH: number, kW: number, stride: number, padding: string): { outH: number; outW: number; padH: number; padW: number } {
  if (padding === "same") {
    const outH = Math.ceil(H / stride);
    const outW = Math.ceil(W / stride);
    const padH = Math.max(0, (outH - 1) * stride + kH - H);
    const padW = Math.max(0, (outW - 1) * stride + kW - W);
    return { outH, outW, padH, padW };
  }
  const outH = Math.floor((H - kH) / stride) + 1;
  const outW = Math.floor((W - kW) / stride) + 1;
  return { outH, outW, padH: 0, padW: 0 };
}

function emptyGrads(params: Record<string, Tensor>): Record<string, Tensor> {
  const g: Record<string, Tensor> = {};
  for (const k of Object.keys(params)) g[k] = Tensor.zeros(params[k]!.shape as number[]);
  return g;
}

function maybeReuse(existing: { params: Record<string, Tensor> } | undefined, name: string, shape: number[]): Tensor | null {
  const t = existing?.params[name];
  if (!t) return null;
  if (t.size !== numel(shape)) return null;
  if (t.shape.length !== shape.length || t.shape.some((d, i) => d !== shape[i])) {
    return t.reshape(shape);
  }
  return t;
}

function compileDense(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  rng: RNG,
  defaultInit: InitMethod,
  existing: { params: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const units = outputShape[0]!;
  const din = inputShape[0]!;
  const useBias = bool(def.config.useBias, true);
  const act = String(def.config.activation ?? "none") as ActivationName | "none";
  const init = (def.config.init as InitMethod) || defaultInit;
  const W = maybeReuse(existing, "W", [din, units]) ?? initTensor([din, units], init, rng, din, units);
  const b = useBias ? (maybeReuse(existing, "b", [units]) ?? Tensor.zeros([units])) : Tensor.zeros([0]);
  const params: Record<string, Tensor> = useBias ? { W, b } : { W };
  const grads = emptyGrads(params);
  let lastX: Tensor | null = null;
  let lastPre: Tensor | null = null;
  let lastY: Tensor | null = null;

  return {
    def,
    inputShape,
    outputShape,
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x: Tensor) {
      assert(x.rank === 2 && x.shape[1] === din, `Dense ${def.name}: expected [B, ${din}], got ${shapeStr(x.shape)}`);
      lastX = x;
      let y = matmul(x, W);
      if (useBias) y = addBias(y, b);
      lastPre = y;
      if (act !== "none") {
        y = applyActivation(act, y);
      }
      lastY = y;
      return y;
    },
    backward(dy: Tensor) {
      let g = dy;
      if (act !== "none" && lastPre && lastY) {
        g = activationBackward(act, lastPre, lastY, dy);
      }
      const x = lastX!;
      grads.W = matmul(transpose2d(x), g);
      if (useBias) grads.b = sumAxis0(g);
      return matmul(g, transpose2d(W));
    },
    resetCache() {
      lastX = lastPre = lastY = null;
    },
  };
}

function compileFlatten(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  let inShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      inShape = x.shape as number[];
      const B = x.shape[0]!;
      return x.reshape([B, x.size / B]);
    },
    backward(dy) {
      return dy.reshape(inShape);
    },
    resetCache() {
      inShape = [];
    },
  };
}

function compileReshape(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  let inShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      inShape = x.shape as number[];
      const B = x.shape[0]!;
      return x.reshape([B, ...outputShape]);
    },
    backward(dy) {
      return dy.reshape(inShape);
    },
    resetCache() {
      inShape = [];
    },
  };
}

function compileDropout(def: LayerDef, inputShape: number[], outputShape: number[], rng: RNG, frozen: boolean): CompiledLayer {
  const rate = Math.min(0.95, Math.max(0, num(def.config.rate, 0.2)));
  let mask: Float32Array | null = null;
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x, training) {
      if (!training || rate <= 0) {
        mask = null;
        return x;
      }
      const keep = 1 - rate;
      const scale = 1 / keep;
      const m = new Float32Array(x.size);
      const out = new Float32Array(x.size);
      for (let i = 0; i < x.size; i++) {
        const keepIt = rng.next() >= rate ? 1 : 0;
        m[i] = keepIt * scale;
        out[i] = x.data[i]! * m[i]!;
      }
      mask = m;
      return new Tensor(x.shape as number[], out);
    },
    backward(dy) {
      if (!mask) return dy;
      const out = new Float32Array(dy.size);
      for (let i = 0; i < dy.size; i++) out[i] = dy.data[i]! * mask[i]!;
      return new Tensor(dy.shape as number[], out);
    },
    resetCache() {
      mask = null;
    },
  };
}

function compileBatchNorm(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  existing: { params?: Record<string, Tensor>; nonTrainable?: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const features = inputShape[inputShape.length - 1]!;
  const eps = num(def.config.epsilon, 1e-5);
  const momentum = num(def.config.momentum, 0.9);
  const gamma = maybeReuse(existing as { params: Record<string, Tensor> }, "gamma", [features]) ?? Tensor.ones([features]);
  const beta = maybeReuse(existing as { params: Record<string, Tensor> }, "beta", [features]) ?? Tensor.zeros([features]);
  const params = { gamma, beta };
  const grads = emptyGrads(params);
  const runningMean =
    existing?.nonTrainable?.runningMean?.clone() ?? Tensor.zeros([features]);
  const runningVar =
    existing?.nonTrainable?.runningVar?.clone() ?? Tensor.ones([features]);
  let cache: { xhat: Tensor; std: Float32Array; x: Tensor; mean: Float32Array } | null = null;

  return {
    def,
    inputShape,
    outputShape,
    params,
    grads,
    nonTrainable: { runningMean, runningVar },
    frozen,
    forward(x, training) {
      const inner = features;
      const outer = x.size / inner;
      const out = new Float32Array(x.size);
      if (training) {
        const mean = new Float32Array(inner);
        for (let i = 0; i < x.size; i++) mean[i % inner] += x.data[i]!;
        for (let c = 0; c < inner; c++) mean[c]! /= outer;
        const var_ = new Float32Array(inner);
        for (let i = 0; i < x.size; i++) {
          const d = x.data[i]! - mean[i % inner]!;
          var_[i % inner]! += d * d;
        }
        for (let c = 0; c < inner; c++) var_[c]! /= outer;
        const std = new Float32Array(inner);
        for (let c = 0; c < inner; c++) {
          std[c] = Math.sqrt(var_[c]! + eps);
          runningMean.data[c] = momentum * runningMean.data[c]! + (1 - momentum) * mean[c]!;
          runningVar.data[c] = momentum * runningVar.data[c]! + (1 - momentum) * var_[c]!;
        }
        const xhat = new Float32Array(x.size);
        for (let i = 0; i < x.size; i++) {
          const c = i % inner;
          const nh = (x.data[i]! - mean[c]!) / std[c]!;
          xhat[i] = nh;
          out[i] = gamma.data[c]! * nh + beta.data[c]!;
        }
        const xhatT = new Tensor(x.shape as number[], xhat);
        cache = { xhat: xhatT, std, x, mean };
        return new Tensor(x.shape as number[], out);
      }
      for (let i = 0; i < x.size; i++) {
        const c = i % inner;
        const nh = (x.data[i]! - runningMean.data[c]!) / Math.sqrt(runningVar.data[c]! + eps);
        out[i] = gamma.data[c]! * nh + beta.data[c]!;
      }
      cache = null;
      return new Tensor(x.shape as number[], out);
    },
    backward(dy) {
      const inner = features;
      const N = dy.size / inner;
      const gGamma = new Float32Array(inner);
      const gBeta = new Float32Array(inner);
      if (!cache) {
        // inference-mode backward (rare): treat running stats as constants
        const dx = new Float32Array(dy.size);
        for (let i = 0; i < dy.size; i++) {
          const c = i % inner;
          gBeta[c]! += dy.data[i]!;
          const std = Math.sqrt(runningVar.data[c]! + eps);
          const xhat = (0); // unknown
          void xhat;
          gGamma[c]! += dy.data[i]!;
          dx[i] = dy.data[i]! * gamma.data[c]! / std;
        }
        grads.gamma = new Tensor([inner], gGamma);
        grads.beta = new Tensor([inner], gBeta);
        return new Tensor(dy.shape as number[], dx);
      }
      const { xhat, std } = cache;
      for (let i = 0; i < dy.size; i++) {
        const c = i % inner;
        gBeta[c]! += dy.data[i]!;
        gGamma[c]! += dy.data[i]! * xhat.data[i]!;
      }
      grads.gamma = new Tensor([inner], gGamma);
      grads.beta = new Tensor([inner], gBeta);
      const dx = new Float32Array(dy.size);
      for (let c = 0; c < inner; c++) {
        let sumDy = 0;
        let sumDyXhat = 0;
        for (let n = 0; n < N; n++) {
          const i = n * inner + c;
          sumDy += dy.data[i]!;
          sumDyXhat += dy.data[i]! * xhat.data[i]!;
        }
        for (let n = 0; n < N; n++) {
          const i = n * inner + c;
          const dyc = dy.data[i]!;
          dx[i] =
            (gamma.data[c]! / std[c]! / N) *
            (N * dyc - sumDy - xhat.data[i]! * sumDyXhat);
        }
      }
      return new Tensor(dy.shape as number[], dx);
    },
    resetCache() {
      cache = null;
    },
  };
}

function compileLayerNorm(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  existing: { params?: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const features = inputShape[inputShape.length - 1]!;
  const eps = num(def.config.epsilon, 1e-5);
  const gamma = maybeReuse(existing as { params: Record<string, Tensor> }, "gamma", [features]) ?? Tensor.ones([features]);
  const beta = maybeReuse(existing as { params: Record<string, Tensor> }, "beta", [features]) ?? Tensor.zeros([features]);
  const params = { gamma, beta };
  const grads = emptyGrads(params);
  let cache: { xhat: Tensor; std: Float32Array } | null = null;

  return {
    def,
    inputShape,
    outputShape,
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x) {
      const inner = features;
      const rows = x.size / inner;
      const out = new Float32Array(x.size);
      const xhat = new Float32Array(x.size);
      const std = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        const off = r * inner;
        let m = 0;
        for (let c = 0; c < inner; c++) m += x.data[off + c]!;
        m /= inner;
        let v = 0;
        for (let c = 0; c < inner; c++) {
          const d = x.data[off + c]! - m;
          v += d * d;
        }
        v /= inner;
        std[r] = Math.sqrt(v + eps);
        for (let c = 0; c < inner; c++) {
          const nh = (x.data[off + c]! - m) / std[r]!;
          xhat[off + c] = nh;
          out[off + c] = gamma.data[c]! * nh + beta.data[c]!;
        }
      }
      cache = { xhat: new Tensor(x.shape as number[], xhat), std };
      return new Tensor(x.shape as number[], out);
    },
    backward(dy) {
      const inner = features;
      const rows = dy.size / inner;
      const gGamma = new Float32Array(inner);
      const gBeta = new Float32Array(inner);
      const { xhat, std } = cache!;
      for (let i = 0; i < dy.size; i++) {
        const c = i % inner;
        gBeta[c]! += dy.data[i]!;
        gGamma[c]! += dy.data[i]! * xhat.data[i]!;
      }
      grads.gamma = new Tensor([inner], gGamma);
      grads.beta = new Tensor([inner], gBeta);
      const dx = new Float32Array(dy.size);
      for (let r = 0; r < rows; r++) {
        const off = r * inner;
        let sumDy = 0;
        let sumDyXhat = 0;
        for (let c = 0; c < inner; c++) {
          sumDy += dy.data[off + c]!;
          sumDyXhat += dy.data[off + c]! * xhat.data[off + c]!;
        }
        for (let c = 0; c < inner; c++) {
          dx[off + c] =
            (gamma.data[c]! / std[r]! / inner) *
            (inner * dy.data[off + c]! - sumDy - xhat.data[off + c]! * sumDyXhat);
        }
      }
      return new Tensor(dy.shape as number[], dx);
    },
    resetCache() {
      cache = null;
    },
  };
}

function compileConv2d(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  rng: RNG,
  defaultInit: InitMethod,
  existing: { params: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const [H, W, Cin] = inputShape;
  const [outH, outW, filters] = outputShape;
  const [kH, kW] = kernelWH(def.config.kernelSize);
  const stride = num(def.config.stride, 1);
  const padding = String(def.config.padding ?? "same");
  const { padH, padW } = convOut(H!, W!, kH, kW, stride, padding);
  const padTop = Math.floor(padH / 2);
  const padLeft = Math.floor(padW / 2);
  const useBias = bool(def.config.useBias, true);
  const act = String(def.config.activation ?? "none") as ActivationName | "none";
  const init = (def.config.init as InitMethod) || defaultInit;
  const fanIn = kH * kW * Cin!;
  const fanOut = kH * kW * filters!;
  const K = maybeReuse(existing, "K", [kH, kW, Cin!, filters!]) ?? initTensor([kH, kW, Cin!, filters!], init, rng, fanIn, fanOut);
  const b = useBias ? (maybeReuse(existing, "b", [filters!]) ?? Tensor.zeros([filters!])) : Tensor.zeros([0]);
  const params: Record<string, Tensor> = useBias ? { K, b } : { K };
  const grads = emptyGrads(params);
  let lastX: Tensor | null = null;
  let lastPre: Tensor | null = null;
  let lastY: Tensor | null = null;

  return {
    def,
    inputShape,
    outputShape,
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x) {
      const B = x.shape[0]!;
      lastX = x;
      const y = conv2dForward(x, K, useBias ? b : null, B, H!, W!, Cin!, outH!, outW!, filters!, kH, kW, stride, padTop, padLeft);
      lastPre = y;
      const out = act !== "none" ? applyActivation(act, y) : y;
      lastY = out;
      return out;
    },
    backward(dy) {
      let g = dy;
      if (act !== "none" && lastPre && lastY) g = activationBackward(act, lastPre, lastY, dy);
      const x = lastX!;
      const B = x.shape[0]!;
      const { dX, dK, dB } = conv2dBackward(x, K, g, B, H!, W!, Cin!, outH!, outW!, filters!, kH, kW, stride, padTop, padLeft);
      grads.K = dK;
      if (useBias) grads.b = dB;
      return dX;
    },
    resetCache() {
      lastX = lastPre = lastY = null;
    },
  };
}

function conv2dForward(
  x: Tensor,
  K: Tensor,
  b: Tensor | null,
  B: number,
  H: number,
  W: number,
  Cin: number,
  outH: number,
  outW: number,
  F: number,
  kH: number,
  kW: number,
  stride: number,
  padTop: number,
  padLeft: number,
): Tensor {
  const out = new Float32Array(B * outH * outW * F);
  const xd = x.data;
  const kd = K.data;
  for (let n = 0; n < B; n++) {
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let f = 0; f < F; f++) {
          let acc = b ? b.data[f]! : 0;
          for (let kh = 0; kh < kH; kh++) {
            const ih = oh * stride + kh - padTop;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < kW; kw++) {
              const iw = ow * stride + kw - padLeft;
              if (iw < 0 || iw >= W) continue;
              for (let c = 0; c < Cin; c++) {
                const xv = xd[((n * H + ih) * W + iw) * Cin + c]!;
                const kv = kd[((kh * kW + kw) * Cin + c) * F + f]!;
                acc += xv * kv;
              }
            }
          }
          out[((n * outH + oh) * outW + ow) * F + f] = acc;
        }
      }
    }
  }
  return new Tensor([B, outH, outW, F], out);
}

function conv2dBackward(
  x: Tensor,
  K: Tensor,
  dy: Tensor,
  B: number,
  H: number,
  W: number,
  Cin: number,
  outH: number,
  outW: number,
  F: number,
  kH: number,
  kW: number,
  stride: number,
  padTop: number,
  padLeft: number,
): { dX: Tensor; dK: Tensor; dB: Tensor } {
  const dX = new Float32Array(x.size);
  const dK = new Float32Array(K.size);
  const dB = new Float32Array(F);
  const xd = x.data;
  const kd = K.data;
  const gd = dy.data;
  for (let n = 0; n < B; n++) {
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let f = 0; f < F; f++) {
          const grad = gd[((n * outH + oh) * outW + ow) * F + f]!;
          dB[f]! += grad;
          for (let kh = 0; kh < kH; kh++) {
            const ih = oh * stride + kh - padTop;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < kW; kw++) {
              const iw = ow * stride + kw - padLeft;
              if (iw < 0 || iw >= W) continue;
              for (let c = 0; c < Cin; c++) {
                const xIdx = ((n * H + ih) * W + iw) * Cin + c;
                const kIdx = ((kh * kW + kw) * Cin + c) * F + f;
                dK[kIdx]! += xd[xIdx]! * grad;
                dX[xIdx]! += kd[kIdx]! * grad;
              }
            }
          }
        }
      }
    }
  }
  return {
    dX: new Tensor(x.shape as number[], dX),
    dK: new Tensor(K.shape as number[], dK),
    dB: new Tensor([F], dB),
  };
}

function compileMaxPool(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  const pool = num(def.config.poolSize, 2);
  const stride = num(def.config.stride, pool);
  let switches: Int32Array | null = null;
  let inShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      inShape = x.shape as number[];
      const B = x.shape[0]!;
      const H = x.shape[1]!;
      const W = x.shape[2]!;
      const C = x.shape[3]!;
      const outH = outputShape[0]!;
      const outW = outputShape[1]!;
      const out = new Float32Array(B * outH * outW * C);
      const sw = new Int32Array(out.length);
      for (let n = 0; n < B; n++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            for (let c = 0; c < C; c++) {
              let best = -Infinity;
              let bestIdx = 0;
              for (let kh = 0; kh < pool; kh++) {
                const ih = oh * stride + kh;
                if (ih >= H) continue;
                for (let kw = 0; kw < pool; kw++) {
                  const iw = ow * stride + kw;
                  if (iw >= W) continue;
                  const idx = ((n * H + ih) * W + iw) * C + c;
                  const v = x.data[idx]!;
                  if (v > best) {
                    best = v;
                    bestIdx = idx;
                  }
                }
              }
              const oidx = ((n * outH + oh) * outW + ow) * C + c;
              out[oidx] = best === -Infinity ? 0 : best;
              sw[oidx] = bestIdx;
            }
          }
        }
      }
      switches = sw;
      return new Tensor([B, outH, outW, C], out);
    },
    backward(dy) {
      const dx = new Float32Array(numel(inShape));
      if (switches) {
        for (let i = 0; i < dy.size; i++) dx[switches[i]!] += dy.data[i]!;
      }
      return new Tensor(inShape, dx);
    },
    resetCache() {
      switches = null;
    },
  };
}

function compileAvgPool(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  const pool = num(def.config.poolSize, 2);
  const stride = num(def.config.stride, pool);
  let inShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      inShape = x.shape as number[];
      const B = x.shape[0]!;
      const H = x.shape[1]!;
      const W = x.shape[2]!;
      const C = x.shape[3]!;
      const outH = outputShape[0]!;
      const outW = outputShape[1]!;
      const out = new Float32Array(B * outH * outW * C);
      const area = pool * pool;
      for (let n = 0; n < B; n++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            for (let c = 0; c < C; c++) {
              let acc = 0;
              for (let kh = 0; kh < pool; kh++) {
                const ih = oh * stride + kh;
                if (ih >= H) continue;
                for (let kw = 0; kw < pool; kw++) {
                  const iw = ow * stride + kw;
                  if (iw >= W) continue;
                  acc += x.data[((n * H + ih) * W + iw) * C + c]!;
                }
              }
              out[((n * outH + oh) * outW + ow) * C + c] = acc / area;
            }
          }
        }
      }
      return new Tensor([B, outH, outW, C], out);
    },
    backward(dy) {
      const B = inShape[0]!;
      const H = inShape[1]!;
      const W = inShape[2]!;
      const C = inShape[3]!;
      const outH = outputShape[0]!;
      const outW = outputShape[1]!;
      const dx = new Float32Array(numel(inShape));
      const area = pool * pool;
      for (let n = 0; n < B; n++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            for (let c = 0; c < C; c++) {
              const g = dy.data[((n * outH + oh) * outW + ow) * C + c]! / area;
              for (let kh = 0; kh < pool; kh++) {
                const ih = oh * stride + kh;
                if (ih >= H) continue;
                for (let kw = 0; kw < pool; kw++) {
                  const iw = ow * stride + kw;
                  if (iw >= W) continue;
                  dx[((n * H + ih) * W + iw) * C + c]! += g;
                }
              }
            }
          }
        }
      }
      return new Tensor(inShape, dx);
    },
    resetCache() {},
  };
}

function compileGap(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  let inShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      inShape = x.shape as number[];
      const B = x.shape[0]!;
      const H = x.shape[1]!;
      const W = x.shape[2]!;
      const C = x.shape[3]!;
      const spatial = H * W;
      const out = new Float32Array(B * C);
      for (let n = 0; n < B; n++) {
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            for (let c = 0; c < C; c++) {
              out[n * C + c] += x.data[((n * H + h) * W + w) * C + c]!;
            }
          }
        }
        for (let c = 0; c < C; c++) out[n * C + c]! /= spatial;
      }
      return new Tensor([B, C], out);
    },
    backward(dy) {
      const B = inShape[0]!;
      const H = inShape[1]!;
      const W = inShape[2]!;
      const C = inShape[3]!;
      const spatial = H * W;
      const dx = new Float32Array(numel(inShape));
      for (let n = 0; n < B; n++) {
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            for (let c = 0; c < C; c++) {
              dx[((n * H + h) * W + w) * C + c] = dy.data[n * C + c]! / spatial;
            }
          }
        }
      }
      return new Tensor(inShape, dx);
    },
    resetCache() {},
  };
}

/**
 * Simple Elman RNN: h_t = tanh(x_t·Wxh + h_{t-1}·Whh + b), h_{-1} = 0.
 * Gradients are computed with full backpropagation through time (BPTT).
 */
function compileRnn(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  rng: RNG,
  existing: { params: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const D = inputShape[1]!;
  const H = num(def.config.units, 32);
  void outputShape;
  const returnSeq = bool(def.config.returnSequences, false);
  const Wxh = maybeReuse(existing, "Wxh", [D, H]) ?? initTensor([D, H], "xavier_uniform", rng, D, H);
  const Whh = maybeReuse(existing, "Whh", [H, H]) ?? initTensor([H, H], "xavier_uniform", rng, H, H);
  const b = maybeReuse(existing, "b", [H]) ?? Tensor.zeros([H]);
  const params = { Wxh, Whh, b };
  const grads = emptyGrads(params);

  let lastX: Tensor | null = null;
  let lastHs: Float32Array | null = null; // [B, T+1, H], slot 0 is the zero initial state
  let lastDims = { B: 0, T: 0 };

  return {
    def,
    inputShape,
    outputShape: returnSeq ? [inputShape[0]!, H] : [H],
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x: Tensor) {
      assert(x.rank === 3, `RNN ${def.name}: expected batched sequence [B, T, ${D}], got ${shapeStr(x.shape)}`);
      const B = x.shape[0]!;
      const steps = x.shape[1]!;
      const hs = new Float32Array(B * (steps + 1) * H);
      const out = returnSeq ? new Float32Array(B * steps * H) : new Float32Array(B * H);
      const xd = x.data;
      const wx = Wxh.data;
      const wh = Whh.data;
      const bd = b.data;
      for (let n = 0; n < B; n++) {
        for (let t = 0; t < steps; t++) {
          const hPrevOff = (n * (steps + 1) + t) * H;
          const hOff = hPrevOff + H;
          const xOff = (n * steps + t) * D;
          for (let j = 0; j < H; j++) {
            let acc = bd[j]!;
            for (let i = 0; i < D; i++) acc += xd[xOff + i]! * wx[i * H + j]!;
            for (let i = 0; i < H; i++) acc += hs[hPrevOff + i]! * wh[i * H + j]!;
            hs[hOff + j] = Math.tanh(acc);
          }
          if (returnSeq) out.set(hs.subarray(hOff, hOff + H), (n * steps + t) * H);
        }
        if (!returnSeq) {
          const hOff = (n * (steps + 1) + steps) * H;
          out.set(hs.subarray(hOff, hOff + H), n * H);
        }
      }
      lastX = x;
      lastHs = hs;
      lastDims = { B, T: steps };
      return returnSeq ? new Tensor([B, steps, H], out) : new Tensor([B, H], out);
    },
    backward(dy: Tensor) {
      const x = lastX!;
      const hs = lastHs!;
      const { B, T: steps } = lastDims;
      const dWxh = new Float32Array(D * H);
      const dWhh = new Float32Array(H * H);
      const db = new Float32Array(H);
      const dx = new Float32Array(x.size);
      const xd = x.data;
      const wx = Wxh.data;
      const wh = Whh.data;
      const dhNext = new Float32Array(H);
      const dhRaw = new Float32Array(H);
      for (let n = 0; n < B; n++) {
        dhNext.fill(0);
        for (let t = steps - 1; t >= 0; t--) {
          const hOff = (n * (steps + 1) + t + 1) * H;
          const hPrevOff = hOff - H;
          const xOff = (n * steps + t) * D;
          for (let j = 0; j < H; j++) {
            let dh = dhNext[j]!;
            if (returnSeq) dh += dy.data[(n * steps + t) * H + j]!;
            else if (t === steps - 1) dh += dy.data[n * H + j]!;
            const h = hs[hOff + j]!;
            dhRaw[j] = dh * (1 - h * h);
          }
          for (let i = 0; i < D; i++) {
            const xv = xd[xOff + i]!;
            let dxv = 0;
            for (let j = 0; j < H; j++) {
              dWxh[i * H + j]! += xv * dhRaw[j]!;
              dxv += wx[i * H + j]! * dhRaw[j]!;
            }
            dx[xOff + i] = dxv;
          }
          for (let i = 0; i < H; i++) {
            const hv = hs[hPrevOff + i]!;
            for (let j = 0; j < H; j++) dWhh[i * H + j]! += hv * dhRaw[j]!;
          }
          for (let j = 0; j < H; j++) {
            db[j]! += dhRaw[j]!;
            let acc = 0;
            for (let i = 0; i < H; i++) acc += dhRaw[i]! * wh[j * H + i]!;
            dhNext[j] = acc;
          }
        }
      }
      grads.Wxh = new Tensor([D, H], dWxh);
      grads.Whh = new Tensor([H, H], dWhh);
      grads.b = new Tensor([H], db);
      return new Tensor(x.shape as number[], dx);
    },
    resetCache() {
      lastX = null;
      lastHs = null;
    },
  };
}

function compileEmbedding(
  def: LayerDef,
  inputShape: number[],
  outputShape: number[],
  rng: RNG,
  defaultInit: InitMethod,
  existing: { params: Record<string, Tensor> } | undefined,
  frozen: boolean,
): CompiledLayer {
  const vocab = num(def.config.vocabSize, 32);
  const dim = outputShape[1]!;
  const init = (def.config.init as InitMethod) || defaultInit;
  const W = maybeReuse(existing, "W", [vocab, dim]) ?? initTensor([vocab, dim], init === "zeros" ? "zeros" : "random_normal", rng, vocab, dim);
  const params = { W };
  const grads = emptyGrads(params);
  let lastIdx: Int32Array | null = null;
  let lastShape: number[] = [];
  return {
    def,
    inputShape,
    outputShape,
    params,
    grads,
    nonTrainable: {},
    frozen,
    forward(x) {
      lastShape = x.shape as number[];
      const seq = x.size;
      const out = new Float32Array(seq * dim);
      const idx = new Int32Array(seq);
      for (let i = 0; i < seq; i++) {
        let id = Math.round(x.data[i]!);
        if (id < 0) id = 0;
        if (id >= vocab) id = vocab - 1;
        idx[i] = id;
        out.set(W.data.subarray(id * dim, id * dim + dim), i * dim);
      }
      lastIdx = idx;
      const B = x.shape[0]!;
      const rest = seq / B;
      return new Tensor([B, rest, dim], out);
    },
    backward(dy) {
      const dW = new Float32Array(W.size);
      const idx = lastIdx!;
      for (let i = 0; i < idx.length; i++) {
        const id = idx[i]!;
        const off = i * dim;
        const woff = id * dim;
        for (let d = 0; d < dim; d++) dW[woff + d]! += dy.data[off + d]!;
      }
      grads.W = new Tensor([vocab, dim], dW);
      return Tensor.zeros(lastShape);
    },
    resetCache() {
      lastIdx = null;
    },
  };
}

function compileIdentity(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      return x;
    },
    backward(dy) {
      return dy;
    },
    resetCache() {},
  };
}

function compileActivation(def: LayerDef, inputShape: number[], outputShape: number[], frozen: boolean): CompiledLayer {
  const type = def.type as ActivationName;
  const alpha = num(def.config.alpha, 0.01);
  let lastX: Tensor | null = null;
  let lastY: Tensor | null = null;
  return {
    def,
    inputShape,
    outputShape,
    params: {},
    grads: {},
    nonTrainable: {},
    frozen,
    forward(x) {
      lastX = x;
      let y: Tensor;
      switch (type) {
        case "relu":
          y = relu(x);
          break;
        case "leakyrelu":
          y = leakyRelu(x, alpha);
          break;
        case "gelu":
          y = gelu(x);
          break;
        case "silu":
          y = silu(x);
          break;
        case "sigmoid":
          y = sigmoid(x);
          break;
        case "tanh":
          y = tanhAct(x);
          break;
        case "softmax":
          y = softmaxLast(x);
          break;
        default:
          y = x;
      }
      lastY = y;
      return y;
    },
    backward(dy) {
      const x = lastX!;
      const y = lastY!;
      switch (type) {
        case "relu":
          return reluBackward(x, dy);
        case "leakyrelu":
          return leakyReluBackward(x, dy, alpha);
        case "gelu":
          return geluBackward(x, dy);
        case "silu":
          return siluBackward(x, dy);
        case "sigmoid":
          return sigmoidBackward(y, dy);
        case "tanh":
          return tanhBackward(y, dy);
        case "softmax":
          return softmaxBackward(y, dy);
        default:
          return dy;
      }
    },
    resetCache() {
      lastX = lastY = null;
    },
  };
}

export const LAYER_DOCS: Record<LayerType, string> = {
  dense: "Fully connected layer. Every input feature connects to every output unit: y = xW + b.",
  flatten: "Collapses all non-batch dimensions into a single feature vector.",
  reshape: "Rearranges the non-batch dimensions without changing the number of elements.",
  dropout: "Randomly zeroes units during training (inverted dropout). Disabled at inference.",
  batchnorm: "Normalizes features across the batch, then scales and shifts with learned γ, β.",
  layernorm: "Normalizes features of each example independently. Common in transformers.",
  conv2d: "2D convolution over [H, W, C] inputs. Learns local spatial filters.",
  maxpool2d: "Takes the maximum in each pooling window. Provides translation invariance.",
  avgpool2d: "Averages each pooling window.",
  gap2d: "Global average pooling: mean over spatial dims, producing one value per channel.",
  embedding: "Maps integer token indices to dense vectors. Used for discrete / language inputs.",
  rnn: "Simple tanh RNN with full backpropagation through time. Processes a sequence step by step and carries a hidden state. Long contexts can suffer vanishing gradients — that is a real property of this architecture, not a bug.",
  identity: "Pass-through. Useful as a placeholder or residual-friendly identity path.",
  relu: "max(0, x)",
  leakyrelu: "x if x>0 else αx",
  gelu: "Gaussian Error Linear Unit (tanh approximation).",
  silu: "x · sigmoid(x)",
  sigmoid: "1 / (1 + e^(−x))",
  tanh: "Hyperbolic tangent.",
  softmax: "Exponential normalize along the last axis.",
};

