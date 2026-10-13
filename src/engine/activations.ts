import type { ActivationName } from "./types";
import { Tensor, softmaxLast, softmaxBackward } from "./tensor";

export function applyActivation(name: ActivationName, x: Tensor): Tensor {
  switch (name) {
    case "linear":
      return x;
    case "relu":
      return relu(x);
    case "leakyrelu":
      return leakyRelu(x, 0.01);
    case "gelu":
      return gelu(x);
    case "silu":
      return silu(x);
    case "sigmoid":
      return sigmoid(x);
    case "tanh":
      return tanhAct(x);
    case "softmax":
      return softmaxLast(x);
  }
}

export function activationBackward(name: ActivationName, x: Tensor, y: Tensor, dy: Tensor, leak = 0.01): Tensor {
  switch (name) {
    case "linear":
      return dy;
    case "relu":
      return reluBackward(x, dy);
    case "leakyrelu":
      return leakyReluBackward(x, dy, leak);
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
  }
}

export function relu(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i]!;
    out[i] = v > 0 ? v : 0;
  }
  return new Tensor(x.shape as number[], out);
}

export function reluBackward(x: Tensor, dy: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = x.data[i]! > 0 ? dy.data[i]! : 0;
  return new Tensor(x.shape as number[], out);
}

export function leakyRelu(x: Tensor, alpha: number): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i]!;
    out[i] = v > 0 ? v : alpha * v;
  }
  return new Tensor(x.shape as number[], out);
}

export function leakyReluBackward(x: Tensor, dy: Tensor, alpha: number): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = (x.data[i]! > 0 ? 1 : alpha) * dy.data[i]!;
  return new Tensor(x.shape as number[], out);
}

export function sigmoid(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i]!;
    out[i] = v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
  }
  return new Tensor(x.shape as number[], out);
}

export function sigmoidBackward(y: Tensor, dy: Tensor): Tensor {
  const out = new Float32Array(y.size);
  for (let i = 0; i < y.size; i++) {
    const s = y.data[i]!;
    out[i] = s * (1 - s) * dy.data[i]!;
  }
  return new Tensor(y.shape as number[], out);
}

export function tanhAct(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = Math.tanh(x.data[i]!);
  return new Tensor(x.shape as number[], out);
}

export function tanhBackward(y: Tensor, dy: Tensor): Tensor {
  const out = new Float32Array(y.size);
  for (let i = 0; i < y.size; i++) {
    const t = y.data[i]!;
    out[i] = (1 - t * t) * dy.data[i]!;
  }
  return new Tensor(y.shape as number[], out);
}

/** GELU tanh approximation (Hendrycks & Gimpel). */
export function gelu(x: Tensor): Tensor {
  const out = new Float32Array(x.size);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i]!;
    out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
  return new Tensor(x.shape as number[], out);
}

export function geluBackward(x: Tensor, dy: Tensor): Tensor {
  const out = new Float32Array(x.size);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i]!;
    const u = c * (v + 0.044715 * v * v * v);
    const th = Math.tanh(u);
    const sech2 = 1 - th * th;
    const du = c * (1 + 3 * 0.044715 * v * v);
    out[i] = dy.data[i]! * (0.5 * (1 + th) + 0.5 * v * sech2 * du);
  }
  return new Tensor(x.shape as number[], out);
}

export function silu(x: Tensor): Tensor {
  const s = sigmoid(x);
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = x.data[i]! * s.data[i]!;
  return new Tensor(x.shape as number[], out);
}

export function siluBackward(x: Tensor, dy: Tensor): Tensor {
  const s = sigmoid(x);
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) {
    const sig = s.data[i]!;
    const v = x.data[i]!;
    out[i] = dy.data[i]! * (sig + v * sig * (1 - sig));
  }
  return new Tensor(x.shape as number[], out);
}

export const ACTIVATION_DOCS: Record<ActivationName, string> = {
  linear: "Identity. Output equals input. Use on the last layer of a regression network.",
  relu: "Rectified Linear Unit: max(0, x). Sparse, cheap, and the default for hidden layers.",
  leakyrelu: "Leaky ReLU: x if x>0 else αx. Avoids dying ReLU by allowing a small negative slope.",
  gelu: "Gaussian Error Linear Unit (tanh approximation). Smooth, used in modern transformers.",
  silu: "Sigmoid Linear Unit / Swish: x · σ(x). Smooth, non-monotonic, strong on deep nets.",
  sigmoid: "Maps to (0, 1). Use for binary classification outputs or as a gating function.",
  tanh: "Maps to (−1, 1), zero-centered. Common in smaller fully-connected networks.",
  softmax: "Normalizes the last axis into a probability distribution. Use for multiclass outputs.",
};
