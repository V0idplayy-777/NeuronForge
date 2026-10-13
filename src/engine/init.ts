import type { InitMethod } from "./types";
import { Tensor } from "./tensor";
import type { RNG } from "./rng";

export function initTensor(shape: number[], method: InitMethod, rng: RNG, fanIn: number, fanOut: number): Tensor {
  switch (method) {
    case "zeros":
      return Tensor.zeros(shape);
    case "random_normal":
      return Tensor.randn(shape, rng, 0.05);
    case "xavier_uniform": {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      return Tensor.randu(shape, rng, -limit, limit);
    }
    case "xavier_normal": {
      const std = Math.sqrt(2 / (fanIn + fanOut));
      return Tensor.randn(shape, rng, std);
    }
    case "he_uniform": {
      const limit = Math.sqrt(6 / fanIn);
      return Tensor.randu(shape, rng, -limit, limit);
    }
    case "he_normal": {
      const std = Math.sqrt(2 / fanIn);
      return Tensor.randn(shape, rng, std);
    }
  }
}

export const INIT_DOCS: Record<InitMethod, string> = {
  zeros: "All parameters start at 0. Only appropriate for biases or as a diagnostic — hidden weights of 0 prevent learning in symmetric layers.",
  random_normal: "i.i.d. Gaussian with σ = 0.05. Simple, but can vanish or explode in deep nets.",
  xavier_uniform: "Glorot uniform. Variance scaled by fan-in + fan-out. Best with tanh / sigmoid / softmax.",
  xavier_normal: "Glorot normal. Same motivation as Xavier uniform, Gaussian samples.",
  he_uniform: "Kaiming uniform. Variance scaled by fan-in. Best with ReLU / Leaky ReLU / GELU / SiLU.",
  he_normal: "Kaiming normal. Preferred default for ReLU-family hidden layers.",
};

export function recommendedInit(activation: string): InitMethod {
  const a = activation.toLowerCase();
  if (a === "relu" || a === "leakyrelu" || a === "gelu" || a === "silu") return "he_normal";
  if (a === "tanh" || a === "sigmoid" || a === "softmax") return "xavier_normal";
  return "xavier_uniform";
}
