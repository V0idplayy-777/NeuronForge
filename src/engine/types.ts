export type InitMethod = "zeros" | "random_normal" | "xavier_uniform" | "xavier_normal" | "he_uniform" | "he_normal";

export type ActivationName =
  | "linear"
  | "relu"
  | "leakyrelu"
  | "gelu"
  | "silu"
  | "sigmoid"
  | "tanh"
  | "softmax";

export type LayerType =
  | "dense"
  | "flatten"
  | "reshape"
  | "dropout"
  | "batchnorm"
  | "layernorm"
  | "conv2d"
  | "maxpool2d"
  | "avgpool2d"
  | "gap2d"
  | "embedding"
  | "attention"
  | "transformer_block"
  | "reasoning_block"
  | "rnn"
  | "lstm"
  | "gru"
  | "residual_dense"
  | "residual_conv"
  | "identity"
  | "relu"
  | "leakyrelu"
  | "gelu"
  | "silu"
  | "sigmoid"
  | "tanh"
  | "softmax";

export type OptimizerName = "sgd" | "momentum" | "adam" | "adamw";

export type LossName = "mse" | "mae" | "bce" | "cce" | "sparse_cce";

export type TaskType = "binary" | "multiclass" | "regression" | "language";

export interface LayerDef {
  id: string;
  name: string;
  type: LayerType;
  config: Record<string, unknown>;
  frozen: boolean;
}

export interface Architecture {
  inputShape: number[];
  inputName: string;
  layers: LayerDef[];
  seed: number;
  initMethod: InitMethod;
}

export interface ShapeIssue {
  layerId: string;
  layerName: string;
  message: string;
  expected?: number[];
  actual?: number[];
}

export interface ParamCount {
  trainable: number;
  nonTrainable: number;
  byLayer: { id: string; name: string; type: LayerType; trainable: number; nonTrainable: number }[];
}

export class EngineError extends Error {
  context?: Record<string, unknown>;
  constructor(message: string, context?: Record<string, unknown>) {
    super(message);
    this.name = "EngineError";
    this.context = context;
  }
}

export function assert(cond: unknown, message: string, context?: Record<string, unknown>): asserts cond {
  if (!cond) throw new EngineError(message, context);
}

export function uid(prefix = "id"): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
  }
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function shapeEq(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function shapeStr(s: readonly number[]): string {
  return `[${s.join(", ")}]`;
}

export function numel(shape: readonly number[]): number {
  if (shape.length === 0) return 1;
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}
