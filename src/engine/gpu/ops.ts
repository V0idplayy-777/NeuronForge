/**
 * Device-agnostic operation surface used by the training accelerator.
 *
 * Everything here is async because a real WebGPU device is async: commands are
 * recorded into a command encoder and only complete when the GPU says so. The
 * CPU reference device (`cpu.ts`) implements the same interface by delegating to
 * the float32 tensor engine, which is what lets the accelerator be tested on
 * machines without a GPU and lets us diff GPU results against the CPU path.
 */

/** Opaque device-resident tensor. Only the owning device may interpret it. */
export interface DevBuffer {
  readonly shape: number[];
  readonly size: number;
  /** Implementation-private handle. */
  readonly handle: unknown;
  /** True for a zero-copy reinterpretation of another buffer: never disposed on its own. */
  readonly view?: boolean;
}

import { convOut, kernelWH } from "../layers";

export type BinaryOp = "add" | "sub" | "mul" | "div";
export type UnaryOp = "neg" | "abs" | "square" | "scale" | "clip";
export type ReduceMode = "sum" | "mean";
export type PoolMode = "max" | "avg";

export interface ConvGeom {
  B: number;
  H: number;
  W: number;
  Cin: number;
  outH: number;
  outW: number;
  F: number;
  kH: number;
  kW: number;
  stride: number;
  padTop: number;
  padLeft: number;
}

export interface PoolGeom {
  B: number;
  H: number;
  W: number;
  C: number;
  outH: number;
  outW: number;
  pool: number;
  stride: number;
}

export interface DeviceInfo {
  kind: "webgpu" | "cpu";
  label: string;
  adapter?: string;
  maxBufferBytes?: number;
}

export interface GpuOps {
  readonly info: DeviceInfo;

  fromHost(data: Float32Array, shape: number[]): Promise<DevBuffer>;
  toHost(t: DevBuffer): Promise<Float32Array>;
  /**
   * Reads several tensors back at once. A real device can then submit once and
   * stall once instead of once per buffer, which is what gradient collection
   * needs; devices that cannot batch it simply omit the method.
   */
  toHostMany?(targets: DevBuffer[]): Promise<Float32Array[]>;
  zeros(shape: number[]): Promise<DevBuffer>;
  dispose(t: DevBuffer): void;
  /** Zero-copy reinterpretation (Flatten / Reshape / Embedding output views). */
  reshapeView(t: DevBuffer, shape: number[]): Promise<DevBuffer>;

  matmul(a: DevBuffer, b: DevBuffer, transposeA: boolean, transposeB: boolean): Promise<DevBuffer>;
  /** Adds a bias vector along the last axis. */
  addBias(m: DevBuffer, bias: DevBuffer): Promise<DevBuffer>;
  binary(op: BinaryOp, a: DevBuffer, b: DevBuffer): Promise<DevBuffer>;
  unary(op: UnaryOp, x: DevBuffer, scalar: number, scalar2: number): Promise<DevBuffer>;
  activationFwd(name: string, x: DevBuffer, alpha: number): Promise<DevBuffer>;
  activationBwd(name: string, x: DevBuffer, y: DevBuffer, dy: DevBuffer, alpha: number): Promise<DevBuffer>;
  softmaxLast(x: DevBuffer): Promise<DevBuffer>;
  softmaxBackward(p: DevBuffer, dy: DevBuffer): Promise<DevBuffer>;
  /**
   * Reduces blocks of axes into the last one: [group, outer…, inner] → [group, inner].
   * `group = 1` is the plain "reduce every leading axis" case (bias gradients);
   * `group = B` reduces only the spatial axes (global average pooling).
   */
  reduceOuter(mode: ReduceMode, m: DevBuffer, group?: number): Promise<DevBuffer>;
  /**
   * Inverse of global average pooling: repeats each (batch, channel) value over
   * the H×W spatial axes of `geom`, scaled by `scale`. Used by the backward pass
   * of GAP2D.
   */
  broadcastPlanes(t: DevBuffer, geom: { B: number; H: number; W: number; C: number }, scale: number): Promise<DevBuffer>;

  conv2d(x: DevBuffer, k: DevBuffer, bias: DevBuffer | null, g: ConvGeom): Promise<DevBuffer>;
  conv2dInputGrad(dy: DevBuffer, k: DevBuffer, g: ConvGeom): Promise<DevBuffer>;
  conv2dKernelGrad(x: DevBuffer, dy: DevBuffer, g: ConvGeom): Promise<DevBuffer>;

  pool2d(mode: PoolMode, x: DevBuffer, g: PoolGeom): Promise<DevBuffer>;
  /**
   * Pooling backward takes the original input rather than a cached argmax index
   * buffer: both devices derive the winner from `x`, which needs no float
   * atomics (WGSL has none) and is deterministic for overlapping windows.
   */
  pool2dBackward(mode: PoolMode, x: DevBuffer, dy: DevBuffer, g: PoolGeom): Promise<DevBuffer>;

  gatherRows(w: DevBuffer, ids: DevBuffer, dim: number): Promise<DevBuffer>;
  scatterRows(grad: DevBuffer, ids: DevBuffer, dim: number, vocab: number): Promise<DevBuffer>;

  /** Flushes queued work and releases all device memory. */
  destroy(): void;
}

export interface BroadcastPlan {
  /** Output shape with leading ones removed. */
  outShape: number[];
  /** Output axes right-aligned to four entries (leading padding of ones). */
  out4: number[];
  /** Row-major strides of the left operand, zeroed on broadcast axes. */
  a4: number[];
  /** Row-major strides of the right operand, zeroed on broadcast axes. */
  b4: number[];
}

function pad4(shape: readonly number[]): number[] {
  const out = [1, 1, 1, 1];
  for (let i = 0; i < shape.length; i++) out[4 - shape.length + i] = Math.max(1, Math.floor(shape[i]!));
  return out;
}

function strides4(padded: readonly number[]): number[] {
  const s = [1, 1, 1, 1];
  for (let d = 2; d >= 0; d--) s[d] = s[d + 1]! * padded[d + 1]!;
  return s;
}

/**
 * Numpy-style broadcasting, described with four right-aligned axes so that a
 * single kernel (and a single CPU loop) can serve every case the engine needs:
 * same-shape arithmetic, bias adds, per-channel scale/shift. Throws when the
 * shapes are not broadcast-compatible.
 */
export function broadcastPlan(a: readonly number[], b: readonly number[]): BroadcastPlan {
  if (a.length > 4 || b.length > 4) {
    throw new Error(`broadcast: tensors are limited to rank 4 (got ${a.length} and ${b.length})`);
  }
  const pa = pad4(a);
  const pb = pad4(b);
  const out4 = [1, 1, 1, 1];
  for (let d = 0; d < 4; d++) {
    const da = pa[d]!;
    const db = pb[d]!;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`broadcast: cannot combine [${a.join(",")}] with [${b.join(",")}]`);
    }
    out4[d] = Math.max(da, db);
  }
  const sa = strides4(pa);
  const sb = strides4(pb);
  const a4 = [0, 0, 0, 0];
  const b4 = [0, 0, 0, 0];
  for (let d = 0; d < 4; d++) {
    a4[d] = pa[d] === out4[d] ? sa[d]! : 0;
    b4[d] = pb[d] === out4[d] ? sb[d]! : 0;
  }
  let lead = 0;
  while (lead < 3 && out4[lead] === 1) lead++;
  return { outShape: out4.slice(lead), out4, a4, b4 };
}

/** Splits a flat index into the four right-aligned axes of `shape4`. */
export function coords4(shape4: readonly number[], i: number): [number, number, number, number] {
  const c3 = i % shape4[3]!;
  const r3 = Math.floor(i / shape4[3]!);
  const c2 = r3 % shape4[2]!;
  const r2 = Math.floor(r3 / shape4[2]!);
  const c1 = r2 % shape4[1]!;
  const c0 = Math.floor(r2 / shape4[1]!);
  return [c0, c1, c2, c3];
}

/**
 * Convolution geometry from a layer config, shared by every device so the two
 * backends always agree on output size and padding.
 */
export function convGeomFor(
  B: number,
  H: number,
  W: number,
  Cin: number,
  filters: number,
  config: Record<string, unknown>,
  strideOverride?: number,
): ConvGeom {
  const [kH, kW] = kernelWH(config.kernelSize);
  const rawStride = Number(config.stride ?? 1);
  const stride = strideOverride ?? Math.max(1, Math.round(Number.isFinite(rawStride) ? rawStride : 1));
  const padding = String(config.padding ?? "same");
  const g = convOut(H, W, kH, kW, stride, padding);
  return {
    B,
    H,
    W,
    Cin,
    outH: g.outH,
    outW: g.outW,
    F: filters,
    kH,
    kW,
    stride,
    padTop: Math.floor(g.padH / 2),
    padLeft: Math.floor(g.padW / 2),
  };
}

export function devNumel(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function sameDevShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
