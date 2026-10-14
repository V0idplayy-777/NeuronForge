/**
 * Dispatch planning: the parameters and grid geometry every kernel is called
 * with, kept apart from the device so it can be tested without a GPU.
 *
 * The parameter order in each function mirrors the `P[...]` documentation on
 * the corresponding kernel in `kernels.ts`. `kernels.sim.test.ts`-style checks
 * (see the development harness) re-run the kernels' arithmetic on the CPU with
 * exactly these parameters, which is what makes an untestable-on-this-machine
 * WGSL path verifiable somewhere.
 */

import { Params, flatDispatch, type KernelName } from "./kernels";
import type { ConvGeom, PoolGeom, PoolMode, ReduceMode } from "../gpu/ops";

export type Grid = [number, number, number];

/** The matmul kernel is tiled 16×16, so one workgroup covers 16 output rows/cols. */
export const MATMUL_TILE = 16;

export interface Step {
  params: Params;
  grid: Grid;
  kernel: KernelName;
}

function ceilDiv(n: number, d: number): number {
  return Math.max(1, Math.ceil(n / d));
}

export function matmulDims(shape: readonly number[], transpose: boolean): [number, number] {
  if (shape.length !== 2) throw new Error(`matmul: rank-2 operands only (got [${shape.join(",")}])`);
  return transpose ? [shape[1]!, shape[0]!] : [shape[0]!, shape[1]!];
}

export function matmulTileCounts(M: number, N: number): Grid {
  return [ceilDiv(N, MATMUL_TILE), ceilDiv(M, MATMUL_TILE), 1];
}

export function matmulStep(M: number, N: number, K: number, transposeA: boolean, transposeB: boolean): Step {
  return {
    kernel: "matmul",
    params: new Params().u32(M, N, K, transposeA ? 1 : 0, transposeB ? 1 : 0),
    grid: matmulTileCounts(M, N),
  };
}

export function fillZeroStep(size: number): Step {
  return { kernel: "fillZero", params: new Params().u32(size), grid: flatDispatch(size) };
}

export function binaryStep(
  size: number,
  op: number,
  out4: readonly number[],
  a4: readonly number[],
  b4: readonly number[],
): Step {
  const [o0, o1, o2, o3] = out4 as [number, number, number, number];
  const [a0, a1, a2, a3] = a4 as [number, number, number, number];
  const [b0, b1, b2, b3] = b4 as [number, number, number, number];
  return {
    kernel: "binary",
    params: new Params().u32(size, op, o0, o1, o2, o3, a0, a1, a2, a3, b0, b1, b2, b3),
    grid: flatDispatch(size),
  };
}

export function unaryStep(size: number, code: number, scalar: number, scalar2: number): Step {
  return {
    kernel: "unary",
    params: new Params().u32(size, code).f32(scalar, scalar2),
    grid: flatDispatch(size),
  };
}

export function unaryBwdStep(size: number, code: number, scalar: number): Step {
  return {
    kernel: "unaryBwd",
    params: new Params().u32(size, code).f32(scalar, 0),
    grid: flatDispatch(size),
  };
}

/** One workgroup per row; rows are folded across the two grid dimensions. */
export function rowDispatch(rows: number): Grid {
  const gx = Math.min(Math.max(1, rows), 65535);
  return [gx, Math.max(1, Math.ceil(rows / 65535)), 1];
}

export function softmaxStep(inner: number, rows: number): Step {
  return { kernel: "softmax", params: new Params().u32(inner), grid: rowDispatch(rows) };
}

export function softmaxBwdStep(inner: number, rows: number): Step {
  return { kernel: "softmaxBwd", params: new Params().u32(inner), grid: rowDispatch(rows) };
}

export function reduceOuterStep(outer: number, inner: number, group: number, mode: ReduceMode): Step {
  return {
    kernel: "reduceOuter",
    params: new Params().u32(outer, inner, mode === "mean" ? 1 : 0, group),
    grid: flatDispatch(group * inner),
  };
}

export function broadcastPlanesStep(size: number, hwc: number, c: number, scale: number): Step {
  return {
    kernel: "broadcastPlanes",
    params: new Params().u32(size, hwc, c).f32(scale),
    grid: flatDispatch(size),
  };
}

export function convParams(g: ConvGeom): Params {
  return new Params().u32(g.B, g.H, g.W, g.Cin, g.kH, g.kW, g.stride, g.padTop, g.padLeft, g.outH, g.outW);
}

export function im2colStep(g: ConvGeom, colSize: number): Step {
  return { kernel: "im2col", params: convParams(g), grid: flatDispatch(colSize) };
}

export function col2imStep(g: ConvGeom, dxSize: number): Step {
  return { kernel: "col2im", params: convParams(g), grid: flatDispatch(dxSize) };
}

export function poolParams(mode: PoolMode, g: PoolGeom): Params {
  return new Params().u32(g.B, g.H, g.W, g.C, g.outH, g.outW, g.pool, g.stride, mode === "max" ? 0 : 1);
}

export function poolFwdStep(mode: PoolMode, g: PoolGeom, size: number): Step {
  return { kernel: "poolFwd", params: poolParams(mode, g), grid: flatDispatch(size) };
}

export function poolBwdStep(mode: PoolMode, g: PoolGeom, size: number): Step {
  return { kernel: "poolBwd", params: poolParams(mode, g), grid: flatDispatch(size) };
}

export function gatherStep(size: number, dim: number, vocab: number): Step {
  return { kernel: "gather", params: new Params().u32(size, dim, vocab), grid: flatDispatch(size) };
}

export function scatterStep(size: number, dim: number, vocab: number, rows: number): Step {
  return { kernel: "scatter", params: new Params().u32(size, dim, vocab, rows), grid: flatDispatch(size) };
}

export function binaryCode(op: string): number {
  return op === "sub" ? 1 : op === "mul" ? 2 : op === "div" ? 3 : 0;
}

export function unaryCode(op: string): number {
  return op === "neg" ? 1 : op === "abs" ? 2 : op === "square" ? 3 : op === "scale" ? 4 : 5;
}

/** Activation name → UNARY op code (0 is a plain copy, i.e. "linear"/"none"). */
export function activationCode(name: string): number {
  switch (name) {
    case "relu":
      return 7;
    case "leakyrelu":
      return 8;
    case "gelu":
      return 9;
    case "silu":
      return 10;
    case "sigmoid":
      return 11;
    case "tanh":
      return 12;
    default:
      return 0;
  }
}

/** Activation name → UNARY_BWD op code. */
export function backwardCode(name: string): number {
  switch (name) {
    case "relu":
      return 1;
    case "leakyrelu":
      return 2;
    case "gelu":
      return 3;
    case "silu":
      return 4;
    case "sigmoid":
      return 5;
    case "tanh":
      return 6;
    default:
      return 0;
  }
}
