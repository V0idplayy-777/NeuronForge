import { assert, EngineError, numel } from "./types";
import type { RNG } from "./rng";

export class Tensor {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly rank: number;
  readonly size: number;

  constructor(shape: number[], data?: Float32Array | number[]) {
    this.shape = Object.freeze(shape.slice());
    this.rank = shape.length;
    this.size = numel(shape);
    if (data) {
      if (data.length !== this.size) {
        throw new EngineError(
          `Tensor data length ${data.length} does not match size ${this.size} for shape [${shape.join(", ")}]`,
        );
      }
      this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    } else {
      this.data = new Float32Array(this.size);
    }
  }

  static zeros(shape: number[]): Tensor {
    return new Tensor(shape);
  }

  static ones(shape: number[]): Tensor {
    const t = new Tensor(shape);
    t.data.fill(1);
    return t;
  }

  static fill(shape: number[], value: number): Tensor {
    const t = new Tensor(shape);
    t.data.fill(value);
    return t;
  }

  static from(arr: number[] | Float32Array, shape?: number[]): Tensor {
    const data = arr instanceof Float32Array ? arr : Float32Array.from(arr);
    return new Tensor(shape ?? [data.length], data);
  }

  static randn(shape: number[], rng: RNG, scale = 1): Tensor {
    const t = new Tensor(shape);
    for (let i = 0; i < t.size; i++) t.data[i] = rng.normal() * scale;
    return t;
  }

  static randu(shape: number[], rng: RNG, min: number, max: number): Tensor {
    const t = new Tensor(shape);
    for (let i = 0; i < t.size; i++) t.data[i] = rng.uniform(min, max);
    return t;
  }

  clone(): Tensor {
    return new Tensor(this.shape as number[], new Float32Array(this.data));
  }

  reshape(shape: number[]): Tensor {
    assert(numel(shape) === this.size, `Cannot reshape [${this.shape.join(", ")}] (size ${this.size}) to [${shape.join(", ")}]`);
    return new Tensor(shape, this.data);
  }

  flatten(startDim = 1): Tensor {
    if (startDim <= 0) return new Tensor([this.size], this.data);
    const keep = (this.shape as number[]).slice(0, startDim);
    const rest = numel((this.shape as number[]).slice(startDim));
    return new Tensor([...keep, rest], this.data);
  }

  at(...indices: number[]): number {
    let off = 0;
    let stride = 1;
    for (let i = this.rank - 1; i >= 0; i--) {
      off += (indices[i] ?? 0) * stride;
      stride *= this.shape[i]!;
    }
    return this.data[off]!;
  }

  setAt(value: number, ...indices: number[]): void {
    let off = 0;
    let stride = 1;
    for (let i = this.rank - 1; i >= 0; i--) {
      off += (indices[i] ?? 0) * stride;
      stride *= this.shape[i]!;
    }
    this.data[off] = value;
  }
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  assert(a.rank === 2 && b.rank === 2, `matmul expects 2D tensors, got rank ${a.rank} and ${b.rank}`);
  const M = a.shape[0]!;
  const K = a.shape[1]!;
  const K2 = b.shape[0]!;
  const N = b.shape[1]!;
  assert(K === K2, `matmul inner dims mismatch: [${M}, ${K}] x [${K2}, ${N}]`);
  const out = new Float32Array(M * N);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < M; i++) {
    const aOff = i * K;
    const cOff = i * N;
    for (let k = 0; k < K; k++) {
      const av = ad[aOff + k]!;
      const bOff = k * N;
      for (let j = 0; j < N; j++) {
        out[cOff + j] += av * bd[bOff + j]!;
      }
    }
  }
  return new Tensor([M, N], out);
}

export function transpose2d(a: Tensor): Tensor {
  assert(a.rank === 2, "transpose2d expects a 2D tensor");
  const M = a.shape[0]!;
  const N = a.shape[1]!;
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      out[j * M + i] = a.data[i * N + j]!;
    }
  }
  return new Tensor([N, M], out);
}

/** Add bias vector [N] to matrix [B, N] in-place-safe (returns new). */
export function addBias(matrix: Tensor, bias: Tensor): Tensor {
  assert(matrix.rank === 2 && bias.rank === 1 && matrix.shape[1] === bias.shape[0], "addBias shape mismatch");
  const B = matrix.shape[0]!;
  const N = matrix.shape[1]!;
  const out = new Float32Array(matrix.size);
  for (let i = 0; i < B; i++) {
    const off = i * N;
    for (let j = 0; j < N; j++) out[off + j] = matrix.data[off + j]! + bias.data[j]!;
  }
  return new Tensor([B, N], out);
}

export function sumAxis0(matrix: Tensor): Tensor {
  assert(matrix.rank === 2, "sumAxis0 expects 2D");
  const B = matrix.shape[0]!;
  const N = matrix.shape[1]!;
  const out = new Float32Array(N);
  for (let i = 0; i < B; i++) {
    const off = i * N;
    for (let j = 0; j < N; j++) out[j] += matrix.data[off + j]!;
  }
  return new Tensor([N], out);
}

export function meanAxis0(matrix: Tensor): Tensor {
  const s = sumAxis0(matrix);
  const B = matrix.shape[0]!;
  if (B === 0) return s;
  for (let i = 0; i < s.size; i++) s.data[i] /= B;
  return s;
}

export function add(a: Tensor, b: Tensor): Tensor {
  const [x, y, shape] = broadcastPair(a, b);
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i++) out[i] = x[i]! + y[i]!;
  return new Tensor(shape, out);
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const [x, y, shape] = broadcastPair(a, b);
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i++) out[i] = x[i]! - y[i]!;
  return new Tensor(shape, out);
}

export function mul(a: Tensor, b: Tensor): Tensor {
  const [x, y, shape] = broadcastPair(a, b);
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i++) out[i] = x[i]! * y[i]!;
  return new Tensor(shape, out);
}

export function div(a: Tensor, b: Tensor): Tensor {
  const [x, y, shape] = broadcastPair(a, b);
  const out = new Float32Array(numel(shape));
  for (let i = 0; i < out.length; i++) out[i] = x[i]! / y[i]!;
  return new Tensor(shape, out);
}

export function scale(a: Tensor, s: number): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i]! * s;
  return new Tensor(a.shape as number[], out);
}

export function addScalar(a: Tensor, s: number): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i]! + s;
  return new Tensor(a.shape as number[], out);
}

export function neg(a: Tensor): Tensor {
  return scale(a, -1);
}

export function square(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) {
    const v = a.data[i]!;
    out[i] = v * v;
  }
  return new Tensor(a.shape as number[], out);
}

export function abs(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.abs(a.data[i]!);
  return new Tensor(a.shape as number[], out);
}

export function clip(a: Tensor, lo: number, hi: number): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.min(hi, Math.max(lo, a.data[i]!));
  return new Tensor(a.shape as number[], out);
}

export function exp(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.exp(a.data[i]!);
  return new Tensor(a.shape as number[], out);
}

export function log(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.log(a.data[i]!);
  return new Tensor(a.shape as number[], out);
}

export function sqrt(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.sqrt(a.data[i]!);
  return new Tensor(a.shape as number[], out);
}

export function tanh(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.tanh(a.data[i]!);
  return new Tensor(a.shape as number[], out);
}

export function sum(a: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i]!;
  return s;
}

export function mean(a: Tensor): number {
  return a.size === 0 ? 0 : sum(a) / a.size;
}

export function maxVal(a: Tensor): number {
  let m = -Infinity;
  for (let i = 0; i < a.size; i++) if (a.data[i]! > m) m = a.data[i]!;
  return m;
}

export function minVal(a: Tensor): number {
  let m = Infinity;
  for (let i = 0; i < a.size; i++) if (a.data[i]! < m) m = a.data[i]!;
  return m;
}

export function l2Norm(a: Tensor): number {
  let s = 0;
  for (let i = 0; i < a.size; i++) {
    const v = a.data[i]!;
    s += v * v;
  }
  return Math.sqrt(s);
}

export function variance(a: Tensor): number {
  if (a.size === 0) return 0;
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.size; i++) {
    const d = a.data[i]! - m;
    s += d * d;
  }
  return s / a.size;
}

export function stddev(a: Tensor): number {
  return Math.sqrt(variance(a));
}

export function argmax1d(row: Float32Array | number[], offset: number, len: number): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = row[offset + i]!;
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

export function softmaxLast(x: Tensor): Tensor {
  const inner = x.shape[x.rank - 1]!;
  const outer = x.size / inner;
  const out = new Float32Array(x.size);
  for (let o = 0; o < outer; o++) {
    const off = o * inner;
    let max = -Infinity;
    for (let i = 0; i < inner; i++) {
      const v = x.data[off + i]!;
      if (v > max) max = v;
    }
    let s = 0;
    for (let i = 0; i < inner; i++) {
      const e = Math.exp(x.data[off + i]! - max);
      out[off + i] = e;
      s += e;
    }
    const inv = 1 / (s || 1e-12);
    for (let i = 0; i < inner; i++) out[off + i]! *= inv;
  }
  return new Tensor(x.shape as number[], out);
}

/** Softmax Jacobian-vector product along last axis. */
export function softmaxBackward(p: Tensor, dy: Tensor): Tensor {
  const inner = p.shape[p.rank - 1]!;
  const outer = p.size / inner;
  const out = new Float32Array(p.size);
  for (let o = 0; o < outer; o++) {
    const off = o * inner;
    let dot = 0;
    for (let i = 0; i < inner; i++) dot += p.data[off + i]! * dy.data[off + i]!;
    for (let i = 0; i < inner; i++) {
      out[off + i] = p.data[off + i]! * (dy.data[off + i]! - dot);
    }
  }
  return new Tensor(p.shape as number[], out);
}

export function concatLast(a: Tensor, b: Tensor): Tensor {
  assert(a.rank === b.rank, "concat rank mismatch");
  for (let i = 0; i < a.rank - 1; i++) {
    assert(a.shape[i] === b.shape[i], "concat leading dims mismatch");
  }
  const innerA = a.shape[a.rank - 1]!;
  const innerB = b.shape[b.rank - 1]!;
  const outer = a.size / innerA;
  const inner = innerA + innerB;
  const out = new Float32Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    const dst = o * inner;
    out.set(a.data.subarray(o * innerA, (o + 1) * innerA), dst);
    out.set(b.data.subarray(o * innerB, (o + 1) * innerB), dst + innerA);
  }
  const shape = (a.shape as number[]).slice();
  shape[shape.length - 1] = inner;
  return new Tensor(shape, out);
}

export function stats(t: Tensor): { min: number; max: number; mean: number; std: number; l2: number; size: number } {
  return {
    min: minVal(t),
    max: maxVal(t),
    mean: mean(t),
    std: stddev(t),
    l2: l2Norm(t),
    size: t.size,
  };
}

export function histogram(t: Tensor, bins = 24): { edges: number[]; counts: number[] } {
  if (t.size === 0) return { edges: [], counts: [] };
  let lo = minVal(t);
  let hi = maxVal(t);
  if (lo === hi) {
    const counts = new Array(bins).fill(0);
    counts[Math.floor(bins / 2)] = t.size;
    return { edges: Array.from({ length: bins + 1 }, (_, i) => lo + i * 0.01), counts };
  }
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < t.size; i++) {
    let b = Math.floor((t.data[i]! - lo) / width);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * width);
  return { edges, counts };
}

function broadcastPair(a: Tensor, b: Tensor): [Float32Array, Float32Array, number[]] {
  if (shapeEqual(a.shape, b.shape)) {
    return [a.data, b.data, a.shape as number[]];
  }
  const shape = broadcastShapes(a.shape, b.shape);
  return [expandTo(a, shape), expandTo(b, shape), shape];
}

function shapeEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function broadcastShapes(a: readonly number[], b: readonly number[]): number[] {
  const ndim = Math.max(a.length, b.length);
  const as = padLeft(a, ndim);
  const bs = padLeft(b, ndim);
  const out: number[] = [];
  for (let i = 0; i < ndim; i++) {
    const x = as[i]!;
    const y = bs[i]!;
    if (x === y) out.push(x);
    else if (x === 1) out.push(y);
    else if (y === 1) out.push(x);
    else throw new EngineError(`Cannot broadcast shapes [${a.join(", ")}] and [${b.join(", ")}]`);
  }
  return out;
}

function padLeft(shape: readonly number[], ndim: number): number[] {
  const out = new Array(ndim).fill(1);
  for (let i = 0; i < shape.length; i++) out[ndim - shape.length + i] = shape[i];
  return out;
}

function expandTo(t: Tensor, shape: number[]): Float32Array {
  if (shapeEqual(t.shape, shape)) return t.data;
  const out = new Float32Array(numel(shape));
  const tShape = padLeft(t.shape, shape.length);
  const tStrides = stridesOf(tShape);
  const outStrides = stridesOf(shape);
  const size = out.length;
  for (let i = 0; i < size; i++) {
    let rem = i;
    let src = 0;
    for (let d = 0; d < shape.length; d++) {
      const coord = Math.floor(rem / outStrides[d]!);
      rem = rem % outStrides[d]!;
      const sCoord = tShape[d] === 1 ? 0 : coord;
      src += sCoord * tStrides[d]!;
    }
    out[i] = t.data[src]!;
  }
  return out;
}

function stridesOf(shape: number[]): number[] {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i]!;
  }
  return s;
}

export function sliceBatch(x: Tensor, start: number, count: number): Tensor {
  assert(x.rank >= 1, "sliceBatch expects at least 1D");
  const B = x.shape[0]!;
  assert(start >= 0 && start + count <= B, `Batch slice [${start}, ${start + count}) out of range ${B}`);
  const rest = x.size / B;
  const out = x.data.subarray(start * rest, (start + count) * rest);
  const shape = [count, ...(x.shape as number[]).slice(1)];
  return new Tensor(shape, new Float32Array(out));
}

export function stackRows(rows: Float32Array[], rowLen: number): Tensor {
  const B = rows.length;
  const out = new Float32Array(B * rowLen);
  for (let i = 0; i < B; i++) out.set(rows[i]!.subarray(0, rowLen), i * rowLen);
  return new Tensor([B, rowLen], out);
}
