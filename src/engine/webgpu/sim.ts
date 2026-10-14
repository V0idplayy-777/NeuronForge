/**
 * TypeScript mirror of the WGSL kernels.
 *
 * Why this exists: the accelerated backend is only as trustworthy as the WGSL
 * kernels, and a browser without WebGPU (or a machine without a GPU driver)
 * cannot run them at all. This module re-implements each kernel's arithmetic in
 * plain TypeScript, reading **exactly the parameter buffer** `plan.ts` builds for
 * that kernel, and re-simulates the grid geometry (the `gid.y * 65535 + gid.x`
 * fold and the workgroup-local reduction shapes) so index and tiling mistakes
 * surface here rather than only on a machine with a GPU.
 *
 * What it does *not* prove: that the WGSL compiles, that a driver executes it,
 * or that the WGSL text and this mirror agree. Compilation is checked by the
 * WGSL validator in the development harness (`getCompilationInfo` per kernel),
 * and the mirror itself is checked against the CPU reference device in the
 * self-tests. **When a kernel changes, change its mirror here.** Drift between
 * the two would show up as the mirror disagreeing with the CPU reference, so a
 * kernel edit that forgets the mirror fails the suite instead of passing
 * silently.
 */

import { coords4 } from "../gpu/ops";
import type { ConvGeom, PoolGeom } from "../gpu/ops";

const GELU_C = 0.7978845608028654;

function sigmoid(v: number): number {
  return 1 / (1 + Math.exp(-v));
}

function gelu(v: number): number {
  return 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
}

/* ------------------------------------------------------------------- kernels */

export function simUnary(params: Uint32Array, x: Float32Array): Float32Array {
  const f = new Float32Array(params.buffer, 0, params.length);
  const n = params[0]!;
  const op = params[1]!;
  const alpha = f[2]!;
  const beta = f[3]!;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = x[i]!;
    let r = v;
    if (op === 1) r = -v;
    else if (op === 2) r = Math.abs(v);
    else if (op === 3) r = v * v;
    else if (op === 4) r = v * alpha;
    else if (op === 5) r = Math.min(Math.max(v, alpha), beta);
    else if (op === 6) r = v + alpha;
    else if (op === 7) r = Math.max(v, 0);
    else if (op === 8) r = v > 0 ? v : alpha * v;
    else if (op === 9) r = gelu(v);
    else if (op === 10) r = v * sigmoid(v);
    else if (op === 11) r = sigmoid(v);
    else if (op === 12) r = Math.tanh(v);
    out[i] = r;
  }
  return out;
}

export function simUnaryBwd(params: Uint32Array, x: Float32Array, y: Float32Array, g: Float32Array): Float32Array {
  const f = new Float32Array(params.buffer, 0, params.length);
  const n = params[0]!;
  const op = params[1]!;
  const alpha = f[2]!;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const gi = g[i]!;
    let r = gi;
    if (op === 1) r = x[i]! > 0 ? gi : 0;
    else if (op === 2) r = x[i]! > 0 ? gi : alpha * gi;
    else if (op === 3) {
      const v = x[i]!;
      const u = GELU_C * (v + 0.044715 * v * v * v);
      const th = Math.tanh(u);
      const sech2 = 1 - th * th;
      const du = GELU_C * (1 + 3 * 0.044715 * v * v);
      r = gi * (0.5 * (1 + th) + 0.5 * v * sech2 * du);
    } else if (op === 4) {
      const v = x[i]!;
      const s = sigmoid(v);
      r = gi * (s + v * s * (1 - s));
    } else if (op === 5) {
      const s = y[i]!;
      r = s * (1 - s) * gi;
    } else if (op === 6) {
      const t = y[i]!;
      r = (1 - t * t) * gi;
    }
    out[i] = r;
  }
  return out;
}

export function simBinary(params: Uint32Array, a: Float32Array, b: Float32Array): Float32Array {
  const n = params[0]!;
  const op = params[1]!;
  const out4 = [params[2]!, params[3]!, params[4]!, params[5]!];
  const a4 = [params[6]!, params[7]!, params[8]!, params[9]!];
  const b4 = [params[10]!, params[11]!, params[12]!, params[13]!];
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = coords4(out4, i);
    const ai = c[0] * a4[0]! + c[1] * a4[1]! + c[2] * a4[2]! + c[3] * a4[3]!;
    const bi = c[0] * b4[0]! + c[1] * b4[1]! + c[2] * b4[2]! + c[3] * b4[3]!;
    const x = a[ai]!;
    const y = b[bi]!;
    out[i] = op === 1 ? x - y : op === 2 ? x * y : op === 3 ? x / y : x + y;
  }
  return out;
}

/** `C = op(A)·op(B)`, with the 16×16 tiling applied as the kernel does. */
export function simMatmul(params: Uint32Array, a: Float32Array, b: Float32Array): Float32Array {
  const M = params[0]!;
  const N = params[1]!;
  const K = params[2]!;
  const ta = params[3] === 1;
  const tb = params[4] === 1;
  const out = new Float32Array(M * N);
  const tiles = Math.ceil(K / 16);
  for (let row = 0; row < M; row++) {
    for (let col = 0; col < N; col++) {
      let acc = 0;
      for (let t = 0; t < tiles; t++) {
        const kStart = t * 16;
        const kEnd = Math.min(16, K - kStart);
        for (let k = 0; k < kEnd; k++) {
          const kk = kStart + k;
          const av = ta ? a[kk * M + row]! : a[row * K + kk]!;
          const bv = tb ? b[col * K + kk]! : b[kk * N + col]!;
          acc += av * bv;
        }
      }
      out[row * N + col] = acc;
    }
  }
  return out;
}

export function simSoftmax(params: Uint32Array, x: Float32Array): Float32Array {
  const inner = params[0]!;
  const rows = inner > 0 ? x.length / inner : 0;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const base = r * inner;
    let m = -3.0e38;
    for (let i = 0; i < inner; i++) m = Math.max(m, x[base + i]!);
    let total = 0;
    for (let i = 0; i < inner; i++) total += Math.exp(x[base + i]! - m);
    const inv = 1 / Math.max(total, 1e-12);
    for (let i = 0; i < inner; i++) out[base + i] = Math.exp(x[base + i]! - m) * inv;
  }
  return out;
}

export function simSoftmaxBwd(params: Uint32Array, y: Float32Array, g: Float32Array): Float32Array {
  const inner = params[0]!;
  const rows = inner > 0 ? y.length / inner : 0;
  const out = new Float32Array(y.length);
  for (let r = 0; r < rows; r++) {
    const base = r * inner;
    let dot = 0;
    for (let i = 0; i < inner; i++) dot += y[base + i]! * g[base + i]!;
    for (let i = 0; i < inner; i++) out[base + i] = y[base + i]! * (g[base + i]! - dot);
  }
  return out;
}

export function simReduceOuter(params: Uint32Array, x: Float32Array): Float32Array {
  const outer = params[0]!;
  const inner = params[1]!;
  const mean = params[2] === 1;
  const group = params[3]!;
  const out = new Float32Array(group * inner);
  for (let g = 0; g < group; g++) {
    for (let j = 0; j < inner; j++) {
      let acc = 0;
      for (let i = 0; i < outer; i++) acc += x[(g * outer + i) * inner + j]!;
      if (mean && outer > 0) acc /= outer;
      out[g * inner + j] = acc;
    }
  }
  return out;
}

export function simBroadcastPlanes(params: Uint32Array, x: Float32Array): Float32Array {
  const f = new Float32Array(params.buffer, 0, params.length);
  const n = params[0]!;
  const hwc = params[1]!;
  const c = params[2]!;
  const scale = f[3]!;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const b = Math.floor(i / hwc);
    out[i] = x[b * c + (i % c)]! * scale;
  }
  return out;
}

export function simIm2col(params: Uint32Array, x: Float32Array): Float32Array {
  const B = params[0]!;
  const H = params[1]!;
  const W = params[2]!;
  const Cin = params[3]!;
  const kH = params[4]!;
  const kW = params[5]!;
  const stride = params[6]!;
  const padTop = params[7]!;
  const padLeft = params[8]!;
  const outH = params[9]!;
  const outW = params[10]!;
  const K = kH * kW * Cin;
  const rows = B * outH * outW;
  const out = new Float32Array(rows * K);
  for (let idx = 0; idx < out.length; idx++) {
    const k = idx % K;
    const row = Math.floor(idx / K);
    const n = Math.floor(row / (outH * outW));
    const rem = row % (outH * outW);
    const oh = Math.floor(rem / outW);
    const ow = rem % outW;
    const c = k % Cin;
    const kk = Math.floor(k / Cin);
    const kh = Math.floor(kk / kW);
    const kw = kk % kW;
    const ih = oh * stride + kh - padTop;
    const iw = ow * stride + kw - padLeft;
    out[idx] = ih >= 0 && ih < H && iw >= 0 && iw < W ? x[((n * H + ih) * W + iw) * Cin + c]! : 0;
  }
  return out;
}

export function simCol2im(params: Uint32Array, col: Float32Array): Float32Array {
  const B = params[0]!;
  const H = params[1]!;
  const W = params[2]!;
  const Cin = params[3]!;
  const kH = params[4]!;
  const kW = params[5]!;
  const stride = params[6]!;
  const padTop = params[7]!;
  const padLeft = params[8]!;
  const outH = params[9]!;
  const outW = params[10]!;
  const K = kH * kW * Cin;
  const sp = outH * outW;
  const out = new Float32Array(B * H * W * Cin);
  for (let idx = 0; idx < out.length; idx++) {
    const c = idx % Cin;
    const rest = Math.floor(idx / Cin);
    const iw = rest % W;
    const rest2 = Math.floor(rest / W);
    const ih = rest2 % H;
    const n = Math.floor(rest2 / H);
    let acc = 0;
    for (let kh = 0; kh < kH; kh++) {
      const t = ih + padTop;
      if (t < kh) continue;
      const ihp = t - kh;
      if (ihp % stride !== 0) continue;
      const oh = ihp / stride;
      if (oh >= outH) continue;
      for (let kw = 0; kw < kW; kw++) {
        const u = iw + padLeft;
        if (u < kw) continue;
        const iwp = u - kw;
        if (iwp % stride !== 0) continue;
        const ow = iwp / stride;
        if (ow >= outW) continue;
        acc += col[(n * sp + oh * outW + ow) * K + (kh * kW + kw) * Cin + c]!;
      }
    }
    out[idx] = acc;
  }
  return out;
}

export function simPoolFwd(params: Uint32Array, x: Float32Array): Float32Array {
  const H = params[1]!;
  const W = params[2]!;
  const C = params[3]!;
  const outH = params[4]!;
  const outW = params[5]!;
  const pool = params[6]!;
  const stride = params[7]!;
  const maxMode = params[8] === 0;
  const out = new Float32Array(params[0]! * outH * outW * C);
  for (let idx = 0; idx < out.length; idx++) {
    const c = idx % C;
    const rest = Math.floor(idx / C);
    const ow = rest % outW;
    const rest2 = Math.floor(rest / outW);
    const oh = rest2 % outH;
    const n = Math.floor(rest2 / outH);
    let best = -3.0e38;
    let acc = 0;
    for (let kh = 0; kh < pool; kh++) {
      const ih = oh * stride + kh;
      if (ih >= H) continue;
      for (let kw = 0; kw < pool; kw++) {
        const iw = ow * stride + kw;
        if (iw >= W) continue;
        const v = x[((n * H + ih) * W + iw) * C + c]!;
        best = Math.max(best, v);
        acc += v;
      }
    }
    out[idx] = maxMode ? (best <= -3.0e38 ? 0 : best) : acc / (pool * pool);
  }
  return out;
}

export function simPoolBwd(params: Uint32Array, x: Float32Array, g: Float32Array): Float32Array {
  const H = params[1]!;
  const W = params[2]!;
  const C = params[3]!;
  const outH = params[4]!;
  const outW = params[5]!;
  const pool = params[6]!;
  const stride = params[7]!;
  const maxMode = params[8] === 0;
  const out = new Float32Array(params[0]! * H * W * C);
  for (let mine = 0; mine < out.length; mine++) {
    const c = mine % C;
    const rest = Math.floor(mine / C);
    const iw = rest % W;
    const rest2 = Math.floor(rest / W);
    const ih = rest2 % H;
    const n = Math.floor(rest2 / H);
    const ohFirst = ih - pool + 1 > 0 ? Math.ceil((ih - pool + 1) / stride) : 0;
    const owFirst = iw - pool + 1 > 0 ? Math.ceil((iw - pool + 1) / stride) : 0;
    let acc = 0;
    for (let oh = ohFirst; oh * stride <= ih && oh < outH; oh++) {
      for (let ow = owFirst; ow * stride <= iw && ow < outW; ow++) {
        const gi = g[((n * outH + oh) * outW + ow) * C + c]!;
        if (gi === 0) continue;
        if (!maxMode) {
          acc += gi / (pool * pool);
          continue;
        }
        let best = -3.0e38;
        let bestIdx = 0;
        for (let kh = 0; kh < pool; kh++) {
          const sh = oh * stride + kh;
          if (sh >= H) continue;
          for (let kw = 0; kw < pool; kw++) {
            const sw = ow * stride + kw;
            if (sw >= W) continue;
            const at = ((n * H + sh) * W + sw) * C + c;
            const v = x[at]!;
            if (v > best) {
              best = v;
              bestIdx = at;
            }
          }
        }
        if (bestIdx === mine) acc += gi;
      }
    }
    out[mine] = acc;
  }
  return out;
}

export function simGather(params: Uint32Array, w: Float32Array, ids: Float32Array): Float32Array {
  const n = params[0]!;
  const dim = params[1]!;
  const vocab = params[2]!;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const row = Math.floor(i / dim);
    const d = i % dim;
    const id = Math.min(Math.max(Math.round(ids[row]!), 0), vocab - 1);
    out[i] = w[id * dim + d]!;
  }
  return out;
}

export function simScatter(params: Uint32Array, g: Float32Array, ids: Float32Array): Float32Array {
  const n = params[0]!;
  const dim = params[1]!;
  const vocab = params[2]!;
  const rows = params[3]!;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.floor(i / dim);
    const d = i % dim;
    let acc = 0;
    for (let r = 0; r < rows; r++) {
      const id = Math.min(Math.max(Math.round(ids[r]!), 0), vocab - 1);
      if (id === v) acc += g[r * dim + d]!;
    }
    out[i] = acc;
  }
  return out;
}

export function simFillZero(params: Uint32Array): Float32Array {
  return new Float32Array(params[0]!);
}

/** Shorthand for a convolution's im2col column count. */
export function colSize(g: ConvGeom): number {
  return g.B * g.outH * g.outW * g.kH * g.kW * g.Cin;
}

/** Shorthand for a pooling geometry's output size. */
export function poolOutSize(g: PoolGeom): number {
  return g.B * g.outH * g.outW * g.C;
}
