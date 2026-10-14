/**
 * WGSL compute kernels for the WebGPU tensor device.
 *
 * Conventions used by every kernel in this file:
 *   • tensors are flat row-major `array<f32>` storage buffers,
 *   • every kernel takes a params buffer `P: array<u32>` at its last binding.
 *     Integers are packed directly and floats are bitcast, because a plain u32
 *     array has no alignment traps and no 64 KiB uniform limit.
 *   • workgroup size is 64 unless the kernel says otherwise (matmul and the
 *     row-wise reductions use shared memory).
 *   • flat kernels fold their 1D index across the two grid dimensions
 *     (`gid.y * 65535 + gid.x`) so a single dispatch can cover more than the
 *     65535-workgroup per-dimension limit. The host mirrors this in
 *     `flatDispatch()`.
 *
 * Pooling backward deliberately re-derives the window winner from the input
 * instead of scatter-adding into an index buffer: WGSL has no floating point
 * atomics, and gathering is also deterministic for overlapping windows.
 */

/**
 * Tiled matmul: C[M,N] = op(A)·op(B).
 *
 * P[0]=M P[1]=N P[2]=K P[3]=transposeA P[4]=transposeB
 * bindings: 0=A 1=B 2=C(write) 3=P
 */
export const MATMUL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = P[0];
  let N = P[1];
  let K = P[2];
  let ta = P[3] == 1u;
  let tb = P[4] == 1u;
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;

  var acc = 0.0;
  let tiles = (K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let ka = t * 16u + lc;
    var va = 0.0;
    if (row < M && ka < K) {
      va = select(A[row * K + ka], A[ka * M + row], ta);
    }
    tileA[lr * 16u + lc] = va;

    let kb = t * 16u + lr;
    var vb = 0.0;
    if (kb < K && col < N) {
      vb = select(B[kb * N + col], B[col * K + kb], tb);
    }
    tileB[lr * 16u + lc] = vb;
    workgroupBarrier();

    for (var k = 0u; k < 16u; k = k + 1u) {
      acc = acc + tileA[lr * 16u + k] * tileB[k * 16u + lc];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
`;

/**
 * Elementwise unary ops plus the forward activations.
 *
 * P[0]=count P[1]=op P[2]=f32 alpha P[3]=f32 beta
 * ops: 0 copy 1 neg 2 abs 3 square 4 scale(·alpha) 5 clip(alpha..beta)
 *      6 add(alpha) 7 relu 8 leakyrelu(alpha) 9 gelu 10 silu 11 sigmoid 12 tanh
 * bindings: 0=X 1=Y(write) 2=P
 */
export const UNARY = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

fn sigmoid(v: f32) -> f32 {
  return 1.0 / (1.0 + exp(-v));
}

fn gelu(v: f32) -> f32 {
  return 0.5 * v * (1.0 + tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)));
}

fn activate(op: u32, v: f32, alpha: f32) -> f32 {
  if (op == 7u) { return max(v, 0.0); }
  if (op == 8u) { return select(alpha * v, v, v > 0.0); }
  if (op == 9u) { return gelu(v); }
  if (op == 10u) { return v * sigmoid(v); }
  if (op == 11u) { return sigmoid(v); }
  if (op == 12u) { return tanh(v); }
  return v;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P[0];
  let i = gid.y * 65535u + gid.x;
  if (i >= n) { return; }
  let op = P[1];
  let alpha = bitcast<f32>(P[2]);
  let beta = bitcast<f32>(P[3]);
  let v = X[i];
  var out = v;
  if (op == 1u) {
    out = -v;
  } else if (op == 2u) {
    out = abs(v);
  } else if (op == 3u) {
    out = v * v;
  } else if (op == 4u) {
    out = v * alpha;
  } else if (op == 5u) {
    out = clamp(v, alpha, beta);
  } else if (op == 6u) {
    out = v + alpha;
  } else {
    out = activate(op, v, alpha);
  }
  Y[i] = out;
}
`;

/**
 * Backward pass of the activations. Only the buffers an activation needs are
 * touched: sigmoid/tanh read the output, relu/leakyrelu/gelu/silu read the pre
 * activation input, softmax is handled by its own kernel.
 *
 * P[0]=count P[1]=op P[2]=f32 alpha
 * ops: 0 copy 1 relu 2 leakyrelu 3 gelu 4 silu 5 sigmoid 6 tanh
 * bindings: 0=X(pre) 1=Y(out) 2=G 3=DX(write) 4=P
 */
export const UNARY_BWD = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> DX: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

fn sigmoid(v: f32) -> f32 {
  return 1.0 / (1.0 + exp(-v));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P[0];
  let i = gid.y * 65535u + gid.x;
  if (i >= n) { return; }
  let op = P[1];
  let alpha = bitcast<f32>(P[2]);
  let g = G[i];
  var out = g;
  if (op == 1u) {
    out = select(0.0, g, X[i] > 0.0);
  } else if (op == 2u) {
    out = select(alpha * g, g, X[i] > 0.0);
  } else if (op == 3u) {
    let v = X[i];
    let u = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
    let th = tanh(u);
    let sech2 = 1.0 - th * th;
    let du = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * v * v);
    out = g * (0.5 * (1.0 + th) + 0.5 * v * sech2 * du);
  } else if (op == 4u) {
    let v = X[i];
    let s = sigmoid(v);
    out = g * (s + v * s * (1.0 - s));
  } else if (op == 5u) {
    let s = Y[i];
    out = s * (1.0 - s) * g;
  } else if (op == 6u) {
    let t = Y[i];
    out = (1.0 - t * t) * g;
  }
  DX[i] = out;
}
`;

/**
 * Broadcasting elementwise binary op.
 *
 * Shapes and strides are described with four right-aligned axes, so the host
 * pads rank-1/2/3 tensors with leading ones and gives broadcast axes a stride
 * of zero. One kernel therefore covers bias adds, per-channel scale/shift and
 * plain same-shape arithmetic.
 *
 * P[0]=count P[1]=op P[2..5]=outShape P[6..9]=aStrides P[10..13]=bStrides
 * ops: 0 add 1 sub 2 mul 3 div
 * bindings: 0=A 1=B 2=OUT(write) 3=P
 */
export const BINARY = /* wgsl */ `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P[0];
  let i = gid.y * 65535u + gid.x;
  if (i >= n) { return; }
  let d3 = P[5];
  let c3 = i % d3;
  let r3 = i / d3;
  let c2 = r3 % P[4];
  let r2 = r3 / P[4];
  let c1 = r2 % P[3];
  let c0 = r2 / P[3];
  let ai = c0 * P[6] + c1 * P[7] + c2 * P[8] + c3 * P[9];
  let bi = c0 * P[10] + c1 * P[11] + c2 * P[12] + c3 * P[13];
  let a = A[ai];
  let b = B[bi];
  var out = a + b;
  if (P[1] == 1u) {
    out = a - b;
  } else if (P[1] == 2u) {
    out = a * b;
  } else if (P[1] == 3u) {
    out = a / b;
  }
  OUT[i] = out;
}
`;

/**
 * Reduces every axis except the last: [d0, …, dk, N] → [N].
 *
 * P[0]=outer P[1]=inner P[2]=mode(0 sum, 1 mean) P[3]=group
 * bindings: 0=X 1=OUT(write) 2=P
 */
export const REDUCE_OUTER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outer = P[0];
  let inner = P[1];
  let k = gid.y * 65535u + gid.x;
  if (k >= P[3] * inner) { return; }
  let g = k / inner;
  let j = k % inner;
  var acc = 0.0;
  for (var i = 0u; i < outer; i = i + 1u) {
    acc = acc + X[(g * outer + i) * inner + j];
  }
  if (P[2] == 1u && outer > 0u) {
    acc = acc / f32(outer);
  }
  OUT[k] = acc;
}
`;

/**
 * Inverse of global average pooling: OUT[b,h,w,c] = scale · X[b,c].
 *
 * P[0]=count(B·H·W·C) P[1]=H·W·C P[2]=C P[3]=f32 scale
 * bindings: 0=X 1=OUT(write) 2=P
 */
export const BROADCAST_PLANES = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u + gid.x;
  if (i >= P[0]) { return; }
  let b = i / P[1];
  let c = i % P[2];
  OUT[i] = X[b * P[2] + c] * bitcast<f32>(P[3]);
}
`;

/**
 * Softmax over the last axis, one workgroup per row.
 *
 * P[0]=inner(columns per row)
 * bindings: 0=X 1=Y(write) 2=P
 */
export const SOFTMAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let inner = P[0];
  let base = (wid.y * 65535u + wid.x) * inner;
  let lane = lid.x;

  var m = -3.0e38;
  for (var i = lane; i < inner; i = i + 64u) {
    m = max(m, X[base + i]);
  }
  scratch[lane] = m;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (lane < s) { scratch[lane] = max(scratch[lane], scratch[lane + s]); }
    workgroupBarrier();
  }
  let mx = scratch[0];

  var total = 0.0;
  for (var i = lane; i < inner; i = i + 64u) {
    total = total + exp(X[base + i] - mx);
  }
  scratch[lane] = total;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (lane < s) { scratch[lane] = scratch[lane] + scratch[lane + s]; }
    workgroupBarrier();
  }
  let sum = scratch[0];
  let inv = 1.0 / max(sum, 1.0e-12);

  for (var i = lane; i < inner; i = i + 64u) {
    Y[base + i] = exp(X[base + i] - mx) * inv;
  }
}
`;

/**
 * Softmax backward: dx = y · (dy − Σ y·dy), one workgroup per row.
 *
 * P[0]=inner
 * bindings: 0=Y 1=G 2=DX(write) 3=P
 */
export const SOFTMAX_BWD = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Y: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let inner = P[0];
  let base = (wid.y * 65535u + wid.x) * inner;
  let lane = lid.x;

  var dot = 0.0;
  for (var i = lane; i < inner; i = i + 64u) {
    dot = dot + Y[base + i] * G[base + i];
  }
  scratch[lane] = dot;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (lane < s) { scratch[lane] = scratch[lane] + scratch[lane + s]; }
    workgroupBarrier();
  }
  let d = scratch[0];
  for (var i = lane; i < inner; i = i + 64u) {
    DX[base + i] = Y[base + i] * (G[base + i] - d);
  }
}
`;

/**
 * im2col: X[B,H,W,Cin] → COL[B·outH·outW, kH·kW·Cin], one thread per entry.
 *
 * P[0]=B P[1]=H P[2]=W P[3]=Cin P[4]=kH P[5]=kW P[6]=stride
 * P[7]=padTop P[8]=padLeft P[9]=outH P[10]=outW
 * bindings: 0=X 1=COL(write) 2=P
 */
export const CONV_IM2COL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> COL: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let H = P[1];
  let W = P[2];
  let Cin = P[3];
  let kH = P[4];
  let kW = P[5];
  let K = kH * kW * Cin;
  let outH = P[9];
  let outW = P[10];
  let sp = outH * outW;
  let rows = P[0] * sp;
  let idx = gid.y * 65535u + gid.x;
  if (idx >= rows * K) { return; }

  let k = idx % K;
  let row = idx / K;
  let n = row / sp;
  let rem = row % sp;
  let oh = rem / outW;
  let ow = rem % outW;
  let c = k % Cin;
  let kk = k / Cin;
  let kh = kk / kW;
  let kw = kk % kW;

  let ih = i32(oh * P[6] + kh) - i32(P[7]);
  let iw = i32(ow * P[6] + kw) - i32(P[8]);
  var v = 0.0;
  if (ih >= 0 && ih < i32(H) && iw >= 0 && iw < i32(W)) {
    v = X[((n * H + u32(ih)) * W + u32(iw)) * Cin + c];
  }
  COL[row * K + k] = v;
}
`;

/**
 * col2im: gathers COL[B·outH·outW, kH·kW·Cin] back into dX[B,H,W,Cin], one
 * thread per input element.
 *
 * P[0]=B P[1]=H P[2]=W P[3]=Cin P[4]=kH P[5]=kW P[6]=stride
 * P[7]=padTop P[8]=padLeft P[9]=outH P[10]=outW
 * bindings: 0=COL 1=DX(write) 2=P
 */
export const CONV_COL2IM = /* wgsl */ `
@group(0) @binding(0) var<storage, read> COL: array<f32>;
@group(0) @binding(1) var<storage, read_write> DX: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let H = P[1];
  let W = P[2];
  let Cin = P[3];
  let kH = P[4];
  let kW = P[5];
  let stride = P[6];
  let outH = P[9];
  let outW = P[10];
  let K = kH * kW * Cin;
  let sp = outH * outW;
  let idx = gid.y * 65535u + gid.x;
  if (idx >= P[0] * H * W * Cin) { return; }

  let c = idx % Cin;
  let rest = idx / Cin;
  let iw = rest % W;
  let rest2 = rest / W;
  let ih = rest2 % H;
  let n = rest2 / H;

  var acc = 0.0;
  for (var kh = 0u; kh < kH; kh = kh + 1u) {
    let t = ih + P[7];
    if (t < kh) { continue; }
    let ihp = t - kh;
    if (ihp % stride != 0u) { continue; }
    let oh = ihp / stride;
    if (oh >= outH) { continue; }
    for (var kw = 0u; kw < kW; kw = kw + 1u) {
      let u = iw + P[8];
      if (u < kw) { continue; }
      let iwp = u - kw;
      if (iwp % stride != 0u) { continue; }
      let ow = iwp / stride;
      if (ow >= outW) { continue; }
      let row = n * sp + oh * outW + ow;
      acc = acc + COL[row * K + (kh * kW + kw) * Cin + c];
    }
  }
  DX[idx] = acc;
}
`;

/**
 * 2D pooling forward (max or average over pool×pool windows).
 *
 * P[0]=B P[1]=H P[2]=W P[3]=C P[4]=outH P[5]=outW P[6]=pool P[7]=stride
 * P[8]=mode(0 max, 1 avg)
 * bindings: 0=X 1=OUT(write) 2=P
 */
export const POOL_FWD = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let H = P[1];
  let W = P[2];
  let C = P[3];
  let outH = P[4];
  let outW = P[5];
  let pool = P[6];
  let stride = P[7];
  let idx = gid.y * 65535u + gid.x;
  if (idx >= P[0] * outH * outW * C) { return; }

  let c = idx % C;
  let rest = idx / C;
  let ow = rest % outW;
  let rest2 = rest / outW;
  let oh = rest2 % outH;
  let n = rest2 / outH;

  var best = -3.0e38;
  var acc = 0.0;
  for (var kh = 0u; kh < pool; kh = kh + 1u) {
    let ih = oh * stride + kh;
    if (ih >= H) { continue; }
    for (var kw = 0u; kw < pool; kw = kw + 1u) {
      let iw = ow * stride + kw;
      if (iw >= W) { continue; }
      let v = X[((n * H + ih) * W + iw) * C + c];
      best = max(best, v);
      acc = acc + v;
    }
  }
  var out = acc / f32(pool * pool);
  if (P[8] == 0u) {
    out = select(best, 0.0, best <= -3.0e38);
  }
  OUT[idx] = out;
}
`;

/**
 * Pooling backward, one thread per input element. Max pooling re-derives each
 * window winner with the engine's "first strict maximum" rule and accumulates
 * every window that selected this element (exact for overlapping windows too);
 * average pooling spreads `dy / pool²` over the valid taps.
 *
 * P[0]=B P[1]=H P[2]=W P[3]=C P[4]=outH P[5]=outW P[6]=pool P[7]=stride
 * P[8]=mode(0 max, 1 avg)
 * bindings: 0=X 1=G 2=DX(write) 3=P
 */
export const POOL_BWD = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let H = P[1];
  let W = P[2];
  let C = P[3];
  let outH = P[4];
  let outW = P[5];
  let pool = P[6];
  let stride = P[7];
  let maxMode = P[8] == 0u;
  let mine = gid.y * 65535u + gid.x;
  if (mine >= P[0] * H * W * C) { return; }

  let c = mine % C;
  let rest = mine / C;
  let iw = rest % W;
  let rest2 = rest / W;
  let ih = rest2 % H;
  let n = rest2 / H;

  // First output row/column whose window reaches this element:
  // ceil((i - pool + 1) / stride), clamped at zero.
  var ohFirst = 0u;
  var owFirst = 0u;
  let th = i32(ih) - i32(pool) + 1;
  if (th > 0) { ohFirst = u32(th + i32(stride) - 1) / stride; }
  let tw = i32(iw) - i32(pool) + 1;
  if (tw > 0) { owFirst = u32(tw + i32(stride) - 1) / stride; }

  var acc = 0.0;
  for (var oh = ohFirst; oh * stride <= ih && oh < outH; oh = oh + 1u) {
    for (var ow = owFirst; ow * stride <= iw && ow < outW; ow = ow + 1u) {
      let g = G[((n * outH + oh) * outW + ow) * C + c];
      if (g == 0.0) { continue; }
      if (!maxMode) {
        acc = acc + g / f32(pool * pool);
        continue;
      }
      var best = -3.0e38;
      var bestIdx = 0u;
      for (var kh = 0u; kh < pool; kh = kh + 1u) {
        let sh = oh * stride + kh;
        if (sh >= H) { continue; }
        for (var kw = 0u; kw < pool; kw = kw + 1u) {
          let sw = ow * stride + kw;
          if (sw >= W) { continue; }
          let at = ((n * H + sh) * W + sw) * C + c;
          let v = X[at];
          if (v > best) {
            best = v;
            bestIdx = at;
          }
        }
      }
      if (bestIdx == mine) { acc = acc + g; }
    }
  }
  DX[mine] = acc;
}
`;

/**
 * Embedding lookup: OUT[i, d] = W[clamp(round(ids[i])), d].
 *
 * P[0]=count(rows·dim) P[1]=dim P[2]=vocab
 * bindings: 0=W 1=IDS 2=OUT(write) 3=P
 */
export const GATHER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Wmat: array<f32>;
@group(0) @binding(1) var<storage, read> IDS: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u + gid.x;
  if (i >= P[0]) { return; }
  let dim = P[1];
  let vocab = P[2];
  let row = i / dim;
  let d = i % dim;
  var id = i32(round(IDS[row]));
  id = clamp(id, 0, i32(vocab) - 1);
  OUT[i] = Wmat[u32(id) * dim + d];
}
`;

/**
 * Embedding gradient: GW[v, d] = Σ_{rows with id == v} G[row, d].
 *
 * One thread per vocabulary entry × dimension, each scanning the sequence
 * (O(vocab·dim·rows) work). WGSL has no float atomics, and the accelerator
 * only exercises this for models that train an embedding table.
 *
 * P[0]=count(vocab·dim) P[1]=dim P[2]=vocab P[3]=rows
 * bindings: 0=G 1=IDS 2=GW(write) 3=P
 */
export const SCATTER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read> IDS: array<f32>;
@group(0) @binding(2) var<storage, read_write> GW: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u + gid.x;
  if (i >= P[0]) { return; }
  let dim = P[1];
  let vocab = P[2];
  let rows = P[3];
  let v = i / dim;
  let d = i % dim;
  var acc = 0.0;
  for (var r = 0u; r < rows; r = r + 1u) {
    var id = i32(round(IDS[r]));
    id = clamp(id, 0, i32(vocab) - 1);
    if (u32(id) == v) {
      acc = acc + G[r * dim + d];
    }
  }
  GW[i] = acc;
}
`;

/** Zeroes a buffer (gradient buffers a block never touches). */
export const FILL_ZERO = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(1) var<storage, read> P: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * 65535u + gid.x;
  if (i >= P[0]) { return; }
  OUT[i] = 0.0;
}
`;

export type KernelName =
  | "matmul"
  | "unary"
  | "unaryBwd"
  | "binary"
  | "reduceOuter"
  | "broadcastPlanes"
  | "softmax"
  | "softmaxBwd"
  | "im2col"
  | "col2im"
  | "poolFwd"
  | "poolBwd"
  | "gather"
  | "scatter"
  | "fillZero";

interface KernelSpec {
  code: string;
  entry: string;
  /** Number of bindings before the trailing params buffer. */
  buffers: number;
  /** Workgroup size along x (used to size dispatches). */
  lanes: number;
}

export const KERNELS: Record<KernelName, KernelSpec> = {
  matmul: { code: MATMUL, entry: "main", buffers: 3, lanes: 16 },
  unary: { code: UNARY, entry: "main", buffers: 2, lanes: 64 },
  unaryBwd: { code: UNARY_BWD, entry: "main", buffers: 4, lanes: 64 },
  binary: { code: BINARY, entry: "main", buffers: 3, lanes: 64 },
  reduceOuter: { code: REDUCE_OUTER, entry: "main", buffers: 2, lanes: 64 },
  broadcastPlanes: { code: BROADCAST_PLANES, entry: "main", buffers: 2, lanes: 64 },
  softmax: { code: SOFTMAX, entry: "main", buffers: 2, lanes: 64 },
  softmaxBwd: { code: SOFTMAX_BWD, entry: "main", buffers: 3, lanes: 64 },
  im2col: { code: CONV_IM2COL, entry: "main", buffers: 2, lanes: 64 },
  col2im: { code: CONV_COL2IM, entry: "main", buffers: 2, lanes: 64 },
  poolFwd: { code: POOL_FWD, entry: "main", buffers: 2, lanes: 64 },
  poolBwd: { code: POOL_BWD, entry: "main", buffers: 3, lanes: 64 },
  gather: { code: GATHER, entry: "main", buffers: 3, lanes: 64 },
  scatter: { code: SCATTER, entry: "main", buffers: 3, lanes: 64 },
  fillZero: { code: FILL_ZERO, entry: "main", buffers: 1, lanes: 64 },
};

/** Splits a flat thread count over the two grid dimensions (65535 max each). */
export function flatDispatch(n: number, lanes = 64): [number, number, number] {
  const groups = Math.max(1, Math.ceil(n / lanes));
  const gx = Math.min(groups, 65535);
  const gy = Math.ceil(groups / 65535);
  return [gx, Math.max(1, gy), 1];
}

/** Packs host numbers into the u32 params buffer the kernels read. */
export class Params {
  private words: number[] = [];

  u32(...values: number[]): this {
    for (const v of values) this.words.push(Math.max(0, Math.floor(v)) >>> 0);
    return this;
  }

  f32(...values: number[]): this {
    const f = new Float32Array(1);
    for (const v of values) {
      f[0] = v;
      this.words.push(new Uint32Array(f.buffer)[0]!);
    }
    return this;
  }

  toU32Array(): Uint32Array {
    return new Uint32Array(this.words);
  }
}
