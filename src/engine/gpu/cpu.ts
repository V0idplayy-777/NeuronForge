import { Tensor, matmul, transpose2d, addBias as cpuAddBias, softmaxLast, softmaxBackward as cpuSoftmaxBackward } from "../tensor";
import { applyActivation, activationBackward, gelu, silu, relu, leakyRelu } from "../activations";
import { conv2dForward, conv2dBackward } from "../layers";
import type { ActivationName } from "../types";
import { broadcastPlan, coords4, devNumel } from "./ops";
import type { BinaryOp, ConvGeom, DevBuffer, DeviceInfo, GpuOps, PoolGeom, PoolMode, ReduceMode, UnaryOp } from "./ops";

/**
 * CPU reference device. Implements the exact same operation surface as the
 * WebGPU device by delegating to the float32 tensor engine, so the accelerator
 * can be exercised (and validated) on a machine with no GPU. It is also the
 * reference the WebGPU device is compared against.
 */
export class CpuOps implements GpuOps {
  readonly info: DeviceInfo = { kind: "cpu", label: "CPU reference (float32 tensor engine)" };

  async fromHost(data: Float32Array, shape: number[]): Promise<DevBuffer> {
    return { shape: shape.slice(), size: data.length, handle: data };
  }

  async toHost(t: DevBuffer): Promise<Float32Array> {
    return new Float32Array(t.handle as Float32Array);
  }

  /** Copies every buffer; there is no round trip to save on the CPU. */
  async toHostMany(targets: DevBuffer[]): Promise<Float32Array[]> {
    return targets.map((t) => new Float32Array(t.handle as Float32Array));
  }

  async zeros(shape: number[]): Promise<DevBuffer> {
    const n = shape.reduce((a, b) => a * b, 1);
    return { shape: shape.slice(), size: n, handle: new Float32Array(n) };
  }

  dispose(): void {
    /* nothing to release */
  }

  async reshapeView(t: DevBuffer, shape: number[]): Promise<DevBuffer> {
    return { shape: shape.slice(), size: t.size, handle: t.handle, view: true };
  }

  private tensor(t: DevBuffer): Tensor {
    return new Tensor(t.shape.slice(), t.handle as Float32Array);
  }

  private wrap(t: Tensor): DevBuffer {
    return { shape: (t.shape as number[]).slice(), size: t.size, handle: t.data };
  }

  async matmul(a: DevBuffer, b: DevBuffer, transposeA: boolean, transposeB: boolean): Promise<DevBuffer> {
    const ta = transposeA ? transpose2d(this.tensor(a)) : this.tensor(a);
    const tb = transposeB ? transpose2d(this.tensor(b)) : this.tensor(b);
    return this.wrap(matmul(ta, tb));
  }

  /**
   * Bias add along the last axis. Rank-2 inputs go through the engine's own
   * `addBias` (so the reference device is comparing the engine's real code
   * path); higher ranks — convolution outputs — use the broadcast plan, which
   * is also what the WebGPU device does.
   */
  async addBias(m: DevBuffer, bias: DevBuffer): Promise<DevBuffer> {
    if (m.shape.length === 2 && bias.shape.length === 1) {
      return this.wrap(cpuAddBias(this.tensor(m), this.tensor(bias)));
    }
    return this.binary("add", m, bias);
  }

  /**
   * Broadcasting elementwise op. Uses the same four-axis broadcast plan as the
   * WebGPU kernel so both devices are directly comparable.
   */
  async binary(op: BinaryOp, a: DevBuffer, b: DevBuffer): Promise<DevBuffer> {
    const plan = broadcastPlan(a.shape, b.shape);
    const out = new Float32Array(devNumel(plan.outShape));
    const av = a.handle as Float32Array;
    const bv = b.handle as Float32Array;
    for (let i = 0; i < out.length; i++) {
      const c = coords4(plan.out4, i);
      const ai = c[0] * plan.a4[0]! + c[1] * plan.a4[1]! + c[2] * plan.a4[2]! + c[3] * plan.a4[3]!;
      const bi = c[0] * plan.b4[0]! + c[1] * plan.b4[1]! + c[2] * plan.b4[2]! + c[3] * plan.b4[3]!;
      const x = av[ai]!;
      const y = bv[bi]!;
      out[i] = op === "add" ? x + y : op === "sub" ? x - y : op === "mul" ? x * y : x / y;
    }
    return this.wrap(new Tensor(plan.outShape, out));
  }

  async unary(op: UnaryOp, x: DevBuffer, scalar: number, scalar2: number): Promise<DevBuffer> {
    const t = this.tensor(x);
    const out = new Float32Array(t.size);
    for (let i = 0; i < t.size; i++) {
      const v = t.data[i]!;
      switch (op) {
        case "neg":
          out[i] = -v;
          break;
        case "abs":
          out[i] = Math.abs(v);
          break;
        case "square":
          out[i] = v * v;
          break;
        case "scale":
          out[i] = v * scalar;
          break;
        case "clip":
          out[i] = Math.min(scalar2, Math.max(scalar, v));
          break;
      }
    }
    return this.wrap(new Tensor(t.shape as number[], out));
  }

  async activationFwd(name: string, x: DevBuffer, alpha: number): Promise<DevBuffer> {
    const t = this.tensor(x);
    switch (name) {
      case "none":
      case "linear":
        return this.wrap(t);
      case "relu":
        return this.wrap(relu(t));
      case "leakyrelu":
        return this.wrap(leakyRelu(t, alpha));
      case "gelu":
        return this.wrap(gelu(t));
      case "silu":
        return this.wrap(silu(t));
      default:
        return this.wrap(applyActivation(name as ActivationName, t));
    }
  }

  async activationBwd(name: string, x: DevBuffer, y: DevBuffer, dy: DevBuffer, alpha: number): Promise<DevBuffer> {
    if (name === "none" || name === "linear") return this.wrap(this.tensor(dy));
    if (name === "softmax") return this.wrap(cpuSoftmaxBackward(this.tensor(y), this.tensor(dy)));
    return this.wrap(activationBackward(name as ActivationName, this.tensor(x), this.tensor(y), this.tensor(dy), alpha));
  }

  async softmaxLast(x: DevBuffer): Promise<DevBuffer> {
    return this.wrap(softmaxLast(this.tensor(x)));
  }

  async softmaxBackward(p: DevBuffer, dy: DevBuffer): Promise<DevBuffer> {
    return this.wrap(cpuSoftmaxBackward(this.tensor(p), this.tensor(dy)));
  }

  async reduceOuter(mode: ReduceMode, m: DevBuffer, group = 1): Promise<DevBuffer> {
    const t = this.tensor(m);
    const inner = t.shape[t.shape.length - 1]!;
    const outer = t.size / Math.max(1, inner) / Math.max(1, group);
    const out = new Float32Array(group * inner);
    for (let g = 0; g < group; g++) {
      for (let i = 0; i < outer; i++) {
        const off = (g * outer + i) * inner;
        for (let j = 0; j < inner; j++) out[g * inner + j]! += t.data[off + j]!;
      }
    }
    if (mode === "mean" && outer > 0) for (let k = 0; k < out.length; k++) out[k]! /= outer;
    return this.wrap(new Tensor([group, inner], out));
  }

  async broadcastPlanes(t: DevBuffer, geom: { B: number; H: number; W: number; C: number }, scale: number): Promise<DevBuffer> {
    const src = this.tensor(t);
    const { B, H, W, C } = geom;
    const plane = H * W * C;
    const out = new Float32Array(B * plane);
    for (let i = 0; i < out.length; i++) {
      const b = Math.floor(i / plane);
      const c = i % C;
      out[i] = src.data[b * C + c]! * scale;
    }
    return this.wrap(new Tensor([B, H, W, C], out));
  }

  async conv2d(x: DevBuffer, k: DevBuffer, bias: DevBuffer | null, g: ConvGeom): Promise<DevBuffer> {
    const out = conv2dForward(
      this.tensor(x),
      this.tensor(k),
      bias ? this.tensor(bias) : null,
      g.B,
      g.H,
      g.W,
      g.Cin,
      g.outH,
      g.outW,
      g.F,
      g.kH,
      g.kW,
      g.stride,
      g.padTop,
      g.padLeft,
    );
    return this.wrap(out);
  }

  async conv2dInputGrad(dy: DevBuffer, k: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    const x = Tensor.zeros([g.B, g.H, g.W, g.Cin]);
    const { dX } = conv2dBackward(x, this.tensor(k), this.tensor(dy), g.B, g.H, g.W, g.Cin, g.outH, g.outW, g.F, g.kH, g.kW, g.stride, g.padTop, g.padLeft);
    return this.wrap(dX);
  }

  async conv2dKernelGrad(x: DevBuffer, dy: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    const k = Tensor.zeros([g.kH, g.kW, g.Cin, g.F]);
    const { dK } = conv2dBackward(this.tensor(x), k, this.tensor(dy), g.B, g.H, g.W, g.Cin, g.outH, g.outW, g.F, g.kH, g.kW, g.stride, g.padTop, g.padLeft);
    return this.wrap(dK);
  }

  async pool2d(mode: PoolMode, x: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    const t = this.tensor(x);
    const out = new Float32Array(g.B * g.outH * g.outW * g.C);
    const d = t.data;
    const area = g.pool * g.pool;
    for (let n = 0; n < g.B; n++) {
      for (let oh = 0; oh < g.outH; oh++) {
        for (let ow = 0; ow < g.outW; ow++) {
          for (let c = 0; c < g.C; c++) {
            let best = -Infinity;
            let acc = 0;
            for (let kh = 0; kh < g.pool; kh++) {
              const ih = oh * g.stride + kh;
              if (ih >= g.H) continue;
              for (let kw = 0; kw < g.pool; kw++) {
                const iw = ow * g.stride + kw;
                if (iw >= g.W) continue;
                const v = d[((n * g.H + ih) * g.W + iw) * g.C + c]!;
                if (v > best) best = v;
                acc += v;
              }
            }
            out[((n * g.outH + oh) * g.outW + ow) * g.C + c] = mode === "max" ? (best === -Infinity ? 0 : best) : acc / area;
          }
        }
      }
    }
    return this.wrap(new Tensor([g.B, g.outH, g.outW, g.C], out));
  }

  /**
   * Pooling backward. Max pooling recomputes the winner from the input (the
   * same "first strict maximum" rule as the engine) and accumulates every
   * window that selected this element; average pooling spreads `dy / pool²`
   * over the valid taps. Both match the engine for overlapping windows.
   */
  async pool2dBackward(mode: PoolMode, x: DevBuffer, dy: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    const src = this.tensor(x).data;
    const grad = this.tensor(dy).data;
    const dx = new Float32Array(g.B * g.H * g.W * g.C);
    const area = g.pool * g.pool;
    const ohMin = (ih: number): number => Math.ceil((ih - g.pool + 1) / g.stride);
    for (let n = 0; n < g.B; n++) {
      for (let ih = 0; ih < g.H; ih++) {
        for (let iw = 0; iw < g.W; iw++) {
          for (let c = 0; c < g.C; c++) {
            const mine = ((n * g.H + ih) * g.W + iw) * g.C + c;
            let acc = 0;
            for (let oh = Math.max(0, ohMin(ih)); oh * g.stride <= ih && oh < g.outH; oh++) {
              for (let ow = Math.max(0, ohMin(iw)); ow * g.stride <= iw && ow < g.outW; ow++) {
                const gy = grad[((n * g.outH + oh) * g.outW + ow) * g.C + c]!;
                if (gy === 0) continue;
                if (mode === "avg") {
                  acc += gy / area;
                  continue;
                }
                let best = -Infinity;
                let bestIdx = -1;
                for (let kh = 0; kh < g.pool; kh++) {
                  const sh = oh * g.stride + kh;
                  if (sh >= g.H) continue;
                  for (let kw = 0; kw < g.pool; kw++) {
                    const sw = ow * g.stride + kw;
                    if (sw >= g.W) continue;
                    const at = ((n * g.H + sh) * g.W + sw) * g.C + c;
                    const v = src[at]!;
                    if (v > best) {
                      best = v;
                      bestIdx = at;
                    }
                  }
                }
                if (bestIdx === mine) acc += gy;
              }
            }
            dx[mine] = acc;
          }
        }
      }
    }
    return this.wrap(new Tensor([g.B, g.H, g.W, g.C], dx));
  }

  async gatherRows(w: DevBuffer, ids: DevBuffer, dim: number): Promise<DevBuffer> {
    const src = this.tensor(w);
    const idx = ids.handle as Float32Array;
    const vocab = src.shape[0]!;
    const out = new Float32Array(idx.length * dim);
    for (let i = 0; i < idx.length; i++) {
      let id = Math.round(idx[i]!);
      if (id < 0) id = 0;
      if (id >= vocab) id = vocab - 1;
      out.set(src.data.subarray(id * dim, id * dim + dim), i * dim);
    }
    return { shape: [idx.length, dim], size: out.length, handle: out };
  }

  async scatterRows(grad: DevBuffer, ids: DevBuffer, dim: number, vocab: number): Promise<DevBuffer> {
    const g = this.tensor(grad);
    const idx = ids.handle as Float32Array;
    const out = new Float32Array(vocab * dim);
    for (let i = 0; i < idx.length; i++) {
      let id = Math.round(idx[i]!);
      if (id < 0) id = 0;
      if (id >= vocab) id = vocab - 1;
      for (let d = 0; d < dim; d++) out[id * dim + d]! += g.data[i * dim + d]!;
    }
    return this.wrap(new Tensor([vocab, dim], out));
  }

  destroy(): void {
    /* nothing to release */
  }
}
