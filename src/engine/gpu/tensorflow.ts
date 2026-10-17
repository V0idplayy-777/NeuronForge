import * as tf from "@tensorflow/tfjs";
import type {
  BinaryOp, ConvGeom, DevBuffer, DeviceInfo, GpuOps, PoolGeom, PoolMode, ReduceMode, UnaryOp,
} from "./ops";
import { broadcastPlan, devNumel } from "./ops";

/** TensorFlow.js kernels behind the same verified operation surface as WGSL.
 * CPU model weights remain authoritative; loss/optimizers/export are unchanged.
 * Views share their owner's tensor. Temporary reshapes live only inside tidy().
 */
export class TensorFlowOps implements GpuOps {
  readonly info: DeviceInfo;
  private tensors = new Set<tf.Tensor>();

  constructor() {
    this.info = { kind: "tensorflow", label: `TensorFlow.js · ${tf.getBackend()}` };
  }

  static async create(): Promise<TensorFlowOps> {
    // WebGL works in browsers without WebGPU. Node/headless environments use
    // TFJS CPU. No CDN, pretrained weights, or remote service is required.
    let webgl = false;
    if (typeof document !== "undefined") {
      try { webgl = await tf.setBackend("webgl"); } catch { /* try CPU */ }
    }
    if (!webgl && !await tf.setBackend("cpu")) throw new Error("TensorFlow.js CPU backend unavailable.");
    await tf.ready();
    return new TensorFlowOps();
  }

  private tensor(b: DevBuffer): tf.Tensor {
    const t = b.handle as tf.Tensor;
    if (!this.tensors.has(t)) throw new Error("TensorFlow.js buffer was disposed or belongs to another backend.");
    return t.reshape(b.shape);
  }

  private run(fn: () => tf.Tensor): DevBuffer {
    const t = tf.tidy(fn);
    this.tensors.add(t);
    return { shape: t.shape.slice(), size: t.size, handle: t };
  }

  async fromHost(data: Float32Array, shape: number[]): Promise<DevBuffer> {
    // Copy: TFJS CPU may retain the input array, while the optimizer mutates it.
    return this.run(() => tf.tensor(new Float32Array(data), shape, "float32"));
  }
  async toHost(b: DevBuffer): Promise<Float32Array> {
    const t = b.handle as tf.Tensor;
    if (!this.tensors.has(t)) throw new Error("TensorFlow.js buffer is no longer live.");
    return new Float32Array(await t.data());
  }
  async toHostMany(bs: DevBuffer[]): Promise<Float32Array[]> { return Promise.all(bs.map(b => this.toHost(b))); }
  async zeros(shape: number[]): Promise<DevBuffer> { return this.run(() => tf.zeros(shape)); }
  dispose(b: DevBuffer): void {
    if (b.view) return;
    const t = b.handle as tf.Tensor;
    if (this.tensors.delete(t)) t.dispose();
  }
  async reshapeView(b: DevBuffer, shape: number[]): Promise<DevBuffer> {
    if (devNumel(shape) !== b.size) throw new Error("TensorFlow.js reshape size mismatch.");
    return { ...b, shape: shape.slice(), view: true };
  }
  async matmul(a: DevBuffer, b: DevBuffer, ta: boolean, tb: boolean): Promise<DevBuffer> {
    return this.run(() => tf.matMul(this.tensor(a), this.tensor(b), ta, tb));
  }
  async addBias(m: DevBuffer, b: DevBuffer): Promise<DevBuffer> { return this.binary("add", m, b); }
  async binary(op: BinaryOp, a: DevBuffer, b: DevBuffer): Promise<DevBuffer> {
    const plan = broadcastPlan(a.shape, b.shape);
    return this.run(() => {
      const x = this.tensor(a), y = this.tensor(b);
      const z = op === "add" ? tf.add(x, y) : op === "sub" ? tf.sub(x, y) : op === "mul" ? tf.mul(x, y) : tf.div(x, y);
      return z.reshape(plan.outShape);
    });
  }
  async unary(op: UnaryOp, b: DevBuffer, s: number, s2: number): Promise<DevBuffer> {
    return this.run(() => {
      const x = this.tensor(b);
      switch (op) {
        case "neg": return tf.neg(x);
        case "abs": return tf.abs(x);
        case "square": return tf.square(x);
        case "scale": return tf.mul(x, s);
        case "clip": return tf.clipByValue(x, s, s2);
      }
    });
  }

  private activate(name: string, x: tf.Tensor, alpha: number): tf.Tensor {
    switch (name) {
      case "none": case "linear": return x.clone();
      case "relu": return tf.relu(x);
      case "leakyrelu": return tf.where(tf.greater(x, 0), x, tf.mul(x, alpha));
      case "sigmoid": return tf.sigmoid(x);
      case "tanh": return tf.tanh(x);
      case "silu": return tf.mul(x, tf.sigmoid(x));
      case "gelu": {
        const u = tf.mul(tf.add(x, tf.mul(tf.pow(x, 3), 0.044715)), Math.sqrt(2 / Math.PI));
        return tf.mul(tf.mul(x, 0.5), tf.add(1, tf.tanh(u)));
      }
      case "softmax": return tf.softmax(x);
      default: throw new Error(`Unsupported TensorFlow.js activation: ${name}`);
    }
  }
  async activationFwd(name: string, b: DevBuffer, alpha: number): Promise<DevBuffer> {
    return this.run(() => this.activate(name, this.tensor(b), alpha));
  }
  async activationBwd(name: string, bx: DevBuffer, by: DevBuffer, bd: DevBuffer, alpha: number): Promise<DevBuffer> {
    if (name === "softmax") return this.softmaxBackward(by, bd);
    return this.run(() => {
      const x = this.tensor(bx), y = this.tensor(by), d = this.tensor(bd);
      // Match the engine's derivatives at zero and use its cached float32 y.
      let slope: tf.Tensor;
      switch (name) {
        case "none": case "linear": return d.clone();
        case "relu": slope = tf.cast(tf.greater(x, 0), "float32"); break;
        case "leakyrelu": slope = tf.where(tf.greater(x, 0), tf.onesLike(x), tf.fill(x.shape, alpha)); break;
        case "sigmoid": slope = tf.mul(y, tf.sub(1, y)); break;
        case "tanh": slope = tf.sub(1, tf.square(y)); break;
        case "silu": {
          const s = tf.sigmoid(x);
          slope = tf.add(s, tf.mul(tf.mul(x, s), tf.sub(1, s))); break;
        }
        case "gelu": {
          const c = Math.sqrt(2 / Math.PI);
          const th = tf.tanh(tf.mul(tf.add(x, tf.mul(tf.pow(x, 3), 0.044715)), c));
          const du = tf.mul(tf.add(1, tf.mul(tf.square(x), 3 * 0.044715)), c);
          slope = tf.add(tf.mul(tf.add(1, th), 0.5), tf.mul(tf.mul(tf.mul(x, 0.5), tf.sub(1, tf.square(th))), du));
          break;
        }
        default: throw new Error(`Unsupported TensorFlow.js derivative: ${name}`);
      }
      return tf.mul(slope, d);
    });
  }
  async softmaxLast(x: DevBuffer): Promise<DevBuffer> { return this.run(() => tf.softmax(this.tensor(x))); }
  async softmaxBackward(p: DevBuffer, dy: DevBuffer): Promise<DevBuffer> {
    return this.run(() => {
      const y = this.tensor(p), d = this.tensor(dy);
      return tf.mul(y, tf.sub(d, tf.sum(tf.mul(y, d), -1, true)));
    });
  }
  async reduceOuter(mode: ReduceMode, b: DevBuffer, group = 1): Promise<DevBuffer> {
    return this.run(() => {
      const inner = b.shape[b.shape.length - 1]!;
      const x = this.tensor(b).reshape([group, b.size / group / inner, inner]);
      return mode === "sum" ? tf.sum(x, 1) : tf.mean(x, 1);
    });
  }
  async broadcastPlanes(b: DevBuffer, g: { B: number; H: number; W: number; C: number }, scale: number): Promise<DevBuffer> {
    return this.run(() => tf.mul(tf.tile(this.tensor(b).reshape([g.B, 1, 1, g.C]), [1, g.H, g.W, 1]), scale));
  }

  private convolution(x: tf.Tensor, k: tf.Tensor, g: ConvGeom): tf.Tensor4D {
    const bottom = Math.max(0, (g.outH - 1) * g.stride + g.kH - g.H - g.padTop);
    const right = Math.max(0, (g.outW - 1) * g.stride + g.kW - g.W - g.padLeft);
    const padded = tf.pad(x, [[0, 0], [g.padTop, bottom], [g.padLeft, right], [0, 0]]) as tf.Tensor4D;
    return tf.conv2d(padded, k as tf.Tensor4D, g.stride, "valid");
  }
  async conv2d(x: DevBuffer, k: DevBuffer, bias: DevBuffer | null, g: ConvGeom): Promise<DevBuffer> {
    return this.run(() => {
      const y = this.convolution(this.tensor(x), this.tensor(k), g);
      return bias ? tf.add(y, this.tensor(bias)) : y;
    });
  }
  async conv2dInputGrad(dy: DevBuffer, k: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    return this.run(() => tf.grad(x => this.convolution(x, this.tensor(k), g))(
      tf.zeros([g.B, g.H, g.W, g.Cin]), this.tensor(dy),
    ));
  }
  async conv2dKernelGrad(x: DevBuffer, dy: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    return this.run(() => tf.grad(k => this.convolution(this.tensor(x), k, g))(
      tf.zeros([g.kH, g.kW, g.Cin, g.F]), this.tensor(dy),
    ));
  }
  private pool(mode: PoolMode, x: tf.Tensor, g: PoolGeom): tf.Tensor {
    return mode === "max" ? tf.maxPool(x as tf.Tensor4D, g.pool, g.stride, "valid")
      : tf.avgPool(x as tf.Tensor4D, g.pool, g.stride, "valid");
  }
  async pool2d(mode: PoolMode, x: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    return this.run(() => this.pool(mode, this.tensor(x), g));
  }
  async pool2dBackward(mode: PoolMode, x: DevBuffer, dy: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    return this.run(() => tf.grad(t => this.pool(mode, t, g))(this.tensor(x), this.tensor(dy)));
  }
  async gatherRows(w: DevBuffer, ids: DevBuffer, dim: number): Promise<DevBuffer> {
    return this.run(() => {
      const indices = tf.clipByValue(tf.floor(tf.add(this.tensor(ids), 0.5)), 0, w.shape[0]! - 1).cast("int32").flatten();
      return tf.gather(this.tensor(w), indices).reshape([ids.size, dim]);
    });
  }
  async scatterRows(grad: DevBuffer, ids: DevBuffer, dim: number, vocab: number): Promise<DevBuffer> {
    return this.run(() => {
      const indices = tf.clipByValue(tf.floor(tf.add(this.tensor(ids), 0.5)), 0, vocab - 1).cast("int32").flatten();
      return tf.unsortedSegmentSum(this.tensor(grad).reshape([ids.size, dim]), indices, vocab);
    });
  }
  destroy(): void {
    for (const t of this.tensors) t.dispose();
    this.tensors.clear();
  }
}
