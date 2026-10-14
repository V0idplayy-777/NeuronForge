/**
 * WebGPU implementation of the `GpuOps` device surface.
 *
 * The fifteen WGSL kernels live in `kernels.ts`; this file is the host side:
 * buffer pooling, command recording, dispatch geometry, readback and device
 * loss handling.
 *
 * Three rules keep the fast path fast:
 *   • every operation records one compute pass into a *shared* command encoder,
 *     which is submitted when a result is read back or when `sync()` is called.
 *     A training batch therefore costs a couple of submissions rather than one
 *     per operation;
 *   • buffers are recycled through a size-class pool, so a steady-state
 *     training loop allocates almost nothing;
 *   • buffers released between two submissions only become recyclable *after*
 *     the next submit. That matters because `queue.writeBuffer` is a queue
 *     operation: reusing a slot immediately could overwrite data that a still
 *     queued dispatch is about to read (commands execute in submission order,
 *     but an unsubmitted command is not ordered against a queued write yet).
 */

import type { BinaryOp, ConvGeom, DevBuffer, DeviceInfo, GpuOps, PoolGeom, PoolMode, ReduceMode, UnaryOp } from "../gpu/ops";
import { broadcastPlan } from "../gpu/ops";
import { KERNELS, type KernelName } from "./kernels";
import {
  activationCode,
  backwardCode,
  binaryCode,
  binaryStep,
  broadcastPlanesStep,
  col2imStep,
  fillZeroStep,
  gatherStep,
  im2colStep,
  matmulDims,
  matmulStep,
  poolBwdStep,
  poolFwdStep,
  reduceOuterStep,
  scatterStep,
  softmaxBwdStep,
  softmaxStep,
  unaryBwdStep,
  unaryCode,
  unaryStep,
  type Step,
} from "./plan";

/** The opaque handle a `DevBuffer` carries on this device. */
interface GpuHandle {
  buf: GPUBuffer;
}

function storageUsage(): GPUBufferUsageFlags {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
}

function paramsUsage(): GPUBufferUsageFlags {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
}

function stagingUsage(): GPUBufferUsageFlags {
  return GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
}

/** Beyond this the pool stops recycling and actually frees memory. */
const MAX_POOL_BYTES = 96 * 1024 * 1024;

function numel(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return shape.length === 0 ? 1 : n;
}

export class WebGpuDevice implements GpuOps {
  readonly info: DeviceInfo;
  /** The raw device. Exposed for diagnostics; the accelerator only uses `GpuOps`. */
  readonly gpu: GPUDevice;
  readonly adapterInfo: GPUAdapterInfo | null;

  /** Set by the `uncapturederror` listener and surfaced on the next flush/sync. */
  lastError: string | null = null;

  private free = new Map<number, GPUBuffer[]>();
  private pendingFree: GPUBuffer[] = [];
  private pooledBytes = 0;

  private stagingFree: GPUBuffer[] = [];
  private enc: GPUCommandEncoder | null = null;
  private pendingParams: GPUBuffer[] = [];

  private pipelines = new Map<KernelName, GPUComputePipeline>();
  private leased = 0;
  private destroyed = false;
  private lostReason: string | null = null;

  private constructor(gpu: GPUDevice, info: DeviceInfo, adapterInfo: GPUAdapterInfo | null) {
    this.gpu = gpu;
    this.info = info;
    this.adapterInfo = adapterInfo;
    gpu.addEventListener?.("uncapturederror", (event) => {
      this.lastError = (event as GPUUncapturedErrorEvent).error.message;
    });
    void gpu.lost?.then((detail) => {
      this.lostReason = detail.message ? `${detail.reason}: ${detail.message}` : detail.reason;
    });
  }

  /* -------------------------------------------------------------- lifecycle */

  /** True when this browser exposes WebGPU at all. */
  static available(): boolean {
    return typeof navigator !== "undefined" && !!(navigator as Navigator & { gpu?: GPU }).gpu;
  }

  /**
   * Requests an adapter and a device. Throws a readable error when the browser
   * has no WebGPU, has it disabled, or has no adapter to hand out — callers
   * turn that into a CPU fallback.
   */
  static async create(options?: { powerPreference?: GPUPowerPreference }): Promise<WebGpuDevice> {
    const gpu = (typeof navigator !== "undefined" ? (navigator as Navigator & { gpu?: GPU }).gpu : undefined) ?? null;
    if (!gpu) throw new Error("WebGPU is not available in this browser (navigator.gpu is missing).");
    const adapter = await gpu.requestAdapter({
      powerPreference: options?.powerPreference ?? "high-performance",
    });
    if (!adapter) throw new Error("WebGPU is present but no adapter was returned (the GPU may be blocked or unavailable).");
    const device = await adapter.requestDevice({ label: "neuronforge" });
    const info = adapter.info ?? null;
    const parts = [info?.vendor, info?.architecture, info?.device, info?.description].filter(
      (p): p is string => typeof p === "string" && p.length > 0 && p !== "unknown",
    );
    return new WebGpuDevice(
      device,
      {
        kind: "webgpu",
        label: parts.length ? `WebGPU (${parts.join(" · ")})` : "WebGPU",
        adapter: parts.length ? parts.join(" ") : undefined,
        maxBufferBytes: device.limits.maxBufferSize,
      },
      info,
    );
  }

  /** Why the device was lost, if it was. */
  get lost(): string | null {
    return this.lostReason;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.enc = null;
    for (const buf of this.pendingParams) buf.destroy();
    this.pendingParams = [];
    for (const list of this.free.values()) for (const buf of list) buf.destroy();
    this.free.clear();
    for (const buf of this.pendingFree) buf.destroy();
    this.pendingFree = [];
    for (const buf of this.stagingFree) buf.destroy();
    this.stagingFree = [];
    this.pipelines.clear();
    try {
      this.gpu.destroy();
    } catch {
      /* already gone */
    }
  }

  /* ---------------------------------------------------------------- buffers */

  private alloc(bytes: number): GPUBuffer {
    const size = Math.max(4, (bytes + 3) & ~3);
    if (size > this.gpu.limits.maxBufferSize) {
      throw new Error(
        `WebGPU: a tensor needs ${size} bytes, more than this device's maxBufferSize (${this.gpu.limits.maxBufferSize}).`,
      );
    }
    const list = this.free.get(size);
    let buf: GPUBuffer;
    if (list && list.length > 0) {
      buf = list.pop()!;
      this.pooledBytes -= size;
    } else {
      buf = this.gpu.createBuffer({ size, usage: storageUsage(), label: "nf-storage" });
    }
    this.leased++;
    return buf;
  }

  private release(buf: GPUBuffer): void {
    if (this.destroyed) {
      buf.destroy();
      return;
    }
    this.leased--;
    this.pendingFree.push(buf);
  }

  /** Recycles (or frees) everything released since the previous submit. */
  private retire(): void {
    if (this.pendingFree.length === 0) return;
    const released = this.pendingFree;
    this.pendingFree = [];
    for (const buf of released) {
      if (this.destroyed || this.pooledBytes + buf.size > MAX_POOL_BYTES) {
        buf.destroy();
        continue;
      }
      const list = this.free.get(buf.size);
      if (list) list.push(buf);
      else this.free.set(buf.size, [buf]);
      this.pooledBytes += buf.size;
    }
  }

  private gpuBuffer(t: DevBuffer): GPUBuffer {
    const h = t.handle as Partial<GpuHandle> | null;
    if (!h || typeof h !== "object" || !h.buf) {
      throw new Error("WebGpuDevice: buffer was not created by this device");
    }
    return h.buf;
  }

  private buffer(shape: readonly number[], size: number): DevBuffer {
    return { shape: shape.slice(), size, handle: { buf: this.alloc(size * 4) } as GpuHandle };
  }

  /* --------------------------------------------------------------- plumbing */

  private encoder(): GPUCommandEncoder {
    if (this.destroyed) throw new Error("WebGpuDevice: device has been destroyed");
    if (!this.enc) this.enc = this.gpu.createCommandEncoder({ label: "nf-batch" });
    return this.enc;
  }

  /** Submits everything recorded since the last submit. */
  flush(): void {
    if (this.destroyed) return;
    if (this.enc) {
      const enc = this.enc;
      this.enc = null;
      this.gpu.queue.submit([enc.finish()]);
    }
    for (const buf of this.pendingParams) this.release(buf);
    this.pendingParams = [];
    this.retire();
    this.throwIfFailed("submit");
  }

  /** Waits for queued work, then surfaces the first device error, if any. */
  async sync(): Promise<void> {
    this.flush();
    await this.gpu.queue.onSubmittedWorkDone();
    this.throwIfFailed("sync");
    if (this.lostReason) throw new Error(`WebGPU device lost — ${this.lostReason}`);
  }

  private throwIfFailed(where: string): void {
    if (this.lastError) {
      const message = this.lastError;
      this.lastError = null;
      throw new Error(`WebGPU error during ${where}: ${message}`);
    }
  }

  private params(values: Uint32Array): GPUBuffer {
    const bytes = Math.max(4, (values.byteLength + 3) & ~3);
    const buf = this.gpu.createBuffer({ size: bytes, usage: paramsUsage(), label: "nf-params" });
    this.gpu.queue.writeBuffer(buf, 0, values as unknown as BufferSource);
    this.pendingParams.push(buf);
    return buf;
  }

  private pipeline(name: KernelName): GPUComputePipeline {
    const cached = this.pipelines.get(name);
    if (cached) return cached;
    const spec = KERNELS[name];
    const module = this.gpu.createShaderModule({ code: spec.code, label: `nf-${name}` });
    const pipeline = this.gpu.createComputePipeline({
      layout: "auto",
      label: `nf-${name}`,
      compute: { module, entryPoint: spec.entry },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  /** Records one compute pass for a planned step, reading `storage` in order. */
  private dispatch(step: Step, storage: DevBuffer[]): void {
    const name = step.kernel;
    if (this.destroyed) throw new Error("WebGpuDevice: device has been destroyed");
    const pipeline = this.pipeline(name);
    const pbuf = this.params(step.params.toU32Array());
    const entries: GPUBindGroupEntry[] = storage.map((t, i) => ({
      binding: i,
      resource: { buffer: this.gpuBuffer(t) },
    }));
    entries.push({ binding: storage.length, resource: { buffer: pbuf } });
    let bindGroup: GPUBindGroup;
    try {
      bindGroup = this.gpu.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries, label: `nf-${name}` });
    } catch (e) {
      throw new Error(`WebGPU: bind group for '${name}' failed (${e instanceof Error ? e.message : String(e)})`);
    }
    const pass = this.encoder().beginComputePass({ label: `nf-${name}` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(step.grid[0], step.grid[1], step.grid[2]);
    pass.end();
  }

  /** Releases an intermediate buffer once its readers are recorded (see the header). */
  private scratch(t: DevBuffer): void {
    if (!t.view) this.release(this.gpuBuffer(t));
  }

  /* -------------------------------------------------------------------- ops */

  async fromHost(data: Float32Array, shape: number[]): Promise<DevBuffer> {
    const t = this.buffer(shape, data.length);
    if (data.length > 0) this.gpu.queue.writeBuffer(this.gpuBuffer(t), 0, data as unknown as BufferSource);
    return t;
  }

  async toHost(t: DevBuffer): Promise<Float32Array> {
    const [data] = await this.toHostMany([t]);
    return data!;
  }

  /**
   * Reads several tensors back in one round trip: one submit, then one
   * `mapAsync` per buffer, all awaited together. Gradient collection is the hot
   * caller — a batch would otherwise pay a submit and a stall per parameter.
   */
  async toHostMany(targets: DevBuffer[]): Promise<Float32Array[]> {
    if (targets.length === 0) return [];
    this.flush();
    const pending: { staging: GPUBuffer | null; bytes: number }[] = [];
    const enc = this.gpu.createCommandEncoder({ label: "nf-readback" });
    for (const target of targets) {
      const bytes = target.size * 4;
      if (bytes === 0) {
        pending.push({ staging: null, bytes: 0 });
        continue;
      }
      const staging = this.stagingBuffer(bytes);
      enc.copyBufferToBuffer(this.gpuBuffer(target), 0, staging, 0, bytes);
      pending.push({ staging, bytes });
    }
    const copies = pending.filter((p) => p.staging !== null);
    if (copies.length === 0) return targets.map(() => new Float32Array(0));
    this.gpu.queue.submit([enc.finish()]);
    await Promise.all(copies.map((p) => p.staging!.mapAsync(GPUMapMode.READ, 0, p.bytes)));
    const out = pending.map(({ staging, bytes }) => {
      if (!staging) return new Float32Array(0);
      const view = new Float32Array(staging.getMappedRange(0, bytes));
      const copy = new Float32Array(view); // copy before unmapping
      staging.unmap();
      this.stagingFree.push(staging);
      return copy;
    });
    this.throwIfFailed("readback");
    return out;
  }

  private stagingBuffer(bytes: number): GPUBuffer {
    const size = Math.max(4, (bytes + 3) & ~3);
    const reused = this.stagingFree.pop();
    if (reused && reused.size >= size) return reused;
    if (reused) reused.destroy();
    return this.gpu.createBuffer({ size, usage: stagingUsage(), label: "nf-staging" });
  }

  async zeros(shape: number[]): Promise<DevBuffer> {
    const size = numel(shape);
    const t = this.buffer(shape, size);
    if (size > 0) this.dispatch(fillZeroStep(size), [t]);
    return t;
  }

  dispose(t: DevBuffer): void {
    // Views share another buffer's storage, and the accelerator also parks
    // plain geometry objects in DevBuffer-shaped slots: neither owns memory.
    if (t.view || t.size === 0) return;
    const h = t.handle as Partial<GpuHandle> | null;
    if (!h || typeof h !== "object" || !h.buf) return;
    this.release(h.buf);
  }

  async reshapeView(t: DevBuffer, shape: number[]): Promise<DevBuffer> {
    const size = numel(shape);
    if (size !== t.size) {
      throw new Error(`WebGpuDevice: reshape ${t.shape.join("×")} → ${shape.join("×")} changes the element count`);
    }
    return { shape: shape.slice(), size, handle: t.handle, view: true };
  }

  async matmul(a: DevBuffer, b: DevBuffer, transposeA: boolean, transposeB: boolean): Promise<DevBuffer> {
    const [M, K] = matmulDims(a.shape, transposeA);
    const [K2, N] = matmulDims(b.shape, transposeB);
    if (K !== K2) throw new Error(`WebGpuDevice.matmul: inner dimensions disagree (${K} vs ${K2})`);
    const out = this.buffer([M, N], M * N);
    if (M === 0 || N === 0) return out;
    if (K === 0) {
      this.dispatch(fillZeroStep(M * N), [out]);
      return out;
    }
    this.dispatch(matmulStep(M, N, K, transposeA, transposeB), [a, b, out]);
    return out;
  }

  async addBias(m: DevBuffer, bias: DevBuffer): Promise<DevBuffer> {
    return this.binary("add", m, bias);
  }

  async binary(op: BinaryOp, a: DevBuffer, b: DevBuffer): Promise<DevBuffer> {
    const plan = broadcastPlan(a.shape, b.shape);
    const size = plan.outShape.reduce((n, d) => n * d, 1);
    const out = this.buffer(plan.outShape, size);
    if (size === 0) return out;
    this.dispatch(binaryStep(size, binaryCode(op), plan.out4, plan.a4, plan.b4), [a, b, out]);
    return out;
  }

  async unary(op: UnaryOp, x: DevBuffer, scalar: number, scalar2: number): Promise<DevBuffer> {
    const out = this.buffer(x.shape, x.size);
    if (x.size === 0) return out;
    this.dispatch(unaryStep(x.size, unaryCode(op), scalar, scalar2), [x, out]);
    return out;
  }

  async activationFwd(name: string, x: DevBuffer, alpha: number): Promise<DevBuffer> {
    if (name === "softmax") return this.softmaxLast(x);
    const out = this.buffer(x.shape, x.size);
    if (x.size === 0) return out;
    this.dispatch(unaryStep(x.size, activationCode(name), alpha, 0), [x, out]);
    return out;
  }

  async activationBwd(name: string, x: DevBuffer, y: DevBuffer, dy: DevBuffer, alpha: number): Promise<DevBuffer> {
    if (name === "softmax") return this.softmaxBackward(y, dy);
    const out = this.buffer(dy.shape, dy.size);
    if (dy.size === 0) return out;
    this.dispatch(unaryBwdStep(dy.size, backwardCode(name), alpha), [x, y, dy, out]);
    return out;
  }

  async softmaxLast(x: DevBuffer): Promise<DevBuffer> {
    const out = this.buffer(x.shape, x.size);
    const inner = lastDim(x.shape);
    const rows = inner > 0 ? x.size / inner : 0;
    if (rows === 0) return out;
    this.dispatch(softmaxStep(inner, rows), [x, out]);
    return out;
  }

  async softmaxBackward(p: DevBuffer, dy: DevBuffer): Promise<DevBuffer> {
    const out = this.buffer(p.shape, p.size);
    const inner = lastDim(p.shape);
    const rows = inner > 0 ? p.size / inner : 0;
    if (rows === 0) return out;
    this.dispatch(softmaxBwdStep(inner, rows), [p, dy, out]);
    return out;
  }

  async reduceOuter(mode: ReduceMode, m: DevBuffer, group = 1): Promise<DevBuffer> {
    const inner = lastDim(m.shape);
    const g = Math.max(1, group);
    const outer = inner > 0 ? m.size / inner / g : 0;
    const out = this.buffer([g, inner], g * inner);
    if (m.size === 0 || inner === 0) return out;
    this.dispatch(reduceOuterStep(outer, inner, g, mode), [m, out]);
    return out;
  }

  async broadcastPlanes(
    t: DevBuffer,
    geom: { B: number; H: number; W: number; C: number },
    scale: number,
  ): Promise<DevBuffer> {
    const size = geom.B * geom.H * geom.W * geom.C;
    const out = this.buffer([geom.B, geom.H, geom.W, geom.C], size);
    if (size === 0) return out;
    this.dispatch(broadcastPlanesStep(size, geom.H * geom.W * geom.C, geom.C, scale), [t, out]);
    return out;
  }

  async conv2d(x: DevBuffer, k: DevBuffer, bias: DevBuffer | null, g: ConvGeom): Promise<DevBuffer> {
    const rows = g.B * g.outH * g.outW;
    const K = g.kH * g.kW * g.Cin;
    const col = this.buffer([rows, K], rows * K);
    if (col.size > 0) this.dispatch(im2colStep(g, col.size), [x, col]);
    const kernel = await this.reshapeView(k, [K, g.F]);
    const flat = await this.matmul(col, kernel, false, false);
    this.scratch(col);
    const shaped = await this.reshapeView(flat, [g.B, g.outH, g.outW, g.F]);
    if (bias) return this.binary("add", shaped, bias);
    return shaped;
  }

  async conv2dInputGrad(dy: DevBuffer, k: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    const rows = g.B * g.outH * g.outW;
    const dyFlat = await this.reshapeView(dy, [rows, g.F]);
    const dcol = await this.matmul(dyFlat, k, false, true); // [rows, K]
    const dx = this.buffer([g.B, g.H, g.W, g.Cin], g.B * g.H * g.W * g.Cin);
    if (dx.size > 0) this.dispatch(col2imStep(g, dx.size), [dcol, dx]);
    this.scratch(dcol);
    return dx;
  }

  async conv2dKernelGrad(x: DevBuffer, dy: DevBuffer, g: ConvGeom): Promise<DevBuffer> {
    const rows = g.B * g.outH * g.outW;
    const K = g.kH * g.kW * g.Cin;
    const col = this.buffer([rows, K], rows * K);
    if (col.size > 0) this.dispatch(im2colStep(g, col.size), [x, col]);
    const dyFlat = await this.reshapeView(dy, [rows, g.F]);
    const dk = await this.matmul(col, dyFlat, true, false); // [K, F]
    this.scratch(col);
    return this.reshapeView(dk, [g.kH, g.kW, g.Cin, g.F]);
  }

  async pool2d(mode: PoolMode, x: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    const size = g.B * g.outH * g.outW * g.C;
    const out = this.buffer([g.B, g.outH, g.outW, g.C], size);
    if (size === 0) return out;
    this.dispatch(poolFwdStep(mode, g, size), [x, out]);
    return out;
  }

  async pool2dBackward(mode: PoolMode, x: DevBuffer, dy: DevBuffer, g: PoolGeom): Promise<DevBuffer> {
    const size = g.B * g.H * g.W * g.C;
    const dx = this.buffer([g.B, g.H, g.W, g.C], size);
    if (size === 0) return dx;
    this.dispatch(poolBwdStep(mode, g, size), [x, dy, dx]);
    return dx;
  }

  async gatherRows(w: DevBuffer, ids: DevBuffer, dim: number): Promise<DevBuffer> {
    const vocab = w.shape[0] ?? 0;
    const rows = dim > 0 ? ids.size : 0;
    const out = this.buffer([rows, dim], rows * dim);
    if (out.size === 0) return out;
    this.dispatch(gatherStep(out.size, dim, vocab), [w, ids, out]);
    return out;
  }

  async scatterRows(grad: DevBuffer, ids: DevBuffer, dim: number, vocab: number): Promise<DevBuffer> {
    const out = this.buffer([vocab, dim], vocab * dim);
    if (out.size === 0) return out;
    this.dispatch(scatterStep(out.size, dim, vocab, dim > 0 ? ids.size : 0), [grad, ids, out]);
    return out;
  }

  /** Live and pooled buffer counts, for diagnostics. */
  bufferStats(): { leased: number; pooled: number; pooledBytes: number } {
    let pooled = 0;
    for (const list of this.free.values()) pooled += list.length;
    return { leased: this.leased, pooled, pooledBytes: this.pooledBytes };
  }
}

function lastDim(shape: readonly number[]): number {
  return shape.length > 0 ? shape[shape.length - 1]! : 1;
}
