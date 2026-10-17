import { Tensor } from "../tensor";
import type { SequentialModel } from "../model";
import type { CompiledLayer } from "../layers";
import { residualConvPlan, convOut } from "../layers";
import type { LayerType } from "../types";
import { convGeomFor } from "./ops";
import type { ConvGeom, DevBuffer, GpuOps, PoolGeom } from "./ops";

/** Layer types the accelerator runs as device kernels. Everything else falls back to the CPU engine. */
export const DEVICE_LAYER_TYPES: LayerType[] = [
  "dense",
  "conv2d",
  "residual_dense",
  "residual_conv",
  "flatten",
  "reshape",
  "identity",
  "relu",
  "leakyrelu",
  "gelu",
  "silu",
  "sigmoid",
  "tanh",
  "softmax",
  "maxpool2d",
  "avgpool2d",
  "gap2d",
  "embedding",
];

export type PlacementMode = "device" | "cpu";

export interface LayerPlacement {
  id: string;
  name: string;
  type: LayerType;
  mode: PlacementMode;
  note: string;
}

export interface AcceleratorStats {
  batches: number;
  deviceLayers: number;
  cpuLayers: number;
  uploadedBytes: number;
  downloadedBytes: number;
  forwardMs: number;
  backwardMs: number;
  totalDeviceMs: number;
}

interface LayerState {
  layer: CompiledLayer;
  mode: PlacementMode;
  buffers: Map<string, DevBuffer>;
  /** Device buffers borrowed from another layer or from the caller: not disposed here. */
  borrowed: Set<string>;
}

function act(name: unknown, fallback = "none"): string {
  const a = String(name ?? fallback);
  return a === "" || a === "undefined" ? fallback : a;
}

function isRealActivation(name: string): boolean {
  return name !== "none" && name !== "linear";
}

function num(v: unknown, d: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function bool(v: unknown, d: boolean): boolean {
  return typeof v === "boolean" ? v : d;
}

/** A layer's real (batched) output shape: the compiled shape plus the batch axis. */
function outShapeOf(input: DevBuffer, layer: CompiledLayer): number[] {
  return [input.shape[0]!, ...layer.outputShape];
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/**
 * Runs a `SequentialModel`'s forward and backward passes on a device that
 * implements `GpuOps`.
 *
 * What happens per training batch:
 *   1. weights are re-uploaded (the CPU tensors stay authoritative, so the
 *      optimiser, save/load and the standalone export are all unchanged),
 *   2. device layers run as kernels and keep their activations resident;
 *      layers the device does not implement round-trip through the CPU engine,
 *   3. the output is downloaded so the real loss can be computed on the CPU,
 *   4. the loss gradient is uploaded and the backward pass runs in reverse,
 *   5. parameter gradients are downloaded into the layer's own `grads` tensors.
 */
export class Accelerator {
  readonly model: SequentialModel;
  readonly ops: GpuOps;
  readonly placements: LayerPlacement[];
  readonly stats: AcceleratorStats = {
    batches: 0,
    deviceLayers: 0,
    cpuLayers: 0,
    uploadedBytes: 0,
    downloadedBytes: 0,
    forwardMs: 0,
    backwardMs: 0,
    totalDeviceMs: 0,
  };
  active = false;
  lastError: string | null = null;
  /** True once parameters have been uploaded at least once. */
  private uploaded = false;

  private states = new Map<string, LayerState>();
  private scratch: DevBuffer[] = [];

  constructor(model: SequentialModel, ops: GpuOps) {
    this.model = model;
    this.ops = ops;
    this.placements = model.layers.map((layer) => {
      const supported = DEVICE_LAYER_TYPES.includes(layer.def.type);
      return {
        id: layer.def.id,
        name: layer.def.name,
        type: layer.def.type,
        mode: supported ? ("device" as const) : ("cpu" as const),
        note: supported
          ? "device kernels"
          : layer.def.type === "dropout"
            ? "Dropout draws from the engine's own RNG, so it stays on the CPU"
            : "runs on the CPU engine and is synchronised around the device segments",
      };
    });
    for (const p of this.placements) {
      const layer = model.layers.find((l) => l.def.id === p.id)!;
      this.states.set(p.id, { layer, mode: p.mode, buffers: new Map(), borrowed: new Set() });
    }
    this.stats.deviceLayers = this.placements.filter((p) => p.mode === "device").length;
    this.stats.cpuLayers = this.placements.filter((p) => p.mode === "cpu").length;
  }

  get usable(): boolean {
    return this.model.issues.length === 0 && !this.lastError;
  }

  /* --------------------------------------------------------------- plumbing */

  /** Uploads every parameter of every device layer. Called once per batch. */
  async syncParams(): Promise<void> {
    for (const state of this.states.values()) {
      if (state.mode !== "device") continue;
      for (const [name, tensor] of Object.entries(state.layer.params)) {
        const key = `param.${name}`;
        const previous = state.buffers.get(key);
        if (previous) this.ops.dispose(previous);
        const buf = await this.ops.fromHost(tensor.data, tensor.shape as number[]);
        state.buffers.set(key, buf);
        this.stats.uploadedBytes += tensor.size * 4;
      }
    }
    this.uploaded = true;
  }

  private async upload(t: Tensor): Promise<DevBuffer> {
    this.stats.uploadedBytes += t.size * 4;
    const buf = await this.ops.fromHost(t.data, t.shape as number[]);
    this.scratch.push(buf);
    return buf;
  }

  private async download(buf: DevBuffer): Promise<Float32Array> {
    this.stats.downloadedBytes += buf.size * 4;
    return this.ops.toHost(buf);
  }

  private keep(buf: DevBuffer): DevBuffer {
    this.scratch.push(buf);
    return buf;
  }

  /** Registers a buffer this layer did not allocate (an input or a weight). */
  private borrow(state: LayerState, key: string, buf: DevBuffer): DevBuffer {
    state.buffers.set(key, buf);
    state.borrowed.add(key);
    return buf;
  }

  private releaseScratch(): void {
    for (const b of this.scratch) this.ops.dispose(b);
    this.scratch = [];
  }

  private param(state: LayerState, name: string): DevBuffer {
    const buf = state.buffers.get(`param.${name}`);
    if (!buf) throw new Error(`Accelerator: parameter '${name}' of layer '${state.layer.def.name}' is not on the device`);
    return buf;
  }

  private cached(state: LayerState, key: string): DevBuffer {
    const buf = state.buffers.get(key);
    if (!buf) throw new Error(`Accelerator: layer '${state.layer.def.name}' has no cached '${key}' buffer`);
    return buf;
  }

  private setGrad(state: LayerState, name: string, buf: DevBuffer): void {
    const key = `grad.${name}`;
    const previous = state.buffers.get(key);
    if (previous) this.ops.dispose(previous);
    state.buffers.set(key, this.keep(buf));
  }

  /* ---------------------------------------------------------------- forward */

  async runForward(x: Tensor, training: boolean): Promise<Tensor> {
    // A new forward invalidates the previous forward cache. Inference and
    // validation have no backward pass to release their temporary buffers.
    this.releaseScratch();
    this.active = true;
    // The first call after construction (or after a model rebuild) uploads the
    // parameters; training loops call syncParams() per batch to refresh them.
    if (!this.uploaded) await this.syncParams();
    const t0 = now();
    let current = await this.upload(x);
    for (let i = 0; i < this.model.layers.length; i++) {
      const state = this.states.get(this.placements[i]!.id)!;
      current = await this.forwardLayer(state, current, training);
    }
    const data = await this.download(current);
    this.stats.forwardMs = now() - t0;
    this.stats.totalDeviceMs += this.stats.forwardMs;
    return new Tensor((current.shape as number[]).slice(), data);
  }

  private async forwardLayer(state: LayerState, input: DevBuffer, training: boolean): Promise<DevBuffer> {
    if (state.mode === "cpu") return this.forwardOnCpu(state, input, training);
    const layer = state.layer;
    const cfg = layer.def.config;
    switch (layer.def.type) {
      case "dense": {
        let y = this.keep(await this.ops.matmul(input, this.param(state, "W"), false, false));
        if (bool(cfg.useBias, true)) y = this.keep(await this.ops.addBias(y, this.param(state, "b")));
        this.borrow(state, "pre", y);
        const a = act(cfg.activation);
        if (isRealActivation(a)) y = this.keep(await this.ops.activationFwd(a, y, 0.01));
        this.borrow(state, "in", input);
        this.borrow(state, "out", y);
        return y;
      }
      case "conv2d": {
        const [H, W, Cin] = layer.inputShape as [number, number, number];
        const g = convGeomFor(input.shape[0]!, H, W, Cin, layer.outputShape[2]!, cfg);
        let y = this.keep(await this.ops.conv2d(input, this.param(state, "K"), bool(cfg.useBias, true) ? this.param(state, "b") : null, g));
        this.borrow(state, "pre", y);
        const a = act(cfg.activation);
        if (isRealActivation(a)) y = this.keep(await this.ops.activationFwd(a, y, 0.01));
        this.borrow(state, "in", input);
        this.borrow(state, "out", y);
        this.borrow(state, "geom", { shape: [], size: 0, handle: g });
        return y;
      }
      case "residual_dense": {
        const useBias = bool(cfg.useBias, true);
        const a1 = act(cfg.activation, "relu");
        const a2 = act(cfg.finalActivation);
        const scale = num(cfg.scale, 1);
        let t1 = this.keep(await this.ops.matmul(input, this.param(state, "W1"), false, false));
        if (useBias) t1 = this.keep(await this.ops.addBias(t1, this.param(state, "b1")));
        const hiddenAct = isRealActivation(a1) ? this.keep(await this.ops.activationFwd(a1, t1, 0.01)) : t1;
        let main = this.keep(await this.ops.matmul(hiddenAct, this.param(state, "W2"), false, false));
        if (useBias) main = this.keep(await this.ops.addBias(main, this.param(state, "b2")));
        const scaled = scale === 1 ? main : this.keep(await this.ops.unary("scale", main, scale, 0));
        const hasProj = !!state.buffers.get("param.Wp");
        const shortcut = hasProj ? this.keep(await this.ops.matmul(input, this.param(state, "Wp"), false, false)) : input;
        let y = this.keep(await this.ops.binary("add", shortcut, scaled));
        this.borrow(state, "z", y);
        if (isRealActivation(a2)) y = this.keep(await this.ops.activationFwd(a2, y, 0.01));
        this.borrow(state, "in", input);
        this.borrow(state, "hiddenPre", t1);
        this.borrow(state, "hiddenAct", hiddenAct);
        this.borrow(state, "out", y);
        return y;
      }
      case "residual_conv": {
        const plan = residualConvPlan(cfg, layer.inputShape);
        const B = input.shape[0]!;
        const useBias = bool(cfg.useBias, true);
        const a1 = act(cfg.activation, "relu");
        const a2 = act(cfg.finalActivation);
        const scale = num(cfg.scale, 1);
        const g1 = convGeomFor(B, plan.H, plan.W, plan.Cin, plan.filters, cfg, plan.stride);
        const conv1 = this.keep(await this.ops.conv2d(input, this.param(state, "K1"), useBias ? this.param(state, "b1") : null, g1));
        const hiddenAct = isRealActivation(a1) ? this.keep(await this.ops.activationFwd(a1, conv1, 0.01)) : conv1;
        const g2 = convGeomFor(B, g1.outH, g1.outW, plan.filters, plan.filters, cfg, 1);
        let main = this.keep(await this.ops.conv2d(hiddenAct, this.param(state, "K2"), useBias ? this.param(state, "b2") : null, g2));
        let shortcut: DevBuffer;
        if (plan.identityShortcut) {
          shortcut = input;
        } else {
          const projOut = convOut(plan.H, plan.W, 1, 1, plan.stride, "valid");
          shortcut = this.keep(
            await this.ops.conv2d(input, this.param(state, "Kp"), null, {
              B,
              H: plan.H,
              W: plan.W,
              Cin: plan.Cin,
              outH: projOut.outH,
              outW: projOut.outW,
              F: plan.filters,
              kH: 1,
              kW: 1,
              stride: plan.stride,
              padTop: 0,
              padLeft: 0,
            }),
          );
        }
        const scaled = scale === 1 ? main : this.keep(await this.ops.unary("scale", main, scale, 0));
        let y = this.keep(await this.ops.binary("add", shortcut, scaled));
        this.borrow(state, "z", y);
        if (isRealActivation(a2)) y = this.keep(await this.ops.activationFwd(a2, y, 0.01));
        this.borrow(state, "in", input);
        this.borrow(state, "conv1", conv1);
        this.borrow(state, "hiddenAct", hiddenAct);
        this.borrow(state, "out", y);
        this.borrow(state, "geom1", { shape: [], size: 0, handle: g1 });
        this.borrow(state, "geom2", { shape: [], size: 0, handle: g2 });
        return y;
      }
      case "flatten":
      case "reshape":
        this.borrow(state, "in", input);
        return this.keep(await this.ops.reshapeView(input, outShapeOf(input, layer)));
      case "identity":
        return input;
      case "maxpool2d":
      case "avgpool2d": {
        const [H, W, C] = layer.inputShape as [number, number, number];
        const pool = num(cfg.poolSize, 2);
        const stride = num(cfg.stride, pool);
        const pg: PoolGeom = { B: input.shape[0]!, H, W, C, outH: layer.outputShape[0]!, outW: layer.outputShape[1]!, pool, stride };
        const out = this.keep(await this.ops.pool2d(layer.def.type === "maxpool2d" ? "max" : "avg", input, pg));
        this.borrow(state, "in", input);
        this.borrow(state, "poolGeom", { shape: [], size: 0, handle: pg });
        return out;
      }
      case "gap2d": {
        const [H, W, C] = layer.inputShape as [number, number, number];
        const B = input.shape[0]!;
        this.borrow(state, "gapGeom", { shape: [], size: 0, handle: { B, H, W, C } });
        const pooled = this.keep(await this.ops.reduceOuter("mean", input, B));
        this.borrow(state, "in", input);
        return this.keep(await this.ops.reshapeView(pooled, [B, C]));
      }
      case "embedding": {
        const dim = layer.outputShape[1]!;
        const gathered = this.keep(await this.ops.gatherRows(this.param(state, "W"), input, dim));
        this.borrow(state, "ids", input);
        return this.keep(await this.ops.reshapeView(gathered, outShapeOf(input, layer)));
      }
      case "relu":
      case "leakyrelu":
      case "gelu":
      case "silu":
      case "sigmoid":
      case "tanh": {
        const y = this.keep(await this.ops.activationFwd(layer.def.type, input, num(cfg.alpha, 0.01)));
        this.borrow(state, "in", input);
        this.borrow(state, "out", y);
        return y;
      }
      case "softmax": {
        const y = this.keep(await this.ops.softmaxLast(input));
        this.borrow(state, "out", y);
        return y;
      }
      default:
        return this.forwardOnCpu(state, input, training);
    }
  }

  private async forwardOnCpu(state: LayerState, input: DevBuffer, training: boolean): Promise<DevBuffer> {
    const host = new Tensor((input.shape as number[]).slice(), await this.download(input));
    const out = state.layer.forward(host, training);
    return this.upload(out);
  }

  /* --------------------------------------------------------------- backward */

  async runBackward(dOut: Tensor): Promise<void> {
    const t0 = now();
    if (!this.uploaded) await this.syncParams();
    let current = await this.upload(dOut);
    for (let i = this.model.layers.length - 1; i >= 0; i--) {
      const state = this.states.get(this.placements[i]!.id)!;
      current = await this.backwardLayer(state, current);
    }
    await this.collectGrads();
    this.releaseScratch();
    this.stats.backwardMs = now() - t0;
    this.stats.totalDeviceMs += this.stats.backwardMs;
    this.stats.batches += 1;
  }

  private async backwardLayer(state: LayerState, dy: DevBuffer): Promise<DevBuffer> {
    if (state.mode === "cpu") return this.backwardOnCpu(state, dy);
    const layer = state.layer;
    const cfg = layer.def.config;
    switch (layer.def.type) {
      case "dense": {
        let g = dy;
        const a = act(cfg.activation);
        if (isRealActivation(a)) {
          g = this.keep(await this.ops.activationBwd(a, this.cached(state, "pre"), this.cached(state, "out"), g, 0.01));
        }
        this.setGrad(state, "W", this.keep(await this.ops.matmul(this.cached(state, "in"), g, true, false)));
        if (bool(cfg.useBias, true)) this.setGrad(state, "b", this.keep(await this.ops.reduceOuter("sum", g)));
        return this.keep(await this.ops.matmul(g, this.param(state, "W"), false, true));
      }
      case "conv2d": {
        const g = this.cached(state, "geom").handle as ConvGeom;
        let dyAct = dy;
        const a = act(cfg.activation);
        if (isRealActivation(a)) {
          dyAct = this.keep(await this.ops.activationBwd(a, this.cached(state, "pre"), this.cached(state, "out"), dyAct, 0.01));
        }
        this.setGrad(state, "K", this.keep(await this.ops.conv2dKernelGrad(this.cached(state, "in"), dyAct, g)));
        if (bool(cfg.useBias, true)) this.setGrad(state, "b", this.keep(await this.ops.reduceOuter("sum", dyAct)));
        return this.keep(await this.ops.conv2dInputGrad(dyAct, this.param(state, "K"), g));
      }
      case "residual_dense": {
        const useBias = bool(cfg.useBias, true);
        const a1 = act(cfg.activation, "relu");
        const a2 = act(cfg.finalActivation);
        const scale = num(cfg.scale, 1);
        const input = this.cached(state, "in");
        const gZ = isRealActivation(a2)
          ? this.keep(await this.ops.activationBwd(a2, this.cached(state, "z"), this.cached(state, "out"), dy, 0.01))
          : dy;
        const gMain = scale === 1 ? gZ : this.keep(await this.ops.unary("scale", gZ, scale, 0));
        this.setGrad(state, "W2", this.keep(await this.ops.matmul(this.cached(state, "hiddenAct"), gMain, true, false)));
        if (useBias) this.setGrad(state, "b2", this.keep(await this.ops.reduceOuter("sum", gMain)));
        const dAct = this.keep(await this.ops.matmul(gMain, this.param(state, "W2"), false, true));
        const dPre = isRealActivation(a1)
          ? this.keep(await this.ops.activationBwd(a1, this.cached(state, "hiddenPre"), this.cached(state, "hiddenAct"), dAct, 0.01))
          : dAct;
        this.setGrad(state, "W1", this.keep(await this.ops.matmul(input, dPre, true, false)));
        if (useBias) this.setGrad(state, "b1", this.keep(await this.ops.reduceOuter("sum", dPre)));
        const dMain = this.keep(await this.ops.matmul(dPre, this.param(state, "W1"), false, true));
        if (state.buffers.get("param.Wp")) {
          this.setGrad(state, "Wp", this.keep(await this.ops.matmul(input, gZ, true, false)));
          const dSkip = this.keep(await this.ops.matmul(gZ, this.param(state, "Wp"), false, true));
          return this.keep(await this.ops.binary("add", dMain, dSkip));
        }
        return this.keep(await this.ops.binary("add", dMain, gZ));
      }
      case "residual_conv": {
        const plan = residualConvPlan(cfg, layer.inputShape);
        const useBias = bool(cfg.useBias, true);
        const a1 = act(cfg.activation, "relu");
        const a2 = act(cfg.finalActivation);
        const scale = num(cfg.scale, 1);
        const input = this.cached(state, "in");
        const g1 = this.cached(state, "geom1").handle as ConvGeom;
        const g2 = this.cached(state, "geom2").handle as ConvGeom;
        const gZ = isRealActivation(a2)
          ? this.keep(await this.ops.activationBwd(a2, this.cached(state, "z"), this.cached(state, "out"), dy, 0.01))
          : dy;
        const gMain = scale === 1 ? gZ : this.keep(await this.ops.unary("scale", gZ, scale, 0));
        this.setGrad(state, "K2", this.keep(await this.ops.conv2dKernelGrad(this.cached(state, "hiddenAct"), gMain, g2)));
        if (useBias) this.setGrad(state, "b2", this.keep(await this.ops.reduceOuter("sum", gMain)));
        const d2x = this.keep(await this.ops.conv2dInputGrad(gMain, this.param(state, "K2"), g2));
        const dAct = isRealActivation(a1)
          ? this.keep(await this.ops.activationBwd(a1, this.cached(state, "conv1"), this.cached(state, "hiddenAct"), d2x, 0.01))
          : d2x;
        this.setGrad(state, "K1", this.keep(await this.ops.conv2dKernelGrad(input, dAct, g1)));
        if (useBias) this.setGrad(state, "b1", this.keep(await this.ops.reduceOuter("sum", dAct)));
        let dx = this.keep(await this.ops.conv2dInputGrad(dAct, this.param(state, "K1"), g1));
        if (plan.identityShortcut) {
          dx = this.keep(await this.ops.binary("add", dx, gZ));
        } else {
          const projOut = convOut(plan.H, plan.W, 1, 1, plan.stride, "valid");
          const pg: ConvGeom = {
            B: input.shape[0]!,
            H: plan.H,
            W: plan.W,
            Cin: plan.Cin,
            outH: projOut.outH,
            outW: projOut.outW,
            F: plan.filters,
            kH: 1,
            kW: 1,
            stride: plan.stride,
            padTop: 0,
            padLeft: 0,
          };
          this.setGrad(state, "Kp", this.keep(await this.ops.conv2dKernelGrad(input, gZ, pg)));
          const dSkip = this.keep(await this.ops.conv2dInputGrad(gZ, this.param(state, "Kp"), pg));
          dx = this.keep(await this.ops.binary("add", dx, dSkip));
        }
        return dx;
      }
      case "flatten":
      case "reshape":
        return this.keep(await this.ops.reshapeView(dy, this.cached(state, "in").shape));
      case "identity":
        return dy;
      case "maxpool2d":
      case "avgpool2d": {
        const pg = this.cached(state, "poolGeom").handle as PoolGeom;
        const mode = layer.def.type === "maxpool2d" ? "max" : "avg";
        return this.keep(await this.ops.pool2dBackward(mode, this.cached(state, "in"), dy, pg));
      }
      case "gap2d": {
        const geom = this.cached(state, "gapGeom").handle as { B: number; H: number; W: number; C: number };
        return this.keep(await this.ops.broadcastPlanes(dy, geom, 1 / (geom.H * geom.W)));
      }
      case "embedding": {
        const dim = layer.outputShape[1]!;
        const vocab = num(cfg.vocabSize, 32);
        this.setGrad(state, "W", this.keep(await this.ops.scatterRows(dy, this.cached(state, "ids"), dim, vocab)));
        // The engine treats integer token ids as constants: no gradient flows to the index input.
        return this.keep(await this.ops.zeros((this.cached(state, "ids").shape as number[]).slice()));
      }
      case "relu":
      case "leakyrelu":
      case "gelu":
      case "silu":
      case "sigmoid":
      case "tanh":
        return this.keep(
          await this.ops.activationBwd(layer.def.type, this.cached(state, "in"), this.cached(state, "out"), dy, num(cfg.alpha, 0.01)),
        );
      case "softmax":
        return this.keep(await this.ops.softmaxBackward(this.cached(state, "out"), dy));
      default:
        return this.backwardOnCpu(state, dy);
    }
  }

  private async backwardOnCpu(state: LayerState, dy: DevBuffer): Promise<DevBuffer> {
    const host = new Tensor((dy.shape as number[]).slice(), await this.download(dy));
    return this.upload(state.layer.backward(host));
  }

  /**
   * Copies every device-layer gradient into the layer's own `grads` tensor, so
   * the untouched CPU optimisers can apply the step.
   */
  async collectGrads(): Promise<void> {
    // Gather every device gradient first, then read them back together: a
    // device that supports batched readback (WebGPU) pays one round trip per
    // batch instead of one per parameter.
    const targets: { state: LayerState; name: string; buf: DevBuffer }[] = [];
    for (const placement of this.placements) {
      const state = this.states.get(placement.id)!;
      if (state.mode !== "device") continue;
      for (const name of Object.keys(state.layer.grads)) {
        const buf = state.buffers.get(`grad.${name}`);
        if (buf) targets.push({ state, name, buf });
      }
    }
    if (targets.length === 0) return;
    let data: Float32Array[];
    if (this.ops.toHostMany) {
      for (const t of targets) this.stats.downloadedBytes += t.buf.size * 4;
      data = await this.ops.toHostMany(targets.map((t) => t.buf));
    } else {
      data = [];
      for (const t of targets) data.push(await this.download(t.buf));
    }
    targets.forEach((t, i) => {
      const values = data[i];
      if (values) t.state.layer.grads[t.name]!.data.set(values);
    });
  }

  destroy(): void {
    for (const state of this.states.values()) {
      for (const [key, buf] of state.buffers) {
        if (state.borrowed.has(key) || buf.size === 0) continue;
        this.ops.dispose(buf);
      }
      state.buffers.clear();
      state.borrowed.clear();
    }
    this.releaseScratch();
    this.active = false;
  }
}
