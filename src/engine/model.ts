import type { Architecture, LayerDef, ParamCount, ShapeIssue } from "./types";
import { assert, numel, shapeEq, shapeStr, uid } from "./types";
import { Tensor, stats, histogram } from "./tensor";
import { RNG } from "./rng";
import { compileLayer, inferOutputShape, makeLayerDef } from "./layers";
import type { CompiledLayer } from "./layers";
import type { InitMethod, LayerType } from "./types";

export interface ForwardResult {
  output: Tensor;
  activations: { id: string; name: string; tensor: Tensor }[];
}

export class SequentialModel {
  arch: Architecture;
  layers: CompiledLayer[];
  issues: ShapeIssue[];
  rng: RNG;

  constructor(arch: Architecture, existing?: SequentialModel | null) {
    this.arch = structuredClone(arch);
    this.rng = new RNG(arch.seed);
    this.issues = [];
    this.layers = [];
    this.rebuild(existing ?? null);
  }

  rebuild(existing: SequentialModel | null): void {
    this.issues = [];
    const compiled: CompiledLayer[] = [];
    let shape = this.arch.inputShape.slice();
    const prevById = new Map<string, CompiledLayer>();
    if (existing) {
      for (const l of existing.layers) prevById.set(l.def.id, l);
    }
    for (const def of this.arch.layers) {
      try {
        const out = inferOutputShape(def.type, def.config, shape);
        const prev = prevById.get(def.id);
        const layer = compileLayer(
          def,
          shape,
          this.rng,
          this.arch.initMethod,
          prev ? { params: prev.params, nonTrainable: prev.nonTrainable } : undefined,
        );
        compiled.push(layer);
        shape = out;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        this.issues.push({ layerId: def.id, layerName: def.name, message: msg, actual: shape.slice() });
        break;
      }
    }
    this.layers = compiled;
  }

  get valid(): boolean {
    return this.issues.length === 0 && this.layers.length === this.arch.layers.length;
  }

  get outputShape(): number[] {
    if (this.layers.length === 0) return this.arch.inputShape.slice();
    return this.layers[this.layers.length - 1]!.outputShape.slice();
  }

  paramCount(): ParamCount {
    const byLayer: ParamCount["byLayer"] = [];
    let trainable = 0;
    let nonTrainable = 0;
    for (const l of this.layers) {
      let t = 0;
      let nt = 0;
      for (const p of Object.values(l.params)) t += p.size;
      for (const p of Object.values(l.nonTrainable)) nt += p.size;
      if (l.frozen) {
        nt += t;
        t = 0;
      }
      trainable += t;
      nonTrainable += nt;
      byLayer.push({ id: l.def.id, name: l.def.name, type: l.def.type, trainable: t, nonTrainable: nt });
    }
    return { trainable, nonTrainable, byLayer };
  }

  memoryEstimate(batchSize: number, optimizer: "sgd" | "momentum" | "adam" | "adamw"): {
    paramBytes: number;
    optimizerBytes: number;
    activationBytes: number;
    totalBytes: number;
  } {
    const pc = this.paramCount();
    const paramBytes = (pc.trainable + pc.nonTrainable) * 4;
    const optMul = optimizer === "sgd" ? 0 : optimizer === "momentum" ? 1 : 2;
    const optimizerBytes = pc.trainable * 4 * optMul;
    let act = 0;
    let shape = this.arch.inputShape;
    act += numel(shape) * batchSize * 4;
    for (const l of this.layers) {
      shape = l.outputShape;
      act += numel(shape) * batchSize * 4;
    }
    return { paramBytes, optimizerBytes, activationBytes: act, totalBytes: paramBytes + optimizerBytes + act };
  }

  forward(x: Tensor, training: boolean, collect = false): ForwardResult {
    assert(this.valid, this.issues[0]?.message ?? "Model architecture is invalid");
    const expected = [x.shape[0], ...this.arch.inputShape];
    if (x.rank !== expected.length || x.shape.some((d, i) => i > 0 && d !== expected[i])) {
      throw new Error(`Model input shape ${shapeStr(x.shape)} does not match expected ${shapeStr(expected)}`);
    }
    const activations: ForwardResult["activations"] = [];
    let h = x;
    if (collect) activations.push({ id: "__input__", name: this.arch.inputName, tensor: h });
    for (const layer of this.layers) {
      h = layer.forward(h, training);
      if (collect) activations.push({ id: layer.def.id, name: layer.def.name, tensor: h });
    }
    return { output: h, activations };
  }

  backward(dOut: Tensor): { id: string; name: string; gradNorm: number }[] {
    let g = dOut;
    const norms: { id: string; name: string; gradNorm: number }[] = [];
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]!;
      g = layer.backward(g);
      let acc = 0;
      for (const t of Object.values(layer.grads)) {
        for (let k = 0; k < t.size; k++) acc += t.data[k]! * t.data[k]!;
      }
      norms.push({ id: layer.def.id, name: layer.def.name, gradNorm: Math.sqrt(acc) });
    }
    return norms.reverse();
  }

  trainableParams(): { layer: CompiledLayer; name: string; data: Tensor; grad: Tensor }[] {
    const out: { layer: CompiledLayer; name: string; data: Tensor; grad: Tensor }[] = [];
    for (const layer of this.layers) {
      if (layer.frozen) continue;
      for (const name of Object.keys(layer.params)) {
        out.push({ layer, name, data: layer.params[name]!, grad: layer.grads[name]! });
      }
    }
    return out;
  }

  zeroGrads(): void {
    for (const layer of this.layers) {
      for (const g of Object.values(layer.grads)) g.data.fill(0);
    }
  }

  reinitialize(seed?: number): void {
    if (seed !== undefined) {
      this.arch.seed = seed;
      this.rng = new RNG(seed);
    } else {
      this.rng = new RNG(this.arch.seed);
    }
    this.rebuild(null);
  }

  snapshotWeights(): Record<string, { shape: number[]; data: Float32Array }> {
    const snap: Record<string, { shape: number[]; data: Float32Array }> = {};
    for (const layer of this.layers) {
      for (const [name, t] of Object.entries(layer.params)) {
        snap[`${layer.def.id}.${name}`] = { shape: t.shape as number[], data: new Float32Array(t.data) };
      }
      for (const [name, t] of Object.entries(layer.nonTrainable)) {
        snap[`${layer.def.id}.${name}`] = { shape: t.shape as number[], data: new Float32Array(t.data) };
      }
    }
    return snap;
  }

  loadWeights(snap: Record<string, { shape: number[]; data: ArrayLike<number> }>): string[] {
    const warnings: string[] = [];
    for (const layer of this.layers) {
      for (const [name, t] of Object.entries(layer.params)) {
        const key = `${layer.def.id}.${name}`;
        const s = snap[key];
        if (!s) {
          warnings.push(`Missing parameter ${key}`);
          continue;
        }
        if (s.data.length !== t.size) {
          warnings.push(`Shape mismatch for ${key}: file has ${s.data.length} values, layer has ${t.size}`);
          continue;
        }
        t.data.set(s.data);
      }
      for (const [name, t] of Object.entries(layer.nonTrainable)) {
        const key = `${layer.def.id}.${name}`;
        const s = snap[key];
        if (s && s.data.length === t.size) t.data.set(s.data);
      }
    }
    return warnings;
  }

  inspectLayer(id: string): {
    layer: CompiledLayer;
    params: { name: string; shape: number[]; stats: ReturnType<typeof stats>; hist: ReturnType<typeof histogram> }[];
  } | null {
    const layer = this.layers.find((l) => l.def.id === id);
    if (!layer) return null;
    const params = Object.entries(layer.params).map(([name, t]) => ({
      name,
      shape: t.shape as number[],
      stats: stats(t),
      hist: histogram(t),
    }));
    return { layer, params };
  }

  clone(): SequentialModel {
    const m = new SequentialModel(this.arch, this);
    return m;
  }
}

/**
 * Per-layer gradient norms. `backward` reports them as it walks the layers; the
 * accelerated path downloads gradients first and uses this instead, so both
 * backends report the same numbers.
 */
export function gradNorms(model: SequentialModel): { id: string; name: string; gradNorm: number }[] {
  return model.layers.map((layer) => {
    let acc = 0;
    for (const t of Object.values(layer.grads)) {
      for (let k = 0; k < t.size; k++) acc += t.data[k]! * t.data[k]!;
    }
    return { id: layer.def.id, name: layer.def.name, gradNorm: Math.sqrt(acc) };
  });
}

export function emptyArchitecture(inputShape: number[] = [2], seed = 42, initMethod: InitMethod = "he_normal"): Architecture {
  return {
    inputShape,
    inputName: "input",
    layers: [],
    seed,
    initMethod,
  };
}

export function addLayer(arch: Architecture, type: LayerType, index?: number, config?: Record<string, unknown>): Architecture {
  const next = structuredClone(arch);
  const def = makeLayerDef(type, config ? { config } : undefined);
  if (index === undefined || index >= next.layers.length) next.layers.push(def);
  else next.layers.splice(index, 0, def);
  return next;
}

export function removeLayer(arch: Architecture, id: string): Architecture {
  const next = structuredClone(arch);
  next.layers = next.layers.filter((l) => l.id !== id);
  return next;
}

export function updateLayer(arch: Architecture, id: string, patch: Partial<LayerDef>): Architecture {
  const next = structuredClone(arch);
  next.layers = next.layers.map((l) => (l.id === id ? { ...l, ...patch, config: { ...l.config, ...(patch.config ?? {}) } } : l));
  return next;
}

export function moveLayer(arch: Architecture, id: string, dir: -1 | 1): Architecture {
  const next = structuredClone(arch);
  const i = next.layers.findIndex((l) => l.id === id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= next.layers.length) return arch;
  const tmp = next.layers[i]!;
  next.layers[i] = next.layers[j]!;
  next.layers[j] = tmp;
  return next;
}

export function duplicateLayer(arch: Architecture, id: string): Architecture {
  const next = structuredClone(arch);
  const i = next.layers.findIndex((l) => l.id === id);
  if (i < 0) return arch;
  const src = next.layers[i]!;
  const copy: LayerDef = {
    ...structuredClone(src),
    id: uid(src.type),
    name: `${src.name}_copy`,
  };
  next.layers.splice(i + 1, 0, copy);
  return next;
}

export function validateArchitecture(arch: Architecture): ShapeIssue[] {
  const issues: ShapeIssue[] = [];
  let shape = arch.inputShape.slice();
  if (shape.some((d) => !Number.isFinite(d) || d <= 0)) {
    issues.push({ layerId: "input", layerName: "input", message: `Invalid input shape ${shapeStr(shape)}` });
    return issues;
  }
  for (const def of arch.layers) {
    try {
      shape = inferOutputShape(def.type, def.config, shape);
    } catch (e) {
      issues.push({
        layerId: def.id,
        layerName: def.name,
        message: e instanceof Error ? e.message : String(e),
        actual: shape.slice(),
      });
      break;
    }
  }
  return issues;
}

export function shapesAlong(arch: Architecture): { id: string; name: string; input: number[]; output: number[] | null; error?: string }[] {
  const rows: { id: string; name: string; input: number[]; output: number[] | null; error?: string }[] = [];
  let shape = arch.inputShape.slice();
  rows.push({ id: "__input__", name: arch.inputName, input: [], output: shape.slice() });
  for (const def of arch.layers) {
    try {
      const out = inferOutputShape(def.type, def.config, shape);
      rows.push({ id: def.id, name: def.name, input: shape.slice(), output: out });
      shape = out;
    } catch (e) {
      rows.push({
        id: def.id,
        name: def.name,
        input: shape.slice(),
        output: null,
        error: e instanceof Error ? e.message : String(e),
      });
      break;
    }
  }
  return rows;
}

export function architectureEquals(a: Architecture, b: Architecture): boolean {
  if (!shapeEq(a.inputShape, b.inputShape)) return false;
  if (a.layers.length !== b.layers.length) return false;
  for (let i = 0; i < a.layers.length; i++) {
    const x = a.layers[i]!;
    const y = b.layers[i]!;
    if (x.type !== y.type || x.id !== y.id) return false;
    if (JSON.stringify(x.config) !== JSON.stringify(y.config)) return false;
  }
  return true;
}
