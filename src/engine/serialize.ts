import type { Architecture } from "./types";
import type { SequentialModel } from "./model";
import type { OptimizerConfig, OptimizerState } from "./optimizers";
import { serializeOptimizerState, loadOptimizerState } from "./optimizers";
import { SequentialModel as SM } from "./model";
import { Tensor } from "./tensor";

export const FORMAT_VERSION = 1;
export const FORMAT_NAME = "neuronforge";

export interface ExportedModel {
  format: typeof FORMAT_NAME;
  version: number;
  exportedAt: number;
  kind: "model" | "checkpoint";
  architecture: Architecture;
  weights: Record<string, { shape: number[]; dtype: "float32"; data: string }>;
  preprocessing?: unknown;
  metadata?: Record<string, unknown>;
  optimizer?: OptimizerConfig;
  optimizerState?: ReturnType<typeof serializeOptimizerState>;
  epoch?: number;
  batchIndex?: number;
  samplesSeen?: number;
}

function f32ToB64(data: Float32Array): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function exportModel(
  model: SequentialModel,
  opts?: {
    kind?: "model" | "checkpoint";
    preprocessing?: unknown;
    metadata?: Record<string, unknown>;
    optimizer?: OptimizerConfig;
    optimizerState?: OptimizerState;
    epoch?: number;
    batchIndex?: number;
    samplesSeen?: number;
  },
): ExportedModel {
  const weights: ExportedModel["weights"] = {};
  const snap = model.snapshotWeights();
  for (const [k, v] of Object.entries(snap)) {
    weights[k] = { shape: v.shape, dtype: "float32", data: f32ToB64(v.data) };
  }
  return {
    format: FORMAT_NAME,
    version: FORMAT_VERSION,
    exportedAt: Date.now(),
    kind: opts?.kind ?? "model",
    architecture: structuredClone(model.arch),
    weights,
    preprocessing: opts?.preprocessing,
    metadata: opts?.metadata,
    optimizer: opts?.optimizer,
    optimizerState: opts?.optimizerState ? serializeOptimizerState(opts.optimizerState) : undefined,
    epoch: opts?.epoch,
    batchIndex: opts?.batchIndex,
    samplesSeen: opts?.samplesSeen,
  };
}

export function parseExport(raw: unknown): ExportedModel {
  if (!raw || typeof raw !== "object") throw new Error("Invalid model file: not an object");
  const obj = raw as Record<string, unknown>;
  if (obj.format !== FORMAT_NAME) throw new Error(`Unrecognized format "${String(obj.format)}". Expected "${FORMAT_NAME}".`);
  if (obj.version !== FORMAT_VERSION) throw new Error(`Unsupported format version ${String(obj.version)}. This build reads v${FORMAT_VERSION}.`);
  if (!obj.architecture || typeof obj.architecture !== "object") throw new Error("Missing architecture");
  if (!obj.weights || typeof obj.weights !== "object") throw new Error("Missing weights");
  return obj as unknown as ExportedModel;
}

export function importModel(exported: ExportedModel): {
  model: SequentialModel;
  warnings: string[];
  optimizerState?: OptimizerState;
} {
  const model = new SM(exported.architecture);
  const snap: Record<string, { shape: number[]; data: Float32Array }> = {};
  for (const [k, v] of Object.entries(exported.weights)) {
    snap[k] = { shape: v.shape, data: b64ToF32(v.data) };
  }
  const warnings = model.loadWeights(snap);
  const optimizerState = exported.optimizerState ? loadOptimizerState(exported.optimizerState) : undefined;
  return { model, warnings, optimizerState };
}

export function verifyRoundTrip(original: SequentialModel, sample: Tensor): {
  ok: boolean;
  maxAbsDiff: number;
  message: string;
} {
  const exported = exportModel(original, { kind: "model" });
  const { model: restored, warnings } = importModel(exported);
  const a = original.forward(sample, false).output;
  const b = restored.forward(sample, false).output;
  if (a.size !== b.size) {
    return { ok: false, maxAbsDiff: Infinity, message: `Output size mismatch ${a.size} vs ${b.size}` };
  }
  let max = 0;
  for (let i = 0; i < a.size; i++) {
    const d = Math.abs(a.data[i]! - b.data[i]!);
    if (d > max) max = d;
  }
  const ok = max < 1e-5 && warnings.length === 0;
  return {
    ok,
    maxAbsDiff: max,
    message: ok
      ? `Round-trip verified. Max absolute difference ${max.toExponential(2)}`
      : `Round-trip discrepancy. Max |Δ| = ${max.toExponential(2)}${warnings.length ? `. Warnings: ${warnings.join("; ")}` : ""}`,
  };
}

export function downloadJson(filename: string, data: unknown): void {
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function architectureJson(arch: Architecture): unknown {
  return {
    format: FORMAT_NAME,
    version: FORMAT_VERSION,
    kind: "architecture",
    architecture: arch,
  };
}
