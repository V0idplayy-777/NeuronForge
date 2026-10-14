import RUNTIME_JS from "./runtime/runtime.js?raw";
import RUNTIME_PY from "./runtime/runtime.py?raw";
import { SequentialModel } from "../engine/model";
import type { PreparedDataset, TokenizerSpec, TabularPreprocessing, ImageSpec } from "../data/datasets";
import type { Architecture, LayerDef, LayerType, LossName } from "../engine/types";
import { numel } from "../engine/types";
import { Tensor } from "../engine/tensor";
import { RNG } from "../engine/rng";
import { makeLayerDef } from "../engine/layers";
import { generateTokens, tokenizeForModel, detokenize } from "../engine/generate";

export { RUNTIME_JS, RUNTIME_PY };

export const STANDALONE_FORMAT = "neuronforge-standalone";
export const STANDALONE_VERSION = 1;
export const STANDALONE_RUNTIME_VERSION = "1.1.0";

/** Every layer type the builder can produce. Anything else is reported as unsupported. */
export const SUPPORTED_LAYER_TYPES: LayerType[] = [
  "dense", "flatten", "reshape", "dropout", "batchnorm", "layernorm", "conv2d", "maxpool2d", "avgpool2d",
  "gap2d", "embedding", "rnn", "lstm", "gru", "residual_dense", "residual_conv",
  "identity", "relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax",
];
const FUSABLE = ["none", "linear", "relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax"];
const ACTIVATION_LAYERS = ["relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax"];

export type ExportTask = "binary" | "multiclass" | "regression" | "reconstruction" | "language" | "raw";

export interface BundleTokenizer extends TokenizerSpec {
  eos: string | null;
}

export interface BundleLayer {
  id: string;
  name: string;
  type: LayerType;
  config: Record<string, unknown>;
  inputShape: number[];
  outputShape: number[];
}

export interface BundleTests {
  tolerance: { atol: number; rtol: number };
  forward: { input: number[]; output: number[] }[];
  tokenizer: { text: string; ids: number[] }[];
  generation: { prompt: string; maxTokens: number; temperature: number; topK: number; seed: number; ids: number[]; text: string }[];
  preprocessing: TabularPreprocessing["samples"];
}

export interface StandaloneBundle {
  format: typeof STANDALONE_FORMAT;
  formatVersion: number;
  runtimeVersion: string;
  generator: string;
  exportedAt: string;
  name: string;
  task: ExportTask;
  architecture: { inputShape: number[]; inputName: string; layers: BundleLayer[] };
  weights: Record<string, { shape: number[]; dtype: "float32"; data: string; checksum: string }>;
  output: { shape: number[]; finalActivation: string | null; outputsAreProbabilities: boolean; classNames?: string[]; targetNames?: string[] };
  preprocessing: Omit<TabularPreprocessing, "samples"> | null;
  image: ImageSpec | null;
  tokenizer: BundleTokenizer | null;
  sampling: { temperature: number; topK: number };
  provenance: { project: string; dataset: string | null; loss: LossName; trainedThisSession: boolean; paramCount: number };
  includes: { inference: true; training: false; optimizerState: false; dataset: boolean };
  limitations: string[];
  tests: BundleTests;
}

export interface LayerSupport {
  name: string;
  type: string;
  js: boolean;
  python: boolean;
  onnx: false;
  note?: string;
}

export interface ExportAnalysis {
  task: ExportTask;
  taskReason: string;
  datasetCompatible: boolean;
  finalActivation: string | null;
  io: string[];
  limitations: string[];
  blockers: string[];
  perLayer: LayerSupport[];
  paramCount: number;
  approxBundleBytes: number;
}

export interface ExportOptions {
  name: string;
  project: string;
  loss: LossName;
  trained: boolean;
  includeTests?: boolean;
  includeDataset?: boolean;
}

/* ------------------------------------------------------------------ analysis */

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function finalActivation(layers: LayerDef[]): string | null {
  for (let i = layers.length - 1; i >= 0; i--) {
    const l = layers[i]!;
    if (l.type === "dropout" || l.type === "identity" || l.type === "flatten" || l.type === "reshape") continue;
    if (ACTIVATION_LAYERS.includes(l.type)) return l.type;
    if (l.type === "dense" || l.type === "conv2d") {
      const a = String(l.config.activation ?? "none");
      return a === "none" || a === "linear" ? null : a;
    }
    if (l.type === "residual_dense" || l.type === "residual_conv") {
      const a = String(l.config.finalActivation ?? "none");
      return a === "none" || a === "linear" ? null : a;
    }
    return null;
  }
  return null;
}

export function analyzeExport(model: SequentialModel, dataset: PreparedDataset | null, loss: LossName, trained: boolean): ExportAnalysis {
  const blockers: string[] = [];
  const limitations: string[] = [];
  if (!model.valid) blockers.push(`Architecture is invalid: ${model.issues[0]?.message ?? "unknown error"}`);

  const perLayer: LayerSupport[] = model.arch.layers.map((l) => {
    const supported = SUPPORTED_LAYER_TYPES.includes(l.type);
    const fused = l.type === "dense" || l.type === "conv2d" ? String(l.config.activation ?? "none") : "none";
    const fusedOk = FUSABLE.includes(fused);
    const ok = supported && fusedOk;
    return {
      name: l.name,
      type: l.type,
      js: ok,
      python: ok,
      onnx: false,
      note: !supported ? "layer type not implemented in the standalone runtimes" : !fusedOk ? `fused activation '${fused}' not supported` : undefined,
    };
  });
  for (const p of perLayer) if (!p.js) blockers.push(`Layer '${p.name}' (${p.type}): ${p.note}`);

  const inShape = model.arch.inputShape;
  const outSize = model.valid ? numel(model.outputShape) : 0;
  const meta = dataset?.meta;
  const datasetCompatible = !!meta && sameShape(meta.inputShape, inShape);
  const fa = finalActivation(model.arch.layers);

  let task: ExportTask = "raw";
  let taskReason = "";
  const tok = datasetCompatible ? meta!.tokenizer : undefined;
  if (tok && outSize === tok.tokens.length) {
    task = "language";
    taskReason = `${tok.level}-level language model: ${tok.tokens.length.toLocaleString()} output classes match the dataset tokenizer.`;
  } else if (datasetCompatible && meta!.reconstruction && outSize === numel(inShape)) {
    task = "reconstruction";
    taskReason = "Autoencoder: the dataset's targets are its inputs and the output size equals the input size.";
  } else if (loss === "mse" || loss === "mae") {
    task = "regression";
    taskReason = `Trained with ${loss.toUpperCase()}: outputs are returned as numeric predictions.`;
  } else if (loss === "bce") {
    task = outSize === 1 ? "binary" : "raw";
    taskReason = outSize === 1 ? "Binary cross-entropy with a single output unit." : `BCE with ${outSize} outputs: exported as raw outputs.`;
  } else {
    task = outSize >= 2 ? "multiclass" : "raw";
    taskReason = outSize >= 2 ? `Categorical cross-entropy over ${outSize} classes.` : "Categorical loss with one output: exported as raw outputs.";
  }
  if (meta?.tokenizer && datasetCompatible && task !== "language") {
    limitations.push(`The dataset tokenizer (${meta.tokenizer.tokens.length} tokens) does not match the model's ${outSize} outputs, so text helpers were not attached.`);
  }

  const io: string[] = [];
  switch (task) {
    case "language":
      io.push(`Text → ${tok!.level} tokens using the exact training vocabulary (${tok!.tokens.length.toLocaleString()} tokens).`);
      io.push(`Context window: last ${inShape[0]} tokens, left-padded with "${tok!.padToken === "\n" ? "\\n" : tok!.padToken}".`);
      io.push(tok!.level === "word" ? "Unknown words map to the <unk> token the model saw during training." : "Characters outside the vocabulary are skipped.");
      io.push("generate(): autoregressive loop with temperature (default 0.7) and top-k (default 50), seedable.");
      break;
    case "binary":
      io.push(`Input: ${numel(inShape)} values${meta?.preprocessing ? " or a raw table row (encoded automatically)" : ""}.`);
      io.push(`Output: ${fa === "sigmoid" ? "probability" : "score"} of class 1, predicted class (threshold 0.5)${meta?.classNames ? " and its label" : ""}.`);
      break;
    case "multiclass":
      io.push(`Input: ${numel(inShape)} values${meta?.preprocessing ? " or a raw table row (encoded automatically)" : ""}${meta?.image ? " or an image (resized and scaled automatically)" : ""}.`);
      io.push(`Output: ${fa === "softmax" ? "class probabilities" : "class scores"}, predicted class, labels, and the top-5 ranking.`);
      break;
    case "regression":
      io.push(`Input: ${numel(inShape)} values${meta?.preprocessing ? " or a raw table row (encoded automatically)" : ""}.`);
      io.push(`Output: ${outSize} predicted value(s)${datasetCompatible && meta?.targetNames ? ` named ${meta.targetNames.slice(0, 4).join(", ")}` : ""}.`);
      break;
    case "reconstruction":
      io.push(`Input: ${numel(inShape)} values. Output: the reconstruction plus its mean squared reconstruction error.`);
      break;
    default:
      io.push(`Input: ${numel(inShape)} values shaped ${JSON.stringify(inShape)}. Output: raw tensor ${JSON.stringify(model.valid ? model.outputShape : [])}.`);
  }

  if (!dataset) limitations.push("No dataset context: exported as a raw tensor model.");
  else if (!datasetCompatible) {
    limitations.push(
      `The project's current dataset (“${meta!.name}”, input ${JSON.stringify(meta!.inputShape)}) does not match the model input ${JSON.stringify(inShape)}, so no preprocessing, tokenizer or class names were attached.`,
    );
  }
  if (datasetCompatible && meta!.preprocessing) {
    const pp = meta!.preprocessing;
    limitations.push(
      `Raw table rows are encoded with the training-split statistics saved at import (${pp.normalize} normalization, missing values: ${pp.missing}, ${pp.columns.filter((c) => c.type === "categorical").length} categorical column(s)). Unseen categories encode as all-zero (one-hot) or index 0 (label), exactly as during training.`,
    );
  }
  if (datasetCompatible && meta!.image && task !== "language") {
    limitations.push(
      `predictImage() resizes with nearest-neighbour sampling${meta!.image.channels === 1 ? " and converts colour to grayscale by averaging RGB" : ""}; 8-bit pixels are scaled to [${meta!.image.range.join(", ")}]. The training images were synthetic, so real photos are out of distribution.`,
    );
  }
  if (task === "language") {
    if (tok!.level === "word") {
      limitations.push(
        `Word vocabulary is frequency-capped (${((tok!.oovRate ?? 0) * 100).toFixed(1)}% of corpus tokens were <unk>). There is no dedicated end-of-sequence token; pass stop: ["."] to end at a sentence boundary.`,
      );
    } else {
      limitations.push('Newline ("\\n") is treated as end-of-sequence by default (stopAtEos). Pass stop: [] to disable.');
    }
  }
  limitations.push("Inference only: the export cannot train, fine-tune, or compute gradients, and contains no optimiser state. Use “Download checkpoint” to resume training inside NeuronForge.");
  limitations.push("BatchNorm uses its running statistics and Dropout is disabled, matching NeuronForge's evaluation behaviour.");
  if (!trained) limitations.push("No training run was recorded in this session: the weights are exported exactly as they are now and may be untrained.");
  limitations.push("Exporting preserves the learned behaviour; it does not make the model more capable.");
  limitations.push("ONNX export is not implemented.");

  const pc = model.valid ? model.paramCount() : { trainable: 0, nonTrainable: 0 };
  const paramCount = pc.trainable + pc.nonTrainable;
  return {
    task,
    taskReason,
    datasetCompatible,
    finalActivation: fa,
    io,
    limitations,
    blockers,
    perLayer,
    paramCount,
    approxBundleBytes: Math.round(paramCount * 4 * 1.34 + 20000),
  };
}

/* ------------------------------------------------------------------ encoding */

const textEncoder = new TextEncoder();

/** Same checksum as the runtimes: over utf8(key) followed by the tensor bytes. */
export function tensorChecksum(key: string, bytes: Uint8Array): string {
  const kb = textEncoder.encode(key);
  let s1 = 0;
  let s2 = 0;
  let idx = 0;
  for (let i = 0; i < kb.length; i++, idx++) {
    s1 += kb[i]!;
    s2 += ((idx % 65521) + 1) * kb[i]!;
  }
  for (let j = 0; j < bytes.length; j++, idx++) {
    s1 += bytes[j]!;
    s2 += ((idx % 65521) + 1) * bytes[j]!;
  }
  const hex = (n: number) => (n % 4294967296).toString(16).padStart(8, "0");
  return hex(s1) + hex(s2);
}

function f32ToBase64LE(data: Float32Array): { b64: string; bytes: Uint8Array } {
  const bytes = new Uint8Array(data.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < data.length; i++) view.setFloat32(i * 4, data[i]!, true);
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return { b64: btoa(s), bytes };
}

/* ------------------------------------------------------------- test vectors */

function makeInputs(model: SequentialModel, dataset: PreparedDataset | null, compatible: boolean, n: number, rng: RNG, offset = 0): number[][] {
  const inShape = model.arch.inputShape;
  const inSize = numel(inShape);
  const out: number[][] = [];
  if (compatible && dataset) {
    for (const split of [dataset.test, dataset.val, dataset.train]) {
      const rows = split.xs.shape[0]!;
      for (let r = offset; r < rows && out.length < n; r++) out.push(Array.from(split.xs.data.subarray(r * inSize, (r + 1) * inSize)));
      if (out.length >= n) break;
    }
  }
  const first = model.arch.layers[0];
  const vocab = first?.type === "embedding" ? Number(first.config.vocabSize ?? 32) : 0;
  while (out.length < n) {
    const row: number[] = [];
    for (let i = 0; i < inSize; i++) row.push(vocab ? rng.int(vocab) : Math.round(rng.normal() * 1e6) / 1e6);
    out.push(row);
  }
  return out;
}

function referenceOutputs(model: SequentialModel, inputs: number[][]): number[][] {
  const inSize = numel(model.arch.inputShape);
  const data = new Float32Array(inputs.length * inSize);
  inputs.forEach((r, i) => data.set(r, i * inSize));
  const y = model.forward(new Tensor([inputs.length, ...model.arch.inputShape], data), false).output;
  const per = y.size / inputs.length;
  return inputs.map((_, i) => Array.from(y.data.subarray(i * per, (i + 1) * per)));
}

const TOKENIZER_TEXTS = [
  "hello there, how are you today?",
  "the quick brown fox jumps over the lazy dog.",
  "q: what is a tensor? a:",
  "zyxqv neuronforge 42!",
];

/* -------------------------------------------------------------- the bundle */

export function buildStandaloneBundle(
  model: SequentialModel,
  dataset: PreparedDataset | null,
  opts: ExportOptions,
): { bundle: StandaloneBundle; analysis: ExportAnalysis } {
  const analysis = analyzeExport(model, dataset, opts.loss, opts.trained);
  if (analysis.blockers.length) throw new Error(`Cannot export: ${analysis.blockers.join("; ")}`);
  const meta = analysis.datasetCompatible ? dataset!.meta : null;

  const weights: StandaloneBundle["weights"] = {};
  for (const [key, t] of Object.entries(model.snapshotWeights())) {
    const { b64, bytes } = f32ToBase64LE(t.data);
    weights[key] = { shape: t.shape.slice(), dtype: "float32", data: b64, checksum: tensorChecksum(key, bytes) };
  }

  const layers: BundleLayer[] = model.layers.map((l) => ({
    id: l.def.id,
    name: l.def.name,
    type: l.def.type,
    config: structuredClone(l.def.config),
    inputShape: l.inputShape.slice(),
    outputShape: l.outputShape.slice(),
  }));

  let tokenizer: BundleTokenizer | null = null;
  if (analysis.task === "language" && meta?.tokenizer) {
    const t = meta.tokenizer;
    tokenizer = { ...structuredClone(t), eos: t.level === "char" && t.tokens.includes("\n") ? "\n" : null };
  }
  let preprocessing: StandaloneBundle["preprocessing"] = null;
  if (meta?.preprocessing && analysis.task !== "language") {
    const { samples: _samples, ...rest } = meta.preprocessing;
    void _samples;
    preprocessing = structuredClone(rest);
  }
  const image = meta?.image && analysis.task !== "language" ? structuredClone(meta.image) : null;
  const fa = analysis.finalActivation;
  const output: StandaloneBundle["output"] = {
    shape: model.outputShape.slice(),
    finalActivation: fa,
    outputsAreProbabilities:
      (analysis.task === "binary" && fa === "sigmoid") || ((analysis.task === "multiclass" || analysis.task === "language") && fa === "softmax"),
  };
  if (meta?.classNames && (analysis.task === "binary" || analysis.task === "multiclass")) output.classNames = meta.classNames.slice();
  if (meta?.targetNames && analysis.task === "regression") output.targetNames = meta.targetNames.slice();

  const tests: BundleTests = { tolerance: { atol: 1e-5, rtol: 1e-4 }, forward: [], tokenizer: [], generation: [], preprocessing: [] };
  if (opts.includeTests !== false) {
    const outSize = numel(model.outputShape);
    const inputs = makeInputs(model, dataset, analysis.datasetCompatible, outSize > 512 ? 3 : 6, new RNG(1234));
    const outs = referenceOutputs(model, inputs);
    tests.forward = inputs.map((input, i) => ({ input, output: outs[i]! }));
    if (tokenizer) {
      tests.tokenizer = TOKENIZER_TEXTS.map((text) => ({ text, ids: tokenizeForModel(tokenizer!, text) }));
      const settings = [
        { prompt: "hello there", maxTokens: tokenizer.level === "char" ? 40 : 20, temperature: 0.7, topK: 50, seed: 7 },
        { prompt: "the", maxTokens: 16, temperature: 0, topK: 50, seed: 1 },
      ];
      for (const st of settings) {
        if (tokenizeForModel(tokenizer, st.prompt).length === 0) continue;
        const r = generateTokens(model, tokenizer, st.prompt, { ...st, stop: [] });
        tests.generation.push({ ...st, ids: r.ids, text: r.text });
      }
    }
    if (meta?.preprocessing && preprocessing) tests.preprocessing = structuredClone(meta.preprocessing.samples);
  }

  const limitations = analysis.limitations.slice();
  limitations.splice(limitations.length - 1, 0, opts.includeDataset ? "The training dataset is included as dataset.json (optional export)." : "The training dataset is not included.");

  const bundle: StandaloneBundle = {
    format: STANDALONE_FORMAT,
    formatVersion: STANDALONE_VERSION,
    runtimeVersion: STANDALONE_RUNTIME_VERSION,
    generator: "NeuronForge",
    exportedAt: new Date().toISOString(),
    name: opts.name,
    task: analysis.task,
    architecture: { inputShape: model.arch.inputShape.slice(), inputName: model.arch.inputName, layers },
    weights,
    output,
    preprocessing,
    image,
    tokenizer,
    sampling: { temperature: 0.7, topK: 50 },
    provenance: { project: opts.project, dataset: meta?.name ?? null, loss: opts.loss, trainedThisSession: opts.trained, paramCount: analysis.paramCount },
    includes: { inference: true, training: false, optimizerState: false, dataset: !!opts.includeDataset },
    limitations,
    tests,
  };
  return { bundle, analysis: { ...analysis, limitations } };
}

/* ---------------------------------------------------------------- runtime */

export interface RuntimeModel {
  info: { layers: { name: string; type: string; outputShape: number[] }[]; outputShape: number[]; paramCount: number };
  forward(input: ArrayLike<number> | number[], batch?: number): { shape: number[]; data: Float32Array };
  predict(input: unknown, opts?: { preprocessed?: boolean }): Record<string, unknown>;
  encode(row: unknown): number[];
  tokenize(text: string): number[];
  detokenize(ids: number[]): string;
  generateDetailed(prompt: string, o: Record<string, unknown>): { ids: number[]; text: string };
  verify(): { ok: boolean; checks: { name: string; status: string; detail: string }[] };
  getWeights(): Record<string, Float32Array>;
}

export interface RuntimeModule {
  createModel(bundle: unknown, options?: { skipChecksums?: boolean }): RuntimeModel;
  NF_RUNTIME_VERSION: string;
}

let cachedRuntime: RuntimeModule | null = null;

/**
 * Evaluate the runtime source text that is shipped inside every JavaScript
 * export. Only NeuronForge's own bundled runtime is evaluated here; imported
 * files are never executed.
 */
export function loadRuntime(): RuntimeModule {
  if (!cachedRuntime) {
    const factory = new Function(`${RUNTIME_JS}\nreturn { createModel: createModel, NF_RUNTIME_VERSION: NF_RUNTIME_VERSION };`) as () => RuntimeModule;
    cachedRuntime = factory();
  }
  return cachedRuntime;
}

function compare(a: ArrayLike<number>, b: ArrayLike<number>, atol: number, rtol: number): { maxAbs: number; bad: number } {
  let maxAbs = 0;
  let bad = 0;
  if (a.length !== b.length) return { maxAbs: Infinity, bad: Math.max(a.length, b.length) };
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > maxAbs || Number.isNaN(d)) maxAbs = Number.isNaN(d) ? Infinity : d;
    if (!(d <= atol + rtol * Math.abs(b[i]!))) bad++;
  }
  return { maxAbs, bad };
}

/* --------------------------------------------------------- coverage suite */

interface CoverageCase {
  name: string;
  inputShape: number[];
  layers: { type: LayerType; config?: Record<string, unknown> }[];
  vocab?: number;
}

const ACTS = ["linear", "relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax"];

export const COVERAGE_CASES: CoverageCase[] = [
  ...ACTS.map((a) => ({ name: `Dense + fused ${a}`, inputShape: [5], layers: [{ type: "dense" as LayerType, config: { units: 4, activation: a } }] })),
  { name: "Dense without bias", inputShape: [5], layers: [{ type: "dense", config: { units: 3, useBias: false } }] },
  ...(["relu", "gelu", "silu", "sigmoid", "tanh", "softmax"] as LayerType[]).map((a) => ({
    name: `${a} layer`,
    inputShape: [6],
    layers: [{ type: "dense" as LayerType, config: { units: 6 } }, { type: a }],
  })),
  { name: "LeakyReLU layer (α = 0.2)", inputShape: [6], layers: [{ type: "dense", config: { units: 6 } }, { type: "leakyrelu", config: { alpha: 0.2 } }] },
  { name: "Identity", inputShape: [4], layers: [{ type: "dense", config: { units: 4 } }, { type: "identity" }] },
  { name: "Dropout (disabled at inference)", inputShape: [4], layers: [{ type: "dense", config: { units: 4 } }, { type: "dropout", config: { rate: 0.5 } }] },
  { name: "BatchNorm (running statistics)", inputShape: [4], layers: [{ type: "dense", config: { units: 4 } }, { type: "batchnorm" }] },
  { name: "LayerNorm", inputShape: [5], layers: [{ type: "dense", config: { units: 5 } }, { type: "layernorm" }] },
  { name: "Conv2D same, stride 1, fused ReLU", inputShape: [6, 6, 2], layers: [{ type: "conv2d", config: { filters: 3, kernelSize: 3, stride: 1, padding: "same", activation: "relu" } }] },
  { name: "Conv2D valid, stride 2, no bias", inputShape: [7, 7, 1], layers: [{ type: "conv2d", config: { filters: 2, kernelSize: 3, stride: 2, padding: "valid", useBias: false } }] },
  { name: "Conv2D even kernel, same (asymmetric pad)", inputShape: [5, 5, 1], layers: [{ type: "conv2d", config: { filters: 2, kernelSize: 2, stride: 1, padding: "same" } }] },
  { name: "Conv2D same, stride 2", inputShape: [5, 5, 2], layers: [{ type: "conv2d", config: { filters: 3, kernelSize: 3, stride: 2, padding: "same" } }] },
  { name: "Conv2D fused softmax over channels", inputShape: [3, 3, 1], layers: [{ type: "conv2d", config: { filters: 4, kernelSize: 3, padding: "same", activation: "softmax" } }] },
  { name: "MaxPool2D", inputShape: [6, 6, 1], layers: [{ type: "conv2d", config: { filters: 2, kernelSize: 3 } }, { type: "maxpool2d", config: { poolSize: 2, stride: 2 } }] },
  { name: "AvgPool2D (overlapping windows)", inputShape: [5, 5, 1], layers: [{ type: "conv2d", config: { filters: 2, kernelSize: 3 } }, { type: "avgpool2d", config: { poolSize: 2, stride: 1 } }] },
  { name: "GlobalAvgPool → Dense", inputShape: [4, 4, 1], layers: [{ type: "conv2d", config: { filters: 3, kernelSize: 3 } }, { type: "gap2d" }, { type: "dense", config: { units: 2 } }] },
  { name: "BatchNorm over conv channels", inputShape: [4, 4, 1], layers: [{ type: "conv2d", config: { filters: 2, kernelSize: 3 } }, { type: "batchnorm" }] },
  { name: "Flatten → Dense", inputShape: [3, 3, 2], layers: [{ type: "flatten" }, { type: "dense", config: { units: 3 } }] },
  {
    name: "Reshape → LayerNorm → Flatten",
    inputShape: [6],
    layers: [{ type: "dense", config: { units: 6 } }, { type: "reshape", config: { targetShape: [2, 3] } }, { type: "layernorm" }, { type: "flatten" }, { type: "dense", config: { units: 2 } }],
  },
  { name: "Embedding (rounding + clamping)", inputShape: [5], vocab: 7, layers: [{ type: "embedding", config: { vocabSize: 7, embeddingDim: 3 } }, { type: "flatten" }, { type: "dense", config: { units: 2 } }] },
  { name: "Embedding → RNN (final state)", inputShape: [4], vocab: 6, layers: [{ type: "embedding", config: { vocabSize: 6, embeddingDim: 3 } }, { type: "rnn", config: { units: 4 } }] },
  { name: "RNN (return sequences) → Flatten", inputShape: [4, 3], layers: [{ type: "rnn", config: { units: 3, returnSequences: true } }, { type: "flatten" }, { type: "dense", config: { units: 2 } }] },
  { name: "Embedding → LSTM (final state)", inputShape: [5], vocab: 6, layers: [{ type: "embedding", config: { vocabSize: 6, embeddingDim: 3 } }, { type: "lstm", config: { units: 4 } }] },
  { name: "LSTM (return sequences) → Flatten → Dense", inputShape: [4, 3], layers: [{ type: "lstm", config: { units: 3, returnSequences: true } }, { type: "flatten" }, { type: "dense", config: { units: 2 } }] },
  { name: "LSTM (forget bias 0, tanh readout)", inputShape: [3, 2], layers: [{ type: "lstm", config: { units: 2, forgetBias: 0 } }, { type: "tanh" }] },
  { name: "Embedding → GRU (final state)", inputShape: [5], vocab: 6, layers: [{ type: "embedding", config: { vocabSize: 6, embeddingDim: 3 } }, { type: "gru", config: { units: 4 } }] },
  { name: "GRU (return sequences) → Flatten → Dense", inputShape: [4, 3], layers: [{ type: "gru", config: { units: 3, returnSequences: true } }, { type: "flatten" }, { type: "dense", config: { units: 2 } }] },
  { name: "Residual Dense (identity shortcut)", inputShape: [5], layers: [{ type: "residual_dense", config: { units: 0, activation: "relu" } }] },
  { name: "Residual Dense (projection + final activation)", inputShape: [4], layers: [{ type: "residual_dense", config: { units: 7, activation: "gelu", finalActivation: "tanh", scale: 0.7 } }, { type: "dense", config: { units: 2 } }] },
  { name: "Stacked residual blocks → softmax", inputShape: [6], layers: [{ type: "residual_dense", config: { units: 0, activation: "silu" } }, { type: "residual_dense", config: { units: 0, activation: "silu" } }, { type: "dense", config: { units: 3 } }, { type: "softmax" }] },
  { name: "Residual Conv (identity shortcut)", inputShape: [6, 6, 2], layers: [{ type: "residual_conv", config: { filters: 0, kernelSize: 3, stride: 1, padding: "same", activation: "relu" } }] },
  { name: "Residual Conv (projection, stride 2)", inputShape: [8, 8, 1], layers: [{ type: "residual_conv", config: { filters: 3, kernelSize: 3, stride: 2, padding: "same", activation: "gelu" } }, { type: "gap2d" }, { type: "dense", config: { units: 2 } }] },
  { name: "Residual Conv (1×1 kernels, valid, no bias)", inputShape: [4, 4, 2], layers: [{ type: "residual_conv", config: { filters: 2, kernelSize: 1, stride: 1, padding: "valid", activation: "linear", finalActivation: "silu", useBias: false } }, { type: "flatten" }, { type: "dense", config: { units: 3 } }] },
];

function randomizeParams(m: SequentialModel, rng: RNG): void {
  for (const l of m.layers) {
    for (const t of Object.values(l.params)) for (let i = 0; i < t.size; i++) t.data[i] = rng.normal() * 0.6;
    const nt = l.nonTrainable;
    if (nt.runningMean) for (let i = 0; i < nt.runningMean.size; i++) nt.runningMean.data[i] = rng.normal() * 0.5;
    if (nt.runningVar) for (let i = 0; i < nt.runningVar.size; i++) nt.runningVar.data[i] = 0.3 + rng.next();
  }
}

export interface CoverageResult {
  name: string;
  ok: boolean;
  maxDiff: number;
  error?: string;
}

export function runLayerCoverage(rt: RuntimeModule = loadRuntime()): CoverageResult[] {
  const rng = new RNG(20240601);
  return COVERAGE_CASES.map((cs, ci) => {
    try {
      const arch: Architecture = {
        inputShape: cs.inputShape,
        inputName: "x",
        seed: 7 + ci,
        initMethod: "xavier_uniform",
        layers: cs.layers.map((l, i) => makeLayerDef(l.type, { id: `cov${ci}_${i}_${l.type}`, name: `${l.type}_${i}`, config: l.config })),
      };
      const m = new SequentialModel(arch);
      if (!m.valid) throw new Error(m.issues[0]?.message ?? "invalid case");
      randomizeParams(m, rng);
      const { bundle } = buildStandaloneBundle(m, null, { name: cs.name, project: "coverage", loss: "mse", trained: false, includeTests: false });
      const rtm = rt.createModel(bundle);
      const B = 3;
      const inSize = numel(cs.inputShape);
      const data = new Float32Array(B * inSize);
      const edge = cs.vocab ? [0, 1, cs.vocab - 1, cs.vocab + 3, -2, 2.5, 1.4999] : [];
      for (let i = 0; i < data.length; i++) data[i] = cs.vocab ? (i < edge.length ? edge[i]! : rng.int(cs.vocab)) : rng.normal();
      const ref = m.forward(new Tensor([B, ...cs.inputShape], data), false).output;
      const got = rtm.forward(data, B);
      if (!sameShape(got.shape, ref.shape as number[])) throw new Error(`shape ${JSON.stringify(got.shape)} vs ${JSON.stringify(ref.shape)}`);
      const c = compare(got.data, ref.data, 1e-6, 1e-6);
      return { name: cs.name, ok: c.bad === 0, maxDiff: c.maxAbs };
    } catch (e) {
      return { name: cs.name, ok: false, maxDiff: NaN, error: e instanceof Error ? e.message : String(e) };
    }
  });
}

/* ------------------------------------------------------------ verification */

export type CheckStatus = "pass" | "fail" | "warn" | "info";
export interface VerifyCheck {
  id: string;
  label: string;
  status: CheckStatus;
  detail: string;
  items?: { label: string; ok: boolean; detail: string }[];
}
export interface VerifyReport {
  ok: boolean;
  checks: VerifyCheck[];
  durationMs: number;
}

function expectThrow(fn: () => unknown, pattern: RegExp): { ok: boolean; detail: string } {
  try {
    fn();
    return { ok: false, detail: "accepted silently (should have been rejected)" };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: pattern.test(msg), detail: msg };
  }
}

export function verifyStandalone(bundle: StandaloneBundle, model: SequentialModel, dataset: PreparedDataset | null): VerifyReport {
  const t0 = performance.now();
  const checks: VerifyCheck[] = [];
  const rt = loadRuntime();

  // 1. load
  let rtm: RuntimeModel;
  try {
    rtm = rt.createModel(bundle);
    checks.push({
      id: "load",
      label: "Exported runtime loads the bundle",
      status: "pass",
      detail: `Runtime v${rt.NF_RUNTIME_VERSION}: format, all ${bundle.architecture.layers.length} layers, every tensor shape and checksum validated.`,
    });
  } catch (e) {
    checks.push({ id: "load", label: "Exported runtime loads the bundle", status: "fail", detail: e instanceof Error ? e.message : String(e) });
    return { ok: false, checks, durationMs: performance.now() - t0 };
  }

  // 2. weights bit-identical
  {
    const snap = model.snapshotWeights();
    const got = rtm.getWeights();
    const keys = Object.keys(snap);
    const bad: string[] = [];
    for (const k of keys) {
      const a = snap[k]!.data;
      const b = got[k];
      if (!b || b.length !== a.length) {
        bad.push(`${k}: missing or wrong length`);
        continue;
      }
      const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
      const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
      for (let i = 0; i < ua.length; i++) {
        if (ua[i] !== ub[i]) {
          bad.push(`${k}[${i}] differs`);
          break;
        }
      }
    }
    const extra = Object.keys(got).filter((k) => !(k in snap));
    checks.push({
      id: "weights",
      label: "Weights are bit-identical and on the right layers",
      status: bad.length || extra.length ? "fail" : "pass",
      detail: bad.length || extra.length ? [...bad, ...extra.map((k) => `${k}: unexpected`)].slice(0, 6).join("; ") : `${keys.length} tensors, ${bundle.provenance.paramCount.toLocaleString()} values, bit-for-bit equal.`,
    });
  }

  // 3. shapes
  {
    const mism = model.layers
      .map((l, i) => ({ l, r: rtm.info.layers[i] }))
      .filter(({ l, r }) => !r || !sameShape(r.outputShape, l.outputShape))
      .map(({ l }) => l.def.name);
    checks.push({
      id: "shapes",
      label: "Tensor shapes agree at every layer",
      status: mism.length ? "fail" : "pass",
      detail: mism.length ? `Mismatch at: ${mism.join(", ")}` : `${model.layers.length} layers, output ${JSON.stringify(model.outputShape)}.`,
    });
  }

  // 4. forward parity on embedded + fresh inputs
  {
    const compatible = !!dataset && sameShape(dataset.meta.inputShape, model.arch.inputShape);
    const outSize = numel(model.outputShape);
    const fresh = makeInputs(model, dataset, compatible, outSize > 512 ? 4 : 12, new RNG(98765), 6);
    const synthetic = makeInputs(model, null, false, outSize > 512 ? 2 : 6, new RNG(4321));
    const all = [...bundle.tests.forward.map((f) => f.input), ...fresh, ...synthetic];
    const ref = referenceOutputs(model, all);
    let maxAbs = 0;
    let bad = 0;
    all.forEach((inp, i) => {
      const got = rtm.forward(inp, 1).data;
      const c = compare(got, ref[i]!, 1e-6, 1e-6);
      maxAbs = Math.max(maxAbs, c.maxAbs);
      bad += c.bad;
    });
    checks.push({
      id: "forward",
      label: "Predictions match NeuronForge",
      status: bad ? "fail" : "pass",
      detail: `${all.length} inputs (${compatible ? "real dataset rows + " : ""}random) · max |Δ| = ${maxAbs.toExponential(2)}${bad ? ` · ${bad} values outside 1e-6` : " · within 1e-6"}.`,
    });
  }

  // 5. layer coverage
  {
    const cov = runLayerCoverage(rt);
    const failed = cov.filter((c) => !c.ok);
    checks.push({
      id: "coverage",
      label: "Every supported layer and activation",
      status: failed.length ? "fail" : "pass",
      detail: failed.length ? `${failed.length}/${cov.length} cases failed` : `${cov.length}/${cov.length} cases match the engine (random weights, edge-case inputs).`,
      items: cov.map((c) => ({ label: c.name, ok: c.ok, detail: c.error ?? `max |Δ| ${Number.isFinite(c.maxDiff) ? c.maxDiff.toExponential(1) : "—"}` })),
    });
  }

  // 6. tokenizer + generation
  if (bundle.tokenizer) {
    const tok = bundle.tokenizer;
    const texts = [...TOKENIZER_TEXTS, "Hello, WORLD... new words like flibbertigibbet"];
    const items = texts.map((text) => {
      const a = tokenizeForModel(tok, text);
      const b = rtm.tokenize(text);
      const same = a.length === b.length && a.every((v, i) => v === b[i]);
      const round = detokenize(tok, a) === rtm.detokenize(b);
      return { label: JSON.stringify(text), ok: same && round, detail: `${a.length} tokens${same ? "" : " (ids differ)"}${round ? "" : " (decode differs)"}` };
    });
    checks.push({
      id: "tokenizer",
      label: `Tokenizer (${tok.level}, ${tok.tokens.length.toLocaleString()} tokens, context ${tok.seqLen})`,
      status: items.every((i) => i.ok) ? "pass" : "fail",
      detail: items.every((i) => i.ok) ? "Encoding and decoding identical to training." : "Tokenizer mismatch.",
      items,
    });
    const gens = [...bundle.tests.generation];
    for (const seed of [11, 12]) {
      const st = { prompt: "hello there", maxTokens: 12, temperature: 0.7, topK: 50, seed };
      if (tokenizeForModel(tok, st.prompt).length) {
        const r = generateTokens(model, tok, st.prompt, { ...st, stop: [] });
        gens.push({ ...st, ids: r.ids, text: r.text });
      }
    }
    const gitems = gens.map((g) => {
      const r = rtm.generateDetailed(g.prompt, { maxTokens: g.maxTokens, temperature: g.temperature, topK: g.topK, seed: g.seed, stop: [] });
      const same = r.ids.length === g.ids.length && r.ids.every((v, i) => v === g.ids[i]);
      return { label: `“${g.prompt}” T=${g.temperature} k=${g.topK} seed=${g.seed}`, ok: same, detail: same ? JSON.stringify(g.text.slice(0, 60)) : "sampled tokens differ" };
    });
    checks.push({
      id: "generation",
      label: "Text generation (temperature + top-k sampling)",
      status: gitems.every((i) => i.ok) ? "pass" : "fail",
      detail: gitems.every((i) => i.ok) ? `${gitems.length} seeded generations produce identical tokens.` : "Generation diverged from NeuronForge.",
      items: gitems,
    });
  }

  // 7. preprocessing
  if (bundle.preprocessing && bundle.tests.preprocessing.length) {
    const items = bundle.tests.preprocessing.map((s, i) => {
      try {
        const enc = rtm.encode(s.raw);
        const c = compare(enc, s.encoded, 1e-12, 1e-9);
        return { label: `training row #${i + 1}`, ok: c.bad === 0, detail: `${enc.length} features, max |Δ| ${c.maxAbs.toExponential(1)}` };
      } catch (e) {
        return { label: `training row #${i + 1}`, ok: false, detail: e instanceof Error ? e.message : String(e) };
      }
    });
    checks.push({
      id: "preprocessing",
      label: "Raw-row preprocessing matches the training pipeline",
      status: items.every((i) => i.ok) ? "pass" : "fail",
      detail: `${items.length} real rows encoded from raw CSV values.`,
      items,
    });
  }

  // 8. rejects bad bundles
  {
    const items: { label: string; ok: boolean; detail: string }[] = [];
    const clone = () => structuredClone(bundle) as StandaloneBundle;
    const unsupported = clone();
    unsupported.architecture.layers.push({ id: "probe", name: "attention_probe", type: "multihead_attention" as LayerType, config: {}, inputShape: [], outputShape: [] });
    const r1 = expectThrow(() => rt.createModel(unsupported), /Unsupported layer type/);
    items.push({ label: "unsupported layer type", ...r1 });
    const keys = Object.keys(bundle.weights);
    if (keys.length) {
      const corrupt = clone();
      const e = corrupt.weights[keys[0]!]!;
      const mid = Math.floor(e.data.length / 2);
      e.data = e.data.slice(0, mid) + (e.data[mid] === "A" ? "B" : "A") + e.data.slice(mid + 1);
      items.push({ label: "corrupted weight bytes", ...expectThrow(() => rt.createModel(corrupt), /Checksum mismatch|Corrupted/) });
      const missing = clone();
      delete missing.weights[keys[0]!];
      items.push({ label: "missing weight tensor", ...expectThrow(() => rt.createModel(missing), /Missing weight tensor/) });
      const wrongShape = clone();
      wrongShape.weights[keys[0]!]!.shape = [1];
      items.push({ label: "wrong tensor shape", ...expectThrow(() => rt.createModel(wrongShape), /has shape|bytes/) });
      const pair = keys.flatMap((a, i) => keys.slice(i + 1).map((b) => [a, b] as const)).find(([a, b]) => sameShape(bundle.weights[a]!.shape, bundle.weights[b]!.shape));
      if (pair) {
        const swapped = clone();
        const tmp = swapped.weights[pair[0]]!;
        swapped.weights[pair[0]] = swapped.weights[pair[1]]!;
        swapped.weights[pair[1]] = tmp;
        items.push({ label: "same-shape tensors swapped between layers", ...expectThrow(() => rt.createModel(swapped), /Checksum mismatch/) });
      }
      const extra = clone();
      extra.weights["ghost.W"] = { ...bundle.weights[keys[0]!]! };
      items.push({ label: "unassigned extra tensor", ...expectThrow(() => rt.createModel(extra), /not used by any layer/) });
    }
    checks.push({
      id: "rejects",
      label: "Detects unsupported and corrupted bundles",
      status: items.every((i) => i.ok) ? "pass" : "fail",
      detail: items.every((i) => i.ok) ? `${items.length} deliberately broken bundles were all rejected with a clear error.` : "A broken bundle was not rejected.",
      items,
    });
  }

  // 9. embedded self-test
  {
    const r = rtm.verify();
    checks.push({
      id: "selftest",
      label: "Shipped verify() passes",
      status: r.ok ? "pass" : "fail",
      detail: `${r.checks.length} embedded test vectors (the same ones verify.mjs / verify.py replay).`,
      items: r.checks.map((c) => ({ label: c.name, ok: c.status !== "fail", detail: c.detail })),
    });
  }

  checks.push({
    id: "python",
    label: "Python / NumPy target",
    status: "info",
    detail: `Generated from the same bundle but cannot run inside the browser. Run “python neuronforge_model.py --verify” to replay ${bundle.tests.forward.length} forward, ${bundle.tests.tokenizer.length} tokenizer, ${bundle.tests.generation.length} generation and ${bundle.tests.preprocessing.length} preprocessing vectors computed here (tolerance 1e-4 abs / 1e-3 rel).`,
  });
  checks.push({
    id: "onnx",
    label: "ONNX",
    status: "info",
    detail: "Not implemented. It would need each layer mapped to ONNX operators (Embedding → Gather, RNN → ONNX RNN with matching gate layout, GELU's tanh approximation) plus its own conformance tests.",
  });

  return { ok: checks.every((c) => c.status !== "fail"), checks, durationMs: performance.now() - t0 };
}
