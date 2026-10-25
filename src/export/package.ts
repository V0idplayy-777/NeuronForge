import { zipSync, strToU8 } from "fflate";
import { RUNTIME_JS, RUNTIME_PY, type StandaloneBundle } from "./standalone";

export type ExportTarget = "js" | "single" | "python" | "bundle" | "all";

export const TARGETS: { id: ExportTarget; label: string; file: string; description: string }[] = [
  { id: "js", label: "JavaScript / TypeScript package", file: ".zip", description: "ES modules for browsers, Node.js, Deno and Bun. Runtime, embedded weights, TypeScript types, verify script, example, README." },
  { id: "single", label: "Single-file JavaScript module", file: ".mjs", description: "One self-contained file: runtime + weights + tokenizer/preprocessing. Drop it into any project and import it." },
  { id: "python", label: "Python package (NumPy)", file: ".zip", description: "Pure Python + NumPy runtime, bundle.json, CLI (--verify / --predict / --generate), example, README." },
  { id: "bundle", label: "Portable model bundle", file: ".json", description: "Language-independent bundle: architecture, weights, preprocessing, tokenizer, test vectors. Needs a compatible runtime to execute." },
  { id: "all", label: "Everything", file: ".zip", description: "JavaScript package, Python package and the portable bundle in one archive." },
];

export function sanitizeName(name: string): string {
  const s = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return s || "neuronforge-model";
}

function bundleJson(b: StandaloneBundle): string {
  return JSON.stringify(b);
}

function round(arr: number[], d = 4): number[] {
  return arr.map((v) => Math.round(v * 10 ** d) / 10 ** d);
}

/* ----------------------------------------------------------------- examples */

function jsExample(b: StandaloneBundle, importPath: string): string {
  const head = `import { loadModel } from "${importPath}";\n\nconst model = await loadModel();\n`;
  const sample = b.tests.forward[0]?.input ?? [];
  switch (b.task) {
    case "language":
      return (
        head +
        `\n// Generate text (temperature + top-k sampling; pass a seed for reproducible output)\n` +
        `const text = model.generate("hello there", { maxTokens: ${b.tokenizer?.level === "char" ? 120 : 40}, temperature: 0.7, topK: 50 });\n` +
        `console.log(text);\n\n` +
        `// Full details: generated ids/tokens, settings and why it stopped\n` +
        `const run = model.generateDetailed("hello there", { maxTokens: 20, seed: 42 });\n` +
        `console.log(run.fullText, run.stoppedBy);\n\n` +
        `// Next-token distribution\n` +
        `console.log(model.predict("hello there").top.slice(0, 5));\n` +
        (b.reasoning
          ? `\n// Deep reasoning: every token above ran ${b.reasoning.defaultSteps} shared-weight deliberation steps.\n` +
            `// Ask for the per-step halting trace of the final forward pass:\n` +
            `const traced = model.generateDetailed("q: seven plus eight think:", { maxTokens: 90, seed: 42, traceReasoning: true });\n` +
            `console.log(traced.fullText);\n` +
            `console.log(traced.reasoning?.[0]?.halt);  // one halting probability per deliberation step\n\n` +
            `// Or run one forward pass on a prompt and inspect the trace directly:\n` +
            `const r = model.reason("q: seven plus eight think:");\n` +
            `console.log(r.traces[0].stepsUsed, r.traces[0].halt);\n`
          : "")
      );
    case "binary":
    case "multiclass":
    case "regression":
    case "reconstruction": {
      const pp = b.preprocessing;
      const rawRow = b.tests.preprocessing[0]?.raw;
      let s = head;
      if (pp && rawRow) {
        const row: Record<string, string> = {};
        for (const c of pp.columns) row[c.name] = rawRow[c.name] ?? "";
        s += `\n// Raw table row: encoding + normalisation are applied automatically\nconst result = model.predict(${JSON.stringify(row, null, 2)});\nconsole.log(result);\n`;
        s += `\n// Already-encoded features (${pp.featureNames.length} values) skip preprocessing:\n// model.predict(features, { preprocessed: true });\n`;
      } else {
        s += `\nconst result = model.predict(${JSON.stringify(round(sample))});\nconsole.log(result);\n`;
      }
      if (b.image) {
        s += `\n// Images: ImageData, { data, width, height } or a pixel array. Resized to ${b.image.width}x${b.image.height}, scaled to [${b.image.range.join(", ")}].\n// const result = model.predictImage(ctx.getImageData(0, 0, w, h));\n`;
      }
      s += `\n// Batches\nconsole.log(model.predictBatch([${JSON.stringify(round(sample))}]));\n`;
      if (b.reasoning) {
        s += `\n// Deep reasoning trace for this input (shared-weight deliberation)\nconst r = model.reason(${JSON.stringify(round(sample))});\nconsole.log(r.traces[0].stepsUsed, r.traces[0].halt);\n`;
      }
      return s;
    }
    default:
      return (
        head +
        `\nconst { shape, data } = model.forward(${JSON.stringify(round(sample))});\nconsole.log(shape, data);\n` +
        (b.reasoning ? `\nconst r = model.reason(${JSON.stringify(round(sample))});\nconsole.log(r.traces[0]);\n` : "")
      );
  }
}

function pyExample(b: StandaloneBundle): string {
  const head = `from neuronforge_model import load_model\n\nmodel = load_model()  # reads bundle.json next to neuronforge_model.py\n`;
  const sample = b.tests.forward[0]?.input ?? [];
  if (b.task === "language") {
    return (
      head +
      `\nprint(model.generate("hello there", max_tokens=${b.tokenizer?.level === "char" ? 120 : 40}, temperature=0.7, top_k=50, include_prompt=True))\n` +
      `print(model.next_token("hello there")["top"][:5])\n` +
      (b.reasoning
        ? `\n# Deep reasoning: the runtime executes the deliberation loop itself.\n` +
          `run = model.generate_detailed("q: seven plus eight think:", max_tokens=90, seed=42, trace_reasoning=True)\n` +
          `print(run["fullText"])\n` +
          `print(run["reasoning"][0]["stepsUsed"], run["reasoning"][0]["halt"])\n\n` +
          `# Or one forward pass with the trace, with an optional depth override:\n` +
          `r = model.reason("q: seven plus eight think:", steps=${b.reasoning.defaultSteps})\n` +
          `print(r["traces"][0]["halt"])\n`
        : "")
    );
  }
  if (b.preprocessing && b.tests.preprocessing[0]) {
    const row: Record<string, string> = {};
    for (const c of b.preprocessing.columns) row[c.name] = b.tests.preprocessing[0].raw[c.name] ?? "";
    return head + `\n# Raw table row: encoding + normalisation applied automatically\nprint(model.predict(${JSON.stringify(row)}))\n`;
  }
  return (
    head +
    `\nprint(model.predict(${JSON.stringify(round(sample))}))\n` +
    (b.reasoning ? `\nprint(model.reason(${JSON.stringify(round(sample))})["traces"][0]["halt"])\n` : "")
  );
}

/* ------------------------------------------------------------------- docs */

function ioSection(b: StandaloneBundle): string {
  const lines: string[] = [];
  const shape = JSON.stringify(b.architecture.inputShape);
  lines.push(`- **Task:** \`${b.task}\``);
  lines.push(`- **Input shape:** \`${shape}\` (batch dimension excluded)`);
  lines.push(`- **Output shape:** \`${JSON.stringify(b.output.shape)}\`${b.output.finalActivation ? `, final activation \`${b.output.finalActivation}\`` : ""}`);
  if (b.output.classNames) lines.push(`- **Classes:** ${b.output.classNames.map((c) => `\`${c}\``).join(", ")}`);
  if (b.output.targetNames) lines.push(`- **Targets:** ${b.output.targetNames.map((c) => `\`${c}\``).join(", ")}`);
  if (b.reasoning) {
    const r = b.reasoning;
    lines.push(
      `- **Deep reasoning:** ${r.layers.length} layer${r.layers.length > 1 ? "s" : ""}, default depth ${r.layers.map((l) => l.steps).join(" + ")} shared-weight deliberation step${r.layers.length > 1 ? "s" : ""} per forward pass. The runtime executes the loop; \`reason(input, { steps?, adaptive?, haltThreshold? })\` returns the per-step halting, delta and state-norm trace (pass a text prompt for language models).`,
    );
  }
  if (b.tokenizer) {
    const t = b.tokenizer;
    lines.push(`- **Tokenizer:** ${t.level}-level, ${t.tokens.length.toLocaleString()} tokens, context ${t.seqLen}, pad token \`${JSON.stringify(t.padToken)}\`${t.eos ? `, end-of-sequence \`${JSON.stringify(t.eos)}\`` : ", no end-of-sequence token"}`);
    const ruleNote = (t.merges ?? 0).toLocaleString();
    lines.push(
      t.level === "word"
        ? "  - Text is lowercased and split into words + punctuation; unknown words become the `<unk>` token (id 0)."
        : t.level === "bpe"
          ? "  - Text is lowercased and split into whitespace / non-whitespace runs. The " +
            ruleNote +
            " learned BPE merge rules are re-applied, so subword tokenisation matches training exactly; unseen characters become the `<unk>` token (id 0)."
          : "  - Text is lowercased; characters outside the vocabulary are skipped.",
    );
    lines.push(`  - Prompts longer than ${t.seqLen} tokens are truncated to the most recent ${t.seqLen}; shorter ones are left-padded.`);
  }
  if (b.preprocessing) {
    const p = b.preprocessing;
    lines.push(`- **Preprocessing:** ${p.columns.length} raw columns → ${p.featureNames.length} features, ${p.normalize} normalisation (training-split statistics), missing values: \`${p.missing}\``);
    lines.push(`  - Columns: ${p.columns.map((c) => `\`${c.name}\`${c.type === "categorical" ? ` (${c.encoding}, ${c.categories.length} categories)` : ""}`).join(", ")}`);
  }
  if (b.image) lines.push(`- **Image input:** ${b.image.width}×${b.image.height}×${b.image.channels}, HWC, values in [${b.image.range.join(", ")}]. ${b.image.description}`);
  return lines.join("\n");
}

function limitationsSection(b: StandaloneBundle): string {
  return b.limitations.map((l) => `- ${l}`).join("\n");
}

function reasoningSample(b: StandaloneBundle): string {
  if (b.tokenizer) return `"q: seven plus eight think:"`;
  return JSON.stringify(round(b.tests.reasoning[0]?.input ?? b.tests.forward[0]?.input ?? []));
}

function reasoningReadmeSection(b: StandaloneBundle): string {
  if (!b.reasoning) return "";
  const layers = b.reasoning.layers;
  const depth = layers.map((l) => l.steps).join(" + ");
  return `## Deep reasoning

This export contains **${layers.length} Deep Reasoning block${layers.length > 1 ? "s" : ""}** with a default depth of **${depth}** shared-weight deliberation step${b.reasoning.defaultSteps === 1 ? "" : "s"}. The loop is part of the runtime, not a description: every forward pass iterates the same learned thought cell and applies its learned halting head. The weights, step code and halting parameters are included in the bundle like every other tensor.

\`\`\`js
const r = model.reason(${reasoningSample(b)}, { steps: ${b.reasoning.defaultSteps} });
console.log(r.traces[0].stepsUsed, r.traces[0].halt);   // per-step halting probabilities
\`\`\`

\`steps\` may be raised to spend more compute at inference (depth is extrapolation beyond the trained value) or lowered to save time; \`adaptive: false\` forces the full depth, and \`haltThreshold\` controls early stopping. The output is still produced by the exported weights — the runtime cannot call a remote model.

`;
}

function jsReadme(b: StandaloneBundle, pkg: string): string {
  return `# ${b.name} — NeuronForge standalone model (JavaScript / TypeScript)

Exported from NeuronForge on ${b.exportedAt}. This package runs the trained model **without NeuronForge**: it contains a dependency-free inference runtime (v${b.runtimeVersion}) that implements every layer of the network, the trained weights, and the input/output handling used during training.

## Files

| File | Purpose |
|---|---|
| \`model.mjs\` | Entry point: \`loadModel()\` |
| \`runtime.mjs\` | Inference engine: matmul, convolution, pooling, RNN, normalisation, activations, tokenizer, sampler |
| \`bundle.mjs\` | Architecture, float32 weights (base64, checksummed), preprocessing/tokenizer, test vectors |
| \`model.d.mts\` | TypeScript declarations |
| \`verify.mjs\` | Replays test vectors computed by NeuronForge |
| \`example.mjs\` | Runnable usage example |
${b.tokenizer ? "| `tokenizer.json` | The exact training vocabulary (also embedded in bundle.mjs) |\n" : ""}${b.preprocessing ? "| `preprocessing.json` | Column encoding + normalisation statistics (also embedded) |\n" : ""}
## Quick start

\`\`\`bash
cd ${pkg}
node verify.mjs     # should print OK
node example.mjs
\`\`\`

\`\`\`js
${jsExample(b, "./model.mjs")}\`\`\`

In a browser, serve the folder over HTTP (ES modules do not load from \`file://\`) and \`import { loadModel } from "./model.mjs"\`. Bundlers (Vite, webpack, esbuild) can import it directly.

## Inputs and outputs

${ioSection(b)}

## API

- \`loadModel()\` / \`loadModelSync()\` → model
- \`model.predict(input, { preprocessed? })\` → task-aware result (\`predicted\`, \`label\`, \`probability\`/\`probabilities\`, \`value\`, \`reconstruction\`, \`top\`, \`raw\` …)
- \`model.predictBatch(inputs)\`
- \`model.forward(values, batchSize)\` → \`{ shape, data: Float32Array }\` raw network output
${b.preprocessing ? "- `model.encode(rawRow)` → the encoded feature vector\n" : ""}${b.tokenizer ? "- `model.tokenize(text)`, `model.detokenize(ids)`, `model.nextToken(text)`\n- `model.generate(prompt, { maxTokens, temperature = 0.7, topK = 50, seed, stop, stopAtEos, includePrompt })`\n- `model.generateDetailed(prompt, { ..., traceReasoning })` → `{ text, fullText, ids, tokens, stoppedBy, settings, reasoning? }`\n" : ""}${b.reasoning ? "- `model.reason(input, { steps?, adaptive?, haltThreshold? })` → `{ shape, data, traces }`, where each trace has `stepsUsed`, `halt`, `cumulativeHalt`, `deltaNorm`, `stateNorm`, `changeNorm`\n- `model.getReasoning()` → the traces from the most recent forward pass\n" : ""}${b.image ? "- `model.predictImage(imageDataOrPixels, { scale, invert })`, `model.imageToInput(...)`\n" : ""}- \`model.verify()\` → \`{ ok, checks }\`
- \`model.info\`, \`model.getWeights()\`

## Numerical behaviour

The runtime mirrors NeuronForge's engine operation for operation: float32 storage, NHWC layout, the same loop order, GELU's tanh approximation, numerically stable sigmoid and softmax, Dropout disabled, BatchNorm using running statistics. In the same JavaScript engine results are normally bit-identical to NeuronForge; across engines \`Math.exp\`/\`Math.tanh\` can differ in the last digit, which is why \`verify()\` uses a tolerance (${b.tests.tolerance.atol} abs + ${b.tests.tolerance.rtol} rel) and reports a sampling divergence as a warning rather than a failure.

On load the runtime validates every layer type, recomputes every tensor shape, checks each weight's shape and checksum (bound to its layer), and rejects unused tensors. Unsupported or corrupted bundles throw descriptive errors instead of producing wrong numbers.

${reasoningReadmeSection(b)}## What this export does not include

${limitationsSection(b)}
`;
}

function pyReadme(b: StandaloneBundle): string {
  return `# ${b.name} — NeuronForge standalone model (Python)

Exported from NeuronForge on ${b.exportedAt}. Runs the trained model without NeuronForge using a self-contained NumPy implementation of every layer (runtime v${b.runtimeVersion}).

## Install

\`\`\`bash
pip install -r requirements.txt   # numpy
python neuronforge_model.py --verify
python neuronforge_model.py --info
${b.task === "language" ? 'python neuronforge_model.py --generate "hello there" --max-tokens 40 --temperature 0.7 --top-k 50' : `python neuronforge_model.py --predict ${round(b.tests.forward[0]?.input ?? []).join(",")}`}${b.reasoning ? (b.tokenizer ? `
python neuronforge_model.py --reason-prompt "q: seven plus eight think:" --reason-steps ${b.reasoning.defaultSteps}` : `
python neuronforge_model.py --reason=${round(b.tests.reasoning[0]?.input ?? []).join(",")} --reason-steps ${b.reasoning.defaultSteps}`) : ""}
\`\`\`

## Usage

\`\`\`python
${pyExample(b)}\`\`\`

## Inputs and outputs

${ioSection(b)}

## Verification and numerical behaviour

\`--verify\` replays test vectors computed by NeuronForge at export time. NumPy's matrix multiply accumulates in a different order from the browser engine, so forward outputs are compared with a tolerance of 1e-4 absolute + 1e-3 relative rather than bit-for-bit. The sampler and random generator are ports of NeuronForge's, so seeded generation usually reproduces the same tokens; a rare divergence (a near-tie flipped by float rounding) is reported as a warning.

**This Python runtime was generated, not executed, by NeuronForge** — the browser cannot run Python. Run \`--verify\` before relying on it.

${b.reasoning ? `## Deep reasoning

This export contains **${b.reasoning.layers.length} Deep Reasoning block${b.reasoning.layers.length > 1 ? "s" : ""}** (default depth ${b.reasoning.layers.map((l) => l.steps).join(" + ")}). \`reason()\` runs the same shared-weight deliberation loop as the browser model and returns the per-step halting trace:

\`\`\`python
r = model.reason(${b.tokenizer ? '"q: seven plus eight think:"' : JSON.stringify(round(b.tests.reasoning[0]?.input ?? b.tests.forward[0]?.input ?? []))})
print(r["traces"][0]["stepsUsed"], r["traces"][0]["halt"])
\`\`\`

Pass \`steps=...\`, \`adaptive=False\` or \`halt_threshold=...\` to override the exported plan for one call. \`generate_detailed(..., trace_reasoning=True)\` attaches the final-token trace to generated text.

` : ""}## What this export does not include

${limitationsSection(b)}
`;
}

const DTS = `export interface PredictionResult {
  task: "binary" | "multiclass" | "regression" | "reconstruction" | "language" | "raw";
  raw: number[];
  predicted?: number;
  label?: string;
  probability?: number;
  probabilities?: number[];
  scores?: number[];
  score?: number;
  value?: number;
  values?: number[];
  named?: Record<string, number>;
  reconstruction?: number[];
  reconstructionError?: number;
  next?: string | null;
  top?: { index?: number; id?: number; label?: string; token?: string; value?: number; probability?: number }[];
  output?: number[];
  shape?: number[];
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  seed?: number;
  stop?: string[];
  stopAtEos?: boolean;
  includePrompt?: boolean;
  /** Include the final forward pass's Deep Reasoning trace in the result. */
  traceReasoning?: boolean;
}

export interface ReasoningTrace {
  layerId: string;
  name: string;
  steps: number;
  stepsUsed: number;
  halted: boolean;
  halt: number[];
  cumulativeHalt: number[];
  deltaNorm: number[];
  stateNorm: number[];
  changeNorm: number[];
}

export interface ReasoningResult {
  shape: number[];
  data: Float32Array;
  traces: ReasoningTrace[];
}

export interface ReasonOptions {
  batch?: number;
  steps?: number;
  adaptive?: boolean;
  haltThreshold?: number;
}

export interface GenerationResult {
  text: string;
  fullText: string;
  ids: number[];
  tokens: string[];
  promptIds: number[];
  stoppedBy: string | null;
  settings: { maxTokens: number; temperature: number; topK: number; seed: number; stop: string[] };
  /** Present for Deep Reasoning models when traceReasoning was requested. */
  reasoning?: ReasoningTrace[];
}

export interface VerifyReport {
  ok: boolean;
  runtimeVersion: string;
  checks: { name: string; status: "pass" | "fail" | "warn"; detail: string }[];
}

export interface ModelInfo {
  name: string;
  task: PredictionResult["task"];
  inputShape: number[];
  outputShape: number[];
  paramCount: number;
  layers: { name: string; type: string; inputShape: number[]; outputShape: number[]; params: Record<string, number> }[];
  classNames: string[] | null;
  targetNames: string[] | null;
  outputsAreProbabilities: boolean;
  tokenizer: { level: "char" | "word" | "bpe"; vocabSize: number; seqLen: number; padToken: string; merges?: number; eos: string | null } | null;
  preprocessing: { kind: "tabular"; columns: string[]; normalize: string; missing: string } | null;
  image: { height: number; width: number; channels: number; layout: "HWC"; range: [number, number]; description: string } | null;
  sampling: { temperature: number; topK: number };
  reasoning: {
    present: true;
    layers: { id: string; name: string; steps: number; hidden: number; adaptive: boolean; haltThreshold: number; minSteps: number; stepCode: boolean; stepCodeScale: number; activation: string }[];
    defaultSteps: number;
    adjustableSteps: true;
    runtimeApi: string;
    description: string;
  } | null;
  runtimeVersion: string;
  exportedAt: string;
  limitations: string[];
}

export type ImageInput = { data: ArrayLike<number>; width: number; height: number; channels?: number } | ArrayLike<number>;

export interface NeuronForgeModel {
  info: ModelInfo;
  forward(input: ArrayLike<number> | number[] | number[][], batchSize?: number): { shape: number[]; data: Float32Array };
  predict(input: unknown, options?: { preprocessed?: boolean }): PredictionResult;
  predictBatch(inputs: unknown[], options?: { preprocessed?: boolean }): PredictionResult[];
  encode(row: Record<string, unknown> | unknown[]): number[];
  tokenize(text: string): number[];
  detokenize(ids: number[]): string;
  nextToken(text: string): PredictionResult;
  generate(prompt: string, options?: GenerateOptions): string;
  generateDetailed(prompt: string, options?: GenerateOptions): GenerationResult;
  /** Run one forward pass through the exported Deep Reasoning loop and return its trace. */
  reason(input: string | ArrayLike<number> | number[], options?: ReasonOptions): ReasoningResult;
  /** Traces from the most recent forward pass. */
  getReasoning(): ReasoningTrace[];
  imageToInput(image: ImageInput, options?: { scale?: number; invert?: boolean }): number[];
  predictImage(image: ImageInput, options?: { scale?: number; invert?: boolean }): PredictionResult;
  verify(): VerifyReport;
  getWeights(): Record<string, Float32Array>;
}

export declare function loadModel(): Promise<NeuronForgeModel>;
export declare function loadModelSync(): NeuronForgeModel;
export declare function createModel(bundle: unknown, options?: { skipChecksums?: boolean }): NeuronForgeModel;
export declare const bundle: unknown;
export declare const NF_RUNTIME_VERSION: string;
export default loadModel;
`;

const VERIFY_MJS = `import { loadModel } from "./model.mjs";

const model = await loadModel();
const report = model.verify();
for (const c of report.checks) console.log("[" + c.status.toUpperCase().padEnd(4) + "] " + c.name + ": " + c.detail);
console.log(report.ok ? "OK" : "FAILED");
if (typeof process !== "undefined" && !report.ok) process.exitCode = 1;
`;

const MODEL_MJS = `import bundle from "./bundle.mjs";
import { createModel, NF_RUNTIME_VERSION } from "./runtime.mjs";

let cached = null;

/** Synchronously build (once) and return the model. */
export function loadModelSync() {
  if (!cached) cached = createModel(bundle);
  return cached;
}

/** Async for API symmetry with loaders that fetch weights; resolves immediately. */
export async function loadModel() {
  return loadModelSync();
}

export { bundle, createModel, NF_RUNTIME_VERSION };
export default loadModel;
`;

/* ---------------------------------------------------------------- files */

export type FileMap = Record<string, string>;

export function runtimeModuleSource(): string {
  return `${RUNTIME_JS}\nexport { createModel, NF_RUNTIME_VERSION };\n`;
}

export function singleFileModule(b: StandaloneBundle): string {
  return (
    `// ${b.name}: NeuronForge standalone model (single file). Exported ${b.exportedAt}.\n` +
    `// Usage: import { loadModel } from "./this-file.mjs"; const model = await loadModel();\n` +
    `${RUNTIME_JS}\n` +
    `var NF_EMBEDDED_BUNDLE = ${bundleJson(b)};\n` +
    `var nfCachedModel = null;\n` +
    `function loadModelSync() { if (!nfCachedModel) nfCachedModel = createModel(NF_EMBEDDED_BUNDLE); return nfCachedModel; }\n` +
    `async function loadModel() { return loadModelSync(); }\n` +
    `export { loadModel, loadModelSync, createModel, NF_RUNTIME_VERSION, NF_EMBEDDED_BUNDLE as bundle };\n` +
    `export default loadModel;\n`
  );
}

export function jsPackageFiles(b: StandaloneBundle, pkg: string): FileMap {
  const files: FileMap = {
    "model.mjs": MODEL_MJS,
    "runtime.mjs": runtimeModuleSource(),
    "bundle.mjs": `// Generated by NeuronForge. Architecture, weights and I/O metadata.\nexport default ${bundleJson(b)};\n`,
    "model.d.mts": DTS,
    "verify.mjs": VERIFY_MJS,
    "example.mjs": jsExample(b, "./model.mjs"),
    "package.json": JSON.stringify(
      {
        name: pkg,
        version: "1.1.0",
        description: `${b.name}: NeuronForge standalone model (${b.task})`,
        type: "module",
        main: "model.mjs",
        types: "model.d.mts",
        exports: { ".": { types: "./model.d.mts", default: "./model.mjs" } },
        scripts: { verify: "node verify.mjs", example: "node example.mjs" },
        license: "UNLICENSED",
      },
      null,
      2,
    ),
    "README.md": jsReadme(b, pkg),
  };
  if (b.tokenizer) files["tokenizer.json"] = JSON.stringify(b.tokenizer, null, 2);
  if (b.preprocessing) files["preprocessing.json"] = JSON.stringify(b.preprocessing, null, 2);
  return files;
}

export function pythonPackageFiles(b: StandaloneBundle): FileMap {
  return {
    "neuronforge_model.py": RUNTIME_PY,
    "bundle.json": bundleJson(b),
    "requirements.txt": "numpy>=1.20\n",
    "example.py": pyExample(b),
    "README.md": pyReadme(b),
  };
}

export function buildExportFiles(b: StandaloneBundle, target: ExportTarget, datasetJson?: string): { filename: string; data: Uint8Array | string; mime: string } {
  const pkg = sanitizeName(b.name);
  if (target === "single") return { filename: `${pkg}.standalone.mjs`, data: singleFileModule(b), mime: "text/javascript" };
  if (target === "bundle") return { filename: `${pkg}.bundle.json`, data: bundleJson(b), mime: "application/json" };
  const tree: Record<string, Uint8Array> = {};
  const add = (prefix: string, files: FileMap) => {
    for (const [k, v] of Object.entries(files)) tree[`${prefix}/${k}`] = strToU8(v);
  };
  if (target === "js" || target === "all") add(target === "all" ? `${pkg}/javascript` : pkg, jsPackageFiles(b, pkg));
  if (target === "python" || target === "all") add(target === "all" ? `${pkg}/python` : `${pkg}-python`, pythonPackageFiles(b));
  if (target === "all") tree[`${pkg}/bundle.json`] = strToU8(bundleJson(b));
  if (datasetJson) tree[`${target === "all" ? pkg : target === "python" ? `${pkg}-python` : pkg}/dataset.json`] = strToU8(datasetJson);
  return { filename: `${pkg}${target === "python" ? "-python" : ""}.zip`, data: zipSync(tree, { level: 6 }), mime: "application/zip" };
}

export function downloadFile(filename: string, data: Uint8Array | string, mime: string): void {
  const blob = new Blob([data as BlobPart], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
