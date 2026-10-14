# NeuronForge

Browser-based neural network laboratory. Models, gradients, losses, and metrics are computed in this page — they are not simulated.

## Stack

- React 19 + TypeScript + Vite + Tailwind CSS 4
- Custom float32 tensor engine with reverse-mode, layer-wise autodiff
- Sequential architectures (Dense, Conv2D, pooling, Dropout, Batch/LayerNorm, Embedding, activations, RNN/LSTM/GRU, residual blocks)
- Optimisers: SGD, momentum, Adam, AdamW
- Losses: MSE, MAE, BCE, categorical CE, sparse CE
- IndexedDB project persistence (localStorage fallback)
- Cooperative training loop (`setTimeout(0)` between batches) so the UI stays responsive
- Optional WebGPU (WGSL) compute backend, verified against the CPU engine before use; the CPU path is the default and the only requirement

## Development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

The Vite config emits a single `dist/index.html` (inlined JS/CSS).

## Custom datasets

- **Draw**: in Dataset Studio, click points onto the canvas per class (2–5 classes) and commit. Points become a real train/val/test split and persist with the project.
- **Formula**: generate `y = f(x)` regression data or `sign(f(x, y))` classification data from a math expression. Expressions are compiled by a safe recursive-descent parser (sin, cos, exp, sqrt, ^, pi, …) — never `eval()`.
- **CSV**: import, select features/targets, choose missing-value/encoding/normalization policy; statistics are fit on the train split only. Imported data now persists with the project (up to ~400k values).

## Standalone Model Export

**Export → Standalone Model Export** produces a model you can *run* outside NeuronForge, not just a file that stores it.

| Target | Contents |
|---|---|
| JavaScript / TypeScript package (.zip) | `model.mjs` (`loadModel()`), `runtime.mjs` (inference engine), `bundle.mjs` (weights + metadata), `model.d.mts`, `verify.mjs`, `example.mjs`, `tokenizer.json` / `preprocessing.json`, `package.json`, README |
| Single-file JavaScript module (.mjs) | Runtime + weights + I/O metadata in one importable file |
| Python package (.zip) | `neuronforge_model.py` (NumPy runtime + CLI), `bundle.json`, `example.py`, `requirements.txt`, README |
| Portable bundle (.json) | Language-independent architecture, checksummed float32 weights, preprocessing, tokenizer and test vectors |
| ONNX | **Not implemented**, and listed as unavailable |

```js
import { loadModel } from "./my-model/model.mjs";
const model = await loadModel();
model.predict([0.25, 0.75]);                       // { predicted, label, probability, raw, ... }
model.generate("hello there", { maxTokens: 100, temperature: 0.7, topK: 50 });
```

**How it works.** The runtimes live in the repo as real source files (`src/export/runtime/runtime.js` and `runtime.py`) and are imported as text (Vite `?raw`). The JavaScript runtime mirrors the engine operation for operation: float32 storage, NHWC layout, the same loop order, GELU's tanh approximation, the stable sigmoid and softmax, Dropout disabled, BatchNorm on running statistics. It implements all 24 layer types the builder can produce (runtime version 1.1.0: the sequence and residual layer types are the 1.1 additions). Input/output handling is task-aware:

- **Classification:** probabilities (when the model ends in sigmoid/softmax) or scores, the predicted class, labels, and a top-5 list.
- **Regression:** named values.
- **Autoencoders:** the reconstruction plus its error.
- **Images:** resizing and scaling into the training layout.
- **Imported tables:** raw rows go through the exact training-split encoding and normalisation (captured at CSV import).
- **Language models:** the training tokenizer, context window and padding, temperature/top-k sampling, the generation loop, and end-of-sequence handling.

**Verification before download.** NeuronForge evaluates the *exact runtime source it ships* and checks:

1. The bundle loads.
2. The weights are bit-identical and attached to the right layers.
3. Every layer's shape matches.
4. Predictions match on real dataset rows plus random inputs.
5. A coverage suite of 45 cases passes (every layer, every fused activation, edge-case inputs).
6. Tokenizer and seeded generation are token-identical.
7. Raw-row preprocessing matches real training rows.
8. Six deliberately broken bundles are all rejected: unsupported layer, corrupted bytes, missing tensor, wrong shape, swapped tensors, unused tensor.
9. The shipped `verify()` passes.

Weight checksums include the layer key, so a tensor moved onto the wrong layer is caught even when the shapes match.

**Not included in this export:** the ability to train (it is inference only), optimiser state, or the dataset (that is an opt-in toggle). The Python runtime is generated but cannot be executed in the browser; run `python neuronforge_model.py --verify` yourself. It replays test vectors computed by NeuronForge.

## Language models

Two text datasets ship with the app:

| | Tiny character LM | Tiny character LM 2 |
|---|---|---|
| Tokenizer | character | **word** |
| Context | 64 characters | 16 words |
| Corpus | ~7.6k chars, hand written | **~150k+ chars**, hand written + templated |
| Vocabulary | ~35 characters | 3,000 word types (frequency-capped) |
| Presets | char MLP, wider char MLP, char RNN | Word LM, Word LM (fast), Word RNN |

The large corpus (`src/data/corpusLarge.ts`) is generated deterministically at runtime from hand-written banks (greetings, Q&A dialogue, narrative, science, computing, advice, history, food, music, sport, health, nature) plus 30 combinatorial sentence templates, then shuffled so topics are interleaved. The Dataset Studio reports its exact size: characters, sentences, word tokens, unique word types, and what fraction of tokens fell outside the vocabulary cap.

**Vocabulary capping is deliberate and reported.** An uncapped word vocabulary makes the output projection enormous; the cap maps rare words to a real `<unk>` token that the model *does* see during training, so inference on unseen words is principled rather than silently wrong. The OOV rate is displayed.

## Sampling

Generation uses real temperature and top-k sampling with a seeded RNG:

- **Temperature** divides the log-probabilities before renormalising. Default **0.7**. At 0 it collapses to greedy argmax.
- **Top-k** keeps only the *k* most likely tokens and renormalises over them. Default **50**.
- The model's output is a softmax, so probabilities are converted back to logits with `log(p)` first — mathematically identical to having had the raw logits, since softmax is shift-invariant.

The self-tests verify: T=0 returns the argmax, k=1 always returns the argmax, the empirical distribution at T=1 with wide k matches the model's probabilities within 4%, low temperature concentrates on the argmax, and k=3 never samples outside the three largest probabilities.

## Language models (architecture notes)

The built-in character LM corpus (`src/data/corpus.ts`) is ~7,600 characters of greetings, Q&A conversations, everyday scenes, science facts, computing explainers, and advice. Models read a **64-character context** (`LM_SEQ_LEN`). The tokenizer is derived from the corpus — `TEXT_CHARS` is the single source of truth for embedding vocab size, output classes, and generation decoding:

- Text presets build their embedding/output sizes from `TEXT_CHARS.length`.
- Selecting the language dataset auto-aligns an existing embedding model's vocab/context (changed layers reinitialize, logged).
- Training refuses to start if embedding vocab, output classes, and tokenizer size disagree.
- Exported model files embed `preprocessing.tokenizer = { chars, seqLen }`; import validates it against the model's output classes.
- Generation skips prompt characters outside the vocabulary (never maps them to id 0) and errors if none remain.

Four sequence architectures are available: character MLPs (flattened context), a genuine tanh **RNN**, and two gated cells — **LSTM** (input/forget/output gates and a cell state) and **GRU** (update/reset gates). All three recurrent layers implement full backpropagation through time with their own gradient caches, and every recurrent parameter is checked against a numerical gradient in the self-tests. Both gated cells support `returnSequences` (per-timestep outputs), a forget-gate bias for the LSTM, an optional GRU reset-gate projection, and packed input projections. Transformers/attention are not implemented.

## Presets

38 presets in six categories (Classification, Regression, Vision, Autoencoder, Text, **Sequence**), from logistic-regression baselines to two-block CNNs, character-level language models, gated-cell language models and RNN/LSTM/GRU sequence forecasters. They include residual MLP and ResNet-pattern presets that stack residual blocks, and char/word language models built on LSTM and GRU cells. Loading a preset also selects its matching dataset and recommended loss, so every preset trains immediately. Autoencoder presets automatically build a reconstruction dataset (targets = inputs).

**Sequence** is a new, non-text category built around the **Wave sequence forecasting** dataset (`wave_seq`): each example is a 12-step window of a noisy two-frequency wave, and the target is the value immediately after it. It includes a baseline RNN forecaster, a deeper RNN forecaster, and a no-recurrence MLP baseline on the identical windows — so you can directly check whether recurrence is earning its complexity on a given task.

## Compute backends

Training and inference run on **one of two backends**, selectable in **Settings → Computation backend**:

| | CPU (default) | WebGPU (opt-in) |
|---|---|---|
| Availability | Every browser | Browsers with `navigator.gpu` and a usable adapter |
| Kernels | JavaScript float32 | 15 WGSL compute shaders |
| Accelerated | everything | matmul (dense, conv inner products, gradient products), conv2d forward / dX / dK via im2col, max & average pooling forward and backward, every fused activation and its backward, softmax, global average pooling, embedding gather/scatter, broadcasting element-wise ops, bias adds, sum/mean reductions |
| Still on the CPU | — | recurrent cells (RNN/LSTM/GRU), Batch/LayerNorm and Dropout fall back per layer; a model with an unsupported layer type keeps those layers on the CPU and synchronises around the device segments |
| Selection | **default** | `Auto` (uses the device once a model is big enough to benefit), `WebGPU` (always prefer it), and `CPU only` |

Three independent things have to be true before the device is allowed to train a model, and each is checked rather than assumed:

1. **The WGSL compiles and pipelines build.** Every kernel's `getCompilationInfo()` is inspected and a compute pipeline plus bind group is created in the development harness (`getCompilationInfo`, not `createComputePipeline`, is the gate — an invalid module does not throw there).
2. **The kernel arithmetic is right.** `src/engine/webgpu/sim.ts` is a TypeScript mirror of the kernels that reads *the exact parameter buffers the device builds* (`src/engine/webgpu/plan.ts`) and re-applies the grid geometry (including the `gid.y * 65535 + gid.x` fold and the 16×16 matmul tiling). The self-tests run every kernel through the mirror and diff it against the CPU reference device, so index, padding and tiling mistakes show up on a machine with no GPU. Its limit is stated in the file: it validates the algorithms and the parameter layout, not the WGSL text — which is why (1) and (3) exist.
3. **The device reproduces the CPU engine.** *Verify device* runs 64 op cases (matmul with every transpose combination, conv forward/dX/dK over four geometries, pooling, softmax, reductions, broadcasting, embedding, reshape, zeros) on the real device and diffs them against the same operations on the CPU reference, then the accelerator runs whole-model forward and backward passes and five Adam steps, comparing weights and losses with a pure-CPU run.

Nothing is trusted on the strength of "it should work": the verification runs once per device — automatically before the first accelerated epoch — and the report is shown in Settings. If it fails, if `navigator.gpu` is missing, or if the device is lost mid-run, training continues on the CPU and says why in the log. Weights, gradients, losses, metrics and the optimiser stay exactly as they are; the device only replaces the arithmetic. Same seed, same numbers — only faster.

**Where the acceleration actually pays.** Every device op is an async round trip, so a batch costs a few submissions and one readback. Under ~20,000 trainable parameters the CPU engine is genuinely faster, so `Auto` keeps small models on the CPU (the log says so) and an explicit `WebGPU` preference overrides it.

## Residual blocks

Sequential architectures were the biggest structural limit, so two residual block layer types are now first-class builder entries (`residual_dense`, `residual_conv`):

- `y = act_last(x + scale · F(x))`, where `F` is a two-layer branch (two Dense layers, or two Conv2D layers with the same kernel/shape policy as a ResNet basic block).
- The shortcut is the identity when the block's output shape matches its input shape.
- When the shape changes (`units` ≠ input width, or `filters` ≠ input channels), the shortcut becomes a linear/1×1 projection, which is what makes channel-changing blocks legal in a chain.
- `scale` (default 1, typical 1/√2 for deep stacks) damps the residual branch.
- Shape inference, validation, parameter counting, serialisation, both export runtimes and the coverage suite understand the two new types, so a residual model exports and verifies like any other.

Free-form graphs (multiple inputs, arbitrary re-wiring) remain **not implemented** — see the limitations below.

## Using the lab

1. A starter project **XOR Lab** loads automatically.
2. Open **Training Lab** and press **Start training**. Loss is mean binary cross-entropy on the four XOR points; weights update with Adam.
3. Open **Inspector** to read the live `W` / `b` tensors.
4. **Settings → Run tests** executes the mathematical self-test suite in this page: matmul, losses, numerical gradients for every layer (including LSTM/GRU backpropagation through time and the residual blocks), XOR learning, serialisation round-trip, the CSV parser, sampling, the export runtimes against the engine, the device op surface, the WGSL kernel arithmetic against the CPU reference, and accelerator-vs-CPU forward, backward and optimiser agreement.

## Honest limitations

- Sequential graphs only (no free-form skip connections or multi-input graphs).
- ONNX export is **not** implemented. The default compute path is CPU JavaScript float32; the WebGPU path is opt-in, verified before use, and only accelerates the layer types and operations listed above.
- The WebGPU path has been validated with a WGSL validator, a kernel-by-kernel numerical mirror and the device verification suite, but the execution itself has only run against the CPU reference device in development (no GPU in the development sandbox): the first real-GPU run is a browser run. That is exactly why the device must pass verification before it is used, and why the CPU path remains the default.
- Residual blocks are a layer type (`residual_dense`, `residual_conv`), not free-form graphs: a block wraps a two-layer branch with a shortcut, but the model is still a chain.
- Transformers / subword tokenisers are **not** implemented. The character MLP preset is a real next-token model, not a transformer.
- Identical seeds are not bit-identical across browsers or future engine changes.
