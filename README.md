# NeuronForge

Browser-based neural network laboratory. Models, gradients, losses, and metrics are computed in this page — they are not simulated.

## Stack

- React 19 + TypeScript + Vite + Tailwind CSS 4
- Custom float32 tensor engine with reverse-mode, layer-wise autodiff
- Sequential architectures (Dense, Conv2D, pooling, Dropout, Batch/LayerNorm, Embedding, activations, RNN/LSTM/GRU, residual blocks)
- Optimisers: SGD, momentum, Adam, AdamW
- Losses: MSE, MAE, BCE, categorical CE, sparse CE
- IndexedDB project persistence (localStorage fallback)
- Cooperative training loop (`setTimeout(0)` between batches) so the UI stays responsive; gradient-norm diagnostics are reduced every 10 steps by default (and at epoch boundaries), avoiding an extra full parameter-memory scan every update without changing gradients or optimizer results
- Optional WebGPU (WGSL) and TensorFlow.js compute backends, verified against the CPU engine before use; the custom CPU path remains available everywhere

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

## TensorFlow.js backend

TensorFlow.js 4.22 is bundled locally with the app — no account, API key, CDN or remote inference service is needed.

1. Open **Settings → Computation backend → Tensor backend**.
2. Select **TensorFlow.js — WebGL with TFJS CPU fallback**.
3. Click **Verify device**, or start training to verify automatically. All 64 operation checks must agree with the custom CPU engine before TensorFlow.js is used.
4. Train normally. The Training Lab resource card shows **TensorFlow.js · webgl** or **TensorFlow.js · cpu**, and the layer-placement counts. The selection persists across reloads.

The adapter (`src/engine/gpu/tensorflow.ts`) implements the same operation interface as the WGSL device: matrix multiplication, convolution and its input/kernel gradients, pooling and its gradients, elementwise operations, activations (including the engine's tanh GELU approximation), softmax, reductions and embedding gather/scatter. TensorFlow.js selects WebGL when available, otherwise its CPU backend. Initialization or verification failures leave training on the custom CPU engine. This explicit selection does not change Auto's existing WebGPU/CPU policy.

The existing CPU model weights remain authoritative. Losses, optimizers, diagnostics, checkpoints and standalone exports retain their existing formats and behavior. Recurrent cells, normalization, dropout, attention and transformer layers still use the custom CPU engine through per-layer fallback. Evaluation/Inspector and exported inference runtimes also remain unchanged. This is a compute-backend integration, **not** TensorFlow.js Layers-model import/export or pretrained-model loading. Performance depends on model size and browser; per-batch weight/gradient transfers can make small models slower than the custom CPU engine.

Run the integration tests (TFJS CPU, suitable for headless Node environments):

```bash
npm test
npm run typecheck
npm run build
```

Tests cover all 64 operations, multi-step gradient/optimizer parity, real Trainer epochs with validation, embedding edge cases, buffer ownership, repeated verification/inference cleanup, and backend selection. Browser-specific WebGL kernels are checked on the actual device by **Verify device** before use.

## Custom datasets

- **Draw**: in Dataset Studio, click points onto the canvas per class (2–5 classes) and commit. Points become a real train/val/test split and persist with the project.
- **Formula**: generate `y = f(x)` regression data or `sign(f(x, y))` classification data from a math expression. Expressions are compiled by a safe recursive-descent parser (sin, cos, exp, sqrt, ^, pi, …) — never `eval()`.
- **CSV**: import, select features/targets, choose missing-value/encoding/normalization policy; statistics are fit on the train split only. Files are streamed through an incremental parser in chunks (the same parser also handles pasted strings, and both agree on quoting, escaped quotes and CRLF newlines — verified at every chunk boundary in the self-tests), so multi-megabyte exports never have to be held in memory twice. Progress is shown while reading, up to 250,000 rows are kept, and imported data persists with the project (up to ~400k values).
- **Images**: import files or a whole folder (up to 4,000 images). They are rasterised in the page, resized to the model's input size (bilinear, or nearest-neighbour to match the exported runtimes), converted to grayscale or RGB, and scaled to [0, 1] or [-1, 1]. Labels come from the file-name prefix, the folder name, or a single class for autoencoders. Augmentation is applied **to the training split only**: horizontal/vertical flips, 90° rotations, small arbitrary rotations, translations, brightness/contrast jitter and Gaussian noise, with a configurable number of extra copies per image. A thumbnail grid previews the result before you commit it.
- **Synthetic generators**: ten seeded families beyond the formula and draw tools — Gaussian clusters, concentric rings, checkerboard, *n*-bit parity, a two-feature wave surface, waveform and random-walk forecasting windows, a 2D manifold autoencoder, imbalanced classes and waveform classification. Each exposes its parameters (class count, separation, jitter, frequencies, window length, …) with a live scatter preview and declares its input shape, task and loss, so the dataset arrives ready to train. Everything is deterministic for a given seed.

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

**How it works.** The runtimes live in the repo as real source files (`src/export/runtime/runtime.js` and `runtime.py`) and are imported as text (Vite `?raw`). The JavaScript runtime mirrors the engine operation for operation: float32 storage, NHWC layout, the same loop order, GELU's tanh approximation, the stable sigmoid and softmax, Dropout disabled, BatchNorm on running statistics. It implements all 26 layer types the builder can produce (runtime version 1.3.0: 1.1 added the sequence and residual layer types, 1.2 the attention and transformer-block layers, 1.3 the subword (BPE) tokenizer). Input/output handling is task-aware:

- **Classification:** probabilities (when the model ends in sigmoid/softmax) or scores, the predicted class, labels, and a top-5 list.
- **Regression:** named values.
- **Autoencoders:** the reconstruction plus its error.
- **Images:** resizing and scaling into the training layout.
- **Imported tables:** raw rows go through the exact training-split encoding and normalisation (captured at CSV import).
- **Language models:** the training tokenizer — character, word or learned BPE merges, whichever the dataset uses — the context window and padding, temperature/top-k sampling, the generation loop, and end-of-sequence handling. BPE bundles carry the merge rules, so the runtime re-tokenises prompts exactly as the app did.

**Verification before download.** NeuronForge evaluates the *exact runtime source it ships* and checks:

1. The bundle loads.
2. The weights are bit-identical and attached to the right layers.
3. Every layer's shape matches.
4. Predictions match on real dataset rows plus random inputs.
5. A coverage suite of 52 cases passes (every layer, every fused activation, edge-case inputs).
6. Tokenizer and seeded generation are token-identical.
7. Raw-row preprocessing matches real training rows.
8. Six deliberately broken bundles are all rejected: unsupported layer, corrupted bytes, missing tensor, wrong shape, swapped tensors, unused tensor.
9. The shipped `verify()` passes.

Weight checksums include the layer key, so a tensor moved onto the wrong layer is caught even when the shapes match.

**Not included in this export:** the ability to train (it is inference only), optimiser state, or the dataset (that is an opt-in toggle). The Python runtime is generated but cannot be executed in the browser; run `python neuronforge_model.py --verify` yourself. It replays test vectors computed by NeuronForge.

## Language models

NeuronForge is a from-scratch educational trainer, not a pretrained assistant service. It can train next-token models and demonstrate how capacity, tokenisation, and optimisation affect loss, but the bundled corpus is small and partly generated. Models trained here should **not** be expected to answer arbitrary questions, maintain reliable multi-turn conversations, or match a production language model. Useful general-purpose assistant quality requires substantially larger, curated instruction/conversation data and compute than a browser tab can provide.

Four text datasets ship with the app, covering all three tokenizer levels — one token per character, one token per word, and learned subword merges:

| | Tiny character LM | Tiny character LM 2 | Tiny BPE LM | Tiny BPE LM 2 |
|---|---|---|---|---|
| Tokenizer | character | **word** | **BPE** (learned merges) | **BPE** (learned merges) |
| Context | 64 characters | 16 words | 64 subword tokens | 16 subword tokens |
| Corpus | ~7.6k chars, hand written | **~150k+ chars**, hand written + templated | ~7.6k chars, hand written | **~150k+ chars**, hand written + templated |
| Vocabulary | ~35 characters | 3,000 word types (frequency-capped) | ~320 subword tokens (64–2048) | ~512 subword tokens (64–2048) |
| Presets | char MLP, wider char MLP, char RNN | Word LM, Word LM (fast), Word RNN | Subword LSTM | Subword transformer (two sizes) |

The BPE tokenizer (`src/data/bpe.ts`) **learns** its merges from the corpus at load time: it counts adjacent character pairs inside non-whitespace runs, repeatedly merges the most frequent pair, and stores the rule list with the dataset and in exported bundles. Training is deterministic (ties are broken by pair order), decoding is lossless — re-joining the tokens reproduces the lower-cased corpus exactly — and the merge count, vocabulary size, characters per token and unknown-character rate are reported in the Dataset Studio. The subword vocabulary size is adjustable per dataset (64–2048 tokens); whitespace is never merged into a token, so a decoded window keeps its original spacing.

The large corpus (`src/data/corpusLarge.ts`) is generated deterministically at runtime from hand-written banks (greetings, Q&A dialogue, narrative, science, computing, advice, history, food, music, sport, health, nature) plus 30 combinatorial sentence templates, then shuffled so topics are interleaved. The Dataset Studio reports its exact size: characters, sentences, word tokens, unique word types, and what fraction of tokens fell outside the vocabulary cap.

**Vocabulary capping is deliberate and reported.** An uncapped word vocabulary makes the output projection enormous; the cap maps rare words to a real `<unk>` token that the model *does* see during training, so inference on unseen words is principled rather than silently wrong. The OOV rate is displayed.

Two additional large-corpus presets are available in **Model Builder → Load preset → Text**: **Forge Text Large** (4 causal transformer blocks, width 64) and **Forge Text XL** (6 blocks, width 128). They use the large-corpus BPE tokenizer and train from scratch; XL has million-plus parameters. Preset loading selects a conservative, hardware-aware starter batch and caps the initial run at three epochs so users can verify that their device stays responsive before scaling up. In Training Lab, the **Fit batch** recommendation estimates a conservative memory budget and lowers the batch size when needed. It preserves model capacity and weights, though smaller batches do change gradient noise. Browser RAM availability is not exposed reliably, so the estimate is advisory—not a promise that a model will fit.

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

Six sequence-capable architectures are available: character MLPs (flattened context), a genuine tanh **RNN**, two gated cells — **LSTM** (input/forget/output gates and a cell state) and **GRU** (update/reset gates) — plus **multi-head attention** and a full **transformer block**. All of them implement their own backward pass with explicit caches, and every parameter is checked against a numerical gradient in the self-tests. The gated cells support `returnSequences`, a forget-gate bias for the LSTM, an optional GRU reset-gate projection and packed input projections.

**Attention** (`src/engine/attention.ts`) is scaled dot-product self-attention over a sequence: `softmax(QKᵀ/√d_head)V` per head, concatenated and projected, with optional causal masking (masked positions are exactly `-Infinity` before the softmax and their gradients are hard-zeroed), an optional learned input projection when the model width differs from the incoming width, pooling to the last or mean position, and attention dropout that is disabled at inference. **`transformer_block`** packages the modern recipe: pre-LayerNorm `x + MHA(LN(x))` followed by `x + FFN(LN(x))` with a fused GELU/ReLU/SiLU feed-forward. Both layers stay on the CPU engine inside the WebGPU path (synchronised around, verified in the self-tests), and the exported JavaScript and NumPy runtimes mirror them operation for operation.

## Presets

51 presets in six categories (Classification, Regression, Vision, Autoencoder, Text, **Sequence**), from logistic-regression baselines to two-block CNNs, character-level language models, gated-cell language models and RNN/LSTM/GRU sequence forecasters. They include residual MLP and ResNet-pattern presets that stack residual blocks, char/word/subword language models built on LSTM and GRU cells, and attention presets: a causal self-attention LM, transformer language models over both the character and subword vocabularies, and an attention forecaster for the sequence dataset. Loading a preset also selects its matching dataset and recommended loss, so every preset trains immediately. Autoencoder presets automatically build a reconstruction dataset (targets = inputs).

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
- Attention and transformer blocks are *layer types inside a chain*, not a graph: no free-form skip connections, multi-branch encoders, positional-encoding layers or multi-input models. The transformer block is one layer containing attention plus a feed-forward network, and the model is still sequential.
- Tokenisation covers character, word and corpus-learned BPE levels. There is no WordPiece/unigram, no byte-level BPE, no training on user text, and word vocabularies remain frequency-capped with a reported `<unk>` rate.
- Identical seeds are not bit-identical across browsers or future engine changes.
