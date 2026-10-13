# NeuronForge

Browser-based neural network laboratory. Models, gradients, losses, and metrics are computed in this page — they are not simulated.

## Stack

- React 19 + TypeScript + Vite + Tailwind CSS 4
- Custom float32 tensor engine with reverse-mode, layer-wise autodiff
- Sequential architectures (Dense, Conv2D, pooling, Dropout, Batch/LayerNorm, Embedding, activations)
- Optimisers: SGD, momentum, Adam, AdamW
- Losses: MSE, MAE, BCE, categorical CE, sparse CE
- IndexedDB project persistence (localStorage fallback)
- Cooperative training loop (`setTimeout(0)` between batches) so the UI stays responsive

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

**How it works.** The runtimes live in the repo as real source files (`src/export/runtime/runtime.js` and `runtime.py`) and are imported as text (Vite `?raw`). The JavaScript runtime mirrors the engine operation for operation: float32 storage, NHWC layout, the same loop order, GELU's tanh approximation, the stable sigmoid and softmax, Dropout disabled, BatchNorm on running statistics. It implements all 20 layer types the builder can produce. Input/output handling is task-aware:

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
5. A coverage suite of 30+ cases passes (every layer, every fused activation, edge-case inputs).
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

Two sequence architectures are available: character MLPs (flattened context) and a genuine tanh **RNN** with full backpropagation through time (new `rnn` layer, gradient-checked in the self-tests). Transformers/attention are not implemented.

## Presets

28 presets in six categories (Classification, Regression, Vision, Autoencoder, Text, **Sequence**), from logistic-regression baselines to two-block CNNs, character-level language models and RNN sequence forecasters. Loading a preset also selects its matching dataset and recommended loss, so every preset trains immediately. Autoencoder presets automatically build a reconstruction dataset (targets = inputs).

**Sequence** is a new, non-text category built around the **Wave sequence forecasting** dataset (`wave_seq`): each example is a 12-step window of a noisy two-frequency wave, and the target is the value immediately after it. It includes a baseline RNN forecaster, a deeper RNN forecaster, and a no-recurrence MLP baseline on the identical windows — so you can directly check whether recurrence is earning its complexity on a given task.

## Using the lab

1. A starter project **XOR Lab** loads automatically.
2. Open **Training Lab** and press **Start training**. Loss is mean binary cross-entropy on the four XOR points; weights update with Adam.
3. Open **Inspector** to read the live `W` / `b` tensors.
4. **Settings → Run tests** executes the mathematical self-test suite (matmul, numerical gradients, XOR learning, serialisation round-trip, CSV parser).

## Honest limitations

- Sequential graphs only (no free-form skip connections or multi-input graphs).
- CPU JavaScript float32 only. WebGPU and ONNX are **not** implemented.
- Transformers / subword tokenisers are **not** implemented. The character MLP preset is a real next-token model, not a transformer.
- Identical seeds are not bit-identical across browsers or future engine changes.
