import type { Architecture, InitMethod } from "./types";
import { makeLayerDef } from "./layers";
import { uid } from "./types";
import type { LayerType } from "./types";
import { LM_SEQ_LEN, TEXT_VOCAB_SIZE } from "../data/corpus";
import { LM2_SEQ_LEN } from "../data/corpusLarge";
import { SEQ_WINDOW } from "../data/datasets";

export interface Preset {
  id: string;
  name: string;
  category: "Classification" | "Regression" | "Vision" | "Autoencoder" | "Text" | "Sequence";
  description: string;
  datasetHint: string;
  arch: Architecture;
}

function arch(inputShape: number[], layers: Architecture["layers"], seed = 42, initMethod: InitMethod = "he_normal"): Architecture {
  return { inputShape, inputName: "input", layers, seed, initMethod };
}

function dense(units: number, name?: string): ReturnType<typeof makeLayerDef> {
  return makeLayerDef("dense", { name: name ?? `dense_${units}`, config: { units, useBias: true, activation: "none" } });
}

function act(type: LayerType, name?: string): ReturnType<typeof makeLayerDef> {
  return makeLayerDef(type, name ? { name } : undefined);
}

export const PRESETS: Preset[] = [
  /* ---------------- Classification ---------------- */
  {
    id: "logistic",
    name: "Logistic regression",
    category: "Classification",
    description: "Single dense unit + sigmoid. A linear baseline — it will fail on XOR and circles, which is the point.",
    datasetHint: "gauss",
    arch: arch([2], [dense(1, "logit"), act("sigmoid")], 42, "xavier_normal"),
  },
  {
    id: "xor",
    name: "Tiny binary classifier",
    category: "Classification",
    description: "One hidden ReLU layer of 8 units then sigmoid. Smallest net that reliably learns XOR.",
    datasetHint: "xor",
    arch: arch([2], [dense(8, "hidden"), act("relu"), dense(1, "out"), act("sigmoid")]),
  },
  {
    id: "moons",
    name: "Two-moons MLP",
    category: "Classification",
    description: "16→16 ReLU + sigmoid. Smooth nonlinear boundary for crescent-shaped classes.",
    datasetHint: "moons",
    arch: arch([2], [dense(16, "h1"), act("relu"), dense(16, "h2"), act("relu"), dense(1, "out"), act("sigmoid")]),
  },
  {
    id: "deep-binary",
    name: "Deep ReLU binary net",
    category: "Classification",
    description: "64→64→32 ReLU stack. Overkill for toy data — useful for watching gradient norms per depth.",
    datasetHint: "circles",
    arch: arch([2], [
      dense(64, "h1"), act("relu"),
      dense(64, "h2"), act("relu"),
      dense(32, "h3"), act("relu"),
      dense(1, "out"), act("sigmoid"),
    ]),
  },
  {
    id: "dropout-reg",
    name: "Dropout-regularized classifier",
    category: "Classification",
    description: "Wide layers with 30% dropout between them. Compare train vs validation loss with and without dropout.",
    datasetHint: "moons",
    arch: arch([2], [
      dense(64, "h1"), act("relu"),
      makeLayerDef("dropout", { name: "drop1", config: { rate: 0.3 } }),
      dense(64, "h2"), act("relu"),
      makeLayerDef("dropout", { name: "drop2", config: { rate: 0.3 } }),
      dense(1, "out"), act("sigmoid"),
    ]),
  },
  {
    id: "multiclass",
    name: "Multiclass MLP",
    category: "Classification",
    description: "Two hidden ReLU layers and softmax over 3 classes.",
    datasetHint: "spiral",
    arch: arch([2], [dense(32, "h1"), act("relu"), dense(16, "h2"), act("relu"), dense(3, "logits"), act("softmax")]),
  },
  {
    id: "spiral-deep",
    name: "Deep spiral solver",
    category: "Classification",
    description: "Three 64-unit ReLU layers + softmax. Enough capacity to carve interlocking spirals.",
    datasetHint: "spiral",
    arch: arch([2], [
      dense(64, "h1"), act("relu"),
      dense(64, "h2"), act("relu"),
      dense(64, "h3"), act("relu"),
      dense(3, "logits"), act("softmax"),
    ]),
  },
  {
    id: "batchnorm-clf",
    name: "BatchNorm classifier",
    category: "Classification",
    description: "Dense → BatchNorm → ReLU blocks. Watch the running mean/var buffers in the Inspector.",
    datasetHint: "blobs",
    arch: arch([2], [
      dense(32, "h1"), makeLayerDef("batchnorm", { name: "bn1" }), act("relu"),
      dense(32, "h2"), makeLayerDef("batchnorm", { name: "bn2" }), act("relu"),
      dense(3, "logits"), act("softmax"),
    ]),
  },
  {
    id: "gelu-modern",
    name: "GELU + LayerNorm MLP",
    category: "Classification",
    description: "Transformer-style feed-forward blocks: Dense → GELU → LayerNorm. Modern smooth activations.",
    datasetHint: "spiral",
    arch: arch([2], [
      dense(48, "ff1"), act("gelu"), makeLayerDef("layernorm", { name: "ln1" }),
      dense(48, "ff2"), act("gelu"), makeLayerDef("layernorm", { name: "ln2" }),
      dense(3, "logits"), act("softmax"),
    ]),
  },
  {
    id: "noise-robust-clf",
    name: "Noise-robust classifier",
    category: "Classification",
    description: "Dense → BatchNorm → LeakyReLU → Dropout blocks, combining three regularisers at once. Built for the noisy-XOR dataset, where a plain MLP tends to overfit the noise.",
    datasetHint: "xor_noise",
    arch: arch([2], [
      dense(32, "h1"), makeLayerDef("batchnorm", { name: "bn1" }), act("leakyrelu", "lrelu1"),
      makeLayerDef("dropout", { name: "drop1", config: { rate: 0.2 } }),
      dense(32, "h2"), makeLayerDef("batchnorm", { name: "bn2" }), act("leakyrelu", "lrelu2"),
      dense(1, "out"), act("sigmoid"),
    ]),
  },

  /* ---------------- Regression ---------------- */
  {
    id: "linreg",
    name: "Linear regression",
    category: "Regression",
    description: "One dense unit, no activation. Learns slope and intercept — check W and b in the Inspector against y = 1.7x − 0.3.",
    datasetHint: "linear",
    arch: arch([1], [dense(1, "wx_plus_b")], 42, "xavier_normal"),
  },
  {
    id: "regression",
    name: "Tanh regression net",
    category: "Regression",
    description: "Two 32-unit tanh layers. Smooth function approximation for the noisy sine.",
    datasetHint: "sine",
    arch: arch([1], [dense(32, "h1"), act("tanh"), dense(32, "h2"), act("tanh"), dense(1, "out")], 42, "xavier_normal"),
  },
  {
    id: "silu-reg",
    name: "Deep SiLU regression",
    category: "Regression",
    description: "64→64 SiLU stack. Compare convergence speed against the tanh net on the same data.",
    datasetHint: "sine",
    arch: arch([1], [dense(64, "h1"), act("silu"), dense(64, "h2"), act("silu"), dense(1, "out")]),
  },
  {
    id: "poly-leaky",
    name: "Polynomial fitter",
    category: "Regression",
    description: "LeakyReLU layers for the cubic dataset. Piecewise-linear approximation of a polynomial.",
    datasetHint: "poly",
    arch: arch([1], [dense(32, "h1"), act("leakyrelu"), dense(32, "h2"), act("leakyrelu"), dense(1, "out")]),
  },
  {
    id: "layernorm-reg",
    name: "GELU + LayerNorm regression net",
    category: "Regression",
    description: "Dense → GELU → LayerNorm blocks, the same feed-forward pattern as the classification preset of the same style, applied to function approximation.",
    datasetHint: "poly",
    arch: arch(
      [1],
      [
        dense(48, "ff1"), act("gelu"), makeLayerDef("layernorm", { name: "ln1" }),
        dense(48, "ff2"), act("gelu"), makeLayerDef("layernorm", { name: "ln2" }),
        dense(1, "out"),
      ],
      42,
      "xavier_normal",
    ),
  },

  /* ---------------- Vision ---------------- */
  {
    id: "image",
    name: "Tiny image classifier",
    category: "Vision",
    description: "Conv2D(8) → ReLU → MaxPool → Flatten → Dense softmax for the 8×8 patterns.",
    datasetHint: "patterns",
    arch: arch([8, 8, 1], [
      makeLayerDef("conv2d", { name: "conv1", config: { filters: 8, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" } }),
      act("relu"),
      makeLayerDef("maxpool2d", { name: "pool1", config: { poolSize: 2, stride: 2 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(4, "logits"),
      act("softmax"),
    ]),
  },
  {
    id: "cnn-deep",
    name: "Two-block CNN",
    category: "Vision",
    description: "Conv(8)→Pool→Conv(16)→Pool→Flatten→Dense. Classic LeNet-style downsampling pyramid.",
    datasetHint: "patterns",
    arch: arch([8, 8, 1], [
      makeLayerDef("conv2d", { name: "conv1", config: { filters: 8, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" } }),
      act("relu"),
      makeLayerDef("maxpool2d", { name: "pool1", config: { poolSize: 2, stride: 2 } }),
      makeLayerDef("conv2d", { name: "conv2", config: { filters: 16, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" } }),
      act("relu"),
      makeLayerDef("maxpool2d", { name: "pool2", config: { poolSize: 2, stride: 2 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(4, "logits"),
      act("softmax"),
    ]),
  },
  {
    id: "cnn-gap",
    name: "Global-average-pool CNN",
    category: "Vision",
    description: "Two convolutions then GlobalAvgPool instead of Flatten — far fewer dense parameters.",
    datasetHint: "patterns",
    arch: arch([8, 8, 1], [
      makeLayerDef("conv2d", { name: "conv1", config: { filters: 16, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" } }),
      act("relu"),
      makeLayerDef("conv2d", { name: "conv2", config: { filters: 16, kernelSize: 3, stride: 1, padding: "same", useBias: true, activation: "none" } }),
      act("relu"),
      makeLayerDef("gap2d", { name: "gap" }),
      dense(4, "logits"),
      act("softmax"),
    ]),
  },
  {
    id: "cnn-batchnorm",
    name: "Conv + BatchNorm classifier",
    category: "Vision",
    description: "A single Conv2D followed by BatchNorm and LeakyReLU before pooling — normalises the feature maps instead of just the activations, unlike the other Vision presets.",
    datasetHint: "patterns",
    arch: arch([8, 8, 1], [
      makeLayerDef("conv2d", { name: "conv1", config: { filters: 12, kernelSize: 3, stride: 1, padding: "same", useBias: false, activation: "none" } }),
      makeLayerDef("batchnorm", { name: "bn1" }),
      act("leakyrelu", "lrelu1"),
      makeLayerDef("maxpool2d", { name: "pool1", config: { poolSize: 2, stride: 2 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(4, "logits"),
      act("softmax"),
    ]),
  },

  /* ---------------- Autoencoder ---------------- */
  {
    id: "autoencoder",
    name: "Autoencoder (2D bottleneck)",
    category: "Autoencoder",
    description: "Compress 2D inputs through a 2-unit bottleneck and reconstruct. Train with MSE, inputs as targets.",
    datasetHint: "circles",
    arch: arch([2], [
      dense(8, "enc1"), act("relu"),
      dense(2, "bottleneck"), act("relu"),
      dense(8, "dec1"), act("relu"),
      dense(2, "recon"),
    ]),
  },
  {
    id: "deep-ae",
    name: "Deep autoencoder",
    category: "Autoencoder",
    description: "16→8→2→8→16 hourglass with tanh. Inspect the bottleneck activations to see the learned code.",
    datasetHint: "moons",
    arch: arch([2], [
      dense(16, "enc1"), act("tanh"),
      dense(8, "enc2"), act("tanh"),
      dense(2, "code"),
      dense(8, "dec1"), act("tanh"),
      dense(16, "dec2"), act("tanh"),
      dense(2, "recon"),
    ], 42, "xavier_normal"),
  },
  {
    id: "ae-layernorm",
    name: "LayerNorm autoencoder",
    category: "Autoencoder",
    description: "Normalises the code at every layer with LayerNorm before the GELU nonlinearity — a different regularisation strategy from the plain and deep autoencoders above.",
    datasetHint: "blobs",
    arch: arch(
      [2],
      [
        dense(12, "enc1"), makeLayerDef("layernorm", { name: "ln_enc1" }), act("gelu"),
        dense(3, "bottleneck"), makeLayerDef("layernorm", { name: "ln_bottleneck" }), act("gelu"),
        dense(12, "dec1"), makeLayerDef("layernorm", { name: "ln_dec1" }), act("gelu"),
        dense(2, "recon"),
      ],
      42,
      "xavier_normal",
    ),
  },

  /* ---------------- Text ---------------- */
  {
    id: "charlm",
    name: "Character MLP language model",
    category: "Text",
    description: `Embeds the previous ${LM_SEQ_LEN} characters, flattens them, and predicts the next character with a softmax over the ${TEXT_VOCAB_SIZE}-character vocabulary. A real next-token MLP — it sees word beginnings and sentence fragments, but has no recurrence.`,
    datasetHint: "tiny_text",
    arch: arch([LM_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: TEXT_VOCAB_SIZE, embeddingDim: 24 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(128, "h1"), act("tanh"),
      dense(TEXT_VOCAB_SIZE, "logits"), act("softmax"),
    ]),
  },
  {
    id: "charlm-big",
    name: "Wider character LM",
    category: "Text",
    description: "32-dim embeddings and a 192-unit GELU hidden layer over the full context window. Lower loss, slower epochs.",
    datasetHint: "tiny_text",
    arch: arch([LM_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: TEXT_VOCAB_SIZE, embeddingDim: 32 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(192, "h1"), act("gelu"),
      dense(TEXT_VOCAB_SIZE, "logits"), act("softmax"),
    ]),
  },
  {
    id: "word-lm",
    name: "Word LM (Tiny character LM 2)",
    category: "Text",
    description: `Word-level model for the massive corpus. Embeds ${LM2_SEQ_LEN} previous words, flattens, one dense hidden layer, softmax over the frequency-capped vocabulary. Word tokens need a larger vocabulary than characters, so the output projection dominates the parameter count.`,
    datasetHint: "tiny_text_2",
    arch: arch([LM2_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: 3000, embeddingDim: 24 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(96, "h1"), act("gelu"),
      dense(3000, "logits"), act("softmax"),
    ]),
  },
  {
    id: "word-lm-fast",
    name: "Word LM (fast)",
    category: "Text",
    description: "Same word-level task with a 16-dim embedding and a 64-unit hidden layer. Trains several batches per second in a browser tab — start here before scaling up.",
    datasetHint: "tiny_text_2",
    arch: arch([LM2_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: 3000, embeddingDim: 16 } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(64, "h1"), act("gelu"),
      dense(3000, "logits"), act("softmax"),
    ]),
  },
  {
    id: "word-rnn",
    name: "Word RNN (Tiny character LM 2)",
    category: "Text",
    description: `A real sequence model on the word corpus: Embedding → tanh RNN over ${LM2_SEQ_LEN} word steps → softmax. The hidden state carries context forward instead of flattening it. Expect slow epochs and watch the gradient norms for vanishing gradients.`,
    datasetHint: "tiny_text_2",
    arch: arch([LM2_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: 3000, embeddingDim: 16 } }),
      makeLayerDef("rnn", { name: "rnn", config: { units: 96, returnSequences: false } }),
      dense(3000, "logits"), act("softmax"),
    ]),
  },
  {
    id: "char-rnn",
    name: "Character RNN language model",
    category: "Text",
    description: `A genuine sequence model: Embedding → tanh RNN (full backpropagation through time) → softmax. The hidden state carries information across all ${LM_SEQ_LEN} steps instead of flattening them. Watch for vanishing gradients on long contexts — a real RNN property.`,
    datasetHint: "tiny_text",
    arch: arch([LM_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: TEXT_VOCAB_SIZE, embeddingDim: 16 } }),
      makeLayerDef("rnn", { name: "rnn", config: { units: 64, returnSequences: false } }),
      dense(TEXT_VOCAB_SIZE, "logits"), act("softmax"),
    ]),
  },
  {
    id: "char-rnn-seq",
    name: "Character RNN (all timesteps)",
    category: "Text",
    description: `Embedding → tanh RNN with returnSequences on, so every one of the ${LM_SEQ_LEN} hidden states (not just the last) is flattened into the softmax. More parameters in the output head than the final-state RNN preset, and a different gradient path through time.`,
    datasetHint: "tiny_text",
    arch: arch([LM_SEQ_LEN], [
      makeLayerDef("embedding", { name: "embed", config: { vocabSize: TEXT_VOCAB_SIZE, embeddingDim: 16 } }),
      makeLayerDef("rnn", { name: "rnn", config: { units: 24, returnSequences: true } }),
      makeLayerDef("flatten", { name: "flatten" }),
      dense(TEXT_VOCAB_SIZE, "logits"), act("softmax"),
    ]),
  },

  /* ---------------- Sequence ---------------- */
  {
    id: "seq-rnn",
    name: "RNN sequence forecaster",
    category: "Sequence",
    description: `A tanh RNN reads the ${SEQ_WINDOW}-step window one point at a time and predicts the next value from its final hidden state. This is a general-purpose numeric sequence model — not language, no embedding.`,
    datasetHint: "wave_seq",
    arch: arch(
      [SEQ_WINDOW, 1],
      [makeLayerDef("rnn", { name: "rnn", config: { units: 24, returnSequences: false } }), dense(16, "h1"), act("tanh"), dense(1, "out")],
      42,
      "xavier_normal",
    ),
  },
  {
    id: "seq-rnn-deep",
    name: "Deep RNN forecaster",
    category: "Sequence",
    description: "A wider RNN (48 units) followed by two dense layers. More capacity than the baseline forecaster — compare validation loss between the two to see whether the extra size actually helps on this dataset.",
    datasetHint: "wave_seq",
    arch: arch(
      [SEQ_WINDOW, 1],
      [
        makeLayerDef("rnn", { name: "rnn", config: { units: 48, returnSequences: false } }),
        dense(32, "h1"), act("gelu"),
        dense(16, "h2"), act("gelu"),
        dense(1, "out"),
      ],
      42,
      "xavier_normal",
    ),
  },
  {
    id: "seq-mlp-baseline",
    name: "Sequence MLP baseline (no recurrence)",
    category: "Sequence",
    description: "Flattens the same window into a plain feed-forward network — no RNN layer at all. A fair baseline to check whether the recurrent presets are actually earning their extra complexity on this forecasting task.",
    datasetHint: "wave_seq",
    arch: arch(
      [SEQ_WINDOW, 1],
      [makeLayerDef("flatten", { name: "flatten" }), dense(32, "h1"), act("relu"), dense(16, "h2"), act("relu"), dense(1, "out")],
      42,
      "xavier_normal",
    ),
  },
];

/**
 * Align an architecture to a language dataset's tokenizer: context length,
 * embedding vocabulary, and final dense output classes must all agree with
 * the dataset's char<->id mapping, or generation decodes gibberish.
 */
export function alignToVocab(a: Architecture, vocabSize: number, seqLen: number): Architecture {
  const copy = structuredClone(a);
  copy.inputShape = [seqLen];
  let lastDense = -1;
  copy.layers.forEach((l, i) => {
    if (l.type === "embedding") l.config = { ...l.config, vocabSize };
    if (l.type === "dense") lastDense = i;
  });
  if (lastDense >= 0) {
    const l = copy.layers[lastDense]!;
    l.config = { ...l.config, units: vocabSize };
  }
  return copy;
}

export const PRESET_CATEGORIES: Preset["category"][] = ["Classification", "Regression", "Vision", "Autoencoder", "Text", "Sequence"];

export function presetById(id: string): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}

export function cloneArch(a: Architecture): Architecture {
  const copy = structuredClone(a);
  copy.layers = copy.layers.map((l) => ({ ...l, id: uid(l.type) }));
  return copy;
}
