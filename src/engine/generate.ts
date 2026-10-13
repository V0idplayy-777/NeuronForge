import type { SequentialModel } from "./model";
import type { TokenizerSpec } from "../data/datasets";
import { Tensor } from "./tensor";
import { RNG } from "./rng";
import { sampleFromProbs } from "./sampling";
import { tokenizeWords, joinWords } from "../data/corpusLarge";

/**
 * The single reference implementation of tokenisation + autoregressive
 * generation inside NeuronForge. The Evaluation playground uses it, and the
 * standalone exporter verifies the exported runtime against it token-for-token.
 */

export interface GenerateOptions {
  maxTokens: number;
  temperature: number;
  topK: number;
  seed: number;
  /** Tokens that end generation (not included in the output). */
  stop?: string[];
}

export interface GenerateResult {
  ids: number[];
  tokens: string[];
  text: string;
  promptIds: number[];
  stoppedBy: string | null;
}

function tokenMap(tok: TokenizerSpec): Map<string, number> {
  // Later duplicates overwrite earlier ones; the exported runtime mirrors this.
  return new Map(tok.tokens.map((t, i) => [t, i]));
}

/** Word level: unknown words -> id 0 (<unk>). Char level: unknown characters are skipped. */
export function tokenizeForModel(tok: TokenizerSpec, text: string): number[] {
  const map = tokenMap(tok);
  if (tok.level === "word") return tokenizeWords(text).map((w) => map.get(w) ?? 0);
  const out: number[] = [];
  for (const ch of text.toLowerCase().split("")) {
    const id = map.get(ch);
    if (id !== undefined) out.push(id);
  }
  return out;
}

export function detokenize(tok: TokenizerSpec, ids: number[]): string {
  const parts: string[] = [];
  for (const id of ids) {
    const t = tok.tokens[id];
    if (t !== undefined) parts.push(t);
  }
  return tok.level === "word" ? joinWords(parts) : parts.join("");
}

export function generateTokens(model: SequentialModel, tok: TokenizerSpec, prompt: string, opts: GenerateOptions): GenerateResult {
  const outUnits = model.outputShape[model.outputShape.length - 1];
  if (outUnits !== tok.tokens.length) {
    throw new Error(
      `Model outputs ${outUnits} classes but the tokenizer has ${tok.tokens.length} tokens. ` +
        "Decoding would map ids to the wrong tokens. Re-select the language dataset so the vocabulary re-aligns.",
    );
  }
  const promptIds = tokenizeForModel(tok, prompt);
  if (promptIds.length === 0) {
    throw new Error(
      tok.level === "word"
        ? `“${prompt}” contains no word tokens. Try plain lowercase English words.`
        : `None of the characters in “${prompt}” exist in the ${tok.tokens.length}-token vocabulary.`,
    );
  }
  const seq = model.arch.inputShape[0] ?? tok.seqLen;
  const map = tokenMap(tok);
  const padId = map.get(tok.padToken) ?? 0;
  const stop = new Set(opts.stop ?? []);
  const rng = new RNG(opts.seed);
  const ctx = promptIds.slice();
  const ids: number[] = [];
  let stoppedBy: string | null = null;

  for (let t = 0; t < opts.maxTokens; t++) {
    const window = ctx.slice(-seq);
    const filled = window.length < seq ? [...new Array<number>(seq - window.length).fill(padId), ...window] : window;
    const x = new Tensor([1, seq], Float32Array.from(filled));
    const probs = model.forward(x, false).output;
    const { index } = sampleFromProbs(probs.data, opts.temperature, opts.topK, rng);
    const token = tok.tokens[index];
    if (token === undefined) throw new Error(`Model produced class id ${index}, outside the tokenizer's ${tok.tokens.length} tokens.`);
    if (stop.has(token)) {
      stoppedBy = token;
      break;
    }
    ids.push(index);
    ctx.push(index);
  }
  return { ids, tokens: ids.map((i) => tok.tokens[i]!), text: detokenize(tok, ids), promptIds, stoppedBy };
}
