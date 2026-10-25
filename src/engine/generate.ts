import type { SequentialModel } from "./model";
import type { TokenizerSpec } from "../data/datasets";
import { Tensor } from "./tensor";
import { RNG } from "./rng";
import { sampleFromProbs } from "./sampling";
import { tokenizeWords, joinWords } from "../data/corpusLarge";
import { bpeEncode, bpeDecode } from "../data/bpe";
import type { ReasoningTrace } from "./reasoning";

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
  /**
   * Capture the Deep Reasoning trace of the last forward pass. Reasoning runs
   * on every token regardless; this only decides whether the per-step halting
   * diagnostics are returned.
   */
  traceReasoning?: boolean;
}

export interface GenerateReasoningTrace {
  id: string;
  name: string;
  trace: ReasoningTrace | null;
}

export interface GenerateResult {
  ids: number[];
  tokens: string[];
  text: string;
  promptIds: number[];
  stoppedBy: string | null;
  /** Present when the model has Deep Reasoning layers and traceReasoning was requested. */
  reasoning?: GenerateReasoningTrace[];
}

function tokenMap(tok: TokenizerSpec): Map<string, number> {
  // Later duplicates overwrite earlier ones; the exported runtime mirrors this.
  return new Map(tok.tokens.map((t, i) => [t, i]));
}

/**
 * The reference tokeniser for every level.
 *
 * - word: unknown words → id 0 (`<unk>`).
 * - char: unknown characters are skipped.
 * - bpe:  the training merge rules are re-applied; the base alphabet is the
 *         corpus character set, so only genuinely unseen characters → id 0.
 */
export function tokenizeForModel(tok: TokenizerSpec, text: string): number[] {
  if (tok.level === "bpe") {
    if (!tok.merges || tok.merges.length === 0) {
      // A BPE tokenizer without merge rules would be a plain character
      // tokenizer; fall through to the character path rather than throwing.
    } else {
      return bpeEncode(text, tok.tokens, tok.merges, tok.caseSensitive === true);
    }
  }
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
  if (tok.level === "bpe") return bpeDecode(ids, tok.tokens);
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
        : tok.level === "bpe"
          ? `“${prompt}” produced no subword tokens. Try plain lowercase text.`
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
  let stepped = false;

  for (let t = 0; t < opts.maxTokens; t++) {
    const window = ctx.slice(-seq);
    const filled = window.length < seq ? [...new Array<number>(seq - window.length).fill(padId), ...window] : window;
    const x = new Tensor([1, seq], Float32Array.from(filled));
    const probs = model.forward(x, false).output;
    stepped = true;
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
  const result: GenerateResult = { ids, tokens: ids.map((i) => tok.tokens[i]!), text: detokenize(tok, ids), promptIds, stoppedBy };
  const reasoningLayers = model.reasoningLayers();
  if (opts.traceReasoning && stepped && reasoningLayers.length > 0) {
    result.reasoning = reasoningLayers.map((l) => ({ id: l.id, name: l.name, trace: l.trace }));
  }
  return result;
}
