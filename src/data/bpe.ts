/**
 * A small, deterministic byte-pair-encoding (BPE) tokenizer.
 *
 * Why BPE: character-level models waste capacity spelling whole words one letter
 * at a time, and word-level models need an enormous output projection plus an
 * `<unk>` token for every rare word. Subword tokenisation sits in between — the
 * vocabulary is learned by repeatedly merging the most frequent adjacent pair of
 * symbols, so common words end up as one or two tokens and rare words fall back
 * to smaller pieces instead of becoming unknown.
 *
 * Design rules (kept deliberately simple so the exported runtimes can mirror
 * them exactly):
 *
 *  - Text is split into runs of whitespace and runs of non-whitespace.
 *  - Only non-whitespace runs are merged. Whitespace is emitted as single
 *    characters, so decoding is a plain concatenation and is always lossless.
 *  - Symbols are characters, so the base alphabet is the corpus's character set;
 *    an unseen character is the only way to produce the `<unk>` token (id 0).
 *  - Merges are chosen by raw frequency, ties broken by the pair's own string
 *    order, which makes training fully deterministic for a given corpus.
 */
import { UNK_TOKEN } from "./corpusLarge";

export type MergeRule = [string, string];

export interface BpeModel {
  level: "bpe";
  /** id -> token. Index 0 is `<unk>`, then the base symbols, then one token per merge. */
  tokens: string[];
  /** Merge rules in the order they were learned (earlier = higher priority). */
  merges: MergeRule[];
  vocabSize: number;
  /** Non-whitespace characters the tokenizer was trained on. */
  baseAlphabet: string[];
  /** Corpus symbols (whitespace runs counted as their characters) before merging. */
  symbolsBefore: number;
  /** Corpus symbols after merging — the compression the merges bought. */
  symbolsAfter: number;
  /** Distinct symbols that actually occur after merging. */
  symbolsInUse: number;
  corpusChars: number;
  /** True when the vocabulary was learned without lowercasing. */
  caseSensitive: boolean;
}

export interface BpeOptions {
  /** Hard cap on the number of merges (vocabulary growth past the base symbols). */
  maxMerges?: number;
  /** Stop early when the best pair occurs fewer than this many times. */
  minFrequency?: number;
  /**
   * Keep letter case instead of lowercasing. Off by default, which preserves
   * the exact behaviour of every existing dataset and exported runtime. The
   * code corpora turn it on so python's `None`, `True` and `ValueError` stay
   * distinct from lowercase words — the flag is stored in the tokenizer spec
   * so exported runtimes encode identically.
   */
  caseSensitive?: boolean;
  /** Progress callback for long trainings. */
  onProgress?: (info: { merges: number; vocabSize: number; pairsConsidered: number }) => void;
  progressEvery?: number;
}

const WHITESPACE = /\s/;

function isWhitespace(ch: string): boolean {
  return WHITESPACE.test(ch);
}

/** Split text into whitespace runs (kept whole) and non-whitespace runs (mergeable). */
export function bpeChunks(text: string): { chunk: string; mergeable: boolean }[] {
  const out: { chunk: string; mergeable: boolean }[] = [];
  let i = 0;
  while (i < text.length) {
    const space = isWhitespace(text[i]!);
    let j = i;
    while (j < text.length && isWhitespace(text[j]!) === space) j++;
    out.push({ chunk: text.slice(i, j), mergeable: !space });
    i = j;
  }
  return out;
}

/** Frequency of every distinct non-whitespace run in the text. */
function wordFrequencies(text: string): Map<string, number> {
  const freq = new Map<string, number>();
  for (const { chunk, mergeable } of bpeChunks(text)) {
    if (!mergeable) continue;
    freq.set(chunk, (freq.get(chunk) ?? 0) + 1);
  }
  return freq;
}

const SEP = "\u0000";

/**
 * Learn a BPE vocabulary from `text`.
 *
 * The token list is `<unk>` first, then every base symbol, then one token per
 * merge, so ids are stable for a given corpus and target size.
 */
export function trainBpe(text: string, targetVocabSize: number, options: BpeOptions = {}): BpeModel {
  const lower = options.caseSensitive ? text : text.toLowerCase();
  const chars = [...new Set(lower.split("").filter((c) => !isWhitespace(c)))].sort();
  const wsChars = [...new Set(lower.split("").filter((c) => isWhitespace(c)))].sort();
  const tokens = [UNK_TOKEN, ...chars, ...wsChars];
  const baseCount = tokens.length;
  const maxMerges = Math.max(0, Math.min(options.maxMerges ?? Number.POSITIVE_INFINITY, targetVocabSize - baseCount));
  const minFrequency = options.minFrequency ?? 2;
  const progressEvery = options.progressEvery ?? 25;

  const counts = wordFrequencies(lower);
  // Words that occur once cannot change which pair is the most frequent, and
  // including them means one extra pass over the whole vocabulary per merge.
  // They are excluded from *training* only: encoding never consults this list,
  // it applies the learned merge ranks to whatever text it is given.
  const allWords = [...counts.entries()];
  const trainSet = allWords.length > 20000 ? allWords.filter(([, c]) => c >= 2) : allWords;
  const words = new Map<string, string[]>();
  let symbolsBefore = 0;
  for (const [word] of trainSet) {
    words.set(word, word.split(""));
    symbolsBefore += (counts.get(word) ?? 1) * word.length;
  }

  const merges: MergeRule[] = [];
  while (merges.length < maxMerges) {
    const pairCounts = new Map<string, number>();
    for (const [word, symbols] of words) {
      const weight = counts.get(word) ?? 1;
      if (symbols.length < 2) continue;
      for (let i = 0; i + 1 < symbols.length; i++) {
        const key = symbols[i]! + SEP + symbols[i + 1]!;
        pairCounts.set(key, (pairCounts.get(key) ?? 0) + weight);
      }
    }
    let bestKey: string | null = null;
    let bestCount = 0;
    for (const [key, count] of pairCounts) {
      if (count > bestCount || (count === bestCount && bestKey !== null && key < bestKey)) {
        bestCount = count;
        bestKey = key;
      }
    }
    if (bestKey === null || bestCount < minFrequency) break;
    const [a, b] = bestKey.split(SEP) as [string, string];
    const merged = a + b;
    merges.push([a, b]);
    tokens.push(merged);

    for (const [word, symbols] of words) {
      let changed = false;
      const next: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i + 1 < symbols.length && symbols[i] === a && symbols[i + 1] === b) {
          next.push(merged);
          i++;
          changed = true;
        } else {
          next.push(symbols[i]!);
        }
      }
      if (changed) words.set(word, next);
    }
    if (options.onProgress && merges.length % progressEvery === 0) {
      options.onProgress({ merges: merges.length, vocabSize: tokens.length, pairsConsidered: pairCounts.size });
    }
  }

  let symbolsAfter = 0;
  const inUse = new Set<string>();
  for (const [word, symbols] of words) {
    symbolsAfter += (counts.get(word) ?? 1) * symbols.length;
    for (const s of symbols) inUse.add(s);
  }

  return {
    level: "bpe",
    tokens,
    merges,
    vocabSize: tokens.length,
    baseAlphabet: chars,
    symbolsBefore,
    symbolsAfter,
    symbolsInUse: inUse.size,
    corpusChars: lower.length,
    caseSensitive: options.caseSensitive === true,
  };
}

export interface EncodeTable {
  /** pair key -> merge rank (lower rank = applied first). */
  ranks: Map<string, number>;
  /** token -> id. */
  stoi: Map<string, number>;
  unknownId: number;
  /** Keep letter case when encoding (mirrors how the vocabulary was learned). */
  caseSensitive: boolean;
}

export function encodeTable(tokens: string[], merges: MergeRule[], caseSensitive = false): EncodeTable {
  const ranks = new Map<string, number>();
  merges.forEach(([a, b], i) => ranks.set(a + SEP + b, i));
  const stoi = new Map<string, number>();
  tokens.forEach((t, i) => stoi.set(t, i));
  return { ranks, stoi, unknownId: 0, caseSensitive };
}

/** Apply the merge rules to one non-whitespace chunk. */
export function bpeEncodeChunk(chunk: string, ranks: Map<string, number>): string[] {
  let symbols = chunk.split("");
  while (symbols.length > 1) {
    let bestRank = Number.POSITIVE_INFINITY;
    let bestAt = -1;
    for (let i = 0; i + 1 < symbols.length; i++) {
      const rank = ranks.get(symbols[i]! + SEP + symbols[i + 1]!);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestAt = i;
      }
    }
    if (bestAt < 0) break;
    symbols = [...symbols.slice(0, bestAt), symbols[bestAt]! + symbols[bestAt + 1]!, ...symbols.slice(bestAt + 2)];
  }
  return symbols;
}

/** Token strings for arbitrary text (whitespace characters stay individual tokens). */
export function bpeTokenizeWith(table: EncodeTable, text: string): string[] {
  const out: string[] = [];
  const source = table.caseSensitive ? text : text.toLowerCase();
  for (const { chunk, mergeable } of bpeChunks(source)) {
    if (!mergeable) {
      for (const ch of chunk) out.push(ch);
      continue;
    }
    for (const symbol of bpeEncodeChunk(chunk, table.ranks)) out.push(symbol);
  }
  return out;
}

export function bpeTokenize(text: string, tokens: string[], merges: MergeRule[], caseSensitive = false): string[] {
  return bpeTokenizeWith(encodeTable(tokens, merges, caseSensitive), text);
}

/** Token ids for arbitrary text; unseen characters become id 0 (`<unk>`). */
export function bpeEncodeWith(table: EncodeTable, text: string): number[] {
  const out: number[] = [];
  for (const symbol of bpeTokenizeWith(table, text)) {
    const id = table.stoi.get(symbol);
    out.push(id === undefined ? table.unknownId : id);
  }
  return out;
}

export function bpeEncode(text: string, tokens: string[], merges: MergeRule[], caseSensitive = false): number[] {
  return bpeEncodeWith(encodeTable(tokens, merges, caseSensitive), text);
}

/** Inverse of encoding: concatenating the tokens rebuilds the text exactly. */
export function bpeDecode(ids: number[], tokens: string[]): string {
  let s = "";
  for (const id of ids) {
    const t = tokens[id];
    if (t !== undefined) s += t;
  }
  return s;
}

/** Tokeniser statistics for the Dataset Studio / Inspector, computed by real encoding. */
export function bpeStats(model: BpeModel, text: string): {
  tokens: number;
  chars: number;
  compression: number;
  unknown: number;
  singleCharTokens: number;
  longestToken: string;
} {
  const table = encodeTable(model.tokens, model.merges, model.caseSensitive);
  const ids = bpeEncodeWith(table, text);
  let unknown = 0;
  let singleCharTokens = 0;
  for (const id of ids) {
    const t = model.tokens[id] ?? "";
    if (id === 0) unknown++;
    if (t.length === 1) singleCharTokens++;
  }
  return {
    tokens: ids.length,
    chars: text.length,
    compression: ids.length ? text.length / ids.length : 0,
    unknown,
    singleCharTokens,
    longestToken: model.tokens.reduce((best, t) => (t.length > best.length ? t : best), ""),
  };
}

/* ------------------------- corpus-level tokenizer (dataset integration) ------------------------- */

export interface BpeTokenizer {
  level: "bpe";
  /** id -> token. Index 0 is `<unk>`. */
  tokens: string[];
  merges: MergeRule[];
  /** token -> id, for the fast encoding path. */
  tokenToId: Record<string, number>;
  seqLen: number;
  vocabSize: number;
  /** Tokens the corpus encodes to (after merging). */
  totalTokens: number;
  /** Distinct tokens the corpus uses. */
  uniqueTokens: number;
  /** Corpus characters / corpus tokens: how much text one token carries. */
  compression: number;
  corpusChars: number;
  mergesLearned: number;
  /** True when the vocabulary was learned without lowercasing. */
  caseSensitive: boolean;
}

/**
 * Train a BPE tokenizer for a corpus. Deterministic for a given (corpus, cap)
 * pair, and memoised because the Dataset Studio rebuilds language datasets on
 * every generator change.
 */
export function buildBpeTokenizer(corpus: string, vocabSizeCap: number, seqLen: number, caseSensitive = false): BpeTokenizer {
  const cacheKey = `${vocabSizeCap}|${seqLen}|${caseSensitive ? "cs" : "ci"}|${corpus.length}|${corpus.charCodeAt(0)}|${corpus.slice(0, 48)}|${corpus.slice(-48)}`;
  const cached = bpeCache.get(cacheKey);
  if (cached) return cloneBpeTokenizer(cached);

  const model = trainBpe(corpus, vocabSizeCap, { minFrequency: 3, caseSensitive });
  const stats = bpeStats(model, corpus);
  const tokenToId: Record<string, number> = {};
  model.tokens.forEach((t, i) => {
    tokenToId[t] = i;
  });
  const tokenizer: BpeTokenizer = {
    level: "bpe",
    tokens: model.tokens,
    merges: model.merges,
    tokenToId,
    seqLen,
    vocabSize: model.tokens.length,
    totalTokens: stats.tokens,
    uniqueTokens: model.symbolsInUse,
    compression: stats.compression,
    corpusChars: model.corpusChars,
    mergesLearned: model.merges.length,
    caseSensitive,
  };
  bpeCache.set(cacheKey, tokenizer);
  return cloneBpeTokenizer(tokenizer);
}

const bpeCache = new Map<string, BpeTokenizer>();

function cloneBpeTokenizer(t: BpeTokenizer): BpeTokenizer {
  return {
    ...t,
    tokens: t.tokens.slice(),
    merges: t.merges.map(([a, b]) => [a, b] as MergeRule),
    tokenToId: { ...t.tokenToId },
  };
}

/** Encode text with a corpus tokenizer, or with a serialised token list. */
export function encodeBpe(text: string, tokens?: string[], merges?: MergeRule[]): number[] {
  if (tokens && merges) return bpeEncode(text, tokens, merges);
  const lower = text.toLowerCase();
  const table = encodeTable(tokens ?? [], merges ?? []);
  return bpeEncodeWith(table, lower);
}

export function encodeBpeWith(tok: BpeTokenizer, text: string): number[] {
  return bpeEncodeWith(encodeTable(tok.tokens, tok.merges, tok.caseSensitive), text);
}

export function decodeBpeIds(ids: number[], tokens: string[]): string {
  return bpeDecode(ids, tokens);
}
