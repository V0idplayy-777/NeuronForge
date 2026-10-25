import { RNG } from "../engine/rng";
import { tokenizeWords, type CorpusStats } from "./corpusLarge";

/**
 * Deep Reasoning corpus — explicit chain-of-thought traces.
 *
 * This corpus exists for one architecture: the Deep Reasoning preset. Every
 * line is a complete question → thought trace → answer example in a strict
 * format, so a small model has a real chance of learning *how to show its work*
 * rather than only memorising final answers:
 *
 *   q: seven plus eight
 *   think: seven and eight make fifteen . so the answer is fifteen .
 *   a: fifteen
 *
 * Design rules:
 *
 *  - Lowercase ascii only, words instead of digits, so the BPE alphabet stays
 *    small and every line is spellable from the corpus itself.
 *  - Correct answers, computed at build time (no fabricated arithmetic).
 *  - Several reasoning *families* — arithmetic, counting, comparison, ordering,
 *    syllogisms and numeric sequences — so the trace is not one memorised
 *    template. Each family has multiple surface forms.
 *  - The `think:` section always explains the intermediate steps before `a:`.
 *    That is what makes generated output auditable: the answer is conditioned
 *    on the trace, not emitted alone.
 *  - Deterministic for a given seed.
 *
 * This is a small synthetic corpus. A 40k-parameter model trained here learns
 * the trace format and the patterns it sees; it is not a general reasoner and
 * cannot transfer to arbitrary unseen maths. The repo says so in the preset and
 * export limitations instead of pretending otherwise.
 */

export const REASON_SEQ_LEN = 48;
export const REASON_VOCAB_TARGET = 148;
export const REASON_MAX_WINDOWS = 12_000;

/* ------------------------------- number words ------------------------------ */

const ONES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/** Spell 0..999 as lowercase words. */
export function reasonNumber(n: number): string {
  if (n < 0) return "minus " + reasonNumber(-n);
  if (n < 20) return ONES[n]!;
  if (n < 100) {
    const t = Math.floor(n / 10);
    const o = n % 10;
    return TENS[t]! + (o ? " " + ONES[o]! : "");
  }
  const h = Math.floor(n / 100);
  const r = n % 100;
  return ONES[h]! + " hundred" + (r ? " " + reasonNumber(r) : "");
}

/* --------------------------------- helpers -------------------------------- */

const NAMES = ["tom", "mia", "sam", "ada", "leo", "nina", "raj", "zoe"];
const CATEGORIES: { one: string; many: string; groupOne: string; groupMany: string }[] = [
  { one: "cat", many: "cats", groupOne: "animal", groupMany: "animals" },
  { one: "dog", many: "dogs", groupOne: "animal", groupMany: "animals" },
  { one: "rose", many: "roses", groupOne: "plant", groupMany: "plants" },
  { one: "oak", many: "oaks", groupOne: "plant", groupMany: "plants" },
  { one: "sparrow", many: "sparrows", groupOne: "bird", groupMany: "birds" },
  { one: "robin", many: "robins", groupOne: "bird", groupMany: "birds" },
];
const ORDERING_SAMPLES = 130;

const ADJECTIVES: { cmp: string; most: string; least: string }[] = [
  { cmp: "taller", most: "tallest", least: "shortest" },
  { cmp: "older", most: "oldest", least: "youngest" },
  { cmp: "faster", most: "fastest", least: "slowest" },
  { cmp: "heavier", most: "heaviest", least: "lightest" },
];

function pick<T>(arr: readonly T[], r: RNG): T {
  return arr[r.int(arr.length)]!;
}

/* ------------------------------ handwritten ------------------------------- */

/**
 * A few hand-written traces anchor the style beyond the generated templates.
 * They are deliberately simple causal / deductive examples.
 */
const HANDWRITTEN: string[] = [
  "q: it rained all night . the ground is wet . why think: rain puts water on the ground . so the ground is wet . a: rain",
  "q: all birds have wings . a robin is a bird . does a robin have wings think: a robin is a bird and all birds have wings . so a robin has wings . a: yes",
  "q: the ice cream is in the freezer . is it cold think: the freezer keeps things cold . so the ice cream is cold . a: yes",
  "q: if it is monday the shop is open . it is monday . is the shop open think: monday means the shop is open . so the shop is open . a: yes",
  "q: sam is taller than leo . leo is taller than zoe . who is shortest think: sam is taller than leo and leo is taller than zoe . so zoe is shortest . a: zoe",
  "q: a train is faster than a bike . a bike is faster than walking . what is fastest think: the train is faster than the bike and the bike is faster than walking . so the train is fastest . a: train",
  "q: every rose is a plant . this flower is a rose . is it a plant think: this flower is a rose and every rose is a plant . so it is a plant . a: yes",
  "q: three red balls and two blue balls . how many balls think: three plus two make five . so the answer is five . a: five",
];

/* ------------------------------ generators -------------------------------- */

function generateArithmetic(r: RNG, out: string[]): void {
  for (let a = 1; a <= 12; a++) {
    for (let b = 1; b <= 12; b++) {
      const wa = reasonNumber(a);
      const wb = reasonNumber(b);
      const sum = a + b;
      const addForms = [
        `q: ${wa} plus ${wb} think: ${wa} and ${wb} make ${reasonNumber(sum)} . so the answer is ${reasonNumber(sum)} . a: ${reasonNumber(sum)}`,
        `q: add ${wa} and ${wb} think: start with ${wa} and add ${wb} more . ${wa} and ${wb} make ${reasonNumber(sum)} . so the answer is ${reasonNumber(sum)} . a: ${reasonNumber(sum)}`,
      ];
      out.push(pick(addForms, r));

      if (a > b) {
        const diff = a - b;
        out.push(
          `q: ${wa} minus ${wb} think: ${wa} take away ${wb} leaves ${reasonNumber(diff)} . so the answer is ${reasonNumber(diff)} . a: ${reasonNumber(diff)}`,
        );
      }
      if (a >= 2 && b >= 2 && a <= 9 && b <= 9) {
        const prod = a * b;
        out.push(
          `q: ${wa} times ${wb} think: ${wb} groups of ${wa} make ${reasonNumber(prod)} . so the answer is ${reasonNumber(prod)} . a: ${reasonNumber(prod)}`,
        );
      }
    }
  }
}

function generateCounting(r: RNG, out: string[]): void {
  const colours = ["red", "blue", "green", "yellow"];
  for (let i = 0; i < 160; i++) {
    const a = r.int(9) + 1;
    const b = r.int(9) + 1;
    const c1 = pick(colours, r);
    let c2 = pick(colours, r);
    if (c2 === c1) c2 = colours[(colours.indexOf(c1) + 1) % colours.length]!;
    const sum = a + b;
    out.push(
      `q: ${reasonNumber(a)} ${c1} balls and ${reasonNumber(b)} ${c2} balls . how many balls think: ${reasonNumber(a)} plus ${reasonNumber(b)} make ${reasonNumber(sum)} . so the answer is ${reasonNumber(sum)} . a: ${reasonNumber(sum)}`,
    );
  }
}

function generateComparison(r: RNG, out: string[]): void {
  for (let i = 0; i < 150; i++) {
    let a = r.int(19) + 1;
    let b = r.int(19) + 1;
    if (a === b) b = (b % 19) + 1;
    const wa = reasonNumber(a);
    const wb = reasonNumber(b);
    const large = a > b ? wa : wb;
    const small = a > b ? wb : wa;
    if (r.next() < 0.5) {
      out.push(
        `q: which is larger , ${wa} or ${wb} think: ${wa} is ${a > b ? "more" : "less"} than ${wb} . the larger is ${large} . so the answer is ${large} . a: ${large}`,
      );
    } else {
      out.push(
        `q: which is smaller , ${wa} or ${wb} think: ${wa} is ${a < b ? "less" : "more"} than ${wb} . the smaller is ${small} . so the answer is ${small} . a: ${small}`,
      );
    }
  }
}

function generateOrdering(r: RNG, out: string[]): void {
  // Sample a balanced subset of name orderings instead of all 336 permutations:
  // a focused corpus that does not drown the other reasoning families.
  const seen = new Set<string>();
  let guard = 0;
  let count = 0;
  while (count < ORDERING_SAMPLES && guard++ < 5000) {
    const names = r.shuffle(NAMES.slice()).slice(0, 3);
    const key = names.join("|") + "|" + count;
    if (seen.has(key)) continue;
    seen.add(key);
    count++;
    const [x, y, z] = names as [string, string, string];
    const adj = pick(ADJECTIVES, r);
    const target = r.next() < 0.5 ? adj.most : adj.least;
    const answer = target === adj.most ? x : z;
    out.push(
      `q: ${x} is ${adj.cmp} than ${y} . ${y} is ${adj.cmp} than ${z} . who is ${target} think: ${x} is ${adj.cmp} than ${y} and ${y} is ${adj.cmp} than ${z} . so ${answer} is ${target} . a: ${answer}`,
    );
  }
}

function generateSyllogism(r: RNG, out: string[]): void {
  for (const cat of CATEGORIES) {
    for (let i = 0; i < 20; i++) {
      const name = pick(NAMES, r);
      out.push(
        `q: all ${cat.many} are ${cat.groupMany} . ${name} is a ${cat.one} . is ${name} a ${cat.groupOne} think: ${name} is a ${cat.one} and all ${cat.many} are ${cat.groupMany} . so ${name} is a ${cat.groupOne} . a: yes`,
      );
    }
  }
}

function generateSequences(out: string[]): void {
  for (let start = 1; start <= 9; start++) {
    for (let step = 1; step <= 5; step++) {
      const t = [start, start + step, start + 2 * step, start + 3 * step];
      const next = start + 4 * step;
      const pattern = t.map(reasonNumber).join(" , ");
      const answer = reasonNumber(next);
      out.push(
        `q: ${pattern} . what is next think: the pattern adds ${reasonNumber(step)} . ${reasonNumber(t[3]!)} plus ${reasonNumber(step)} make ${answer} . so the answer is ${answer} . a: ${answer}`,
      );
      out.push(
        `q: the pattern is ${pattern} . what comes next think: each number goes up by ${reasonNumber(step)} . ${reasonNumber(t[3]!)} and ${reasonNumber(step)} make ${answer} . so the answer is ${answer} . a: ${answer}`,
      );
    }
  }
}

/* ------------------------------- the corpus -------------------------------- */

const cache = new Map<number, { text: string; stats: CorpusStats }>();

export function buildReasonCorpus(seed = 1234): { text: string; stats: CorpusStats } {
  const cached = cache.get(seed);
  if (cached) return cached;

  const r = new RNG(seed);
  const generated: string[] = [];
  generateArithmetic(r, generated);
  generateCounting(r, generated);
  generateComparison(r, generated);
  generateOrdering(r, generated);
  generateSyllogism(r, generated);
  generateSequences(generated);

  // Repeat the hand-written anchors so their style is not a single rare line.
  const blocks: string[] = [];
  for (let pass = 0; pass < 3; pass++) for (const line of HANDWRITTEN) blocks.push(line);
  for (const line of generated) blocks.push(line);
  const mixed = r.shuffle(blocks);
  const text = mixed.join("\n");

  const words = tokenizeWords(text);
  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const result = {
    text,
    stats: {
      chars: text.length,
      lines: mixed.length,
      templateLines: generated.length,
      handwrittenLines: HANDWRITTEN.length * 3,
      tokens: words.length,
      uniqueTokens: freq.size,
      topTokens: sorted.slice(0, 12).map(([token, count]) => ({ token, count })),
    } satisfies CorpusStats,
  };
  cache.set(seed, result);
  return result;
}
