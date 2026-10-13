import { RNG } from "./rng";

/**
 * Sample a token index from a probability distribution with temperature and
 * top-k filtering.
 *
 * The model's final layer is a softmax, so we convert its probabilities back to
 * logits with log(p) before applying temperature. This is exactly equivalent to
 * having had the raw logits: log(p_i) = z_i - log Z, and softmax is invariant to
 * a constant subtracted from every score, so dividing by T behaves identically.
 *
 * temperature -> 0 collapses to greedy argmax.
 * temperature -> 1 is unmodified sampling from the model.
 * temperature > 1 flattens the distribution and produces rarer tokens.
 *
 * top-k keeps only the k highest-probability candidates and renormalises over
 * them, which removes the long tail of very unlikely tokens.
 */
export function sampleFromProbs(
  probs: Float32Array | number[],
  temperature: number,
  topK: number,
  rng: RNG,
): { index: number; probability: number } {
  if (probs.length === 0) throw new Error("Cannot sample from an empty distribution");

  if (!Number.isFinite(temperature) || temperature <= 0.01) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
    return { index: best, probability: probs[best]! };
  }

  const invT = 1 / temperature;
  const logits = new Float64Array(probs.length);
  for (let i = 0; i < probs.length; i++) {
    const p = Math.max(Number(probs[i]), 1e-9);
    logits[i] = Math.log(p) * invT;
  }

  let candidates: number[];
  const k = Math.max(1, Math.min(Math.floor(topK) || probs.length, probs.length));
  if (k >= probs.length) {
    candidates = Array.from({ length: probs.length }, (_, i) => i);
  } else {
    // Partial selection: O(n) average with a bounded max-heap is overkill here,
    // a simple scan over the k largest thresholds is plenty for vocab sizes used.
    candidates = topKIndices(logits, k);
  }

  let max = -Infinity;
  for (const i of candidates) if (logits[i]! > max) max = logits[i]!;
  let sum = 0;
  const weights = candidates.map((i) => {
    const w = Math.exp(logits[i]! - max);
    sum += w;
    return w;
  });
  if (!(sum > 0)) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
    return { index: best, probability: probs[best]! };
  }

  let r = rng.next() * sum;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i]!;
    if (r <= 0) {
      const idx = candidates[i]!;
      return { index: idx, probability: probs[idx]! };
    }
  }
  const last = candidates[candidates.length - 1]!;
  return { index: last, probability: probs[last]! };
}

function topKIndices(logits: Float64Array, k: number): number[] {
  // Threshold-based selection: find the k-th largest value, then collect.
  const sample: number[] = [];
  const stride = Math.max(1, Math.floor(logits.length / 512));
  for (let i = 0; i < logits.length; i += stride) sample.push(logits[i]!);
  sample.sort((a, b) => b - a);
  const threshold = sample[Math.min(k - 1, sample.length - 1)]!;

  const idx: number[] = [];
  for (let i = 0; i < logits.length; i++) if (logits[i]! >= threshold) idx.push(i);
  if (idx.length > k) {
    idx.sort((a, b) => logits[b]! - logits[a]!);
    idx.length = k;
  }
  return idx;
}

export const SAMPLING_DOCS = {
  temperature:
    "Divides the log-probabilities before renormalising. Below 1 sharpens the distribution toward the model's best guess; above 1 flattens it and admits rarer words. 0.7 is a common default for coherent but varied text.",
  topK:
    "Restricts sampling to the k most likely tokens, then renormalises over just those. 50 discards the long tail of improbable words that would otherwise occasionally produce nonsense.",
};
