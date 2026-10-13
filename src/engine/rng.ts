/** Mulberry32 PRNG — deterministic, seedable, fast. */
export class RNG {
  private s: number;

  constructor(seed = 42) {
    this.s = (seed >>> 0) || 1;
  }

  get seed(): number {
    return this.s >>> 0;
  }

  setSeed(seed: number): void {
    this.s = (seed >>> 0) || 1;
  }

  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal via Box–Muller. */
  normal(): number {
    const u = Math.max(this.next(), 1e-9);
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  uniform(min = 0, max = 1): number {
    return min + (max - min) * this.next();
  }

  int(max: number): number {
    return Math.floor(this.next() * max);
  }

  shuffle<T>(arr: T[]): T[] {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const tmp = a[i];
      a[i] = a[j]!;
      a[j] = tmp!;
    }
    return a;
  }

  clone(): RNG {
    const r = new RNG(1);
    r.s = this.s;
    return r;
  }
}
