export function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return n.toExponential(2);
  return n.toFixed(digits).replace(/\.?0+$/, (m) => (m.includes(".") ? "" : m));
}

export function fmtFixed(n: number, d = 4): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString();
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function fmtDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  if (m < 60) return `${m}m ${r}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function fmtShape(s: readonly number[]): string {
  return `[${s.join(" × ")}]`;
}
