import { useMemo } from "react";
import { fmt } from "../lib/format";

export interface Series {
  name: string;
  color: string;
  values: { x: number; y: number }[];
}

export function LineChart({
  series,
  height = 180,
  yLabel,
}: {
  series: Series[];
  height?: number;
  yLabel?: string;
}) {
  const w = 640;
  const h = height;
  const pad = { l: 42, r: 12, t: 12, b: 24 };
  const pts = series.flatMap((s) => s.values);
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y).filter((v) => Number.isFinite(v));
  const minX = xs.length ? Math.min(...xs) : 0;
  const maxX = xs.length ? Math.max(...xs) : 1;
  const minY = ys.length ? Math.min(0, ...ys) : 0;
  const maxY = ys.length ? Math.max(...ys, 1e-6) : 1;
  const xspan = maxX - minX || 1;
  const yspan = maxY - minY || 1;
  const sx = (x: number) => pad.l + ((x - minX) / xspan) * (w - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (y - minY) / yspan) * (h - pad.t - pad.b);

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => minY + (yspan * i) / ticks);

  return (
    <div className="w-full overflow-hidden">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto" role="img">
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.06)" />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fill="#64748b" fontSize="10" fontFamily="IBM Plex Mono, monospace">
              {fmt(t, 3)}
            </text>
          </g>
        ))}
        {series.map((s) => {
          const finite = s.values.filter((p) => Number.isFinite(p.y));
          if (finite.length < 2) return null;
          const d = finite.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x)},${sy(p.y)}`).join(" ");
          return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={1.6} />;
        })}
        {yLabel && (
          <text x={12} y={h / 2} fill="#64748b" fontSize="10" transform={`rotate(-90 12 ${h / 2})`} textAnchor="middle">
            {yLabel}
          </text>
        )}
      </svg>
      <div className="flex gap-3 px-2 pb-1">
        {series.map((s) => (
          <span key={s.name} className="flex items-center gap-1.5 text-[10px] text-slate-400">
            <i className="h-1.5 w-3 rounded-sm" style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Histogram({ counts, edges, color = "#22d3ee" }: { counts: number[]; edges: number[]; color?: string }) {
  const max = Math.max(1, ...counts);
  return (
    <div className="flex items-end gap-px h-20 w-full">
      {counts.map((c, i) => (
        <div
          key={i}
          title={`${fmt(edges[i] ?? 0)}–${fmt(edges[i + 1] ?? 0)}: ${c}`}
          className="flex-1 min-w-0 rounded-t-sm"
          style={{ height: `${(c / max) * 100}%`, background: color, opacity: 0.75 }}
        />
      ))}
    </div>
  );
}

export function Heatmap({ data, rows, cols, title }: { data: ArrayLike<number>; rows: number; cols: number; title?: string }) {
  const { min, max } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!Number.isFinite(min)) {
      min = 0;
      max = 1;
    }
    return { min, max };
  }, [data]);
  const span = max - min || 1;
  const show = Math.min(rows, 24);
  const showC = Math.min(cols, 24);
  return (
    <div>
      {title && <div className="text-[10px] text-slate-500 mb-1 font-mono">{title}</div>}
      <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${showC}, minmax(0, 1fr))` }}>
        {Array.from({ length: show * showC }, (_, i) => {
          const r = Math.floor(i / showC);
          const c = i % showC;
          const v = data[r * cols + c] ?? 0;
          const t = (v - min) / span;
          const col = t < 0.5 ? lerp("#0ea5e9", "#0f172a", t * 2) : lerp("#0f172a", "#f59e0b", (t - 0.5) * 2);
          return <div key={i} className="aspect-square rounded-[1px]" style={{ background: col }} title={`${r},${c}: ${fmt(v)}`} />;
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] font-mono text-slate-500">
        <span>{fmt(min)}</span>
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

function lerp(a: string, b: string, t: number): string {
  const pa = hex(a);
  const pb = hex(b);
  const u = Math.min(1, Math.max(0, t));
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * u);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * u);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * u);
  return `rgb(${r},${g},${bl})`;
}
function hex(h: string): [number, number, number] {
  const x = h.replace("#", "");
  return [parseInt(x.slice(0, 2), 16), parseInt(x.slice(2, 4), 16), parseInt(x.slice(4, 6), 16)];
}

export function Confusion({ matrix, labels }: { matrix: number[][]; labels?: string[] }) {
  const n = matrix.length;
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-auto">
      <div className="inline-grid gap-px" style={{ gridTemplateColumns: `repeat(${n + 1}, minmax(28px, 1fr))` }}>
        <div />
        {Array.from({ length: n }, (_, j) => (
          <div key={`c${j}`} className="text-[9px] text-slate-500 text-center">
            {labels?.[j] ?? j}
          </div>
        ))}
        {matrix.map((row, i) => (
          <div key={`r${i}`} className="contents">
            <div className="text-[9px] text-slate-500 flex items-center">{labels?.[i] ?? i}</div>
            {row.map((v, j) => (
              <div
                key={j}
                className="h-8 min-w-[28px] flex items-center justify-center text-[10px] font-mono rounded-sm"
                style={{ background: `rgba(34,211,238,${0.08 + (v / max) * 0.7})`, color: v / max > 0.5 ? "#0f172a" : "#e2e8f0" }}
              >
                {v}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="text-[10px] text-slate-500 mt-1">Rows = true class, columns = predicted</div>
    </div>
  );
}

export function Scatter({
  points,
  colors,
  width = 320,
  height = 240,
}: {
  points: { x: number; y: number; cls: number }[];
  colors: string[];
  width?: number;
  height?: number;
}) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs, -1);
  const maxX = Math.max(...xs, 1);
  const minY = Math.min(...ys, -1);
  const maxY = Math.max(...ys, 1);
  const pad = 12;
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (width - pad * 2);
  const sy = (y: number) => pad + (1 - (y - minY) / (maxY - minY || 1)) * (height - pad * 2);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-black/30 rounded-md">
      {points.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.4} fill={colors[p.cls % colors.length]} opacity={0.85} />
      ))}
    </svg>
  );
}
