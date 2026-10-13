import { useMemo, useState } from "react";
import { useStore } from "../store";
import { Panel, Select, Stat, Badge } from "../components/ui";
import { Heatmap, Histogram } from "../components/Charts";
import { stats, histogram, Tensor } from "../engine/tensor";
import { fmt, fmtInt, fmtShape } from "../lib/format";
import { sliceBatch } from "../engine/tensor";

export function InspectorView() {
  const s = useStore();
  const [layerId, setLayerId] = useState(s.model.layers[0]?.def.id ?? "");
  const [paramName, setParamName] = useState("W");
  const layer = s.model.layers.find((l) => l.def.id === layerId) ?? s.model.layers[0];
  const paramNames = layer ? Object.keys(layer.params) : [];
  const tensor = layer?.params[paramName] ?? layer?.params[paramNames[0] ?? ""];

  const act = useMemo(() => {
    if (!s.model.valid) return null;
    try {
      const n = s.dataset.train.xs.shape[0]!;
      const idx = Math.min(s.inspectSample, n - 1);
      const x = sliceBatch(s.dataset.train.xs, idx, 1);
      return s.model.forward(x, false, true);
    } catch {
      return null;
    }
  }, [s.model, s.dataset, s.inspectSample]);

  const actLayer = act?.activations.find((a) => a.id === layerId);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[240px_minmax(0,1fr)] gap-3 min-h-0">
      <Panel title="Parameters">
        <div className="space-y-1">
          {s.model.layers.map((l) => {
            const n = Object.values(l.params).reduce((a, t) => a + t.size, 0);
            return (
              <button
                key={l.def.id}
                onClick={() => {
                  setLayerId(l.def.id);
                  setParamName(Object.keys(l.params)[0] ?? "W");
                }}
                className={`w-full text-left rounded-md px-2 py-1.5 text-xs border ${l.def.id === layerId ? "border-cyan-400/40 bg-cyan-500/10" : "border-transparent hover:bg-white/5"}`}
              >
                <div className="text-slate-200">{l.def.name}</div>
                <div className="text-[10px] text-slate-500 font-mono">
                  {l.def.type} · {fmtInt(n)} values
                </div>
              </button>
            );
          })}
        </div>
      </Panel>

      <div className="space-y-3 overflow-auto min-w-0">
        {!layer && <p className="text-xs text-slate-500">No compiled layers.</p>}
        {layer && (
          <>
            <div className="flex items-center gap-2">
              <h2 className="text-sm text-slate-100">{layer.def.name}</h2>
              <Badge>{layer.def.type}</Badge>
              {layer.frozen && <Badge tone="amber">frozen</Badge>}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <Stat label="Input" value={fmtShape(layer.inputShape)} />
              <Stat label="Output" value={fmtShape(layer.outputShape)} />
              <Stat label="Params" value={fmtInt(Object.values(layer.params).reduce((a, t) => a + t.size, 0))} />
              <Stat label="Sample" value={s.inspectSample} />
            </div>
            {paramNames.length > 0 && (
              <Panel title="Weight tensor">
                <Select value={paramName} onChange={(e) => setParamName(e.target.value)}>
                  {paramNames.map((n) => (
                    <option key={n} value={n}>
                      {n} {fmtShape(layer.params[n]!.shape)}
                    </option>
                  ))}
                </Select>
                {tensor && <TensorView t={tensor} />}
              </Panel>
            )}
            {paramNames.length === 0 && <p className="text-xs text-slate-500">This layer has no trainable parameters.</p>}
            <Panel title="Activations on train sample">
              <label className="text-[11px] text-slate-400 flex items-center gap-2 mb-2">
                index
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, s.dataset.train.xs.shape[0]! - 1)}
                  value={s.inspectSample}
                  onChange={(e) => s.setInspectSample(Number(e.target.value))}
                  className="flex-1"
                />
              </label>
              {actLayer ? (
                <TensorView t={actLayer.tensor} />
              ) : (
                <p className="text-xs text-slate-500">Could not collect activations. Check that the dataset matches the model input shape.</p>
              )}
            </Panel>
            {s.progress.gradNorms.length > 0 && (
              <Panel title="Last-step gradient norms">
                <p className="text-[10px] text-slate-500 mb-2">
                  Gradients are captured during training before the optimiser step. They are not reconstructed from weight deltas.
                </p>
                {s.progress.gradNorms.map((g) => (
                  <div key={g.id} className="flex justify-between text-xs font-mono text-slate-300">
                    <span>{g.name}</span>
                    <span>{fmt(g.gradNorm)}</span>
                  </div>
                ))}
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TensorView({ t }: { t: Tensor }) {
  const st = stats(t);
  const hist = histogram(t);
  const rows = t.rank >= 2 ? t.shape[t.rank - 2]! : 1;
  const cols = t.shape[t.rank - 1]!;
  const maxShow = 80;
  const show = Math.min(t.size, maxShow);
  return (
    <div className="mt-3 space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <Stat label="min" value={fmt(st.min)} />
        <Stat label="max" value={fmt(st.max)} />
        <Stat label="mean" value={fmt(st.mean)} />
        <Stat label="std" value={fmt(st.std)} />
        <Stat label="L2" value={fmt(st.l2)} />
      </div>
      <Histogram counts={hist.counts} edges={hist.edges} />
      {t.rank >= 2 && rows * cols === t.size && rows <= 64 && cols <= 64 && (
        <Heatmap data={t.data} rows={rows} cols={cols} title={`heatmap ${fmtShape(t.shape)}`} />
      )}
      <div className="text-[10px] font-mono text-slate-400 break-all leading-relaxed max-h-32 overflow-auto">
        {Array.from(t.data.subarray(0, show))
          .map((v) => fmt(v, 4))
          .join("  ")}
        {t.size > show ? `  … +${t.size - show} more` : ""}
      </div>
    </div>
  );
}
