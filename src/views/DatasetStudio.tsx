import { useMemo, useState } from "react";
import { useStore } from "../store";
import { Button, Field, Input, Select, Panel, Stat, Badge } from "../components/ui";
import { BUILTIN_INFO, datasetStats, generateBuiltin, type BuiltinId } from "../data/datasets";
import { DEFAULT_CSV_CONFIG, parseCsv, type CsvImportConfig } from "../data/csv";
import { materializeCsv } from "../data/preprocess";
import { compileExpr } from "../data/expr";
import type { FormulaSpec } from "../data/custom";
import { Scatter } from "../components/Charts";
import { fmt, fmtInt, fmtShape } from "../lib/format";
import { cn } from "../utils/cn";

const COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f43f5e"];

export function DatasetStudio() {
  const s = useStore();
  const ds = s.dataset;
  const st = datasetStats(ds);
  const [csvCfg, setCsvCfg] = useState<CsvImportConfig>(DEFAULT_CSV_CONFIG);
  const [csvPreview, setCsvPreview] = useState<{ headers: string[]; rows: string[][] } | null>(null);
  const [csvWarn, setCsvWarn] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const scatter = useMemo(() => {
    if (ds.meta.inputShape.length !== 1 || ds.meta.inputShape[0] !== 2) return null;
    const pts: { x: number; y: number; cls: number }[] = [];
    const B = Math.min(400, ds.allX.shape[0]!);
    const yDim = ds.allY.size / ds.allY.shape[0]!;
    for (let i = 0; i < B; i++) {
      let cls = 0;
      if (yDim === 1) cls = ds.allY.data[i]! >= 0.5 ? 1 : 0;
      else {
        let best = 0;
        let bv = -Infinity;
        for (let c = 0; c < yDim; c++) {
          const v = ds.allY.data[i * yDim + c]!;
          if (v > bv) {
            bv = v;
            best = c;
          }
        }
        cls = best;
      }
      pts.push({ x: ds.allX.data[i * 2]!, y: ds.allX.data[i * 2 + 1]!, cls });
    }
    return pts;
  }, [ds]);

  const hist = useMemo(() => classHist(ds), [ds]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)] gap-3 min-h-0">
      <div className="space-y-3 overflow-auto">
        <Panel title="Built-in datasets">
          <div className="space-y-1 max-h-[360px] overflow-auto pr-1">
            {(Object.keys(BUILTIN_INFO) as BuiltinId[]).map((id) => {
              const info = BUILTIN_INFO[id];
              const active = s.builtinId === id && ds.meta.source === "builtin";
              return (
                <button
                  key={id}
                  onClick={() => s.setDataset(id)}
                  className={`w-full text-left rounded-md border px-2 py-2 ${active ? "border-cyan-400/50 bg-cyan-500/10" : "border-white/8 hover:bg-white/5"}`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-100">{info.name}</span>
                    <Badge tone={info.task === "regression" ? "amber" : info.task === "multiclass" ? "violet" : "cyan"}>{info.task}</Badge>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5 leading-snug">{info.description}</div>
                </button>
              );
            })}
          </div>
        </Panel>
        <Panel title="Generator">
          <div className="grid grid-cols-2 gap-2">
            <Field label="Samples">
              <Input type="number" value={s.synth.n} onChange={(e) => s.setSynth({ n: Number(e.target.value) || 1 }, false)} />
            </Field>
            <Field label="Noise">
              <Input type="number" step={0.01} value={s.synth.noise} onChange={(e) => s.setSynth({ noise: Number(e.target.value) }, false)} />
            </Field>
            <Field label="Seed">
              <Input type="number" value={s.synth.seed} onChange={(e) => s.setSynth({ seed: Number(e.target.value) || 1 }, false)} />
            </Field>
            <Field label="Classes">
              <Input type="number" min={2} max={5} value={s.synth.classes} onChange={(e) => s.setSynth({ classes: Number(e.target.value) || 2 }, false)} />
            </Field>
            <Field label="Train %">
              <Input type="number" step={0.05} value={s.synth.trainFrac} onChange={(e) => s.setSynth({ trainFrac: Number(e.target.value) }, false)} />
            </Field>
            <Field label="Val %">
              <Input type="number" step={0.05} value={s.synth.valFrac} onChange={(e) => s.setSynth({ valFrac: Number(e.target.value) }, false)} />
            </Field>
          </div>
          <Button className="mt-3 w-full" onClick={() => s.setDataset(s.builtinId, s.synth)}>
            Regenerate
          </Button>
          <p className="text-[10px] text-slate-500 mt-2">Regeneration replaces the actual training data. Test split is held out and never used to fit preprocessing.</p>
        </Panel>
      </div>

      <div className="space-y-3 overflow-auto min-w-0">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <Stat label="Train" value={fmtInt(st.nTrain)} />
          <Stat label="Validation" value={fmtInt(st.nVal)} />
          <Stat label="Test" value={fmtInt(st.nTest)} />
          <Stat label="Input shape" value={fmtShape(ds.meta.inputShape)} />
        </div>
        <Panel title={ds.meta.name} actions={<Badge>{ds.meta.source}</Badge>}>
          <p className="text-xs text-slate-400 leading-relaxed">{ds.meta.description}</p>
          {ds.meta.attribution && <p className="text-[10px] text-slate-500 mt-2">{ds.meta.attribution}</p>}
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Feature range</div>
              <div className="text-xs font-mono text-slate-300">
                x ∈ [{fmt(st.xMin)}, {fmt(st.xMax)}] · y ∈ [{fmt(st.yMin)}, {fmt(st.yMax)}]
              </div>
              {hist && (
                <div className="mt-3 space-y-1">
                  {hist.map((h) => (
                    <div key={h.label} className="flex items-center gap-2 text-[11px]">
                      <span className="w-16 text-slate-400 truncate">{h.label}</span>
                      <div className="flex-1 h-2 rounded bg-white/5">
                        <div className="h-2 rounded bg-cyan-400/70" style={{ width: `${(h.n / Math.max(1, st.nTrain + st.nVal + st.nTest)) * 100}%` }} />
                      </div>
                      <span className="font-mono text-slate-500 w-8 text-right">{h.n}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {scatter && (
              <div>
                <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Scatter (first 400)</div>
                <Scatter points={scatter} colors={COLORS} />
              </div>
            )}
          </div>
          <SampleTable ds={ds} />
        </Panel>

        <Panel title="Import CSV">
          <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
            Parsing is local. Quoted fields and missing values are handled. Normalization statistics are fit on the training split only.
          </p>
          <input
            type="file"
            accept=".csv,text/csv"
            className="text-xs text-slate-300"
            onChange={async (e) => {
              setErr(null);
              const f = e.target.files?.[0];
              if (!f) return;
              const text = await f.text();
              try {
                const parsed = parseCsv(text);
                if (!parsed.headers.length) throw new Error("No columns found.");
                setCsvPreview(parsed);
                const nums = parsed.headers.map((_, i) => i);
                setCsvCfg({
                  ...DEFAULT_CSV_CONFIG,
                  featureIdx: nums.slice(0, Math.max(1, nums.length - 1)),
                  targetIdx: [nums[nums.length - 1]!],
                });
              } catch (er) {
                setErr(er instanceof Error ? er.message : String(er));
              }
            }}
          />
          {err && <div className="mt-2 text-xs text-rose-300">{err}</div>}
          {csvPreview && (
            <div className="mt-3 space-y-3">
              <div className="text-[11px] text-slate-400">
                {csvPreview.headers.length} columns · {csvPreview.rows.length} rows
              </div>
              <div className="overflow-auto max-h-40 border border-white/8 rounded-md">
                <table className="text-[10px] font-mono w-full">
                  <thead>
                    <tr className="text-slate-400">
                      {csvPreview.headers.map((h) => (
                        <th key={h} className="px-2 py-1 text-left font-medium">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvPreview.rows.slice(0, 8).map((r, i) => (
                      <tr key={i} className="text-slate-300">
                        {r.map((c, j) => (
                          <td key={j} className="px-2 py-0.5 whitespace-nowrap">
                            {c}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Feature columns">
                  <select
                    multiple
                    className="h-24 w-full rounded-md bg-black/40 border border-white/10 text-xs"
                    value={csvCfg.featureIdx.map(String)}
                    onChange={(e) =>
                      setCsvCfg({
                        ...csvCfg,
                        featureIdx: [...e.target.selectedOptions].map((o) => Number(o.value)),
                      })
                    }
                  >
                    {csvPreview.headers.map((h, i) => (
                      <option key={i} value={i}>
                        {h}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Target columns">
                  <select
                    multiple
                    className="h-24 w-full rounded-md bg-black/40 border border-white/10 text-xs"
                    value={csvCfg.targetIdx.map(String)}
                    onChange={(e) =>
                      setCsvCfg({
                        ...csvCfg,
                        targetIdx: [...e.target.selectedOptions].map((o) => Number(o.value)),
                      })
                    }
                  >
                    {csvPreview.headers.map((h, i) => (
                      <option key={i} value={i}>
                        {h}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Missing values">
                  <Select value={csvCfg.missing} onChange={(e) => setCsvCfg({ ...csvCfg, missing: e.target.value as CsvImportConfig["missing"] })}>
                    <option value="mean">fill with train mean</option>
                    <option value="zero">fill with 0</option>
                    <option value="drop">drop row</option>
                  </Select>
                </Field>
                <Field label="Categorical">
                  <Select value={csvCfg.categorical} onChange={(e) => setCsvCfg({ ...csvCfg, categorical: e.target.value as CsvImportConfig["categorical"] })}>
                    <option value="onehot">one-hot</option>
                    <option value="label">label encode</option>
                    <option value="drop">drop</option>
                  </Select>
                </Field>
                <Field label="Normalize">
                  <Select value={csvCfg.normalize} onChange={(e) => setCsvCfg({ ...csvCfg, normalize: e.target.value as CsvImportConfig["normalize"] })}>
                    <option value="zscore">standardize (train stats)</option>
                    <option value="minmax">min-max (train stats)</option>
                    <option value="none">none</option>
                  </Select>
                </Field>
              </div>
              <Button
                onClick={() => {
                  try {
                    const { dataset, warnings } = materializeCsv(csvPreview.headers, csvPreview.rows, csvCfg);
                    setCsvWarn(warnings);
                    s.loadCustomDataset(dataset);
                    s.setInputShape(dataset.meta.inputShape);
                  } catch (er) {
                    setErr(er instanceof Error ? er.message : String(er));
                  }
                }}
              >
                Commit imported dataset
              </Button>
              {csvWarn.map((w) => (
                <div key={w} className="text-[11px] text-amber-200">
                  {w}
                </div>
              ))}
            </div>
          )}
        </Panel>

        <DrawPanel />
        <FormulaPanel />
      </div>
    </div>
  );
}

/* ------------------------------- Draw your own 2D dataset ------------------------------- */

const DRAW_W = 360;
const DRAW_H = 260;
const DOM_X: [number, number] = [-2, 2];
const DOM_Y: [number, number] = [-1.5, 1.5];

function DrawPanel() {
  const s = useStore();
  const initial = s.customSpec?.kind === "points" ? s.customSpec.points : [];
  const [points, setPoints] = useState<{ x: number; y: number; cls: number }[]>(initial);
  const [cls, setCls] = useState(0);
  const [classes, setClasses] = useState(s.customSpec?.kind === "points" ? s.customSpec.classes : 2);
  const [name, setName] = useState(s.customSpec?.kind === "points" ? s.customSpec.name : "My drawn dataset");
  const [err, setErr] = useState<string | null>(null);

  const toSvg = (x: number, y: number) => ({
    px: ((x - DOM_X[0]) / (DOM_X[1] - DOM_X[0])) * DRAW_W,
    py: (1 - (y - DOM_Y[0]) / (DOM_Y[1] - DOM_Y[0])) * DRAW_H,
  });

  const counts = useMemo(() => {
    const c = new Array(classes).fill(0);
    for (const p of points) if (p.cls < classes) c[p.cls]++;
    return c as number[];
  }, [points, classes]);

  return (
    <Panel title="Draw a dataset" actions={<Badge tone="violet">custom</Badge>}>
      <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
        Click to place real training points. Pick a class, click the canvas, then commit. The dataset is stored with the project and reloads after a refresh.
      </p>
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <Field label="Classes">
          <Select
            value={classes}
            onChange={(e) => {
              const k = Number(e.target.value);
              setClasses(k);
              if (cls >= k) setCls(0);
              setPoints((ps) => ps.filter((p) => p.cls < k));
            }}
            className="w-20"
          >
            {[2, 3, 4, 5].map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </Select>
        </Field>
        <div className="flex items-end gap-1 pb-0.5">
          {Array.from({ length: classes }, (_, i) => (
            <button
              key={i}
              onClick={() => setCls(i)}
              aria-label={`Draw class ${i}`}
              className={cn(
                "h-7 px-2.5 rounded-md border text-[11px] font-mono",
                cls === i ? "border-white/60" : "border-white/10 opacity-70",
              )}
              style={{ background: `${COLORS[i % COLORS.length]}22`, color: COLORS[i % COLORS.length] }}
            >
              {i} · {counts[i] ?? 0}
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-1">
          <Button size="sm" variant="ghost" onClick={() => setPoints((p) => p.slice(0, -1))} disabled={!points.length}>
            Undo
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setPoints([])} disabled={!points.length}>
            Clear
          </Button>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${DRAW_W} ${DRAW_H}`}
        className="w-full h-auto bg-black/40 rounded-md cursor-crosshair border border-white/8"
        onClick={(e) => {
          const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
          const fx = (e.clientX - rect.left) / rect.width;
          const fy = (e.clientY - rect.top) / rect.height;
          const x = DOM_X[0] + fx * (DOM_X[1] - DOM_X[0]);
          const y = DOM_Y[0] + (1 - fy) * (DOM_Y[1] - DOM_Y[0]);
          setPoints((ps) => [...ps, { x: Math.round(x * 1000) / 1000, y: Math.round(y * 1000) / 1000, cls }]);
        }}
      >
        <line x1={DRAW_W / 2} y1={0} x2={DRAW_W / 2} y2={DRAW_H} stroke="rgba(255,255,255,0.07)" />
        <line x1={0} y1={DRAW_H / 2} x2={DRAW_W} y2={DRAW_H / 2} stroke="rgba(255,255,255,0.07)" />
        {points.map((p, i) => {
          const { px, py } = toSvg(p.x, p.y);
          return <circle key={i} cx={px} cy={py} r={4} fill={COLORS[p.cls % COLORS.length]} opacity={0.9} />;
        })}
      </svg>
      <div className="mt-2 flex flex-wrap items-end gap-2">
        <Field label="Name">
          <Input value={name} onChange={(e) => setName(e.target.value)} className="w-44" />
        </Field>
        <Button
          onClick={() => {
            setErr(null);
            try {
              s.setCustomDataset({
                kind: "points",
                name: name || "Drawn dataset",
                points,
                classes,
                trainFrac: s.synth.trainFrac,
                valFrac: s.synth.valFrac,
                seed: s.synth.seed,
              });
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
          disabled={points.length < 6}
        >
          Commit {points.length} points
        </Button>
        <span className="text-[10px] text-slate-500 pb-2">Split uses the Generator fractions ({Math.round(s.synth.trainFrac * 100)}/{Math.round(s.synth.valFrac * 100)}/{Math.round(Math.max(0, 1 - s.synth.trainFrac - s.synth.valFrac) * 100)}).</span>
      </div>
      {err && <div className="mt-2 text-xs text-rose-300">{err}</div>}
    </Panel>
  );
}

/* ------------------------------- Formula generator ------------------------------- */

function FormulaPanel() {
  const s = useStore();
  const prev = s.customSpec?.kind === "formula" ? s.customSpec : null;
  const [spec, setSpec] = useState<FormulaSpec>(
    prev ?? {
      kind: "formula",
      name: "",
      mode: "regression",
      expr: "sin(3*x) * exp(-x^2/4)",
      n: 300,
      noise: 0.05,
      seed: 7,
      xMin: -3,
      xMax: 3,
      yMin: -2,
      yMax: 2,
      trainFrac: 0.7,
      valFrac: 0.15,
    },
  );
  const [err, setErr] = useState<string | null>(null);

  const exprCheck = useMemo(() => {
    try {
      const { vars } = compileExpr(spec.expr);
      const allowed = spec.mode === "regression" ? ["x"] : ["x", "y"];
      const bad = vars.filter((v) => !allowed.includes(v));
      if (bad.length) return { ok: false, msg: `Unknown variable(s): ${bad.join(", ")}. ${spec.mode === "regression" ? "Use x only." : "Use x and y."}` };
      return { ok: true, msg: `Parsed OK. Variables: ${vars.join(", ") || "none"}` };
    } catch (e) {
      return { ok: false, msg: e instanceof Error ? e.message : String(e) };
    }
  }, [spec.expr, spec.mode]);

  const upd = (p: Partial<FormulaSpec>) => setSpec((x) => ({ ...x, ...p }));

  return (
    <Panel title="Formula dataset" actions={<Badge tone="violet">custom</Badge>}>
      <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
        Generate data from a math expression. The expression is parsed with a safe recursive-descent parser — it is never executed as JavaScript.
        Regression: <code className="text-cyan-300">y = f(x)</code>. Classification: label = <code className="text-cyan-300">sign(f(x, y))</code>.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Field label="Mode">
          <Select value={spec.mode} onChange={(e) => upd({ mode: e.target.value as FormulaSpec["mode"] })}>
            <option value="regression">regression y=f(x)</option>
            <option value="classify">classify sign f(x,y)</option>
          </Select>
        </Field>
        <Field label="Samples">
          <Input type="number" min={8} max={5000} value={spec.n} onChange={(e) => upd({ n: Number(e.target.value) || 100 })} />
        </Field>
        <Field label="Noise σ">
          <Input type="number" step={0.01} value={spec.noise} onChange={(e) => upd({ noise: Number(e.target.value) })} />
        </Field>
        <Field label="Seed">
          <Input type="number" value={spec.seed} onChange={(e) => upd({ seed: Number(e.target.value) || 1 })} />
        </Field>
        <Field label="x min">
          <Input type="number" step={0.5} value={spec.xMin} onChange={(e) => upd({ xMin: Number(e.target.value) })} />
        </Field>
        <Field label="x max">
          <Input type="number" step={0.5} value={spec.xMax} onChange={(e) => upd({ xMax: Number(e.target.value) })} />
        </Field>
        {spec.mode === "classify" && (
          <>
            <Field label="y min">
              <Input type="number" step={0.5} value={spec.yMin} onChange={(e) => upd({ yMin: Number(e.target.value) })} />
            </Field>
            <Field label="y max">
              <Input type="number" step={0.5} value={spec.yMax} onChange={(e) => upd({ yMax: Number(e.target.value) })} />
            </Field>
          </>
        )}
      </div>
      <div className="mt-2">
        <Field label={spec.mode === "regression" ? "f(x)" : "f(x, y)"} hint="sin cos tan exp log sqrt abs tanh min max pow · constants pi, e · operators + − * / % ^">
          <Input value={spec.expr} onChange={(e) => upd({ expr: e.target.value })} className="font-mono" />
        </Field>
        <div className={cn("text-[11px] mt-1", exprCheck.ok ? "text-emerald-300" : "text-rose-300")}>{exprCheck.msg}</div>
      </div>
      <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
        {(spec.mode === "regression"
          ? ["sin(3*x) * exp(-x^2/4)", "x^3 - 2*x", "abs(x) * sin(5*x)", "tanh(2*x) + 0.3*sin(8*x)"]
          : ["x^2 + y^2 - 1", "sin(3*x) - y", "x*y", "abs(x) + abs(y) - 1.2"]
        ).map((ex) => (
          <button key={ex} onClick={() => upd({ expr: ex })} className="rounded border border-white/10 px-1.5 py-0.5 font-mono text-slate-400 hover:text-cyan-300">
            {ex}
          </button>
        ))}
      </div>
      <div className="mt-3 flex items-end gap-2">
        <Field label="Name (optional)">
          <Input value={spec.name} onChange={(e) => upd({ name: e.target.value })} className="w-44" placeholder="auto from formula" />
        </Field>
        <Button
          disabled={!exprCheck.ok}
          onClick={() => {
            setErr(null);
            try {
              s.setCustomDataset({ ...spec, trainFrac: s.synth.trainFrac, valFrac: s.synth.valFrac });
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
        >
          Generate dataset
        </Button>
      </div>
      {err && <div className="mt-2 text-xs text-rose-300">{err}</div>}
    </Panel>
  );
}

function SampleTable({ ds }: { ds: ReturnType<typeof generateBuiltin> }) {
  const n = Math.min(8, ds.train.xs.shape[0]!);
  const feat = Math.min(6, ds.meta.featureNames.length);
  const rows = [];
  const xRest = ds.train.xs.size / ds.train.xs.shape[0]!;
  const yRest = ds.train.ys.size / ds.train.ys.shape[0]!;
  for (let i = 0; i < n; i++) {
    const x = Array.from(ds.train.xs.data.subarray(i * xRest, i * xRest + feat)).map((v) => fmt(v, 3));
    const y = Array.from(ds.train.ys.data.subarray(i * yRest, (i + 1) * yRest)).map((v) => fmt(v, 3));
    rows.push({ x, y });
  }
  return (
    <div className="mt-4 overflow-auto">
      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Train split preview</div>
      <table className="text-[11px] font-mono w-full">
        <thead>
          <tr className="text-slate-500">
            {ds.meta.featureNames.slice(0, feat).map((h) => (
              <th key={h} className="text-left font-medium pr-3">
                {h}
              </th>
            ))}
            <th className="text-left font-medium">target</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="text-slate-300">
              {r.x.map((v, j) => (
                <td key={j} className="pr-3">
                  {v}
                </td>
              ))}
              <td>{r.y.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function classHist(ds: ReturnType<typeof generateBuiltin>): { label: string; n: number }[] | null {
  if (ds.meta.task === "regression") return null;
  const yDim = ds.allY.size / ds.allY.shape[0]!;
  const B = ds.allY.shape[0]!;
  if (yDim === 1) {
    let a = 0;
    let b = 0;
    for (let i = 0; i < B; i++) (ds.allY.data[i]! >= 0.5 ? b++ : a++);
    const names = ds.meta.classNames ?? ["0", "1"];
    return [
      { label: names[0] ?? "0", n: a },
      { label: names[1] ?? "1", n: b },
    ];
  }
  const counts = new Array(yDim).fill(0);
  for (let i = 0; i < B; i++) {
    let best = 0;
    let bv = -Infinity;
    for (let c = 0; c < yDim; c++) {
      const v = ds.allY.data[i * yDim + c]!;
      if (v > bv) {
        bv = v;
        best = c;
      }
    }
    counts[best]++;
  }
  return counts.map((n, i) => ({ label: ds.meta.classNames?.[i] ?? String(i), n }));
}
