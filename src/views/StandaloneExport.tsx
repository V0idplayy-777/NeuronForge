import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { Badge, Button, Field, Input, Panel, Stat, Toggle } from "../components/ui";
import { analyzeExport, buildStandaloneBundle, verifyStandalone, type StandaloneBundle, type VerifyReport, type CheckStatus } from "../export/standalone";
import { TARGETS, buildExportFiles, downloadFile, sanitizeName, type ExportTarget } from "../export/package";
import { serializePrepared } from "../data/custom";
import { fmtBytes, fmtInt } from "../lib/format";
import { cn } from "../utils/cn";

const STATUS_TONE: Record<CheckStatus, "emerald" | "rose" | "amber" | "slate"> = { pass: "emerald", fail: "rose", warn: "amber", info: "slate" };

export function StandaloneExport() {
  const s = useStore();
  const trained = s.progress.history.length > 0;
  const [name, setName] = useState(s.projectName);
  const [target, setTarget] = useState<ExportTarget>("js");
  const [includeDataset, setIncludeDataset] = useState(false);
  const [bundle, setBundle] = useState<StandaloneBundle | null>(null);
  const [report, setReport] = useState<VerifyReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const analysis = useMemo(() => analyzeExport(s.model, s.dataset, s.trainingConfig.loss, trained), [s.model, s.dataset, s.trainingConfig.loss, trained]);

  // Any change to the model or data invalidates a previous verification.
  useEffect(() => {
    setBundle(null);
    setReport(null);
  }, [s.model, s.dataset, s.trainingConfig.loss, s.progress.history.length]);

  const run = () => {
    setBusy(true);
    setError(null);
    setReport(null);
    setTimeout(() => {
      try {
        const { bundle: b } = buildStandaloneBundle(s.model, s.dataset, {
          name: name || s.projectName,
          project: s.projectName,
          loss: s.trainingConfig.loss,
          trained,
          includeDataset,
        });
        const r = verifyStandalone(b, s.model, s.dataset);
        setBundle(b);
        setReport(r);
        s.log(r.ok ? "info" : "error", `Standalone export verification ${r.ok ? "passed" : "FAILED"} (${r.checks.filter((c) => c.status === "pass").length} checks passed, ${Math.round(r.durationMs)} ms)`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    }, 30);
  };

  const download = () => {
    if (!bundle) return;
    try {
      const ds = includeDataset ? serializePrepared(s.dataset) : null;
      if (includeDataset && !ds) s.log("warn", "Dataset is too large to include; exported without dataset.json.");
      const out = buildExportFiles(bundle, target, ds ? JSON.stringify(ds) : undefined);
      downloadFile(out.filename, out.data, out.mime);
      const size = typeof out.data === "string" ? out.data.length : out.data.length;
      s.log("info", `Downloaded ${out.filename} (${fmtBytes(size)})`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const failed = report && !report.ok;

  return (
    <Panel title="Standalone Model Export" actions={<Badge tone="cyan">runs without NeuronForge</Badge>}>
      <p className="text-xs text-slate-400 leading-relaxed mb-3">
        Exports the trained model <em>together with the code that runs it</em>: an inference runtime that implements every layer, the weights, and the exact input/output handling used in training
        (preprocessing, tokenizer, sampling). Each export is checked against NeuronForge before you download it.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
        <Stat label="Detected task" value={analysis.task} />
        <Stat label="Parameters" value={fmtInt(analysis.paramCount)} />
        <Stat label="Est. bundle" value={fmtBytes(analysis.approxBundleBytes)} sub="weights as base64" />
        <Stat label="Dataset context" value={analysis.datasetCompatible ? "attached" : "none"} sub={s.dataset.meta.name} />
      </div>
      <p className="text-[11px] text-slate-400 mb-1">{analysis.taskReason}</p>
      <ul className="text-[11px] text-slate-300 space-y-0.5 mb-3 list-disc pl-4">
        {analysis.io.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>

      {analysis.blockers.length > 0 && (
        <div className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200 mb-3">
          {analysis.blockers.map((b) => (
            <div key={b}>{b}</div>
          ))}
        </div>
      )}

      <details className="mb-3">
        <summary className="text-[11px] uppercase tracking-widest text-slate-500 cursor-pointer">Layer support by target ({analysis.perLayer.length} layers)</summary>
        <table className="w-full text-[11px] mt-2">
          <thead className="text-slate-500">
            <tr>
              <th className="text-left font-medium py-1">Layer</th>
              <th className="text-left font-medium">Type</th>
              <th className="font-medium">JS / TS</th>
              <th className="font-medium">Python</th>
              <th className="font-medium">ONNX</th>
            </tr>
          </thead>
          <tbody>
            {analysis.perLayer.map((l) => (
              <tr key={l.name} className="border-t border-white/5 text-slate-300">
                <td className="py-1">{l.name}</td>
                <td className="font-mono">{l.type}</td>
                <td className="text-center">{l.js ? "✓" : "✗"}</td>
                <td className="text-center">{l.python ? "✓" : "✗"}</td>
                <td className="text-center text-slate-600" title="ONNX export not implemented">—</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <details className="mb-3" open>
        <summary className="text-[11px] uppercase tracking-widest text-slate-500 cursor-pointer">Limitations (read before downloading)</summary>
        <ul className="text-[11px] text-amber-100/80 space-y-1 mt-2 list-disc pl-4">
          {analysis.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
          <li>{includeDataset ? "The training dataset will be included as dataset.json." : "The training dataset is not included."}</li>
        </ul>
      </details>

      <div className="grid md:grid-cols-2 gap-2 mb-3">
        {TARGETS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTarget(t.id)}
            className={cn("text-left rounded-md border px-3 py-2", target === t.id ? "border-cyan-400/60 bg-cyan-500/10" : "border-white/8 hover:bg-white/5")}
          >
            <div className="text-xs text-slate-100">
              {t.label} <span className="font-mono text-slate-500">{t.file}</span>
            </div>
            <div className="text-[10px] text-slate-500 leading-snug mt-0.5">{t.description}</div>
          </button>
        ))}
        <div className="rounded-md border border-white/5 px-3 py-2 opacity-50">
          <div className="text-xs text-slate-300">
            ONNX <Badge>not available</Badge>
          </div>
          <div className="text-[10px] text-slate-500 leading-snug mt-0.5">Needs a faithful operator mapping and its own conformance tests. Not offered rather than half-working.</div>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3 mb-3">
        <Field label="Model name">
          <Input value={name} onChange={(e) => setName(e.target.value)} className="w-56" />
        </Field>
        <span className="text-[10px] font-mono text-slate-500 pb-2">package: {sanitizeName(name || s.projectName)}</span>
        <div className="pb-1.5">
          <Toggle checked={includeDataset} onChange={setIncludeDataset} label="Include dataset (optional)" />
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button onClick={run} disabled={busy || analysis.blockers.length > 0}>
          {busy ? "Building & verifying…" : report ? "Re-run verification" : "Build & verify export"}
        </Button>
        <Button variant={failed ? "danger" : "accent"} onClick={download} disabled={!bundle || busy}>
          {failed ? "Download anyway (verification failed)" : `Download ${TARGETS.find((t) => t.id === target)?.label ?? ""}`}
        </Button>
      </div>
      {!bundle && !busy && <p className="text-[10px] text-slate-500 mt-2">Download unlocks after the export has been built and verified against the live model.</p>}
      {error && <div className="mt-3 text-xs text-rose-300">{error}</div>}

      {report && (
        <div className="mt-4 space-y-1.5">
          <div className={cn("text-xs font-medium", report.ok ? "text-emerald-300" : "text-rose-300")}>
            {report.ok ? "Verification passed" : "Verification failed"} · {report.checks.filter((c) => c.status === "pass").length} passed ·{" "}
            {report.checks.filter((c) => c.status === "fail").length} failed · {Math.round(report.durationMs)} ms
          </div>
          {report.checks.map((c) => (
            <div key={c.id} className="rounded-md border border-white/8 bg-black/20">
              <button className="w-full flex items-start gap-2 px-3 py-2 text-left" onClick={() => setOpen(open === c.id ? null : c.id)} disabled={!c.items}>
                <Badge tone={STATUS_TONE[c.status]}>{c.status}</Badge>
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-slate-100">{c.label}</div>
                  <div className="text-[11px] text-slate-400 leading-snug break-words">{c.detail}</div>
                </div>
                {c.items && <span className="text-[10px] text-slate-500">{open === c.id ? "▾" : "▸"} {c.items.length}</span>}
              </button>
              {open === c.id && c.items && (
                <div className="px-3 pb-2 space-y-0.5">
                  {c.items.map((it, i) => (
                    <div key={i} className={cn("text-[10px] font-mono flex gap-2", it.ok ? "text-slate-400" : "text-rose-300")}>
                      <span className="w-8">{it.ok ? "ok" : "FAIL"}</span>
                      <span className="flex-1 truncate">{it.label}</span>
                      <span className="truncate max-w-[50%]">{it.detail}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
