import { StoreProvider, useStore } from "./store";
import { Overview } from "./views/Overview";
import { Builder } from "./views/Builder";
import { DatasetStudio } from "./views/DatasetStudio";
import { TrainingLab } from "./views/TrainingLab";
import { Evaluation } from "./views/Evaluation";
import { InspectorView } from "./views/Inspector";
import { Experiments } from "./views/Experiments";
import { ExportView } from "./views/ExportView";
import { Settings } from "./views/Settings";
import { Badge, Button, Input } from "./components/ui";
import { LineChart } from "./components/Charts";
import { fmt, fmtDuration } from "./lib/format";
import { cn } from "./utils/cn";
import type { ViewId } from "./persist/types";
import { useState } from "react";

const NAV: { id: ViewId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "builder", label: "Model Builder" },
  { id: "dataset", label: "Dataset Studio" },
  { id: "train", label: "Training Lab" },
  { id: "eval", label: "Evaluation" },
  { id: "inspect", label: "Inspector" },
  { id: "experiments", label: "Experiments" },
  { id: "export", label: "Export" },
  { id: "settings", label: "Settings" },
];

export default function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  );
}

function Shell() {
  const s = useStore();
  const [navOpen, setNavOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);

  if (!s.ready) {
    return (
      <div className="min-h-screen bg-nf grid place-items-center text-slate-400 text-sm">
        Restoring local laboratory…
      </div>
    );
  }

  return (
    <div className="h-screen overflow-hidden bg-nf text-slate-200 flex flex-col">
      <header className="h-12 shrink-0 border-b border-white/8 flex items-center gap-3 px-3">
        <button className="xl:hidden text-xs px-2" onClick={() => setNavOpen((v) => !v)} aria-label="Menu">
          ☰
        </button>
        <Logo />
        <div className="h-5 w-px bg-white/10" />
        {renaming ? (
          <Input
            autoFocus
            value={s.projectName}
            onChange={(e) => s.renameProject(e.target.value)}
            onBlur={() => setRenaming(false)}
            onKeyDown={(e) => e.key === "Enter" && setRenaming(false)}
            className="w-48"
          />
        ) : (
          <button className="text-sm text-slate-100 font-medium" onClick={() => setRenaming(true)} title="Rename">
            {s.projectName}
            {s.unsaved && <span className="text-amber-400 ml-1">●</span>}
          </button>
        )}
        <select
          className="h-7 rounded-md bg-black/40 border border-white/10 text-[11px] px-1 max-w-[160px]"
          value={s.projectId}
          onChange={(e) => void s.openProject(e.target.value)}
        >
          {s.projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        <Button size="sm" variant="ghost" onClick={() => void s.saveNow()}>
          Save
        </Button>
        <Button size="sm" variant="ghost" onClick={() => s.createProject()}>
          New
        </Button>
        <div className="ml-auto flex items-center gap-3 text-[11px]">
          <TrainMini />
          <span className="hidden md:inline font-mono text-slate-500">CPU · float32</span>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        <aside
          className={cn(
            "w-48 shrink-0 border-r border-white/8 p-2 overflow-auto bg-nf-panel",
            navOpen ? "absolute z-20 inset-y-12 left-0 shadow-2xl" : "hidden xl:block",
          )}
        >
          <nav className="flex flex-col gap-0.5">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => {
                  s.setView(n.id);
                  setNavOpen(false);
                }}
                className={cn(
                  "text-left rounded-md px-2 py-1.5 text-xs",
                  s.view === n.id ? "bg-cyan-500/15 text-cyan-200" : "text-slate-400 hover:bg-white/5 hover:text-slate-200",
                )}
              >
                {n.label}
              </button>
            ))}
          </nav>
        </aside>

        <main className="flex-1 min-w-0 min-h-0 overflow-auto p-3">{renderView(s.view)}</main>
      </div>

      <Bottom />
    </div>
  );
}

function renderView(view: ViewId) {
  switch (view) {
    case "overview":
      return <Overview />;
    case "builder":
      return <Builder />;
    case "dataset":
      return <DatasetStudio />;
    case "train":
      return <TrainingLab />;
    case "eval":
      return <Evaluation />;
    case "inspect":
      return <InspectorView />;
    case "experiments":
      return <Experiments />;
    case "export":
      return <ExportView />;
    case "settings":
      return <Settings />;
  }
}

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
        <circle cx="5" cy="12" r="2.2" fill="#22d3ee" />
        <circle cx="12" cy="5" r="2.2" fill="#a78bfa" />
        <circle cx="12" cy="19" r="2.2" fill="#f59e0b" />
        <circle cx="19" cy="12" r="2.2" fill="#34d399" />
        <path d="M7 12h3M14 12h3M12 7v3M12 14v3M6.6 10.4l3-3M14.4 7.4l3 3M6.6 13.6l3 3M14.4 16.6l3-3" stroke="rgba(148,163,184,0.6)" strokeWidth="1" />
      </svg>
      <span className="font-semibold tracking-[0.18em] text-[11px] uppercase text-slate-300">NeuronForge</span>
    </div>
  );
}

function TrainMini() {
  const s = useStore();
  const p = s.progress;
  const running = p.status === "running";
  const paused = p.status === "paused";
  return (
    <div className="flex items-center gap-2">
      <Badge tone={running ? "cyan" : paused ? "amber" : p.status === "failed" ? "rose" : p.status === "completed" ? "emerald" : "slate"}>
        {p.status}
      </Badge>
      {running && <span className="font-mono text-slate-400 hidden sm:inline">loss {fmt(p.trainLoss, 3)}</span>}
      {!running && !paused && (
        <Button size="sm" onClick={() => void s.startTraining()} disabled={!s.model.valid}>
          Train
        </Button>
      )}
      {running && (
        <Button size="sm" variant="ghost" onClick={s.pauseTraining}>
          Pause
        </Button>
      )}
      {paused && (
        <Button size="sm" onClick={s.resumeTraining}>
          Resume
        </Button>
      )}
      {(running || paused) && (
        <Button size="sm" variant="danger" onClick={s.stopTraining}>
          Stop
        </Button>
      )}
    </div>
  );
}

function Bottom() {
  const s = useStore();
  const [open, setOpen] = useState(true);
  const p = s.progress;
  return (
    <div className={cn("border-t border-white/8 bg-nf-panel shrink-0", open ? "h-40" : "h-8")}>
      <div className="h-8 flex items-center gap-3 px-3 text-[11px]">
        {(["metrics", "logs", "diag"] as const).map((t) => (
          <button
            key={t}
            className={cn("uppercase tracking-widest", s.bottomTab === t ? "text-cyan-300" : "text-slate-500")}
            onClick={() => {
              s.setBottomTab(t);
              setOpen(true);
            }}
          >
            {t}
          </button>
        ))}
        <span className="ml-auto font-mono text-slate-500">
          {p.epoch > 0 && `ep ${p.epoch}/${p.epochs} · ${fmtDuration(p.elapsedMs)}`}
        </span>
        <button className="text-slate-500" onClick={() => setOpen((v) => !v)} aria-label="Toggle panel">
          {open ? "▾" : "▴"}
        </button>
      </div>
      {open && (
        <div className="h-[calc(100%-2rem)] overflow-auto px-3 pb-2">
          {s.bottomTab === "metrics" && (
            p.history.length ? (
              <LineChart
                height={110}
                series={[
                  { name: "train", color: "#22d3ee", values: p.history.map((h) => ({ x: h.epoch, y: h.trainLoss })) },
                  { name: "val", color: "#f59e0b", values: p.history.filter((h) => h.valLoss !== undefined).map((h) => ({ x: h.epoch, y: h.valLoss! })) },
                ]}
              />
            ) : (
              <p className="text-[11px] text-slate-500">No metrics yet. Start training to stream real losses.</p>
            )
          )}
          {s.bottomTab === "logs" && (
            <div className="font-mono text-[11px] space-y-0.5">
              {s.logs.slice(-40).map((l, i) => (
                <div key={i} className={l.level === "error" ? "text-rose-300" : l.level === "warn" ? "text-amber-200" : "text-slate-400"}>
                  {new Date(l.t).toLocaleTimeString()}  {l.msg}
                </div>
              ))}
              {s.logs.length === 0 && <div className="text-slate-600">No log entries.</div>}
            </div>
          )}
          {s.bottomTab === "diag" && (
            <div className="text-[11px] font-mono text-slate-400 space-y-1">
              <div>backend: CPU JS float32 · yielding trainer</div>
              <div>params: {s.model.paramCount().trainable} trainable / {s.model.paramCount().nonTrainable} frozen-or-buffers</div>
              <div>input {JSON.stringify(s.architecture.inputShape)} · valid={String(s.model.valid)}</div>
              <div>dataset {s.dataset.meta.name} train={s.dataset.train.xs.shape[0]}</div>
              {s.model.issues.map((i) => (
                <div key={i.layerId} className="text-amber-300">
                  {i.layerName}: {i.message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
