import { useStore } from "../store";
import { Button, Field, Input, Select, Panel, Stat, Badge } from "../components/ui";
import { LineChart } from "../components/Charts";
import { LOSS_DOCS } from "../engine/losses";
import { OPTIMIZER_DOCS } from "../engine/optimizers";
import { fmt, fmtDuration, fmtInt, fmtBytes } from "../lib/format";
import type { LossName, OptimizerName } from "../engine/types";

export function TrainingLab() {
  const s = useStore();
  const p = s.progress;
  const running = p.status === "running";
  const paused = p.status === "paused";
  const cfg = s.trainingConfig;
  const mem = s.model.memoryEstimate(cfg.batchSize, cfg.optimizer.name);
  const cls = cfg.loss !== "mse" && cfg.loss !== "mae";

  const series = [
    { name: "train loss", color: "#22d3ee", values: p.history.map((h) => ({ x: h.epoch, y: h.trainLoss })) },
    { name: "val loss", color: "#f59e0b", values: p.history.filter((h) => h.valLoss !== undefined).map((h) => ({ x: h.epoch, y: h.valLoss! })) },
  ];
  const accSeries = cls
    ? [
        { name: "train acc", color: "#34d399", values: p.history.filter((h) => h.trainAcc !== undefined).map((h) => ({ x: h.epoch, y: h.trainAcc! })) },
        { name: "val acc", color: "#a78bfa", values: p.history.filter((h) => h.valAcc !== undefined).map((h) => ({ x: h.epoch, y: h.valAcc! })) },
      ]
    : [];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[320px_minmax(0,1fr)] gap-3 min-h-0">
      <div className="space-y-3 overflow-auto">
        <Panel title="Configuration">
          <div className="space-y-2">
            <Field label="Loss" hint={LOSS_DOCS[cfg.loss]}>
              <Select value={cfg.loss} onChange={(e) => s.setTrainingConfig({ loss: e.target.value as LossName })} disabled={running || paused}>
                <option value="mse">MSE</option>
                <option value="mae">MAE</option>
                <option value="bce">Binary cross-entropy</option>
                <option value="cce">Categorical cross-entropy</option>
                <option value="sparse_cce">Sparse categorical CE</option>
              </Select>
            </Field>
            <Field label="Optimiser" hint={OPTIMIZER_DOCS[cfg.optimizer.name]}>
              <Select
                value={cfg.optimizer.name}
                disabled={running || paused}
                onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, name: e.target.value as OptimizerName } })}
              >
                <option value="sgd">SGD</option>
                <option value="momentum">SGD + momentum</option>
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
              </Select>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Epochs">
                <Input type="number" min={1} value={cfg.epochs} disabled={running} onChange={(e) => s.setTrainingConfig({ epochs: Math.max(1, Number(e.target.value) || 1) })} />
              </Field>
              <Field label="Batch size">
                <Input type="number" min={1} value={cfg.batchSize} disabled={running} onChange={(e) => s.setTrainingConfig({ batchSize: Math.max(1, Number(e.target.value) || 1) })} />
              </Field>
              <Field label="Learning rate">
                <Input
                  type="number"
                  step={0.001}
                  value={cfg.optimizer.learningRate}
                  disabled={running}
                  onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, learningRate: Number(e.target.value) } })}
                />
              </Field>
              <Field label="Seed">
                <Input type="number" value={cfg.seed} disabled={running} onChange={(e) => s.setTrainingConfig({ seed: Number(e.target.value) || 1 })} />
              </Field>
              {(cfg.optimizer.name === "momentum") && (
                <Field label="Momentum">
                  <Input
                    type="number"
                    step={0.05}
                    value={cfg.optimizer.momentum}
                    disabled={running}
                    onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, momentum: Number(e.target.value) } })}
                  />
                </Field>
              )}
              {(cfg.optimizer.name === "adam" || cfg.optimizer.name === "adamw") && (
                <>
                  <Field label="β1">
                    <Input
                      type="number"
                      step={0.01}
                      value={cfg.optimizer.beta1}
                      disabled={running}
                      onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, beta1: Number(e.target.value) } })}
                    />
                  </Field>
                  <Field label="β2">
                    <Input
                      type="number"
                      step={0.001}
                      value={cfg.optimizer.beta2}
                      disabled={running}
                      onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, beta2: Number(e.target.value) } })}
                    />
                  </Field>
                </>
              )}
              <Field label="Weight decay">
                <Input
                  type="number"
                  step={0.0001}
                  value={cfg.optimizer.weightDecay}
                  disabled={running}
                  onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, weightDecay: Number(e.target.value) } })}
                />
              </Field>
              <Field label="Grad clip (0=off)">
                <Input
                  type="number"
                  step={0.1}
                  value={cfg.optimizer.gradClip}
                  disabled={running}
                  onChange={(e) => s.setTrainingConfig({ optimizer: { ...cfg.optimizer, gradClip: Number(e.target.value) } })}
                />
              </Field>
              <Field label="Early stop patience">
                <Input
                  type="number"
                  min={0}
                  value={cfg.earlyStoppingPatience}
                  disabled={running}
                  onChange={(e) => s.setTrainingConfig({ earlyStoppingPatience: Math.max(0, Number(e.target.value) || 0) })}
                />
              </Field>
              <Field label="Shuffle">
                <Select value={cfg.shuffle ? "1" : "0"} disabled={running} onChange={(e) => s.setTrainingConfig({ shuffle: e.target.value === "1" })}>
                  <option value="1">yes</option>
                  <option value="0">no</option>
                </Select>
              </Field>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 mt-4">
            {!running && !paused && (
              <Button onClick={() => void s.startTraining()} disabled={!s.model.valid}>
                Start training
              </Button>
            )}
            {running && (
              <Button variant="ghost" onClick={s.pauseTraining}>
                Pause
              </Button>
            )}
            {paused && (
              <Button onClick={s.resumeTraining}>Resume</Button>
            )}
            {(running || paused) && (
              <Button variant="danger" onClick={s.stopTraining}>
                Stop
              </Button>
            )}
            <Button variant="ghost" onClick={s.reinitWeights} disabled={running || paused}>
              Reset weights
            </Button>
          </div>
          {!s.model.valid && <p className="text-[11px] text-rose-300 mt-2">{s.model.issues[0]?.message}</p>}
          <p className="text-[10px] text-slate-500 mt-3 leading-relaxed">
            Stop ends the run but keeps the current weights. Reset weights discards learned parameters. Pause preserves optimiser state and continues from the next batch.
            Identical seeds are not bit-identical across browsers.
          </p>
        </Panel>
        <Panel title="Resources">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Params" value={fmtInt(s.model.paramCount().trainable)} />
            <Stat label="Est. memory" value={fmtBytes(mem.totalBytes)} />
            <Stat label="Backend" value="CPU · JS" sub="Cooperative yielding, float32" />
            <Stat label="Dataset" value={s.dataset.meta.name} />
          </div>
        </Panel>
      </div>

      <div className="space-y-3 min-w-0 overflow-auto">
        <div className="flex items-center gap-2">
          <StatusBadge status={p.status} />
          <span className="text-xs text-slate-400">{p.message}</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <Stat label="Epoch" value={`${p.epoch} / ${p.epochs || cfg.epochs}`} />
          <Stat label="Batch" value={`${p.batch} / ${p.batches || "—"}`} />
          <Stat label="Train loss" value={fmt(p.trainLoss)} />
          <Stat label={cls ? "Train acc" : "Val loss"} value={cls ? (p.trainAcc !== undefined ? fmt(p.trainAcc, 3) : "—") : fmt(p.valLoss ?? NaN)} />
          <Stat label="Elapsed" value={fmtDuration(p.elapsedMs)} />
          <Stat label="ETA" value={fmtDuration(p.etaMs ?? NaN)} />
          <Stat label="Throughput" value={`${fmt(p.samplesPerSec, 1)} smp/s`} />
          <Stat label="LR" value={fmt(p.lr, 5)} />
        </div>
        <Panel title="Loss">
          {p.history.length < 1 ? (
            <p className="text-xs text-slate-500">Charts update from real epoch metrics after the first epoch completes.</p>
          ) : (
            <LineChart series={series} yLabel="loss" />
          )}
        </Panel>
        {cls && (
          <Panel title="Accuracy">
            {accSeries.every((x) => x.values.length === 0) ? (
              <p className="text-xs text-slate-500">Accuracy is computed from actual argmax / thresholded predictions.</p>
            ) : (
              <LineChart series={accSeries} />
            )}
          </Panel>
        )}
        {p.gradNorms.length > 0 && (
          <Panel title="Gradient L2 by layer">
            <div className="space-y-1">
              {p.gradNorms.map((g) => (
                <div key={g.id} className="flex items-center gap-2 text-[11px]">
                  <span className="w-28 truncate text-slate-400">{g.name}</span>
                  <div className="flex-1 h-1.5 rounded bg-white/5">
                    <div
                      className="h-1.5 rounded bg-amber-400/80"
                      style={{ width: `${Math.min(100, (g.gradNorm / (Math.max(...p.gradNorms.map((x) => x.gradNorm), 1e-9))) * 100)}%` }}
                    />
                  </div>
                  <span className="font-mono text-slate-500 w-16 text-right">{fmt(g.gradNorm, 3)}</span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 mt-2">These are parameter-gradient norms from the last training step, not finite differences of weights.</p>
          </Panel>
        )}
        {p.history.length > 0 && (
          <div>
            <Button variant="ghost" size="sm" onClick={s.captureExperiment}>
              Record experiment
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const tone = status === "running" ? "cyan" : status === "paused" ? "amber" : status === "completed" ? "emerald" : status === "failed" ? "rose" : "slate";
  return <Badge tone={tone as "cyan"}>{status}</Badge>;
}
