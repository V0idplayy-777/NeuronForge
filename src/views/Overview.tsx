import { useStore } from "../store";
import { Button, Panel, Stat, Badge } from "../components/ui";
import { fmtInt, fmtBytes, fmtShape } from "../lib/format";
import { layerDisplayName } from "../engine/layers";

export function Overview() {
  const s = useStore();
  const pc = s.model.paramCount();
  const mem = s.model.memoryEstimate(s.trainingConfig.batchSize, s.trainingConfig.optimizer.name);

  return (
    <div className="space-y-4 overflow-auto max-w-5xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-100">Laboratory</h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl leading-relaxed">
          NeuronForge trains real neural networks in this browser. Weights, losses, and gradients are computed here — nothing is simulated. Start with the XOR starter project or build your own sequential model.
        </p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Project" value={s.projectName} />
        <Stat label="Trainable params" value={fmtInt(pc.trainable)} />
        <Stat label="Dataset" value={s.dataset.meta.name} sub={`${s.dataset.train.xs.shape[0]} train`} />
        <Stat label="Last train loss" value={s.progress.history.length ? s.progress.trainLoss.toFixed(4) : "—"} />
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <Panel title="Current model">
          <div className="text-[11px] font-mono text-slate-400 mb-2">
            in {fmtShape(s.architecture.inputShape)} → out {fmtShape(s.model.valid ? s.model.outputShape : [])}
          </div>
          <ol className="space-y-1">
            {s.architecture.layers.map((l, i) => (
              <li key={l.id} className="flex items-center gap-2 text-xs text-slate-300">
                <span className="text-slate-600 w-5">{i + 1}</span>
                <span>{l.name}</span>
                <Badge tone="slate">{layerDisplayName(l.type)}</Badge>
              </li>
            ))}
            {s.architecture.layers.length === 0 && <li className="text-xs text-slate-500">No layers yet.</li>}
          </ol>
          {!s.model.valid && <p className="text-xs text-amber-300 mt-2">{s.model.issues[0]?.message}</p>}
          <div className="flex gap-2 mt-3">
            <Button size="sm" onClick={() => s.setView("builder")}>
              Open builder
            </Button>
            <Button size="sm" variant="ghost" onClick={() => s.setView("train")}>
              Train
            </Button>
          </div>
        </Panel>
        <Panel title="Memory estimate">
          <div className="space-y-1 text-xs text-slate-300 font-mono">
            <div className="flex justify-between"><span>Parameters</span><span>{fmtBytes(mem.paramBytes)}</span></div>
            <div className="flex justify-between"><span>Optimiser state</span><span>{fmtBytes(mem.optimizerBytes)}</span></div>
            <div className="flex justify-between"><span>Activations (batch {s.trainingConfig.batchSize})</span><span>{fmtBytes(mem.activationBytes)}</span></div>
            <div className="flex justify-between text-slate-100"><span>Total (lower bound)</span><span>{fmtBytes(mem.totalBytes)}</span></div>
          </div>
          <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
            Computation runs on the CPU in float32 JavaScript. The UI yields between batches so the page stays responsive. WebGPU is not wired in this build — it is not listed as available.
          </p>
        </Panel>
      </div>
      <Panel title="Suggested path">
        <ol className="text-sm text-slate-300 space-y-2 list-decimal pl-4">
          <li>Inspect the XOR classifier in the Model Builder.</li>
          <li>Open Training Lab and press Start. Loss should fall; XOR accuracy should approach 1.</li>
          <li>Open Model Inspector and look at the actual weight matrices.</li>
          <li>Evaluate on the same 4 points, then export a NeuronForge JSON model.</li>
        </ol>
      </Panel>
    </div>
  );
}
