import { useStore } from "../store";
import { Button, Panel, Empty, Badge } from "../components/ui";
import { LineChart } from "../components/Charts";
import { fmt, fmtDuration, fmtInt } from "../lib/format";

export function Experiments() {
  const s = useStore();
  const exps = s.experiments;
  if (!exps.length) {
    return (
      <Empty
        title="No experiments recorded"
        body="After a training run, press Record experiment in the Training Lab. Each record stores the architecture, hyperparameters, and the real metric history from that run."
        action={
          <Button onClick={() => s.setView("train")} variant="ghost">
            Go to Training Lab
          </Button>
        }
      />
    );
  }
  const lossSeries = exps.slice(0, 5).map((e, i) => ({
    name: e.name,
    color: ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f43f5e"][i % 5]!,
    values: e.trainLoss.map((y, x) => ({ x: x + 1, y })),
  }));
  return (
    <div className="space-y-3 overflow-auto">
      <Panel title="Train loss overlay (up to 5 latest)">
        <LineChart series={lossSeries} />
      </Panel>
      <div className="overflow-auto rounded-lg border border-white/8">
        <table className="w-full text-xs">
          <thead className="text-[10px] uppercase tracking-widest text-slate-500">
            <tr>
              {["Name", "Dataset", "Opt", "LR", "Params", "Train loss", "Val loss", "Acc", "Time", "Status"].map((h) => (
                <th key={h} className="text-left font-medium px-3 py-2">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {exps.map((e) => (
              <tr key={e.id} className="border-t border-white/5 text-slate-300">
                <td className="px-3 py-2">{e.name}</td>
                <td className="px-3 py-2">{e.datasetName}</td>
                <td className="px-3 py-2 font-mono">{e.optimizer}</td>
                <td className="px-3 py-2 font-mono">{e.learningRate}</td>
                <td className="px-3 py-2 font-mono">{fmtInt(e.paramCount)}</td>
                <td className="px-3 py-2 font-mono">{fmt(e.finalTrainLoss ?? NaN)}</td>
                <td className="px-3 py-2 font-mono">{fmt(e.finalValLoss ?? NaN)}</td>
                <td className="px-3 py-2 font-mono">{fmt(e.finalValAcc ?? e.finalTrainAcc ?? NaN, 3)}</td>
                <td className="px-3 py-2">{fmtDuration(e.durationMs)}</td>
                <td className="px-3 py-2">
                  <Badge tone={e.status === "completed" ? "emerald" : e.status === "failed" ? "rose" : "slate"}>{e.status}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
