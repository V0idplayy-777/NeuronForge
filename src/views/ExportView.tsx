import { useState } from "react";
import { useStore } from "../store";
import { Button, Panel, Field, Input } from "../components/ui";
import { exportModel, importModel, parseExport, downloadJson, verifyRoundTrip, architectureJson } from "../engine/serialize";
import { Tensor } from "../engine/tensor";
import { fmt } from "../lib/format";
import { StandaloneExport } from "./StandaloneExport";

export function ExportView() {
  const s = useStore();
  const [msg, setMsg] = useState<string | null>(null);
  const [ok, setOk] = useState<boolean | null>(null);
  const [filename, setFilename] = useState("neuronforge-model.json");

  const doExport = (kind: "model" | "checkpoint") => {
    const exported = exportModel(s.model, {
      kind,
      metadata: {
        project: s.projectName,
        dataset: s.dataset.meta.name,
        loss: s.trainingConfig.loss,
        // The architecture already carries the full reasoning plan; this is a
        // human-readable summary for anyone inspecting the file.
        reasoning: s.model.reasoningLayers().length
          ? {
              layers: s.model.reasoningLayers().map((l) => ({
                id: l.id,
                name: l.name,
                steps: l.info.plan.steps,
                hidden: l.info.plan.hidden,
                adaptive: l.info.plan.adaptive,
                haltThreshold: l.info.plan.haltThreshold,
                activation: l.info.plan.activation,
              })),
            }
          : undefined,
      },
      optimizer: kind === "checkpoint" ? s.trainingConfig.optimizer : undefined,
      optimizerState: kind === "checkpoint" ? s.trainer?.optState : undefined,
      epoch: s.progress.epoch,
      preprocessing: {
        datasetId: s.dataset.meta.id,
        // The token<->id mapping travels with the model. Without it, a reloaded
        // LM would decode its own outputs against the wrong vocabulary.
        tokenizer: s.dataset.meta.tokenizer ? { ...s.dataset.meta.tokenizer } : undefined,
      },
    });
    downloadJson(filename, exported);
    setMsg(
      `Wrote ${filename}. ${
        kind === "model"
          ? `Contains architecture + float32 weights (base64)${s.model.reasoningLayers().length ? `, including the Deep Reasoning plan and ${s.model.reasoningLayers().length} reasoning block${s.model.reasoningLayers().length > 1 ? "s" : ""}` : ""}. This is not ONNX and is not a resumable training checkpoint.`
          : `Includes architecture, weights, optimiser state and epoch for resuming inside NeuronForge${s.model.reasoningLayers().length ? ", including the Deep Reasoning plan" : ""}.`
      }`,
    );
    setOk(true);
  };

  return (
    <div className="space-y-3 max-w-4xl overflow-auto">
      <StandaloneExport />
      <Panel title="NeuronForge files (reload inside the editor)">
        <p className="text-xs text-slate-400 leading-relaxed mb-3">
          Native format: <code className="text-cyan-300">neuronforge</code> v1 JSON. Tensors are little-endian float32 encoded as base64. ONNX export is not implemented.
        </p>
        <Field label="Filename">
          <Input value={filename} onChange={(e) => setFilename(e.target.value)} />
        </Field>
        <div className="flex flex-wrap gap-2 mt-3">
          <Button onClick={() => doExport("model")} disabled={!s.model.valid}>
            Download model
          </Button>
          <Button variant="ghost" onClick={() => doExport("checkpoint")} disabled={!s.model.valid}>
            Download checkpoint
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              downloadJson("architecture.json", architectureJson(s.architecture));
              setMsg("Architecture-only JSON (no weights).");
              setOk(true);
            }}
          >
            Architecture JSON
          </Button>
        </div>
      </Panel>

      <Panel title="Round-trip verification">
        <p className="text-xs text-slate-400 mb-3 leading-relaxed">
          Serializes the live model, constructs a new instance, and compares a forward pass on a real training batch. Success means max |Δ| &lt; 1e-5.
        </p>
        <Button
          onClick={() => {
            try {
              const n = Math.min(4, s.dataset.train.xs.shape[0]!);
              const xRest = s.dataset.train.xs.size / s.dataset.train.xs.shape[0]!;
              const x = new Tensor([n, ...s.architecture.inputShape], new Float32Array(s.dataset.train.xs.data.subarray(0, n * xRest)));
              const r = verifyRoundTrip(s.model, x);
              setOk(r.ok);
              setMsg(r.message + (r.ok ? "" : ""));
            } catch (e) {
              setOk(false);
              setMsg(e instanceof Error ? e.message : String(e));
            }
          }}
        >
          Verify export → reload → infer
        </Button>
      </Panel>

      <Panel title="Import">
        <p className="text-xs text-slate-400 mb-3">Parsed as JSON only. Imported code is never executed. Incompatible shapes are reported, not silently patched.</p>
        <input
          type="file"
          accept="application/json,.json"
          className="text-xs text-slate-300"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            try {
              const raw = JSON.parse(await f.text());
              const exported = parseExport(raw);
              const { model, warnings } = importModel(exported);
              const pre = exported.preprocessing as { tokenizer?: { tokens: string[]; seqLen: number; level?: string } } | undefined;
              let tokNote = "";
              if (pre?.tokenizer) {
                const outUnits = model.outputShape[model.outputShape.length - 1]!;
                const n = pre.tokenizer.tokens.length;
                tokNote =
                  outUnits === n
                    ? ` Tokenizer restored: ${n.toLocaleString()} ${pre.tokenizer.level ?? "char"} tokens, context ${pre.tokenizer.seqLen} — matches the model's ${outUnits} output classes.`
                    : ` WARNING: file's tokenizer has ${n.toLocaleString()} tokens but the model outputs ${outUnits} classes. Generation against this mapping would be wrong.`;
                if (outUnits !== n) warnings.push(tokNote);
              }
              s.loadImportedModel(model);
              setOk(warnings.length === 0);
              setMsg(
                `Loaded ${exported.kind} (format ${exported.format} v${exported.version}). ${model.paramCount().trainable} trainable params.` +
                  tokNote +
                  (warnings.length && !tokNote.includes("WARNING") ? ` Warnings: ${warnings.join("; ")}` : warnings.length ? "" : " Weights restored."),
              );
            } catch (er) {
              setOk(false);
              setMsg(er instanceof Error ? er.message : String(er));
            }
          }}
        />
      </Panel>

      {msg && (
        <div className={`text-xs rounded-md border px-3 py-2 ${ok ? "border-emerald-500/40 text-emerald-200" : "border-rose-500/40 text-rose-200"}`}>
          {msg}
        </div>
      )}
      <p className="text-[10px] text-slate-600">{fmt(1e-5)} absolute tolerance on verification.</p>
    </div>
  );
}
