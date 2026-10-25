import { useMemo, useState } from "react";
import { useStore } from "../store";
import { Button, Field, Input, Select, Panel, Badge, Stat } from "../components/ui";
import { layerDisplayName, LAYER_DOCS, defaultConfig } from "../engine/layers";
import { shapesAlong } from "../engine/model";
import { PRESETS, PRESET_CATEGORIES } from "../engine/presets";
import { INIT_DOCS } from "../engine/init";
import { ACTIVATION_DOCS } from "../engine/activations";
import type { LayerType, InitMethod, ActivationName } from "../engine/types";
import { fmtInt, fmtBytes, fmtShape } from "../lib/format";
import { cn } from "../utils/cn";

const LIBRARY: { type: LayerType; group: string }[] = [
  { type: "dense", group: "Core" },
  { type: "flatten", group: "Core" },
  { type: "reshape", group: "Core" },
  { type: "dropout", group: "Core" },
  { type: "identity", group: "Core" },
  { type: "relu", group: "Activation" },
  { type: "leakyrelu", group: "Activation" },
  { type: "gelu", group: "Activation" },
  { type: "silu", group: "Activation" },
  { type: "sigmoid", group: "Activation" },
  { type: "tanh", group: "Activation" },
  { type: "softmax", group: "Activation" },
  { type: "conv2d", group: "Convolution" },
  { type: "maxpool2d", group: "Convolution" },
  { type: "avgpool2d", group: "Convolution" },
  { type: "gap2d", group: "Convolution" },
  { type: "residual_conv", group: "Residual" },
  { type: "residual_dense", group: "Residual" },
  { type: "batchnorm", group: "Normalization" },
  { type: "layernorm", group: "Normalization" },
  { type: "embedding", group: "Sequence" },
  { type: "attention", group: "Attention" },
  { type: "transformer_block", group: "Attention" },
  { type: "reasoning_block", group: "Reasoning" },
  { type: "rnn", group: "Sequence" },
  { type: "lstm", group: "Sequence" },
  { type: "gru", group: "Sequence" },
];

const TONE: Record<string, string> = {
  dense: "border-cyan-400/40 bg-cyan-500/10",
  conv2d: "border-emerald-400/40 bg-emerald-500/10",
  maxpool2d: "border-teal-400/40 bg-teal-500/10",
  avgpool2d: "border-teal-400/40 bg-teal-500/10",
  gap2d: "border-teal-400/40 bg-teal-500/10",
  dropout: "border-rose-400/40 bg-rose-500/10",
  batchnorm: "border-amber-400/40 bg-amber-500/10",
  layernorm: "border-amber-400/40 bg-amber-500/10",
  embedding: "border-indigo-400/40 bg-indigo-500/10",
  attention: "border-pink-400/40 bg-pink-500/10",
  transformer_block: "border-pink-400/40 bg-pink-500/10",
  reasoning_block: "border-orange-400/40 bg-orange-500/10",
  rnn: "border-sky-400/40 bg-sky-500/10",
  lstm: "border-blue-400/40 bg-blue-500/10",
  gru: "border-blue-400/40 bg-blue-500/10",
  residual_dense: "border-fuchsia-400/40 bg-fuchsia-500/10",
  residual_conv: "border-fuchsia-400/40 bg-fuchsia-500/10",
  flatten: "border-slate-400/30 bg-slate-500/10",
  reshape: "border-slate-400/30 bg-slate-500/10",
  identity: "border-slate-400/30 bg-white/5",
};

function toneFor(t: string): string {
  if (TONE[t]) return TONE[t]!;
  return "border-violet-400/40 bg-violet-500/10";
}

export function Builder() {
  const s = useStore();
  const [q, setQ] = useState("");
  const shapes = useMemo(() => shapesAlong(s.architecture), [s.architecture]);
  const pc = s.model.paramCount();
  const mem = s.model.memoryEstimate(s.trainingConfig.batchSize, s.trainingConfig.optimizer.name);
  const selected = s.architecture.layers.find((l) => l.id === s.selectedLayerId);
  const selectedCompiled = s.model.layers.find((l) => l.def.id === s.selectedLayerId);
  const filtered = LIBRARY.filter((l) => layerDisplayName(l.type).toLowerCase().includes(q.toLowerCase()) || l.type.includes(q.toLowerCase()));
  const groups = [...new Set(filtered.map((l) => l.group))];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[220px_minmax(0,1fr)_300px] gap-3 h-full min-h-0">
      <Panel title="Layer library" className="overflow-auto">
        <Input placeholder="Search layers" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="mt-3 space-y-3">
          {groups.map((g) => (
            <div key={g}>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{g}</div>
              <div className="flex flex-col gap-1">
                {filtered
                  .filter((l) => l.group === g)
                  .map((l) => (
                    <button
                      key={l.type}
                      onClick={() => s.add(l.type)}
                      title={LAYER_DOCS[l.type]}
                      className={cn("text-left rounded-md border px-2 py-1.5 text-xs text-slate-200 hover:bg-white/5", toneFor(l.type))}
                    >
                      {layerDisplayName(l.type)}
                    </button>
                  ))}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-[10px] text-slate-500 leading-relaxed">
          Click a layer to append it. Residual Dense / Residual Conv are skip-connection blocks you can drop into a
          sequential stack; arbitrary branch/graph topologies are still out of scope.
        </div>
      </Panel>

      <div className="flex flex-col min-h-0 gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" size="sm" onClick={s.undo} disabled={!s.canUndo}>
            Undo
          </Button>
          <Button variant="ghost" size="sm" onClick={s.redo} disabled={!s.canRedo}>
            Redo
          </Button>
          <Button variant="ghost" size="sm" onClick={s.reinitWeights}>
            Reinit weights
          </Button>
          <Select
            className="w-auto"
            value=""
            onChange={(e) => {
              if (e.target.value) s.applyPreset(e.target.value);
            }}
          >
            <option value="">Load preset…</option>
            {PRESET_CATEGORIES.map((cat) => (
              <optgroup key={cat} label={cat}>
                {PRESETS.filter((p) => p.category === cat).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </Select>
          <div className="ml-auto flex gap-3 text-[11px] font-mono text-slate-400">
            <span>{fmtInt(pc.trainable)} trainable</span>
            <span>{fmtBytes(mem.totalBytes)} est.</span>
          </div>
        </div>

        {s.model.issues.length > 0 && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
            {s.model.issues.map((i) => (
              <div key={i.layerId}>
                <strong>{i.layerName}:</strong> {i.message}
              </div>
            ))}
          </div>
        )}

        <div className="flex-1 overflow-auto rounded-lg border border-white/8 bg-[radial-gradient(ellipse_at_top,rgba(34,211,238,0.05),transparent_50%)] p-6">
          <div className="flex flex-col items-center gap-0 max-w-xl mx-auto">
            <NodeCard
              title={s.architecture.inputName}
              subtitle={`Input ${fmtShape(s.architecture.inputShape)}`}
              active={s.selectedLayerId === "__input__"}
              tone="border-cyan-300/30 bg-black/40"
              onClick={() => s.setSelectedLayerId("__input__")}
            />
            {s.architecture.layers.map((l, idx) => {
              const sh = shapes.find((x) => x.id === l.id);
              return (
                <div key={l.id} className="flex flex-col items-center w-full">
                  <div className="h-6 w-px bg-white/20" />
                  <button
                    className="text-[10px] text-slate-500 hover:text-cyan-300"
                    onClick={() => s.add("dense", idx)}
                    title="Insert dense layer here"
                  >
                    +
                  </button>
                  <div className="h-2 w-px bg-white/20" />
                  <NodeCard
                    title={l.name}
                    subtitle={`${layerDisplayName(l.type)}  →  ${sh?.output ? fmtShape(sh.output) : "invalid"}`}
                    meta={paramHint(s, l.id)}
                    active={s.selectedLayerId === l.id}
                    tone={toneFor(l.type)}
                    frozen={l.frozen}
                    error={!!sh?.error}
                    onClick={() => s.setSelectedLayerId(l.id)}
                  />
                </div>
              );
            })}
            <div className="h-6 w-px bg-white/20" />
            <Button variant="ghost" size="sm" onClick={() => s.add("dense")}>
              Add layer
            </Button>
            <div className="mt-4 text-[11px] font-mono text-slate-500">Output {fmtShape(s.model.valid ? s.model.outputShape : [])}</div>
          </div>
        </div>
      </div>

      <div className="overflow-auto space-y-3">
        <Panel title="Architecture">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Trainable" value={fmtInt(pc.trainable)} />
            <Stat label="Non-trainable" value={fmtInt(pc.nonTrainable)} />
            <Stat label="Est. memory" value={fmtBytes(mem.totalBytes)} sub="params + Adam state + activations" />
            <Stat label="Layers" value={s.architecture.layers.length} />
          </div>
          <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
            Memory is an estimate for batch {s.trainingConfig.batchSize} in float32. Intermediate buffers and the JS heap will use more.
          </p>
          <div className="mt-3 space-y-2">
            <Field label="Init method" hint={INIT_DOCS[s.architecture.initMethod]}>
              <Select
                value={s.architecture.initMethod}
                onChange={(e) => {
                  if (confirm("Changing init reinitializes all weights and discards trained parameters. Continue?")) {
                    s.setInit(e.target.value as InitMethod);
                  }
                }}
              >
                {Object.keys(INIT_DOCS).map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Seed">
              <Input
                type="number"
                value={s.architecture.seed}
                onChange={(e) => s.setInit(s.architecture.initMethod, Number(e.target.value) || 1)}
              />
            </Field>
            <Field label="Input shape (comma-separated)" hint="Do not include the batch dimension.">
              <Input
                defaultValue={s.architecture.inputShape.join(", ")}
                key={s.architecture.inputShape.join(",")}
                onBlur={(e) => {
                  const parts = e.target.value.split(/[×x,\s]+/).map((x) => Number(x.trim())).filter((n) => n > 0);
                  if (parts.length) s.setInputShape(parts);
                }}
              />
            </Field>
          </div>
        </Panel>

        {s.selectedLayerId === "__input__" && (
          <Panel title="Input">
            <p className="text-xs text-slate-400 leading-relaxed">
              Runtime tensors are batched: {fmtShape([s.trainingConfig.batchSize, ...s.architecture.inputShape])}. Match this to the dataset feature layout.
            </p>
          </Panel>
        )}

        {selected && (
          <Panel
            title="Layer"
            actions={
              <div className="flex gap-1">
                <Button variant="quiet" size="sm" onClick={() => s.move(selected.id, -1)}>
                  Up
                </Button>
                <Button variant="quiet" size="sm" onClick={() => s.move(selected.id, 1)}>
                  Down
                </Button>
                <Button variant="quiet" size="sm" onClick={() => s.duplicate(selected.id)}>
                  Dup
                </Button>
                <Button variant="danger" size="sm" onClick={() => s.remove(selected.id)}>
                  Delete
                </Button>
              </div>
            }
          >
            <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">{LAYER_DOCS[selected.type]}</p>
            <div className="space-y-2">
              <Field label="Name">
                <Input value={selected.name} onChange={(e) => s.update(selected.id, { name: e.target.value })} />
              </Field>
              <Field label="Frozen">
                <Select value={selected.frozen ? "1" : "0"} onChange={(e) => s.update(selected.id, { frozen: e.target.value === "1" })}>
                  <option value="0">Trainable</option>
                  <option value="1">Frozen</option>
                </Select>
              </Field>
              {selectedCompiled && (
                <div className="text-[11px] font-mono text-slate-400">
                  {fmtShape(selectedCompiled.inputShape)} → {fmtShape(selectedCompiled.outputShape)}
                </div>
              )}
              <LayerFields def={selected} onChange={(config) => s.update(selected.id, { config })} />
            </div>
          </Panel>
        )}

        <Panel title={`Presets (${PRESETS.length})`}>
          <p className="text-[10px] text-slate-500 mb-2 leading-snug">
            Loading a preset also switches to its matching dataset and recommended loss, so it trains immediately.
          </p>
          <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
            {PRESET_CATEGORIES.map((cat) => (
              <div key={cat}>
                <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{cat}</div>
                <div className="space-y-1.5">
                  {PRESETS.filter((p) => p.category === cat).map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        if (confirm(`Replace the current architecture with “${p.name}”? Trained weights will be discarded.`)) s.applyPreset(p.id);
                      }}
                      className="w-full text-left rounded-md border border-white/8 px-2 py-1.5 hover:bg-white/5"
                    >
                      <div className="text-xs text-slate-200">{p.name}</div>
                      <div className="text-[10px] text-slate-500 leading-snug">{p.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function NodeCard({
  title,
  subtitle,
  meta,
  active,
  tone,
  frozen,
  error,
  onClick,
}: {
  title: string;
  subtitle: string;
  meta?: string;
  active?: boolean;
  tone: string;
  frozen?: boolean;
  error?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full max-w-sm rounded-lg border px-3 py-2 text-left transition",
        tone,
        active && "ring-2 ring-cyan-400/70",
        error && "border-rose-400/70",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-slate-100">{title}</span>
        {frozen && <Badge tone="amber">frozen</Badge>}
        {error && <Badge tone="rose">invalid</Badge>}
      </div>
      <div className="text-[11px] font-mono text-slate-400 mt-0.5">{subtitle}</div>
      {meta && <div className="text-[10px] text-slate-500">{meta}</div>}
    </button>
  );
}

function paramHint(s: ReturnType<typeof useStore>, id: string): string {
  const row = s.model.paramCount().byLayer.find((l) => l.id === id);
  if (!row) return "";
  const n = row.trainable + row.nonTrainable;
  return n ? `${fmtInt(n)} params` : "no parameters";
}

function LayerFields({ def, onChange }: { def: { type: LayerType; config: Record<string, unknown> }; onChange: (c: Record<string, unknown>) => void }) {
  const c = { ...defaultConfig(def.type), ...def.config };
  const set = (k: string, v: unknown) => onChange({ ...c, [k]: v });
  return (
    <>
      {"units" in c && (
        <Field label="Units">
          <Input type="number" min={1} value={Number(c.units)} onChange={(e) => set("units", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"useBias" in c && (
        <Field label="Bias">
          <Select value={c.useBias ? "1" : "0"} onChange={(e) => set("useBias", e.target.value === "1")}>
            <option value="1">Yes</option>
            <option value="0">No</option>
          </Select>
        </Field>
      )}
      {"activation" in c && (
        <Field label={"steps" in c ? "Thought activation" : "Fused activation"} hint={c.activation !== "none" ? ACTIVATION_DOCS[c.activation as ActivationName] : "Prefer a dedicated activation layer so you can inspect it."}>
          <Select value={String(c.activation)} onChange={(e) => set("activation", e.target.value)}>
            <option value="none">none</option>
            {Object.keys(ACTIVATION_DOCS).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {"rate" in c && (
        <Field label="Dropout rate">
          <Input type="number" min={0} max={0.95} step={0.05} value={Number(c.rate)} onChange={(e) => set("rate", Number(e.target.value))} />
        </Field>
      )}
      {"filters" in c && (
        <Field label="Filters">
          <Input type="number" min={1} value={Number(c.filters)} onChange={(e) => set("filters", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"kernelSize" in c && (
        <Field label="Kernel size">
          <Input type="number" min={1} value={Number(c.kernelSize)} onChange={(e) => set("kernelSize", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"stride" in c && (
        <Field label="Stride">
          <Input type="number" min={1} value={Number(c.stride)} onChange={(e) => set("stride", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"padding" in c && (
        <Field label="Padding">
          <Select value={String(c.padding)} onChange={(e) => set("padding", e.target.value)}>
            <option value="same">same</option>
            <option value="valid">valid</option>
          </Select>
        </Field>
      )}
      {"poolSize" in c && (
        <Field label="Pool size">
          <Input type="number" min={1} value={Number(c.poolSize)} onChange={(e) => set("poolSize", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"vocabSize" in c && (
        <Field label="Vocab size">
          <Input type="number" min={2} value={Number(c.vocabSize)} onChange={(e) => set("vocabSize", Math.max(2, Number(e.target.value) || 2))} />
        </Field>
      )}
      {"heads" in c && (
        <Field label="Heads" hint="Independent attention heads. Every head gets its own slice of the model width, so the head count must divide it.">
          <Input type="number" min={1} value={Number(c.heads)} onChange={(e) => set("heads", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"dModel" in c && (
        <Field label="Model width" hint="Width the attention mixes internally. 0 = keep the incoming feature width (an embedding's dim, or the last axis of the input shape). A different value adds a learned input projection.">
          <Input type="number" min={0} value={Number(c.dModel)} onChange={(e) => set("dModel", Math.max(0, Number(e.target.value) || 0))} />
        </Field>
      )}
      {"ffDim" in c && (
        <Field label="Feed-forward width" hint="Hidden width of the block's MLP (dModel → ffDim → dModel). 0 = the usual 4 × dModel.">
          <Input type="number" min={0} value={Number(c.ffDim)} onChange={(e) => set("ffDim", Math.max(0, Number(e.target.value) || 0))} />
        </Field>
      )}
      {"causal" in c && (
        <Field label="Masking" hint="Causal hides every future position from the current one — required for next-token prediction to be honest. Bidirectional lets every position see the whole sequence.">
          <Select value={c.causal ? "1" : "0"} onChange={(e) => set("causal", e.target.value === "1")}>
            <option value="1">causal — no looking ahead</option>
            <option value="0">bidirectional — full context</option>
          </Select>
        </Field>
      )}
      {"pool" in c && (
        <Field label="Output" hint="Sequence keeps [T, dModel] for the next layer. Last hands the final position to a Dense head (GPT-style); mean averages every position.">
          <Select value={String(c.pool)} onChange={(e) => set("pool", e.target.value)}>
            <option value="none">sequence [T, dModel]</option>
            <option value="last">last position</option>
            <option value="mean">mean of positions</option>
          </Select>
        </Field>
      )}
      {"dropout" in c && (
        <Field label="Attention dropout" hint="Applied to the attention probabilities while training, disabled at inference. 0 switches it off.">
          <Input type="number" min={0} max={0.9} step={0.05} value={Number(c.dropout)} onChange={(e) => set("dropout", Math.min(0.9, Math.max(0, Number(e.target.value))))} />
        </Field>
      )}
      {"scale" in c && (
        <Field label="Score scale" hint="Multiplies the 1/√d_k normaliser on the attention scores. Below 1 sharpens the distribution, above 1 flattens it — useful for seeing what temperature does to attention.">
          <Input type="number" min={0.05} step={0.05} value={Number(c.scale)} onChange={(e) => set("scale", Math.max(0.05, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"embeddingDim" in c && (
        <Field label="Embedding dim">
          <Input type="number" min={1} value={Number(c.embeddingDim)} onChange={(e) => set("embeddingDim", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"alpha" in c && (
        <Field label="Leaky slope α">
          <Input type="number" step={0.01} value={Number(c.alpha)} onChange={(e) => set("alpha", Number(e.target.value))} />
        </Field>
      )}
      {"epsilon" in c && (
        <Field label="Epsilon">
          <Input type="number" step={1e-6} value={Number(c.epsilon)} onChange={(e) => set("epsilon", Number(e.target.value))} />
        </Field>
      )}
      {"returnSequences" in c && (
        <Field label="Return sequences" hint="Off: only the last timestep reaches the next layer. On: every timestep does (add a Flatten before a Dense layer).">
          <Select value={c.returnSequences ? "1" : "0"} onChange={(e) => set("returnSequences", e.target.value === "1")}>
            <option value="0">No — final state only</option>
            <option value="1">Yes — all timesteps</option>
          </Select>
        </Field>
      )}
      {"forgetBias" in c && (
        <Field label="Forget-gate bias" hint="Initial value of the forget gate's bias. 1.0 (default) keeps the cell state alive early in training; 0 makes the cell forget immediately.">
          <Input type="number" step={0.1} value={Number(c.forgetBias)} onChange={(e) => set("forgetBias", Number(e.target.value))} />
        </Field>
      )}
      {"steps" in c && (
        <Field label="Reasoning depth (steps)" hint="How many sequential deliberation steps one forward pass runs. The weights are shared across steps, so depth costs compute, not parameters. Exported models can override this at inference.">
          <Input type="number" min={1} max={64} value={Number(c.steps)} onChange={(e) => set("steps", Math.max(1, Math.min(64, Number(e.target.value) || 1)))} />
        </Field>
      )}
      {"hidden" in c && (
        <Field label="Thought width" hint="Hidden width of the shared thought cell (width → hidden → width). 0 = 2 × the incoming feature width.">
          <Input type="number" min={0} value={Number(c.hidden)} onChange={(e) => set("hidden", Math.max(0, Number(e.target.value) || 0))} />
        </Field>
      )}
      {"adaptive" in c && (
        <Field label="Adaptive halting" hint="At inference, stop early once cumulative halting confidence crosses the threshold. Training always runs every step so gradients see the full depth.">
          <Select value={c.adaptive ? "1" : "0"} onChange={(e) => set("adaptive", e.target.value === "1")}>
            <option value="1">On — stop when confident</option>
            <option value="0">Off — always use full depth</option>
          </Select>
        </Field>
      )}
      {"haltThreshold" in c && (
        <Field label="Halt threshold" hint="Cumulative halting probability that ends the loop early (0.99 = wait until 99% confident).">
          <Input type="number" min={0.5} max={0.9999} step={0.005} value={Number(c.haltThreshold)} onChange={(e) => set("haltThreshold", Math.min(0.9999, Math.max(0.5, Number(e.target.value) || 0.99)))} />
        </Field>
      )}
      {"minSteps" in c && (
        <Field label="Minimum steps" hint="Never halt before this many steps, even if the halting head is confident.">
          <Input type="number" min={1} max={64} value={Number(c.minSteps)} onChange={(e) => set("minSteps", Math.max(1, Number(e.target.value) || 1))} />
        </Field>
      )}
      {"stepCode" in c && (
        <Field label="Step signal" hint="Adds a parameter-free sinusoidal code for the step index, so step t is a different function from step t−1 and the cell keeps refining instead of reaching a fixed point.">
          <Select value={c.stepCode ? "1" : "0"} onChange={(e) => set("stepCode", e.target.value === "1")}>
            <option value="1">Sinusoidal step code</option>
            <option value="0">None</option>
          </Select>
        </Field>
      )}
      {"stepCodeScale" in c && (
        <Field label="Step-code scale" hint="Multiplier on the sinusoidal step signal.">
          <Input type="number" step={0.1} value={Number(c.stepCodeScale)} onChange={(e) => set("stepCodeScale", Number(e.target.value))} />
        </Field>
      )}
      {"projection" in c && (
        <Field label="Shortcut" hint="Identity needs the block to keep its width. Projection adds a learned 1×1 conv / linear layer so the block may change width (and resolution).">
          <Select value={String(c.projection)} onChange={(e) => set("projection", e.target.value)}>
            <option value="auto">auto — project only when shapes differ</option>
            <option value="always">always — always learn the shortcut</option>
            <option value="never">never — require an identity shortcut</option>
          </Select>
        </Field>
      )}
      {"scale" in c && (
        <Field label="Branch scale" hint="Multiplies the residual branch before the add. Values slightly below 1 (e.g. 0.7) keep deep stacks stable.">
          <Input type="number" step={0.1} value={Number(c.scale)} onChange={(e) => set("scale", Number(e.target.value))} />
        </Field>
      )}
      {"finalActivation" in c && (
        <Field label="Post-add activation" hint="Applied after the shortcut is added. Standard ResNet blocks leave this as none.">
          <Select value={String(c.finalActivation)} onChange={(e) => set("finalActivation", e.target.value)}>
            <option value="none">none</option>
            {Object.keys(ACTIVATION_DOCS).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {"targetShape" in c && (
        <Field label="Target shape">
          <Input
            defaultValue={(c.targetShape as number[]).join(", ")}
            onBlur={(e) => {
              const parts = e.target.value.split(/[×x,\s]+/).map((x) => Number(x.trim())).filter((n) => n > 0);
              if (parts.length) set("targetShape", parts);
            }}
          />
        </Field>
      )}
    </>
  );
}
