import { useMemo, useState } from "react";
import { useStore } from "../store";
import { Button, Field, Input, Panel, Stat } from "../components/ui";
import { Tensor } from "../engine/tensor";
import { evaluateLoss, predict } from "../engine/train";
import { classificationFromPred, regressionFromPred } from "../engine/metrics";
import { Confusion } from "../components/Charts";
import { fmt, fmtShape } from "../lib/format";
import { SAMPLING_DOCS } from "../engine/sampling";
import { generateTokens } from "../engine/generate";

const COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f43f5e"];

export function Evaluation() {
  const s = useStore();
  const [split, setSplit] = useState<"train" | "val" | "test">("test");
  const [result, setResult] = useState<ReturnType<typeof evalSplit> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [inputs, setInputs] = useState<number[]>(() => Array(s.architecture.inputShape.reduce((a, b) => a * b, 1)).fill(0));
  const [pred, setPred] = useState<number[] | null>(null);
  const [prompt, setPrompt] = useState("hello there");
  const [gen, setGen] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [genLen, setGenLen] = useState(60);
  const [seed, setSeed] = useState(1);
  const tokLevel = s.dataset.meta.tokenizer?.level;
  const wordLevel = tokLevel === "word";
  const bpeLevel = tokLevel === "bpe";
  const task = s.dataset.meta.task;
  const twoD = s.architecture.inputShape.length === 1 && s.architecture.inputShape[0] === 2;

  const boundary = useMemo(() => {
    if (!twoD || !result || !s.model.valid) return null;
    try {
      return decisionBoundary(s);
    } catch {
      return null;
    }
  }, [s.model, result, twoD, s]);

  return (
    <div className="space-y-3 overflow-auto">
      <div className="flex flex-wrap items-center gap-2">
        {(["train", "val", "test"] as const).map((k) => (
          <Button key={k} variant={split === k ? "primary" : "ghost"} size="sm" onClick={() => setSplit(k)}>
            {k}
          </Button>
        ))}
        <Button
          onClick={() => {
            setErr(null);
            try {
              setResult(evalSplit(s, split));
            } catch (e) {
              setErr(e instanceof Error ? e.message : String(e));
            }
          }}
          disabled={!s.model.valid}
        >
          Evaluate {split}
        </Button>
        <span className="text-[11px] text-slate-500">Test is held out from training. Do not tune on it and then treat the score as unbiased.</span>
      </div>
      {err && <div className="text-xs text-rose-300">{err}</div>}

      {result && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <Stat label="Loss" value={fmt(result.loss)} />
          {"accuracy" in result.report && <Stat label="Accuracy" value={fmt((result.report as { accuracy: number }).accuracy, 3)} />}
          {"f1Macro" in result.report && <Stat label="Macro F1" value={fmt((result.report as { f1Macro: number }).f1Macro, 3)} />}
          {"mse" in result.report && <Stat label="RMSE" value={fmt((result.report as { rmse: number }).rmse)} />}
          {"mae" in result.report && <Stat label="MAE" value={fmt((result.report as { mae: number }).mae)} />}
          <Stat label="Samples" value={result.n} />
        </div>
      )}

      {result && "confusion" in result.report && (
        <Panel title="Confusion matrix">
          <Confusion matrix={(result.report as { confusion: number[][] }).confusion} labels={s.dataset.meta.classNames} />
        </Panel>
      )}

      {twoD && result && (
        <Panel title="Decision region (grid of real forward passes)">
          <BoundaryVis points={boundary?.points ?? []} grid={boundary?.grid} />
        </Panel>
      )}

      <Panel title="Prediction playground">
        <p className="text-[11px] text-slate-400 mb-3">
          Runs the current weights with training=false. Input shape {fmtShape(s.architecture.inputShape)}. Values are fed as-is — if you trained on standardized CSV features, enter already-transformed numbers.
        </p>
        {s.dataset.meta.task === "language" ? (
          <div className="space-y-2">
            <Field
              label={`Seed ${wordLevel ? "words" : bpeLevel ? "subword text" : "text"}`}
              hint={
                wordLevel
                  ? `Word-level tokenizer: ${s.dataset.meta.tokenizer?.tokens.length.toLocaleString()} tokens, ${s.dataset.meta.tokenizer?.seqLen} words of context. Unknown words become ${s.dataset.meta.tokenizer?.tokens[0]}.`
                  : bpeLevel
                    ? `BPE subword tokenizer: ${s.dataset.meta.tokenizer?.tokens.length.toLocaleString()} tokens (${(s.dataset.meta.tokenizer?.merges?.length ?? 0).toLocaleString()} learned merges), ${s.dataset.meta.tokenizer?.seqLen} tokens of context. Text is lowercased and split with the training merge rules; unseen characters become ${s.dataset.meta.tokenizer?.tokens[0]}.`
                    : `Character-level tokenizer: ${s.dataset.meta.tokenizer?.tokens.length ?? 0} tokens, ${s.dataset.meta.tokenizer?.seqLen} characters of context.`
              }
            >
              <Input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </Field>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <Field label={`Temperature ${temperature.toFixed(2)}`} hint={SAMPLING_DOCS.temperature}>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full"
                />
              </Field>
              <Field label="Top-k" hint={SAMPLING_DOCS.topK}>
                <Input type="number" min={1} value={topK} onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))} />
              </Field>
              <Field label={wordLevel ? "Words to generate" : bpeLevel ? "Tokens to generate" : "Characters to generate"}>
                <Input type="number" min={1} max={800} value={genLen} onChange={(e) => setGenLen(Math.max(1, Number(e.target.value) || 1))} />
              </Field>
              <Field label="Sampling seed" hint="Same seed + same prompt reproduces the same text.">
                <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 1)} />
              </Field>
            </div>
            <div className="flex flex-wrap gap-2 text-[10px]">
              {[0, 0.3, 0.7, 1, 1.4].map((t) => (
                <button
                  key={t}
                  onClick={() => setTemperature(t)}
                  className={`rounded border px-1.5 py-0.5 font-mono ${temperature === t ? "border-cyan-400/60 text-cyan-200" : "border-white/10 text-slate-400"}`}
                >
                  T={t}
                </button>
              ))}
              {[1, 20, 50, 200].map((k) => (
                <button
                  key={k}
                  onClick={() => setTopK(k)}
                  className={`rounded border px-1.5 py-0.5 font-mono ${topK === k ? "border-cyan-400/60 text-cyan-200" : "border-white/10 text-slate-400"}`}
                >
                  k={k}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => {
                  try {
                    setGen(generateText(s, prompt, genLen, temperature, topK, seed));
                  } catch (e) {
                    setErr(e instanceof Error ? e.message : String(e));
                  }
                }}
              >
                Generate
              </Button>
              <span className="text-[10px] text-slate-500 self-end pb-1">
                {temperature <= 0.01 ? "greedy argmax" : `sampling · T=${temperature} · k=${Math.min(topK, s.model.outputShape[s.model.outputShape.length - 1] || topK)}`}
              </span>
            </div>
            {gen && <pre className="text-xs text-cyan-100 whitespace-pre-wrap bg-black/40 p-2 rounded-md leading-relaxed">{gen}</pre>}
          </div>
        ) : (
          <div className="space-y-2">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {inputs.map((v, i) => (
                <Field key={i} label={s.dataset.meta.featureNames[i] ?? `x${i}`}>
                  <Input
                    type="number"
                    step={0.1}
                    value={v}
                    onChange={(e) => {
                      const next = inputs.slice();
                      next[i] = Number(e.target.value);
                      setInputs(next);
                    }}
                  />
                </Field>
              ))}
            </div>
            <Button
              onClick={() => {
                try {
                  const x = new Tensor([1, ...s.architecture.inputShape], Float32Array.from(inputs));
                  const y = predict(s.model, x);
                  setPred(Array.from(y.data));
                } catch (e) {
                  setErr(e instanceof Error ? e.message : String(e));
                }
              }}
            >
              Predict
            </Button>
            {pred && (
              <div className="text-sm font-mono text-cyan-200">
                raw output: [{pred.map((v) => fmt(v, 4)).join(", ")}]
                {interpret(pred, task, s.dataset.meta.classNames)}
              </div>
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}

function evalSplit(s: ReturnType<typeof useStore>, split: "train" | "val" | "test") {
  const data = s.dataset[split];
  const loss = evaluateLoss(s.model, data, s.trainingConfig.loss);
  const pred = predict(s.model, data.xs);
  if (s.trainingConfig.loss === "mse" || s.trainingConfig.loss === "mae") {
    return { loss: loss.loss, n: data.xs.shape[0]!, report: regressionFromPred(pred, data.ys) };
  }
  const binary = s.trainingConfig.loss === "bce" || pred.shape[pred.rank - 1] === 1;
  return { loss: loss.loss, n: data.xs.shape[0]!, report: classificationFromPred(pred, data.ys, binary) };
}

function interpret(pred: number[], task: string, names?: string[]) {
  if (task === "regression") return `  →  ${fmt(pred[0] ?? 0)}`;
  if (pred.length === 1) return `  →  class ${pred[0]! >= 0.5 ? names?.[1] ?? 1 : names?.[0] ?? 0} (p=${fmt(pred[0]!, 3)})`;
  let b = 0;
  pred.forEach((v, i) => {
    if (v > pred[b]!) b = i;
  });
  return `  →  ${names?.[b] ?? b} (${fmt(pred[b]!, 3)})`;
}

function decisionBoundary(s: ReturnType<typeof useStore>) {
  const xs = s.dataset.allX;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < xs.shape[0]!; i++) {
    const x = xs.data[i * 2]!;
    const y = xs.data[i * 2 + 1]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const pad = 0.3;
  minX -= pad; maxX += pad; minY -= pad; maxY += pad;
  const G = 36;
  const grid: { x: number; y: number; cls: number }[] = [];
  const data = new Float32Array(G * G * 2);
  let k = 0;
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const x = minX + (j / (G - 1)) * (maxX - minX);
      const y = minY + (i / (G - 1)) * (maxY - minY);
      data[k++] = x;
      data[k++] = y;
    }
  }
  const pred = predict(s.model, new Tensor([G * G, 2], data));
  const outDim = pred.size / (G * G);
  k = 0;
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const x = minX + (j / (G - 1)) * (maxX - minX);
      const y = minY + (i / (G - 1)) * (maxY - minY);
      let cls = 0;
      if (outDim === 1) cls = pred.data[k]! >= 0.5 ? 1 : 0;
      else {
        let best = 0;
        let bv = -Infinity;
        for (let c = 0; c < outDim; c++) {
          const v = pred.data[k * outDim + c]!;
          if (v > bv) {
            bv = v;
            best = c;
          }
        }
        cls = best;
      }
      grid.push({ x, y, cls });
      k++;
    }
  }
  const points: { x: number; y: number; cls: number }[] = [];
  const yDim = s.dataset.allY.size / s.dataset.allY.shape[0]!;
  const B = Math.min(300, s.dataset.allX.shape[0]!);
  for (let i = 0; i < B; i++) {
    let cls = 0;
    if (yDim === 1) cls = s.dataset.allY.data[i]! >= 0.5 ? 1 : 0;
    else {
      let best = 0;
      let bv = -Infinity;
      for (let c = 0; c < yDim; c++) {
        const v = s.dataset.allY.data[i * yDim + c]!;
        if (v > bv) {
          bv = v;
          best = c;
        }
      }
      cls = best;
    }
    points.push({ x: s.dataset.allX.data[i * 2]!, y: s.dataset.allX.data[i * 2 + 1]!, cls });
  }
  return { grid, points };
}

function BoundaryVis({ points, grid }: { points: { x: number; y: number; cls: number }[]; grid?: { x: number; y: number; cls: number }[] }) {
  const all = [...(grid ?? []), ...points];
  if (!all.length) return null;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const w = 420;
  const h = 300;
  const pad = 8;
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (w - pad * 2);
  const sy = (y: number) => pad + (1 - (y - minY) / (maxY - minY || 1)) * (h - pad * 2);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto bg-black/40 rounded-md">
      {grid?.map((p, i) => (
        <rect key={i} x={sx(p.x) - 4} y={sy(p.y) - 4} width={8} height={8} fill={COLORS[p.cls % COLORS.length]} opacity={0.18} />
      ))}
      {points.map((p, i) => (
        <circle key={`p${i}`} cx={sx(p.x)} cy={sy(p.y)} r={2.6} fill={COLORS[p.cls % COLORS.length]} />
      ))}
    </svg>
  );
}

function generateText(
  s: ReturnType<typeof useStore>,
  prompt: string,
  n: number,
  temperature: number,
  topK: number,
  seed: number,
): string {
  const tok = s.dataset.meta.tokenizer;
  if (!tok) throw new Error("This dataset has no tokenizer attached. Select a language dataset.");
  // Same function the standalone exporter verifies the exported runtime against.
  return generateTokens(s.model, tok, prompt, { maxTokens: n, temperature, topK, seed, stop: [] }).text;
}
