import { useState } from "react";
import { useStore } from "../store";
import { Button, Panel, Toggle, Field, Input } from "../components/ui";
import { runSelfTests, type TestResult } from "../engine/selftest";
import { fmt } from "../lib/format";

export function Settings() {
  const s = useStore();
  const [results, setResults] = useState<TestResult[] | null>(null);
  const [busy, setBusy] = useState(false);

  return (
    <div className="space-y-3 max-w-3xl overflow-auto">
      <Panel title="Interface">
        <Toggle checked={s.advanced} onChange={s.setAdvanced} label="Advanced mode (extra diagnostics and optimiser knobs are already visible in Training Lab)" />
        <div className="mt-3">
          <Field label="Project name">
            <Input value={s.projectName} onChange={(e) => s.renameProject(e.target.value)} />
          </Field>
        </div>
        <div className="flex flex-wrap gap-2 mt-3">
          <Button onClick={() => void s.saveNow()}>Save project</Button>
          <Button variant="ghost" onClick={() => s.createProject()}>
            New project
          </Button>
          <Button variant="ghost" onClick={s.duplicateProject}>
            Duplicate
          </Button>
          <Button
            variant="danger"
            onClick={() => {
              if (confirm("Delete this project permanently from local storage?")) void s.deleteCurrent();
            }}
          >
            Delete project
          </Button>
        </div>
      </Panel>

      <Panel title="Computation backend">
        <dl className="text-xs text-slate-300 space-y-1">
          <div className="flex justify-between"><dt className="text-slate-500">Device</dt><dd>CPU (JavaScript float32)</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Autodiff</dt><dd>Reverse-mode, layer-wise</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">WebGPU</dt><dd>Not implemented — not advertised</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Web Worker</dt><dd>Not used; the loop yields with setTimeout(0)</dd></div>
          <div className="flex justify-between"><dt className="text-slate-500">Storage</dt><dd>IndexedDB + localStorage fallback</dd></div>
        </dl>
      </Panel>

      <Panel title="Mathematical self-tests">
        <p className="text-xs text-slate-400 mb-3 leading-relaxed">
          These tests execute the real tensor engine in this page: matmul, losses, numerical gradient check, XOR learning, serialisation round-trip, CSV parsing.
        </p>
        <Button
          disabled={busy}
          onClick={() => {
            setBusy(true);
            setTimeout(() => {
              setResults(runSelfTests());
              setBusy(false);
            }, 20);
          }}
        >
          {busy ? "Running…" : "Run tests"}
        </Button>
        {results && (
          <div className="mt-3 space-y-1">
            <div className="text-xs text-slate-300">
              {results.filter((r) => r.passed).length}/{results.length} passed
            </div>
            {results.map((r) => (
              <div key={r.name} className={`text-[11px] font-mono flex gap-2 ${r.passed ? "text-emerald-300" : "text-rose-300"}`}>
                <span className="w-14">{r.passed ? "PASS" : "FAIL"}</span>
                <span className="flex-1">{r.name}</span>
                <span className="text-slate-500">{fmt(r.ms, 1)}ms</span>
                {!r.passed && <span className="text-rose-200">{r.message}</span>}
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Implemented / not implemented">
        <ul className="text-xs text-slate-300 space-y-1.5 leading-relaxed">
          <li>Implemented: sequential models, Dense, Conv2D, pooling, Dropout, Batch/LayerNorm, Embedding, activations, SGD/Momentum/Adam/AdamW, MSE/MAE/BCE/CCE, synthetic datasets, CSV import, custom datasets (hand-drawn 2D points and safe formula generator, persisted with the project), 20 architecture presets, IndexedDB projects, JSON export/import, live charts from real metrics.</li>
          <li>Implemented for sequences: Embedding, a genuine tanh RNN with backpropagation through time, and two language datasets — a character-level one (64-char context) and a massive word-level one (16-word context, ~150k+ character generated corpus, frequency-capped vocabulary). The tokenizer (token↔id mapping, level, and padding token) is stored in exported model files and validated on import.</li>
          <li>Standalone Model Export: JavaScript/TypeScript package, single-file module, Python/NumPy package and portable bundle. Each includes an inference runtime for all 20 layer types plus task-aware input/output handling (preprocessing, image layout, tokenizer, sampling). Before download, the shipped runtime is checked against the engine: weights bit-for-bit, forward-pass agreement, a 30+-case layer suite, tokenizer and generation, and rejection of corrupted bundles. The Python target is generated but not executed in the browser.</li>
          <li>Sampling: temperature and top-k decoding with a seeded RNG. Temperature 0 collapses to greedy argmax; the distribution at T=1 is verified against the model's softmax in the self-tests.</li>
          <li>Not implemented: graph/branching topologies, residual skip connections as free-form edges, ONNX, WebGPU, transformer/attention blocks, LSTM/GRU cells, subword tokenisers, remote training.</li>
        </ul>
      </Panel>
    </div>
  );
}
