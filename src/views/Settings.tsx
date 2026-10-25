import { useState } from "react";
import { useStore } from "../store";
import { Button, Panel, Toggle, Field, Input, Select } from "../components/ui";
import { runSelfTests, runAsyncSelfTests, type TestResult } from "../engine/selftest";
import { summarizeChecks } from "../engine/gpu/verify";
import type { BackendPreference } from "../engine/gpu/backend";
import { PRESETS } from "../engine/presets";
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
        <Field
          label="Tensor backend"
          hint="CPU is the default and works everywhere. Auto brings up WebGPU when the browser offers an adapter, keeps small models on the CPU (a per-batch device round trip costs more than a small model's arithmetic saves) and verifies every kernel against the CPU reference before it is used. If verification fails, training stays on the CPU — never on unverified kernels."
        >
          <Select disabled={s.verifyingDevice || s.progress.status === "running" || s.progress.status === "paused" || s.progress.status === "stopping"} value={s.backendPreference} onChange={(e) => s.setBackendPreference(e.target.value as BackendPreference)}>
            <option value="auto">Auto — use WebGPU once a model is big enough to benefit</option>
            <option value="webgpu">WebGPU — prefer it, fall back if unavailable</option>
            <option value="tensorflow">TensorFlow.js — WebGL with TFJS CPU fallback</option>
            <option value="cpu">CPU only — pure JavaScript float32</option>
          </Select>
        </Field>
        <dl className="text-xs text-slate-300 space-y-1 mt-3">
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Device</dt>
            <dd className="text-right">{s.backendStatus ? s.backendStatus.label : "CPU (JavaScript float32) — not probed yet"}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Selection</dt>
            <dd className="text-right">{s.backendStatus ? s.backendStatus.detail : "The backend is created when training starts or when you verify it below."}</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Autodiff</dt>
            <dd className="text-right">Reverse-mode, layer-wise; the accelerator keeps the same gradients</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">WebGPU in this browser</dt>
            <dd className="text-right">
              {s.webgpuAvailable ? "navigator.gpu present — adapter requested on demand" : "not available (no navigator.gpu)"}
            </dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">WGSL path</dt>
            <dd className="text-right">matmul, conv2d (+ dX/dK), pooling, activations, softmax, embedding and the elementwise ops run as WGSL compute kernels</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">TensorFlow.js path</dt>
            <dd className="text-right">Local TFJS kernels for dense, convolution, pooling, activations and embeddings; the existing CPU loss, optimizer and exports stay unchanged. No account or network service needed.</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Still on the CPU</dt>
            <dd className="text-right">recurrent cells (RNN/LSTM/GRU), Batch/LayerNorm and Dropout fall back per layer</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Web Worker</dt>
            <dd className="text-right">Not used; the loop yields with setTimeout(0)</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-slate-500">Storage</dt>
            <dd className="text-right">IndexedDB + localStorage fallback</dd>
          </div>
        </dl>

        {s.backendStatus?.error && (
          <p className="text-[11px] text-amber-300 mt-3 leading-relaxed">Backend note: {s.backendStatus.error}</p>
        )}

        <div className="flex flex-wrap items-center gap-2 mt-3">
          <Button variant="ghost" disabled={s.verifyingDevice || s.progress.status === "running" || s.progress.status === "paused" || s.progress.status === "stopping"} onClick={() => void s.verifyDeviceNow()}>
            {s.verifyingDevice ? "Verifying…" : "Verify device"}
          </Button>
          <span className="text-[11px] text-slate-500">
            Runs all 64 device op cases (matmul, conv forward/dX/dK, pooling, activations, softmax, embedding, reductions, broadcasting, reshape) on small random inputs and diffs every result against the CPU engine. Nothing is accelerated until this passes.
          </span>
        </div>
        {s.deviceReport && <DeviceReportList />}
      </Panel>

      <Panel title="Mathematical self-tests">
        <p className="text-xs text-slate-400 mb-3 leading-relaxed">
          These tests execute the real tensor engine in this page: matmul, losses, numerical gradient checks (including the
          recurrent cells and the residual blocks), XOR learning, presets, the export runtime against the engine,
          serialisation round-trip, CSV parsing, the accelerator's whole operation surface and accelerator-vs-CPU
          forward/backward/training agreement. The accelerator checks run against the CPU reference device here; the same
          suite runs against the WGSL kernels in <em>Verify device</em> above.
        </p>
        <Button
          disabled={busy}
          onClick={() => {
            setBusy(true);
            setResults(null);
            // The synchronous suite first (it is the fast one), then the checks
            // that have to await the accelerator's device surface.
            setTimeout(() => {
              const sync = runSelfTests();
              setResults(sync);
              void runAsyncSelfTests().then((extra) => {
                setResults([...sync, ...extra]);
                setBusy(false);
              });
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
          <li>Implemented: sequential models, Dense, Conv2D, pooling, Dropout, Batch/LayerNorm, Embedding, multi-head attention and transformer blocks (pre-LayerNorm attention + feed-forward, causal masking, optional input projection and pooling), residual blocks (Residual Dense, Residual Conv with identity or learned 1×1 projection shortcuts), activations, SGD/Momentum/Adam/AdamW, MSE/MAE/BCE/CCE, dataset generators, streamed CSV import, image import with resizing and training-only augmentation, custom datasets (hand-drawn 2D points and safe formula generator, persisted with the project), {PRESETS.length} architecture presets, IndexedDB projects, JSON export/import, live charts from real metrics.</li>
          <li>Implemented for sequences: Embedding, a tanh RNN, genuine gated cells — LSTM (input/forget/output gates plus a cell state) and GRU (update/reset gates) — and multi-head attention / transformer blocks, all with their own backpropagation and numerically gradient-checked parameters. Four language datasets ship with it across three tokenizer levels: character (64-char context), word (16-word context, huge generated corpus, frequency-capped vocabulary) and subword BPE (learned merges, adjustable vocabulary size, lossless decoding). The tokenizer — token↔id mapping, level, padding token and any merge rules — is stored in exported model files and validated on import, and the exported runtimes re-tokenise prompts exactly as the app does.</li>
          <li>Standalone Model Export: JavaScript/TypeScript package, single-file module, Python/NumPy package and portable bundle. Each includes an inference runtime for all 27 layer types — including the Deep Reasoning loop and its trace API — plus task-aware input/output handling (preprocessing, image layout, tokenizer, sampling). Before download, the shipped runtime is checked against the engine: weights bit-for-bit, forward-pass agreement, a 54-case layer suite, tokenizer and generation, deep-reasoning output/trace parity, and rejection of corrupted bundles. The Python target is generated but not executed in the browser.</li>
          <li>Compute backend: pure JavaScript float32 on the CPU (default, always available) plus optional WebGPU and TensorFlow.js paths. TensorFlow.js uses WebGL when available and its own CPU backend otherwise; it passes the same verification gate before training. The WebGPU backend runs matmul/conv/pooling/softmax/embedding as WGSL kernels with a per-layer CPU fallback, and it must pass the "Verify device" suite above before the first accelerated epoch — otherwise training stays on the CPU. See the README for what is and is not accelerated.</li>
          <li>Sampling: temperature and top-k decoding with a seeded RNG. Temperature 0 collapses to greedy argmax; the distribution at T=1 is verified against the model's softmax in the self-tests.</li>
          <li>Not implemented: free-form graph topologies (fan-out, merges, multiple inputs/outputs), ONNX import/export, positional-encoding and encoder/decoder layers, WordPiece/unigram tokenizers, remote/distributed training, and a Web Worker for the training loop. Attention and transformer blocks are single chain layers; the WebGPU path keeps them on the CPU engine and synchronises around them.</li>
        </ul>
      </Panel>
    </div>
  );
}

/** Compact rendering of the last device verification run. */
function DeviceReportList() {
  const s = useStore();
  const [showAll, setShowAll] = useState(false);
  const report = s.deviceReport ?? [];
  const summary = summarizeChecks(report);
  const failures = report.filter((r) => !r.passed);
  const shown = showAll ? report : failures;
  return (
    <div className="mt-3 space-y-1">
      <div className={`text-xs ${summary.failed === 0 ? "text-emerald-300" : "text-rose-300"}`}>
        {summary.passed}/{report.length} device ops match the CPU engine
        {summary.failed === 0 ? ` · worst |diff| ${summary.worst.toExponential(1)}` : ` · ${summary.failed} mismatches`}
      </div>
      {report.length > 0 && failures.length === 0 && (
        <p className="text-[11px] text-slate-500 leading-relaxed">
          Every op agrees with the float32 reference implementation within 3·10⁻⁴ relative tolerance, so the accelerated
          training path is producing the same numbers as the CPU path on this machine. This report comes from the device
          that is actually installed here — a different machine can produce different numbers, which is why the check runs
          again on every device and after every preference change.
        </p>
      )}
      <div className="space-y-0.5 max-h-64 overflow-auto">
        {shown.map((r) => (
          <div key={r.name} className={`text-[11px] font-mono flex gap-2 ${r.passed ? "text-slate-400" : "text-rose-300"}`}>
            <span className="w-12">{r.passed ? "OK" : "DIFF"}</span>
            <span className="flex-1">{r.name}</span>
            <span className="text-slate-500">{r.maxDiff.toExponential(1)}</span>
          </div>
        ))}
      </div>
      {report.length > 0 && (
        <button className="text-[11px] text-cyan-300 hover:underline" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Show only problems" : `Show all ${report.length} checks`}
        </button>
      )}
    </div>
  );
}
