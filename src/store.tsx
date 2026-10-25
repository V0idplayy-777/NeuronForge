import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Architecture, LayerDef, LayerType } from "./engine/types";
import { uid } from "./engine/types";
import { SequentialModel, addLayer, removeLayer, updateLayer, moveLayer, duplicateLayer } from "./engine/model";
import { Trainer, DEFAULT_TRAINING, type TrainingConfig, type TrainProgress } from "./engine/train";
import { PRESETS, cloneArch, alignToVocab } from "./engine/presets";
import { generateBuiltin, DEFAULT_SYNTH, BUILTIN_INFO, type BuiltinId, type PreparedDataset, type SynthParams } from "./data/datasets";
import { buildCustom, serializePrepared, deserializePrepared, type CustomSpec } from "./data/custom";
import { listProjects, saveProject, deleteProject, snapshotWeights, type StoredProject } from "./persist/db";
import type { Experiment, ViewId } from "./persist/types";
import type { OptimizerState } from "./engine/optimizers";
import { acquireBackend, releaseBackend, hasWebGpu, type BackendPreference, type BackendStatus } from "./engine/gpu/backend";
import { verifyDevice, summarizeChecks, type DeviceCheck } from "./engine/gpu/verify";
import { Accelerator } from "./engine/gpu/executor";

export interface LogEntry {
  t: number;
  level: "info" | "warn" | "error";
  msg: string;
}

interface Store {
  ready: boolean;
  view: ViewId;
  setView: (v: ViewId) => void;
  advanced: boolean;
  setAdvanced: (v: boolean) => void;
  projects: StoredProject[];
  projectId: string;
  projectName: string;
  unsaved: boolean;
  model: SequentialModel;
  architecture: Architecture;
  dataset: PreparedDataset;
  builtinId: BuiltinId;
  synth: SynthParams;
  trainingConfig: TrainingConfig;
  progress: TrainProgress;
  trainer: Trainer | null;
  selectedLayerId: string | null;
  setSelectedLayerId: (id: string | null) => void;
  logs: LogEntry[];
  log: (level: LogEntry["level"], msg: string) => void;
  experiments: Experiment[];
  bottomTab: "logs" | "metrics" | "diag";
  setBottomTab: (t: "logs" | "metrics" | "diag") => void;
  inspectSample: number;
  setInspectSample: (n: number) => void;
  lastActivations: { id: string; name: string; tensor: { shape: number[]; data: Float32Array } }[] | null;

  createProject: (name?: string) => void;
  renameProject: (name: string) => void;
  duplicateProject: () => void;
  openProject: (id: string) => Promise<void>;
  saveNow: () => Promise<void>;
  deleteCurrent: () => Promise<void>;

  applyPreset: (id: string) => void;
  setInputShape: (shape: number[]) => void;
  setInit: (method: Architecture["initMethod"], seed?: number) => void;
  add: (type: LayerType, index?: number) => void;
  remove: (id: string) => void;
  update: (id: string, patch: Partial<LayerDef>) => void;
  move: (id: string, dir: -1 | 1) => void;
  duplicate: (id: string) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reinitWeights: () => void;

  setDataset: (id: BuiltinId, params?: Partial<SynthParams>) => void;
  setSynth: (p: Partial<SynthParams>, regen?: boolean) => void;
  loadCustomDataset: (ds: PreparedDataset) => void;
  setCustomDataset: (spec: CustomSpec) => void;
  customSpec: CustomSpec | null;

  setTrainingConfig: (p: Partial<TrainingConfig>) => void;
  startTraining: () => Promise<void>;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stopTraining: () => void;
  captureExperiment: () => void;

  setOptState: (s: OptimizerState | null) => void;
  loadImportedModel: (m: SequentialModel) => void;

  /** Compute backend (CPU by default, WebGPU opt-in). */
  backendPreference: BackendPreference;
  setBackendPreference: (p: BackendPreference) => void;
  backendStatus: BackendStatus | null;
  webgpuAvailable: boolean;
  /** Result of the last device verification (kernels vs the CPU reference). */
  deviceReport: DeviceCheck[] | null;
  verifyingDevice: boolean;
  verifyDeviceNow: () => Promise<void>;
  /** How the last training run was split between device and CPU layers. */
  backendPlacements: { device: number; cpu: number } | null;
}

const Ctx = createContext<Store | null>(null);

/**
 * Parameters below which "auto" mode keeps training on the CPU. Uploading a
 * batch, reading the output back and collecting gradients costs several device
 * round trips per batch; under this size the arithmetic is too small to hide
 * them and the CPU engine is genuinely faster.
 */
export const DEVICE_PARAM_THRESHOLD = 20000;

function starterProject(): StoredProject {
  const preset = PRESETS[0]!;
  return {
    id: uid("proj"),
    name: "XOR Lab",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    architecture: cloneArch(preset.arch),
    weights: null,
    datasetKind: "builtin",
    builtinId: "xor",
    synth: { ...DEFAULT_SYNTH, n: 4 },
    trainingConfig: {
      ...DEFAULT_TRAINING,
      epochs: 200,
      batchSize: 4,
      loss: "bce",
      optimizer: { ...DEFAULT_TRAINING.optimizer, name: "adam", learningRate: 0.05 },
    },
    experiments: [],
    logs: [],
  };
}

const idleProgress = (): TrainProgress => ({
  status: "idle",
  epoch: 0,
  epochs: 0,
  batch: 0,
  batches: 0,
  step: 0,
  trainLoss: 0,
  lr: 0.01,
  elapsedMs: 0,
  samplesSeen: 0,
  samplesPerSec: 0,
  gradNorms: [],
  history: [],
});

export function StoreProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [view, setView] = useState<ViewId>("overview");
  const [advanced, setAdvanced] = useState(false);
  const [projects, setProjects] = useState<StoredProject[]>([]);
  const [project, setProject] = useState<StoredProject>(() => starterProject());
  const [model, setModel] = useState<SequentialModel>(() => new SequentialModel(project.architecture));
  const [dataset, setDs] = useState<PreparedDataset>(() => generateBuiltin("xor", { ...DEFAULT_SYNTH, n: 4 }));
  const [progress, setProgress] = useState<TrainProgress>(idleProgress);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [unsaved, setUnsaved] = useState(false);
  const [bottomTab, setBottomTab] = useState<"logs" | "metrics" | "diag">("metrics");
  const [inspectSample, setInspectSample] = useState(0);
  const [lastActivations, setLastActivations] = useState<Store["lastActivations"]>(null);
  const [undoStack, setUndo] = useState<Architecture[]>([]);
  const [redoStack, setRedo] = useState<Architecture[]>([]);
  const trainerRef = useRef<Trainer | null>(null);
  const [backendPreference, setBackendPreferenceState] = useState<BackendPreference>(() => {
    try {
      const stored = localStorage.getItem("nf.backend");
      return stored === "webgpu" || stored === "cpu" || stored === "tensorflow" ? stored : "auto";
    } catch {
      return "auto";
    }
  });
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [deviceReport, setDeviceReport] = useState<DeviceCheck[] | null>(null);
  const [verifyingDevice, setVerifyingDevice] = useState(false);
  const backendOperationRef = useRef(false);
  const [backendPlacements, setBackendPlacements] = useState<{ device: number; cpu: number } | null>(null);
  const verifiedRef = useRef<{ preference: BackendPreference; report: DeviceCheck[] } | null>(null);
  const modelRef = useRef(model);
  modelRef.current = model;
  const projectRef = useRef(project);
  projectRef.current = project;
  const datasetRef = useRef(dataset);
  datasetRef.current = dataset;

  const log = useCallback((level: LogEntry["level"], msg: string) => {
    setLogs((l) => [...l.slice(-199), { t: Date.now(), level, msg }]);
  }, []);

  const restoreDataset = useCallback((p: StoredProject): PreparedDataset => {
    if (p.datasetKind !== "builtin" && p.storedDataset) {
      try {
        return deserializePrepared(p.storedDataset);
      } catch {
        /* fall through to builtin */
      }
    }
    if (p.datasetKind === "custom" && p.customSpec) {
      try {
        return buildCustom(p.customSpec);
      } catch {
        /* fall through */
      }
    }
    try {
      return generateBuiltin(p.builtinId, p.synth);
    } catch {
      return generateBuiltin("xor", { ...DEFAULT_SYNTH, n: 4 });
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const existing = await listProjects();
        if (cancelled) return;
        if (existing.length) {
          setProjects(existing);
          const p = existing[0]!;
          setProject(p);
          const m = new SequentialModel(p.architecture);
          if (p.weights) {
            const snap: Record<string, { shape: number[]; data: Float32Array }> = {};
            for (const [k, v] of Object.entries(p.weights)) snap[k] = { shape: v.shape, data: Float32Array.from(v.data) };
            m.loadWeights(snap);
          }
          setModel(m);
          setDs(restoreDataset(p));
          log("info", `Restored project “${p.name}”`);
        } else {
          setProjects([projectRef.current]);
          log("info", "Created starter project XOR Lab. Open Training Lab and press Start to watch real gradient descent.");
        }
      } catch (e) {
        log("warn", `Storage unavailable (${e instanceof Error ? e.message : String(e)}). Working in-memory.`);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [log]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void persist();
      }
      if (meta && e.key.toLowerCase() === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      }
      if (meta && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint won't run; undo/redo defined later — we bind via refs below
  });

  const persist = useCallback(async () => {
    const p = projectRef.current;
    const m = modelRef.current;
    const stored: StoredProject = {
      ...p,
      architecture: m.arch,
      weights: snapshotWeights(m.snapshotWeights()),
      updatedAt: Date.now(),
      logs: logs.slice(-50).map((l) => ({ t: l.t, level: l.level, msg: l.msg })),
    };
    await saveProject(stored);
    setProject(stored);
    setProjects((ps) => {
      const rest = ps.filter((x) => x.id !== stored.id);
      return [stored, ...rest];
    });
    setUnsaved(false);
    log("info", "Project saved locally (IndexedDB)");
  }, [logs, log]);

  const pushArch = useCallback(
    (next: Architecture, rebuildFromCurrent = true) => {
      setUndo((u) => [...u.slice(-49), modelRef.current.arch]);
      setRedo([]);
      const m = new SequentialModel(next, rebuildFromCurrent ? modelRef.current : null);
      setModel(m);
      setProject((p) => ({ ...p, architecture: next }));
      setUnsaved(true);
      if (m.issues.length) log("warn", m.issues[0]!.message);
    },
    [log],
  );

  const createProject = useCallback(
    (name?: string) => {
      const p = starterProject();
      p.name = name || "Untitled Lab";
      p.id = uid("proj");
      setProject(p);
      setModel(new SequentialModel(p.architecture));
      setDs(generateBuiltin("xor", { ...DEFAULT_SYNTH, n: 4 }));
      setProgress(idleProgress());
      setUnsaved(true);
      setSelectedLayerId(null);
      setUndo([]);
      setRedo([]);
      setProjects((ps) => [p, ...ps]);
      log("info", `Created project “${p.name}”`);
      setView("builder");
    },
    [log],
  );

  const openProject = useCallback(
    async (id: string) => {
      const p = projects.find((x) => x.id === id);
      if (!p) return;
      setProject(p);
      const m = new SequentialModel(p.architecture);
      if (p.weights) {
        const snap: Record<string, { shape: number[]; data: Float32Array }> = {};
        for (const [k, v] of Object.entries(p.weights)) snap[k] = { shape: v.shape, data: Float32Array.from(v.data) };
        const w = m.loadWeights(snap);
        w.forEach((msg) => log("warn", msg));
      }
      setModel(m);
      setDs(restoreDataset(p));
      setProgress(idleProgress());
      setUnsaved(false);
      log("info", `Opened “${p.name}”`);
    },
    [projects, log],
  );

  const renameProject = useCallback((name: string) => {
    setProject((p) => ({ ...p, name }));
    setUnsaved(true);
  }, []);

  const duplicateProject = useCallback(() => {
    const p = { ...projectRef.current, id: uid("proj"), name: `${projectRef.current.name} copy`, createdAt: Date.now(), updatedAt: Date.now() };
    setProject(p);
    setProjects((ps) => [p, ...ps]);
    setUnsaved(true);
    log("info", `Duplicated as “${p.name}”`);
  }, [log]);

  const deleteCurrent = useCallback(async () => {
    const id = projectRef.current.id;
    await deleteProject(id);
    const rest = projects.filter((p) => p.id !== id);
    setProjects(rest);
    if (rest[0]) await openProject(rest[0].id);
    else createProject("XOR Lab");
    log("info", "Project deleted");
  }, [projects, openProject, createProject, log]);

  const applyPreset = useCallback(
    (id: string) => {
      const pr = PRESETS.find((p) => p.id === id);
      if (!pr) return;
      let archToLoad = cloneArch(pr.arch);
      let note = `Loaded preset “${pr.name}”. Weights freshly initialized.`;
      // Switch to the matching built-in dataset and its recommended loss so the preset trains out of the box.
      if (pr.datasetHint in BUILTIN_INFO) {
        const hint = pr.datasetHint as BuiltinId;
        try {
          const synth = { ...projectRef.current.synth };
          let ds = generateBuiltin(hint, hint === "xor" || hint === "and" || hint === "or" ? { ...synth, n: Math.max(4, synth.n) } : synth);
          if (pr.category === "Autoencoder") {
            // Reconstruction task: targets are the inputs themselves.
            ds = {
              ...ds,
              meta: {
                ...ds.meta,
                name: `${ds.meta.name} (reconstruction)`,
                description: `${ds.meta.description} Targets = inputs for autoencoder training.`,
                task: "regression",
                recommendedLoss: "mse",
                targetNames: ds.meta.featureNames,
                classNames: undefined,
                outputUnits: ds.meta.inputShape.reduce((a, b) => a * b, 1),
                reconstruction: true,
              },
              train: { xs: ds.train.xs, ys: ds.train.xs },
              val: { xs: ds.val.xs, ys: ds.val.xs },
              test: { xs: ds.test.xs, ys: ds.test.xs },
              allY: ds.allX,
            };
          }
          setDs(ds);
          const isRecon = pr.category === "Autoencoder";
          const stored = isRecon ? serializePrepared(ds) : null;
          setProject((p) => ({
            ...p,
            builtinId: hint,
            datasetKind: isRecon ? "custom" : "builtin",
            storedDataset: stored,
            customSpec: null,
            trainingConfig: { ...p.trainingConfig, loss: ds.meta.recommendedLoss },
          }));
          note += ` Dataset set to “${ds.meta.name}”, loss set to ${ds.meta.recommendedLoss}.`;
          if (ds.meta.tokenizer) {
            const tok = ds.meta.tokenizer;
            archToLoad = alignToVocab(archToLoad, tok.tokens.length, tok.seqLen);
            note += ` Vocabulary aligned: ${tok.tokens.length.toLocaleString()} ${tok.level} tokens, ${tok.seqLen}-token context. Tokenizer is stored with exported models.`;
          }
        } catch (e) {
          log("warn", e instanceof Error ? e.message : String(e));
        }
      }
      pushArch(archToLoad, false);
      // Text and code models can have large optimizer state and attention workspaces.
      // Start with a bounded, device-sensitive batch and short smoke-test run;
      // users can increase epochs after confirming the run fits their device.
      if (pr.category === "Text" || pr.category === "Code") {
        const parameterCount = new SequentialModel(archToLoad).paramCount().trainable;
        const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
        const cores = navigator.hardwareConcurrency || 4;
        const lowMemory = (deviceMemory !== undefined && deviceMemory <= 4) || cores <= 4;
        const batchSize = parameterCount >= 800_000 ? (lowMemory ? 1 : 2) : parameterCount >= 250_000 ? (lowMemory ? 2 : 4) : 8;
        const epochs = Math.min(projectRef.current.trainingConfig.epochs, parameterCount >= 250_000 ? 3 : 5);
        setTrainingConfig({ batchSize, epochs });
        note += ` Safe starter run set to batch ${batchSize}, ${epochs} epochs for this ${parameterCount.toLocaleString()}-parameter text model; raise these gradually if your device has room.`;
      }
      log("info", note);
      setView("builder");
    },
    [pushArch, log],
  );

  const setInputShape = useCallback(
    (shape: number[]) => {
      pushArch({ ...modelRef.current.arch, inputShape: shape });
    },
    [pushArch],
  );

  const setInit = useCallback(
    (method: Architecture["initMethod"], seed?: number) => {
      const arch = { ...modelRef.current.arch, initMethod: method, seed: seed ?? modelRef.current.arch.seed };
      pushArch(arch, false);
      log("info", `Reinitialized with ${method}, seed ${arch.seed}`);
    },
    [pushArch, log],
  );

  const add = useCallback(
    (type: LayerType, index?: number) => {
      const next = addLayer(modelRef.current.arch, type, index);
      pushArch(next);
      const added = next.layers[index ?? next.layers.length - 1]!;
      setSelectedLayerId(added.id);
    },
    [pushArch],
  );

  const remove = useCallback(
    (id: string) => {
      pushArch(removeLayer(modelRef.current.arch, id));
      setSelectedLayerId(null);
    },
    [pushArch],
  );

  const update = useCallback(
    (id: string, patch: Partial<LayerDef>) => {
      pushArch(updateLayer(modelRef.current.arch, id, patch));
    },
    [pushArch],
  );

  const move = useCallback(
    (id: string, dir: -1 | 1) => {
      pushArch(moveLayer(modelRef.current.arch, id, dir));
    },
    [pushArch],
  );

  const duplicate = useCallback(
    (id: string) => {
      pushArch(duplicateLayer(modelRef.current.arch, id));
    },
    [pushArch],
  );

  const undo = useCallback(() => {
    setUndo((u) => {
      if (!u.length) return u;
      const prev = u[u.length - 1]!;
      setRedo((r) => [...r, modelRef.current.arch]);
      setModel(new SequentialModel(prev, modelRef.current));
      setProject((p) => ({ ...p, architecture: prev }));
      setUnsaved(true);
      return u.slice(0, -1);
    });
  }, []);

  const redo = useCallback(() => {
    setRedo((r) => {
      if (!r.length) return r;
      const next = r[r.length - 1]!;
      setUndo((u) => [...u, modelRef.current.arch]);
      setModel(new SequentialModel(next, modelRef.current));
      setProject((p) => ({ ...p, architecture: next }));
      setUnsaved(true);
      return r.slice(0, -1);
    });
  }, []);

  const reinitWeights = useCallback(() => {
    const m = modelRef.current;
    m.reinitialize(m.arch.seed);
    setModel(new SequentialModel(m.arch));
    setUnsaved(true);
    log("info", "Weights reinitialized. Trained parameters discarded.");
  }, [log]);

  const setDataset = useCallback(
    (id: BuiltinId, params?: Partial<SynthParams>) => {
      const synth = { ...projectRef.current.synth, ...params };
      const ds = generateBuiltin(id, id === "xor" || id === "and" || id === "or" ? { ...synth, n: Math.max(4, synth.n) } : synth);
      setDs(ds);
      setProject((p) => ({ ...p, builtinId: id, synth, datasetKind: "builtin", storedDataset: null, customSpec: null }));
      setUnsaved(true);
      log("info", `Loaded dataset “${ds.meta.name}” (${ds.train.xs.shape[0]} train / ${ds.val.xs.shape[0]} val / ${ds.test.xs.shape[0]} test)`);
      if (ds.meta.tokenizer) {
        const tok = ds.meta.tokenizer;
        const arch = modelRef.current.arch;
        const hasEmbedding = arch.layers.some((l) => l.type === "embedding");
        if (hasEmbedding) {
          const before = JSON.stringify(arch);
          const aligned = alignToVocab(arch, tok.tokens.length, tok.seqLen);
          if (JSON.stringify(aligned) !== before) {
            pushArch(aligned);
            log("info", `Model vocabulary aligned to the dataset tokenizer: ${tok.tokens.length.toLocaleString()} ${tok.level} tokens, ${tok.seqLen} context. Changed layers were reinitialized.`);
          }
        } else {
          log("warn", `Language dataset loaded (${tok.tokens.length.toLocaleString()} ${tok.level} tokens, context ${tok.seqLen}) but the current model has no Embedding layer. Load a Text preset to train it.`);
        }
      }
    },
    [log, pushArch],
  );

  const setSynth = useCallback(
    (p: Partial<SynthParams>, regen = true) => {
      const synth = { ...projectRef.current.synth, ...p };
      setProject((pr) => ({ ...pr, synth }));
      if (regen) setDataset(projectRef.current.builtinId, synth);
      else setUnsaved(true);
    },
    [setDataset],
  );

  const loadCustomDataset = useCallback(
    (ds: PreparedDataset) => {
      setDs(ds);
      const stored = serializePrepared(ds);
      setProject((p) => ({ ...p, datasetKind: "csv", csvName: ds.meta.name, storedDataset: stored, customSpec: null }));
      setUnsaved(true);
      log("info", `Imported “${ds.meta.name}”${stored ? " (persisted with project)" : " (too large to persist — will need re-import after reload)"}`);
    },
    [log],
  );

  const setCustomDataset = useCallback(
    (spec: CustomSpec) => {
      const ds = buildCustom(spec); // throws with a clear message on invalid specs
      setDs(ds);
      const stored = serializePrepared(ds);
      setProject((p) => ({
        ...p,
        datasetKind: "custom",
        customSpec: spec,
        storedDataset: stored,
        trainingConfig: { ...p.trainingConfig, loss: ds.meta.recommendedLoss },
      }));
      setUnsaved(true);
      log("info", `Built custom dataset “${ds.meta.name}” (${ds.train.xs.shape[0]} train / ${ds.val.xs.shape[0]} val / ${ds.test.xs.shape[0]} test). Loss set to ${ds.meta.recommendedLoss}.`);
    },
    [log],
  );

  const setTrainingConfig = useCallback((patch: Partial<TrainingConfig>) => {
    setProject((p) => {
      const trainingConfig: TrainingConfig = {
        ...p.trainingConfig,
        ...patch,
        optimizer: { ...p.trainingConfig.optimizer, ...(patch.optimizer ?? {}) },
      };
      return { ...p, trainingConfig };
    });
    setUnsaved(true);
  }, []);

  const startTraining = useCallback(async () => {
    if (backendOperationRef.current) return;
    if (trainerRef.current && (trainerRef.current.status === "running" || trainerRef.current.status === "paused")) {
      log("warn", "Training is already in progress.");
      return;
    }
    const m = modelRef.current;
    if (!m.valid) {
      log("error", m.issues[0]?.message ?? "Architecture is invalid");
      setView("builder");
      return;
    }
    const ds = datasetRef.current;
    const inShape = m.arch.inputShape;
    const got = ds.train.xs.shape.slice(1);
    if (got.length !== inShape.length || got.some((d, i) => d !== inShape[i])) {
      log("error", `Dataset input ${JSON.stringify(got)} does not match model input ${JSON.stringify(inShape)}. Adjust the model or pick a compatible dataset.`);
      return;
    }
    if (ds.meta.tokenizer) {
      const vocab = ds.meta.tokenizer.tokens.length;
      const emb = m.arch.layers.find((l) => l.type === "embedding");
      const outUnits = m.outputShape[m.outputShape.length - 1]!;
      if (emb && Number(emb.config.vocabSize) !== vocab) {
        log("error", `Embedding vocabSize is ${String(emb.config.vocabSize)} but the dataset tokenizer has ${vocab} tokens. Re-select the dataset to auto-align, or fix the layer.`);
        return;
      }
      if (outUnits !== vocab) {
        log("error", `Model outputs ${outUnits} classes but the tokenizer has ${vocab} characters. Predictions would decode against the wrong mapping.`);
        return;
      }
    }
    backendOperationRef.current = true;
    setVerifyingDevice(true);
    try {
      const backend = await acquireBackend(backendPreference);
      setBackendStatus(backend.status);
      setBackendPlacements(null);
      let accelerator: Accelerator | null = null;
      // A device round trip costs a stall per batch; on a small model that is
      // more than the arithmetic saves. "Auto" therefore waits until the model is
      // big enough to hide the transfer, while an explicit WebGPU preference
      // always uses the device (the user asked for it).
      const trainable = m.paramCount().trainable;
      const worthDevice = backendPreference !== "auto" || trainable >= DEVICE_PARAM_THRESHOLD;
      if (backend.status.kind === "webgpu" && !worthDevice) {
        setBackendPlacements(null);
        log(
          "info",
          `WebGPU is available, but ${trainable.toLocaleString()} trainable parameters is below the ${DEVICE_PARAM_THRESHOLD.toLocaleString()}-parameter point where a per-batch device round trip pays off — training on the CPU. Pick "WebGPU" in Settings to force it anyway.`,
        );
      }
      if (backend.status.kind !== "cpu" && worthDevice) {
        // Verify the kernels against the CPU engine once per device before use.
        if (!verifiedRef.current || verifiedRef.current.preference !== backendPreference) {
          setVerifyingDevice(true);
          log("info", `Checking ${backend.status.label} against the CPU engine before the first accelerated epoch…`);
          try {
            const checks = await verifyDevice(backend.ops);
            verifiedRef.current = { preference: backendPreference, report: checks };
            setDeviceReport(checks);
          } catch (e) {
            log("warn", `Backend verification errored (${e instanceof Error ? e.message : String(e)}); continuing on the CPU.`);
            verifiedRef.current = null;
          } finally {
            setVerifyingDevice(false);
          }
        }
        const report = verifiedRef.current?.report ?? null;
        const summary = report ? summarizeChecks(report) : null;
        if (report && summary && summary.failed === 0) {
          accelerator = new Accelerator(m, backend.ops);
          setBackendPlacements({ device: accelerator.stats.deviceLayers, cpu: accelerator.stats.cpuLayers });
          log("info", `${backend.status.label} active · ${accelerator.stats.deviceLayers} layers on the device, ${accelerator.stats.cpuLayers} on the CPU · kernels verified (${summary.passed} checks, worst |diff| ${summary.worst.toExponential(1)}).`);
        } else if (report) {
          log("error", `Backend kernels failed verification (${summary?.failed ?? "?"} mismatches) — training on the CPU backend.`);
        }
      } else {
        setBackendPlacements(null);
        if (backend.status.error && backendPreference !== "auto") log("warn", backend.status.error);
      }

      setVerifyingDevice(false);
      const t = new Trainer(m, projectRef.current.trainingConfig);
      t.accelerator = accelerator;
      trainerRef.current = t;
      t.on((p) => setProgress({ ...p, history: p.history.slice() }));
      log("info", `Training started · ${projectRef.current.trainingConfig.epochs} epochs · ${projectRef.current.trainingConfig.optimizer.name} · lr ${projectRef.current.trainingConfig.optimizer.learningRate}`);
      setView("train");
      setBottomTab("metrics");
      const result = await t.run(ds.train, ds.val).finally(() => accelerator?.destroy());
      setModel(m);
      setUnsaved(true);
      if (result.status === "completed") log("info", `Training complete. Final train loss ${result.trainLoss.toFixed(4)}${result.valLoss !== undefined ? ` · val ${result.valLoss.toFixed(4)}` : ""}`);
      else if (result.status === "failed") log("error", result.message ?? "Training failed");
      else log("info", result.message ?? "Training stopped");
      try {
        const fwd = m.forward(ds.train.xs, false, true);
        setLastActivations(
          fwd.activations.map((a) => ({
            id: a.id,
            name: a.name,
            tensor: { shape: a.tensor.shape as number[], data: new Float32Array(a.tensor.data) },
          })),
        );
      } catch {
        setLastActivations(null);
      }
    } finally {
      backendOperationRef.current = false;
      setVerifyingDevice(false);
    }
  }, [log, backendPreference]);

  const setBackendPreference = useCallback((p: BackendPreference) => {
    if (backendOperationRef.current) return;
    if (trainerRef.current?.status === "running" || trainerRef.current?.status === "paused" || trainerRef.current?.status === "stopping") return;
    setBackendPreferenceState(p);
    verifiedRef.current = null;
    setDeviceReport(null);
    setBackendPlacements(null);
    setBackendStatus(null);
    releaseBackend();
    try {
      localStorage.setItem("nf.backend", p);
    } catch {
      /* storage may be unavailable; the preference still applies for this session */
    }
  }, []);

  /**
   * Runs the device's operations against the CPU reference implementation.
   * Nothing is trusted until this passes: a failing kernel check keeps training
   * on the CPU rather than producing silently wrong numbers.
   */
  const verifyDeviceNow = useCallback(async () => {
    if (backendOperationRef.current) return;
    backendOperationRef.current = true;
    setVerifyingDevice(true);
    verifiedRef.current = null;
    setDeviceReport(null);
    const pref = backendPreference;
    try {
      const backend = await acquireBackend(pref);
      setBackendStatus(backend.status);
      if (backend.status.kind === "cpu") {
        if (backend.status.error) log("warn", backend.status.error);
        return;
      }
      const checks = await verifyDevice(backend.ops);
      const summary = summarizeChecks(checks);
      verifiedRef.current = { preference: pref, report: checks };
      setDeviceReport(checks);
      if (summary.failed === 0) {
        log("info", `Device verified: ${summary.passed} kernel checks match the CPU engine (worst |diff| ${summary.worst.toExponential(1)}).`);
      } else {
        log("error", `Device verification failed: ${summary.failed} of ${checks.length} checks differ from the CPU engine. Training stays on the CPU.`);
      }
    } catch (e) {
      log("error", `Device verification errored: ${e instanceof Error ? e.message : String(e)}`);
      setDeviceReport(null);
    } finally {
      backendOperationRef.current = false;
      setVerifyingDevice(false);
    }
  }, [backendPreference, log]);

  const pauseTraining = useCallback(() => trainerRef.current?.pause(), []);
  const resumeTraining = useCallback(() => trainerRef.current?.resume(), []);
  const stopTraining = useCallback(() => trainerRef.current?.stop(), []);

  const captureExperiment = useCallback(() => {
    const p = progress;
    const m = modelRef.current;
    const exp: Experiment = {
      id: uid("exp"),
      name: `${projectRef.current.name} · run ${projectRef.current.experiments.length + 1}`,
      createdAt: Date.now(),
      architecture: structuredClone(m.arch),
      datasetId: datasetRef.current.meta.id,
      datasetName: datasetRef.current.meta.name,
      loss: projectRef.current.trainingConfig.loss,
      optimizer: projectRef.current.trainingConfig.optimizer.name,
      learningRate: projectRef.current.trainingConfig.optimizer.learningRate,
      batchSize: projectRef.current.trainingConfig.batchSize,
      epochs: projectRef.current.trainingConfig.epochs,
      seed: projectRef.current.trainingConfig.seed,
      trainLoss: p.history.map((h) => h.trainLoss),
      valLoss: p.history.map((h) => h.valLoss ?? NaN),
      trainAcc: p.history.map((h) => h.trainAcc ?? NaN),
      valAcc: p.history.map((h) => h.valAcc ?? NaN),
      finalTrainLoss: p.trainLoss,
      finalValLoss: p.valLoss,
      finalTrainAcc: p.trainAcc,
      finalValAcc: p.valAcc,
      durationMs: p.elapsedMs,
      status: p.status === "completed" ? "completed" : p.status === "failed" ? "failed" : "stopped",
      error: p.status === "failed" ? p.message : undefined,
      paramCount: m.paramCount().trainable,
    };
    setProject((pr) => ({ ...pr, experiments: [exp, ...pr.experiments] }));
    setUnsaved(true);
    log("info", `Recorded experiment “${exp.name}”`);
    setView("experiments");
  }, [progress, log]);

  const setOptState = useCallback((s: OptimizerState | null) => {
    if (!s) return;
    if (trainerRef.current) trainerRef.current.optState = s;
  }, []);

  const loadImportedModel = useCallback(
    (m: SequentialModel) => {
      setUndo((u) => [...u.slice(-49), modelRef.current.arch]);
      setRedo([]);
      setModel(m);
      setProject((p) => ({ ...p, architecture: m.arch }));
      setUnsaved(true);
      setSelectedLayerId(null);
      log("info", `Imported model with ${m.paramCount().trainable} trainable parameters`);
    },
    [log],
  );

  const value = useMemo<Store>(
    () => ({
      ready,
      view,
      setView,
      advanced,
      setAdvanced,
      projects,
      projectId: project.id,
      projectName: project.name,
      unsaved,
      model,
      architecture: model.arch,
      dataset,
      builtinId: project.builtinId,
      synth: project.synth,
      trainingConfig: project.trainingConfig,
      progress,
      trainer: trainerRef.current,
      selectedLayerId,
      setSelectedLayerId,
      logs,
      log,
      experiments: project.experiments,
      bottomTab,
      setBottomTab,
      inspectSample,
      setInspectSample,
      lastActivations,
      createProject,
      renameProject,
      duplicateProject,
      openProject,
      saveNow: persist,
      deleteCurrent,
      applyPreset,
      setInputShape,
      setInit,
      add,
      remove,
      update,
      move,
      duplicate,
      undo,
      redo,
      canUndo: undoStack.length > 0,
      canRedo: redoStack.length > 0,
      reinitWeights,
      setDataset,
      setSynth,
      loadCustomDataset,
      setCustomDataset,
      customSpec: project.customSpec ?? null,
      setTrainingConfig,
      startTraining,
      pauseTraining,
      resumeTraining,
      stopTraining,
      captureExperiment,
      setOptState,
      loadImportedModel,
      backendPreference,
      setBackendPreference,
      backendStatus,
      webgpuAvailable: hasWebGpu(),
      deviceReport,
      verifyingDevice,
      verifyDeviceNow,
      backendPlacements,
    }),
    [
      ready,
      view,
      advanced,
      projects,
      project,
      unsaved,
      model,
      dataset,
      progress,
      selectedLayerId,
      logs,
      log,
      bottomTab,
      inspectSample,
      lastActivations,
      createProject,
      renameProject,
      duplicateProject,
      openProject,
      persist,
      deleteCurrent,
      applyPreset,
      setInputShape,
      setInit,
      add,
      remove,
      update,
      move,
      duplicate,
      undo,
      redo,
      undoStack.length,
      redoStack.length,
      reinitWeights,
      setDataset,
      setSynth,
      loadCustomDataset,
      setCustomDataset,
      setTrainingConfig,
      startTraining,
      pauseTraining,
      resumeTraining,
      stopTraining,
      captureExperiment,
      setOptState,
      loadImportedModel,
      backendPreference,
      setBackendPreference,
      backendStatus,
      deviceReport,
      verifyingDevice,
      verifyDeviceNow,
      backendPlacements,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside provider");
  return s;
}
  
