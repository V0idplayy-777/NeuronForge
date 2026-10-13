import type { Architecture } from "../engine/types";
import type { TrainingConfig } from "../engine/train";
import type { OptimizerState } from "../engine/optimizers";
import type { BuiltinId, SynthParams } from "../data/datasets";
import type { CustomSpec, SerializedDataset } from "../data/custom";
import type { Experiment } from "./types";

const DB_NAME = "neuronforge";
const DB_VERSION = 1;

export interface StoredProject {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  architecture: Architecture;
  weights: Record<string, { shape: number[]; data: number[] }> | null;
  datasetKind: "builtin" | "csv" | "custom";
  builtinId: BuiltinId;
  synth: SynthParams;
  csvName?: string;
  customSpec?: CustomSpec | null;
  storedDataset?: SerializedDataset | null;
  trainingConfig: TrainingConfig;
  experiments: Experiment[];
  logs: { t: number; level: "info" | "warn" | "error"; msg: string }[];
  optimizerState?: { t: number; slots: Record<string, { m?: number[]; v?: number[] }> };
  lastProgress?: {
    epoch: number;
    step: number;
    history: { epoch: number; trainLoss: number; valLoss?: number; trainAcc?: number; valAcc?: number }[];
  };
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("projects")) {
        db.createObjectStore("projects", { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
  });
}

export async function listProjects(): Promise<StoredProject[]> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction("projects", "readonly");
      const req = tx.objectStore("projects").getAll();
      req.onsuccess = () => {
        const rows = (req.result as StoredProject[]).sort((a, b) => b.updatedAt - a.updatedAt);
        resolve(rows);
      };
      req.onerror = () => reject(req.error);
    });
  } catch {
    return readFallback();
  }
}

export async function getProject(id: string): Promise<StoredProject | undefined> {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction("projects", "readonly");
      const req = tx.objectStore("projects").get(id);
      req.onsuccess = () => resolve(req.result as StoredProject | undefined);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return readFallback().find((p) => p.id === id);
  }
}

export async function saveProject(p: StoredProject): Promise<void> {
  p.updatedAt = Date.now();
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction("projects", "readwrite");
      tx.objectStore("projects").put(p);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    writeFallback(p);
  } catch (e) {
    writeFallback(p);
    if (!(e instanceof Error && /quota/i.test(e.message))) {
      // still persist via localStorage fallback
    }
  }
}

export async function deleteProject(id: string): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction("projects", "readwrite");
      tx.objectStore("projects").delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    /* ignore */
  }
  const all = readFallback().filter((p) => p.id !== id);
  try {
    localStorage.setItem("nf_projects_index", JSON.stringify(all.map((p) => p.id)));
    localStorage.removeItem(`nf_project_${id}`);
  } catch {
    /* ignore */
  }
}

function writeFallback(p: StoredProject): void {
  try {
    const slim: StoredProject = {
      ...p,
      weights: p.weights && weightSize(p.weights) < 1_500_000 ? p.weights : p.weights,
    };
    localStorage.setItem(`nf_project_${p.id}`, JSON.stringify(slim));
    const ids = new Set(readFallback().map((x) => x.id));
    ids.add(p.id);
    localStorage.setItem("nf_projects_index", JSON.stringify([...ids]));
  } catch {
    /* quota */
  }
}

function readFallback(): StoredProject[] {
  try {
    const ids = JSON.parse(localStorage.getItem("nf_projects_index") || "[]") as string[];
    const out: StoredProject[] = [];
    for (const id of ids) {
      const raw = localStorage.getItem(`nf_project_${id}`);
      if (!raw) continue;
      out.push(JSON.parse(raw) as StoredProject);
    }
    return out.sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

function weightSize(w: Record<string, { data: number[] }>): number {
  let n = 0;
  for (const v of Object.values(w)) n += v.data.length;
  return n;
}

export function snapshotWeights(weights: Record<string, { shape: number[]; data: Float32Array }>): Record<string, { shape: number[]; data: number[] }> {
  const out: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [k, v] of Object.entries(weights)) {
    out[k] = { shape: v.shape, data: Array.from(v.data) };
  }
  return out;
}

export type { OptimizerState };
