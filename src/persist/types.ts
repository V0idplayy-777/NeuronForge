import type { Architecture, LossName, OptimizerName } from "../engine/types";

export interface Experiment {
  id: string;
  name: string;
  createdAt: number;
  architecture: Architecture;
  datasetId: string;
  datasetName: string;
  loss: LossName;
  optimizer: OptimizerName;
  learningRate: number;
  batchSize: number;
  epochs: number;
  seed: number;
  trainLoss: number[];
  valLoss: number[];
  trainAcc: number[];
  valAcc: number[];
  finalTrainLoss?: number;
  finalValLoss?: number;
  finalTrainAcc?: number;
  finalValAcc?: number;
  durationMs: number;
  status: "running" | "completed" | "failed" | "stopped";
  error?: string;
  paramCount: number;
  notes?: string;
}

export type ViewId =
  | "overview"
  | "builder"
  | "dataset"
  | "train"
  | "eval"
  | "inspect"
  | "experiments"
  | "export"
  | "settings";
