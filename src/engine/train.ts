import type { LossName } from "./types";
import { Tensor, sliceBatch } from "./tensor";
import { SequentialModel, gradNorms } from "./model";
import { computeLoss } from "./losses";
import { batchAccuracy } from "./metrics";
import {
  DEFAULT_OPTIMIZER,
  optimizerStep,
  createOptimizerState,
  type OptimizerConfig,
  type OptimizerState,
} from "./optimizers";
import { RNG } from "./rng";
import type { Accelerator } from "./gpu/executor";

export interface TensorDataset {
  xs: Tensor;
  ys: Tensor;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  shuffle: boolean;
  seed: number;
  loss: LossName;
  optimizer: OptimizerConfig;
  validationEvery: number;
  earlyStoppingPatience: number;
  collectDiagnosticsEvery: number;
}

export const DEFAULT_TRAINING: TrainingConfig = {
  epochs: 50,
  batchSize: 16,
  shuffle: true,
  seed: 42,
  loss: "bce",
  optimizer: { ...DEFAULT_OPTIMIZER },
  validationEvery: 1,
  earlyStoppingPatience: 0,
  collectDiagnosticsEvery: 10,
};

export type TrainStatus = "idle" | "running" | "paused" | "stopping" | "completed" | "failed";

export interface MetricPoint {
  epoch: number;
  batch: number;
  step: number;
  trainLoss: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  lr: number;
  timeMs: number;
}

export interface LayerGradNorm {
  id: string;
  name: string;
  gradNorm: number;
}

export interface TrainProgress {
  status: TrainStatus;
  epoch: number;
  epochs: number;
  batch: number;
  batches: number;
  step: number;
  trainLoss: number;
  valLoss?: number;
  trainAcc?: number;
  valAcc?: number;
  bestValLoss?: number;
  lr: number;
  elapsedMs: number;
  etaMs?: number;
  samplesSeen: number;
  samplesPerSec: number;
  gradNorms: LayerGradNorm[];
  message?: string;
  history: MetricPoint[];
}

export type TrainListener = (p: TrainProgress) => void;

export class Trainer {
  status: TrainStatus = "idle";
  /**
   * Optional accelerated backend. When present, each training batch runs its
   * forward and backward passes through the device instead of the CPU engine.
   * Any device failure drops back to the CPU mid-run rather than losing the run.
   */
  accelerator: Accelerator | null = null;
  private pauseGate: Promise<void> | null = null;
  private resumeFn: (() => void) | null = null;
  config: TrainingConfig;
  model: SequentialModel;
  optState: OptimizerState;
  progress: TrainProgress;
  private listener: TrainListener | null = null;
  private rng: RNG;

  constructor(model: SequentialModel, config: TrainingConfig) {
    this.model = model;
    this.config = structuredClone(config);
    this.optState = createOptimizerState();
    this.rng = new RNG(config.seed);
    this.progress = blankProgress(config);
  }

  on(listener: TrainListener): void {
    this.listener = listener;
  }

  private emit(patch: Partial<TrainProgress>): void {
    this.progress = { ...this.progress, ...patch, status: this.status };
    this.listener?.(this.progress);
  }

  private shouldStop(): boolean {
    return this.status === "stopping";
  }

  startPauseResume(): void {
    if (this.status === "running") this.pause();
    else if (this.status === "paused") this.resume();
  }

  pause(): void {
    if (this.status !== "running") return;
    this.status = "paused";
    this.pauseGate = new Promise((resolve) => {
      this.resumeFn = resolve;
    });
    this.emit({ message: "Paused" });
  }

  resume(): void {
    if (this.status !== "paused") return;
    this.status = "running";
    this.resumeFn?.();
    this.resumeFn = null;
    this.pauseGate = null;
    this.emit({ message: "Resumed" });
  }

  stop(): void {
    if (this.status === "running" || this.status === "paused") {
      this.status = "stopping";
      this.resumeFn?.();
      this.resumeFn = null;
      this.pauseGate = null;
      this.emit({ message: "Stopping…" });
    }
  }

  async waitIfPaused(): Promise<void> {
    while (this.status === "paused" && this.pauseGate) {
      await this.pauseGate;
    }
  }

  /** Forward pass of one batch, falling back to the CPU engine if the device fails. */
  private async forwardBatch(bx: Tensor, epoch: number, batch: number): Promise<{ output: Tensor }> {
    const accel = this.accelerator;
    if (!accel) return { output: this.model.forward(bx, true, false).output };
    try {
      await accel.syncParams();
      const output = await accel.runForward(bx, true);
      return { output };
    } catch (e) {
      this.dropAccelerator(e, epoch, batch);
      return { output: this.model.forward(bx, true, false).output };
    }
  }

  /** Backward pass of one batch, falling back to the CPU engine if the device fails. */
  private async backwardBatch(grad: Tensor, collectDiagnostics: boolean): Promise<LayerGradNorm[]> {
    const accel = this.accelerator;
    if (!accel) return this.model.backward(grad, collectDiagnostics);
    try {
      await accel.runBackward(grad);
      return collectDiagnostics ? gradNorms(this.model) : [];
    } catch (e) {
      this.dropAccelerator(e);
      return this.model.backward(grad, collectDiagnostics);
    }
  }

  /** Disables the accelerator for the rest of the run and reports why. */
  private dropAccelerator(e: unknown, epoch?: number, batch?: number): void {
    const msg = e instanceof Error ? e.message : String(e);
    this.accelerator = null;
    const where = epoch !== undefined ? ` at epoch ${epoch} batch ${batch}` : "";
    this.emit({ message: `Accelerator disabled${where}: ${msg}. Continuing on the CPU backend.` });
  }

  async run(train: TensorDataset, val?: TensorDataset | null): Promise<TrainProgress> {
    if (!this.model.valid) {
      this.status = "failed";
      this.progress.status = "failed";
      this.progress.message = this.model.issues[0]?.message ?? "Invalid architecture";
      this.emit({});
      return this.progress;
    }
    this.status = "running";
    const cfg = this.config;
    const N = train.xs.shape[0]!;
    const batchSize = Math.max(1, Math.min(cfg.batchSize, N));
    const batches = Math.ceil(N / batchSize);
    const start = performance.now();
    this.progress = {
      ...blankProgress(cfg),
      status: "running",
      batches,
      lr: cfg.optimizer.learningRate,
    };
    this.rng = new RNG(cfg.seed);
    let indices = Array.from({ length: N }, (_, i) => i);
    let bestVal = Infinity;
    let patienceLeft = cfg.earlyStoppingPatience;
    let epochTrainLoss = 0;
    let epochTrainAccAcc = 0;
    let epochTrainAccN = 0;
    let latestGradNorms: LayerGradNorm[] = [];
    let step = this.progress.step;

    try {
      for (let epoch = 1; epoch <= cfg.epochs; epoch++) {
        if (this.shouldStop()) break;
        if (cfg.shuffle) indices = this.rng.shuffle(indices);
        epochTrainLoss = 0;
        epochTrainAccAcc = 0;
        epochTrainAccN = 0;
        let epochSamples = 0;

        for (let b = 0; b < batches; b++) {
          await this.waitIfPaused();
          if (this.shouldStop()) break;

          const startIdx = b * batchSize;
          const endIdx = Math.min(N, startIdx + batchSize);
          const batchIdx = indices.slice(startIdx, endIdx);
          const { bx, by } = gather(train, batchIdx);

          this.model.zeroGrads();
          const fwd = await this.forwardBatch(bx, epoch, b + 1);
          const loss = computeLoss(cfg.loss, fwd.output, by);
          if (!Number.isFinite(loss.value)) {
            throw new Error(`Non-finite loss (${loss.value}) at epoch ${epoch} batch ${b + 1}. Try a lower learning rate or gradient clipping.`);
          }
          const diagnosticsEvery = Math.max(1, cfg.collectDiagnosticsEvery || 1);
          const collectDiagnostics = (step + 1) % diagnosticsEvery === 0 || b === batches - 1;
          const gradNorms = await this.backwardBatch(loss.grad, collectDiagnostics);
          if (gradNorms.length) latestGradNorms = gradNorms;
          optimizerStep(this.model, cfg.optimizer, this.optState);

          const acc = batchAccuracy(fwd.output, by, cfg.loss);
          const bs = batchIdx.length;
          epochTrainLoss += loss.value * bs;
          epochSamples += bs;
          if (acc !== null) {
            epochTrainAccAcc += acc * bs;
            epochTrainAccN += bs;
          }
          step += 1;
          const elapsed = performance.now() - start;
          const samplesSeen = (epoch - 1) * N + endIdx;
          const sps = samplesSeen / (elapsed / 1000 || 1);
          const remainingSamples = cfg.epochs * N - samplesSeen;
          const eta = sps > 0 ? (remainingSamples / sps) * 1000 : undefined;

          this.progress = {
            ...this.progress,
            status: this.status,
            epoch,
            epochs: cfg.epochs,
            batch: b + 1,
            batches,
            step,
            trainLoss: loss.value,
            trainAcc: acc ?? undefined,
            lr: cfg.optimizer.learningRate,
            elapsedMs: elapsed,
            etaMs: eta,
            samplesSeen,
            samplesPerSec: sps,
            gradNorms: latestGradNorms,
          };

          if (b === batches - 1 || b % 2 === 0) {
            this.emit({});
            await yieldToMain();
          }
        }

        if (this.shouldStop()) break;

        const trainLoss = epochSamples ? epochTrainLoss / epochSamples : 0;
        const trainAcc = epochTrainAccN ? epochTrainAccAcc / epochTrainAccN : undefined;
        let valLoss: number | undefined;
        let valAcc: number | undefined;
        if (val && val.xs.size > 0 && (epoch % cfg.validationEvery === 0 || epoch === cfg.epochs)) {
          const v = this.accelerator
            ? await evaluateLossAccelerated(val, cfg.loss, this.accelerator)
            : evaluateLoss(this.model, val, cfg.loss);
          valLoss = v.loss;
          valAcc = v.acc ?? undefined;
          if (valLoss < bestVal) {
            bestVal = valLoss;
            patienceLeft = cfg.earlyStoppingPatience;
          } else if (cfg.earlyStoppingPatience > 0) {
            patienceLeft -= 1;
          }
        }

        const point: MetricPoint = {
          epoch,
          batch: batches,
          step,
          trainLoss,
          valLoss,
          trainAcc,
          valAcc,
          lr: cfg.optimizer.learningRate,
          timeMs: performance.now() - start,
        };
        this.progress.history.push(point);
        this.progress.trainLoss = trainLoss;
        this.progress.trainAcc = trainAcc;
        this.progress.valLoss = valLoss;
        this.progress.valAcc = valAcc;
        this.progress.bestValLoss = Number.isFinite(bestVal) ? bestVal : undefined;
        this.progress.epoch = epoch;
        this.emit({ message: `Epoch ${epoch}/${cfg.epochs}  loss ${trainLoss.toFixed(4)}` });

        if (cfg.earlyStoppingPatience > 0 && patienceLeft <= 0) {
          this.progress.message = `Early stopping at epoch ${epoch} (validation loss did not improve)`;
          break;
        }
        await yieldToMain();
      }

      if (this.shouldStop()) {
        this.status = "idle";
        this.progress.status = "idle";
        this.progress.message = "Stopped";
      } else {
        this.status = "completed";
        this.progress.status = "completed";
        this.progress.message = "Training complete";
      }
      this.emit({});
      return this.progress;
    } catch (e) {
      this.status = "failed";
      this.progress.status = "failed";
      this.progress.message = e instanceof Error ? e.message : String(e);
      this.emit({});
      return this.progress;
    }
  }
}

/**
 * Same metric as `evaluateLoss`, but batches run through the accelerator when
 * one is supplied.
 */
export async function evaluateLossAccelerated(
  data: TensorDataset,
  loss: LossName,
  accelerator: Accelerator,
): Promise<{ loss: number; acc: number | null }> {
  const B = data.xs.shape[0]!;
  const chunk = Math.min(256, B);
  let lossAcc = 0;
  let accAcc = 0;
  let accN = 0;
  let n = 0;
  for (let i = 0; i < B; i += chunk) {
    const c = Math.min(chunk, B - i);
    const bx = sliceBatch(data.xs, i, c);
    const by = sliceBatch(data.ys, i, c);
    const pred = await accelerator.runForward(bx, false);
    const L = computeLoss(loss, pred, by);
    lossAcc += L.value * c;
    n += c;
    const acc = batchAccuracy(pred, by, loss);
    if (acc !== null) {
      accAcc += acc * c;
      accN += c;
    }
  }
  return { loss: n ? lossAcc / n : 0, acc: accN ? accAcc / accN : null };
}

export function evaluateLoss(model: SequentialModel, data: TensorDataset, loss: LossName): { loss: number; acc: number | null } {
  const B = data.xs.shape[0]!;
  const chunk = Math.min(256, B);
  let lossAcc = 0;
  let accAcc = 0;
  let accN = 0;
  let n = 0;
  for (let i = 0; i < B; i += chunk) {
    const c = Math.min(chunk, B - i);
    const bx = sliceBatch(data.xs, i, c);
    const by = sliceBatch(data.ys, i, c);
    const pred = model.forward(bx, false).output;
    const L = computeLoss(loss, pred, by);
    lossAcc += L.value * c;
    n += c;
    const acc = batchAccuracy(pred, by, loss);
    if (acc !== null) {
      accAcc += acc * c;
      accN += c;
    }
  }
  return { loss: n ? lossAcc / n : 0, acc: accN ? accAcc / accN : null };
}

export function predict(model: SequentialModel, xs: Tensor): Tensor {
  return model.forward(xs, false).output;
}

function gather(data: TensorDataset, idx: number[]): { bx: Tensor; by: Tensor } {
  const xRest = data.xs.size / data.xs.shape[0]!;
  const yRest = data.ys.size / data.ys.shape[0]!;
  const xd = new Float32Array(idx.length * xRest);
  const yd = new Float32Array(idx.length * yRest);
  for (let i = 0; i < idx.length; i++) {
    const s = idx[i]!;
    xd.set(data.xs.data.subarray(s * xRest, (s + 1) * xRest), i * xRest);
    yd.set(data.ys.data.subarray(s * yRest, (s + 1) * yRest), i * yRest);
  }
  const xShape = [idx.length, ...(data.xs.shape as number[]).slice(1)];
  const yShape = [idx.length, ...(data.ys.shape as number[]).slice(1)];
  return { bx: new Tensor(xShape, xd), by: new Tensor(yShape, yd) };
}

function blankProgress(cfg: TrainingConfig): TrainProgress {
  return {
    status: "idle",
    epoch: 0,
    epochs: cfg.epochs,
    batch: 0,
    batches: 0,
    step: 0,
    trainLoss: 0,
    lr: cfg.optimizer.learningRate,
    elapsedMs: 0,
    samplesSeen: 0,
    samplesPerSec: 0,
    gradNorms: [],
    history: [],
  };
}

export function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
