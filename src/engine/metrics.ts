import type { LossName, TaskType } from "./types";
import { Tensor, argmax1d } from "./tensor";

export interface ClassificationReport {
  accuracy: number;
  precisionMacro: number;
  recallMacro: number;
  f1Macro: number;
  confusion: number[][];
  perClass: { precision: number; recall: number; f1: number; support: number }[];
}

export interface RegressionReport {
  mse: number;
  rmse: number;
  mae: number;
}

export function taskFromLoss(loss: LossName, outputUnits: number): TaskType {
  if (loss === "mse" || loss === "mae") return "regression";
  if (loss === "bce" || outputUnits === 1) return "binary";
  return "multiclass";
}

export function classificationFromPred(pred: Tensor, target: Tensor, binary: boolean): ClassificationReport {
  const B = pred.shape[0]!;
  let classes: number;
  const yTrue: number[] = [];
  const yPred: number[] = [];

  if (binary || pred.shape[pred.rank - 1] === 1) {
    classes = 2;
    for (let i = 0; i < B; i++) {
      yPred.push(pred.data[i]! >= 0.5 ? 1 : 0);
      yTrue.push(target.data[i]! >= 0.5 ? 1 : 0);
    }
  } else {
    classes = pred.shape[pred.rank - 1]!;
    const tLast = target.shape[target.rank - 1]!;
    for (let i = 0; i < B; i++) {
      yPred.push(argmax1d(pred.data, i * classes, classes));
      if (target.size === B) {
        yTrue.push(Math.round(target.data[i]!));
      } else if (tLast === classes) {
        yTrue.push(argmax1d(target.data, i * classes, classes));
      } else {
        yTrue.push(Math.round(target.data[i]!));
      }
    }
  }

  const confusion: number[][] = Array.from({ length: classes }, () => Array(classes).fill(0));
  let correct = 0;
  for (let i = 0; i < B; i++) {
    const t = Math.min(classes - 1, Math.max(0, yTrue[i]!));
    const p = Math.min(classes - 1, Math.max(0, yPred[i]!));
    confusion[t]![p]! += 1;
    if (t === p) correct++;
  }

  const perClass = [];
  let precSum = 0;
  let recSum = 0;
  let f1Sum = 0;
  let present = 0;
  for (let c = 0; c < classes; c++) {
    let tp = confusion[c]![c]!;
    let fp = 0;
    let fn = 0;
    for (let i = 0; i < classes; i++) {
      if (i !== c) {
        fp += confusion[i]![c]!;
        fn += confusion[c]![i]!;
      }
    }
    const support = tp + fn;
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = support === 0 ? 0 : tp / support;
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    perClass.push({ precision, recall, f1, support });
    if (support > 0) {
      precSum += precision;
      recSum += recall;
      f1Sum += f1;
      present++;
    }
  }
  const d = present || 1;
  return {
    accuracy: B === 0 ? 0 : correct / B,
    precisionMacro: precSum / d,
    recallMacro: recSum / d,
    f1Macro: f1Sum / d,
    confusion,
    perClass,
  };
}

export function regressionFromPred(pred: Tensor, target: Tensor): RegressionReport {
  const n = Math.min(pred.size, target.size) || 1;
  let se = 0;
  let ae = 0;
  for (let i = 0; i < n; i++) {
    const d = pred.data[i]! - target.data[i]!;
    se += d * d;
    ae += Math.abs(d);
  }
  const mse = se / n;
  return { mse, rmse: Math.sqrt(mse), mae: ae / n };
}

export function batchAccuracy(pred: Tensor, target: Tensor, loss: LossName): number | null {
  if (loss === "mse" || loss === "mae") return null;
  const binary = loss === "bce" || pred.shape[pred.rank - 1] === 1;
  return classificationFromPred(pred, target, binary).accuracy;
}
