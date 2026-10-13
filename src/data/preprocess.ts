import { RNG } from "../engine/rng";
import type { DatasetMeta, PreparedDataset, TabularColumn, TabularPreprocessing } from "./datasets";
import { pack } from "./datasets";
import type { CsvImportConfig } from "./csv";
import { inferNumeric } from "./csv";
import type { LossName, TaskType } from "../engine/types";

export interface PreprocessFit {
  featureNames: string[];
  targetNames: string[];
  numericMeans: number[];
  numericStds: number[];
  numericMins: number[];
  numericMaxs: number[];
  catMaps: { col: number; values: string[] }[];
  droppedRows: number;
  missingFilled: number;
  normalize: CsvImportConfig["normalize"];
}

export function materializeCsv(
  headers: string[],
  rows: string[][],
  cfg: CsvImportConfig,
): { dataset: PreparedDataset; fit: PreprocessFit; warnings: string[] } {
  const warnings: string[] = [];
  if (cfg.featureIdx.length === 0) throw new Error("Select at least one feature column.");
  if (cfg.targetIdx.length === 0) throw new Error("Select at least one target column.");

  const featCols = cfg.featureIdx.map((i) => ({ i, name: headers[i] ?? `f${i}`, numeric: inferNumeric(rows.map((r) => r[i] ?? "")) }));
  const tgtCols = cfg.targetIdx.map((i) => ({ i, name: headers[i] ?? `y${i}`, numeric: inferNumeric(rows.map((r) => r[i] ?? "")) }));

  const catFeat = featCols.filter((c) => !c.numeric);
  if (catFeat.length && cfg.categorical === "drop") {
    warnings.push(`Dropped ${catFeat.length} categorical feature column(s): ${catFeat.map((c) => c.name).join(", ")}`);
  }

  const usedFeat = featCols.filter((c) => c.numeric || cfg.categorical !== "drop");
  if (usedFeat.length === 0) throw new Error("No feature columns remain after dropping categoricals.");

  // Fit categorical maps and numeric stats on ALL rows first, then split — wait, that leaks.
  // Correct: split first on raw rows, fit on train only.
  const rng = new RNG(cfg.seed);
  const indexed = rows.map((_, i) => i);
  const order = cfg.shuffle ? rng.shuffle(indexed) : indexed;
  const n = rows.length;
  const nTrain = Math.max(1, Math.floor(n * cfg.trainFrac));
  const nVal = Math.max(0, Math.floor(n * cfg.valFrac));
  const trainIdx = order.slice(0, nTrain);
  const valIdx = order.slice(nTrain, nTrain + nVal);
  const testIdx = order.slice(nTrain + nVal);

  const trainRows = trainIdx.map((i) => rows[i]!);
  const catMaps: { col: number; values: string[] }[] = [];
  for (const c of usedFeat) {
    if (c.numeric) continue;
    const set = new Set<string>();
    for (const r of trainRows) {
      const v = (r[c.i] ?? "").trim();
      if (v !== "") set.add(v);
    }
    catMaps.push({ col: c.i, values: [...set].sort() });
  }

  function encodeFeatures(subset: string[][]): { X: number[][]; dropped: number; filled: number } {
    const X: number[][] = [];
    let dropped = 0;
    let filled = 0;
    for (const r of subset) {
      const vec: number[] = [];
      let drop = false;
      for (const c of usedFeat) {
        const raw = (r[c.i] ?? "").trim();
        if (c.numeric) {
          if (raw === "") {
            if (cfg.missing === "drop") {
              drop = true;
              break;
            }
            vec.push(Number.NaN);
            filled++;
          } else {
            const v = Number(raw);
            if (!Number.isFinite(v)) {
              if (cfg.missing === "drop") {
                drop = true;
                break;
              }
              vec.push(Number.NaN);
              filled++;
            } else vec.push(v);
          }
        } else if (cfg.categorical === "label") {
          const map = catMaps.find((m) => m.col === c.i)!;
          vec.push(raw === "" ? 0 : Math.max(0, map.values.indexOf(raw)));
        } else {
          const map = catMaps.find((m) => m.col === c.i)!;
          const oh = new Array(Math.max(1, map.values.length)).fill(0);
          const idx = map.values.indexOf(raw);
          if (idx >= 0) oh[idx] = 1;
          vec.push(...oh);
        }
      }
      if (drop) {
        dropped++;
        continue;
      }
      X.push(vec);
    }
    return { X, dropped, filled };
  }

  function encodeTarget(subset: string[][], keepMask: boolean[]): { Y: number[][]; names: string[]; task: TaskType; loss: LossName; classNames?: string[] } {
    const t0 = tgtCols[0]!;
    if (tgtCols.length === 1 && !t0.numeric) {
      const values = [...new Set(trainRows.map((r) => (r[t0.i] ?? "").trim()).filter(Boolean))].sort();
      const Y: number[][] = [];
      let k = 0;
      for (let i = 0; i < subset.length; i++) {
        if (!keepMask[i]) continue;
        const raw = (subset[i]![t0.i] ?? "").trim();
        const idx = Math.max(0, values.indexOf(raw));
        if (values.length <= 2) Y.push([idx === values.length - 1 ? 1 : 0]);
        else {
          const oh = new Array(values.length).fill(0);
          oh[idx] = 1;
          Y.push(oh);
        }
        k++;
      }
      void k;
      if (values.length <= 2) {
        return { Y, names: [t0.name], task: "binary", loss: "bce", classNames: values };
      }
      return { Y, names: values, task: "multiclass", loss: "cce", classNames: values };
    }
    const Y: number[][] = [];
    for (let i = 0; i < subset.length; i++) {
      if (!keepMask[i]) continue;
      Y.push(tgtCols.map((c) => {
        const v = Number((subset[i]![c.i] ?? "").trim());
        return Number.isFinite(v) ? v : 0;
      }));
    }
    const multi = tgtCols.length === 1 && Y.some((row) => row[0] !== 0 && row[0] !== 1);
    if (tgtCols.length === 1 && !multi) {
      const uniq = new Set(Y.map((r) => r[0]));
      if ([...uniq].every((v) => v === 0 || v === 1)) {
        return { Y, names: [t0.name], task: "binary", loss: "bce", classNames: ["0", "1"] };
      }
    }
    return { Y, names: tgtCols.map((c) => c.name), task: "regression", loss: "mse" };
  }

  const trainEnc = encodeFeatures(trainRows);
  // Fit numeric stats on train (after encoding, before filling NaN)
  const featDim = trainEnc.X[0]?.length ?? 0;
  const means = new Array(featDim).fill(0);
  const counts = new Array(featDim).fill(0);
  const mins = new Array(featDim).fill(Infinity);
  const maxs = new Array(featDim).fill(-Infinity);
  for (const row of trainEnc.X) {
    for (let j = 0; j < featDim; j++) {
      const v = row[j]!;
      if (Number.isFinite(v)) {
        means[j] += v;
        counts[j]++;
        if (v < mins[j]!) mins[j] = v;
        if (v > maxs[j]!) maxs[j] = v;
      }
    }
  }
  for (let j = 0; j < featDim; j++) means[j] = counts[j] ? means[j]! / counts[j]! : 0;
  const vars = new Array(featDim).fill(0);
  for (const row of trainEnc.X) {
    for (let j = 0; j < featDim; j++) {
      const v = row[j]!;
      if (Number.isFinite(v)) vars[j] += (v - means[j]!) * (v - means[j]!);
    }
  }
  const stds = vars.map((v, j) => {
    const c = counts[j]! || 1;
    return Math.sqrt(v / c) || 1;
  });

  function fillAndNorm(X: number[][]): number[][] {
    return X.map((row) =>
      row.map((v, j) => {
        let x = Number.isFinite(v) ? v : cfg.missing === "zero" ? 0 : means[j]!;
        if (cfg.normalize === "minmax") {
          const d = maxs[j]! - mins[j]! || 1;
          x = (x - mins[j]!) / d;
        } else if (cfg.normalize === "zscore") {
          x = (x - means[j]!) / (stds[j] || 1);
        }
        return x;
      }),
    );
  }

  function keepMaskFor(subset: string[][], enc: { X: number[][]; dropped: number }): boolean[] {
    // encodeFeatures already dropped rows; we need a parallel mask of original subset length.
    // Re-run a lightweight drop check.
    const mask: boolean[] = [];
    let k = 0;
    for (const r of subset) {
      let drop = false;
      if (cfg.missing === "drop") {
        for (const c of usedFeat) {
          if (!c.numeric) continue;
          const raw = (r[c.i] ?? "").trim();
          if (raw === "" || !Number.isFinite(Number(raw))) {
            drop = true;
            break;
          }
        }
      }
      mask.push(!drop);
      if (!drop) k++;
    }
    if (k !== enc.X.length) {
      // fallback: all true matching enc length
      return enc.X.map(() => true);
    }
    return mask;
  }

  const valRows = valIdx.map((i) => rows[i]!);
  const testRows = testIdx.map((i) => rows[i]!);
  const valEnc = encodeFeatures(valRows);
  const testEnc = encodeFeatures(testRows);

  const trainX = fillAndNorm(trainEnc.X);
  const valX = fillAndNorm(valEnc.X);
  const testX = fillAndNorm(testEnc.X);

  const trainMask = keepMaskFor(trainRows, trainEnc);
  const valMask = keepMaskFor(valRows, valEnc);
  const testMask = keepMaskFor(testRows, testEnc);
  const yTrain = encodeTarget(trainRows, trainMask);
  const yVal = encodeTarget(valRows, valMask);
  const yTest = encodeTarget(testRows, testMask);

  const featNames: string[] = [];
  for (const c of usedFeat) {
    if (c.numeric || cfg.categorical === "label") featNames.push(c.name);
    else {
      const map = catMaps.find((m) => m.col === c.i)!;
      if (map.values.length === 0) featNames.push(c.name);
      else for (const v of map.values) featNames.push(`${c.name}=${v}`);
    }
  }

  const task = yTrain.task;
  const loss = yTrain.loss;
  const outUnits = yTrain.Y[0]?.length ?? 1;

  // Record the exact transformation so exported models can encode raw rows
  // identically. Non-finite stats (e.g. an all-missing column) become 0 so the
  // spec stays valid JSON; that matches the fill behaviour for such a column.
  const finite = (arr: number[]) => arr.map((v) => (Number.isFinite(v) ? v : 0));
  const columns: TabularColumn[] = usedFeat.map((c) => {
    if (c.numeric) return { name: c.name, type: "numeric" as const };
    const map = catMaps.find((m) => m.col === c.i)!;
    return { name: c.name, type: "categorical" as const, encoding: cfg.categorical === "label" ? ("label" as const) : ("onehot" as const), categories: map.values.slice() };
  });
  const samples: TabularPreprocessing["samples"] = [];
  {
    let k = 0;
    for (let i = 0; i < trainRows.length && samples.length < 6; i++) {
      if (!trainMask[i]) continue;
      const raw: Record<string, string> = {};
      headers.forEach((h, j) => {
        raw[h] = trainRows[i]![j] ?? "";
      });
      if (trainX[k]) samples.push({ raw, encoded: trainX[k]!.slice() });
      k++;
    }
  }
  const preprocessing: TabularPreprocessing = {
    kind: "tabular",
    columns,
    featureNames: featNames.slice(),
    normalize: cfg.normalize,
    mean: finite(means),
    std: finite(stds),
    min: finite(mins),
    max: finite(maxs),
    missing: cfg.missing,
    samples,
  };

  const meta: DatasetMeta = {
    id: `csv_${Date.now()}`,
    name: "Imported CSV",
    description: `Imported table with ${featNames.length} features and ${nTrain} training rows. Preprocessing statistics were fit on the training split only.`,
    source: "csv",
    task,
    featureNames: featNames,
    targetNames: yTrain.names,
    classNames: yTrain.classNames,
    recommendedLoss: loss,
    inputShape: [featNames.length],
    outputUnits: outUnits,
    preprocessing,
  };

  const dataset = pack(meta, {
    train: [trainX, alignY(yTrain.Y, trainX.length)],
    val: [valX.length ? valX : trainX.slice(0, 1), valX.length ? alignY(yVal.Y, valX.length) : alignY(yTrain.Y, 1)],
    test: [testX.length ? testX : trainX.slice(0, 1), testX.length ? alignY(yTest.Y, testX.length) : alignY(yTrain.Y, 1)],
  });
  dataset.xMin = mins;
  dataset.xMax = maxs;
  dataset.xMean = means;
  dataset.xStd = stds;
  dataset.rawPreview = { headers, rows: rows.slice(0, 12) };

  const fit: PreprocessFit = {
    featureNames: featNames,
    targetNames: yTrain.names,
    numericMeans: means,
    numericStds: stds,
    numericMins: mins,
    numericMaxs: maxs,
    catMaps,
    droppedRows: trainEnc.dropped + valEnc.dropped + testEnc.dropped,
    missingFilled: trainEnc.filled + valEnc.filled + testEnc.filled,
    normalize: cfg.normalize,
  };
  if (fit.droppedRows) warnings.push(`Dropped ${fit.droppedRows} row(s) with missing values.`);
  if (fit.missingFilled) warnings.push(`Filled ${fit.missingFilled} missing numeric value(s) using training-set statistics.`);
  warnings.push("Normalization / encoding statistics were computed on the training split and applied to validation and test.");

  return { dataset, fit, warnings };
}

function alignY(Y: number[][], n: number): number[][] {
  if (Y.length === n) return Y;
  if (Y.length === 0) return Array.from({ length: n }, () => [0]);
  const out = Y.slice(0, n);
  while (out.length < n) out.push(Y[Y.length - 1]!.slice());
  return out;
}

export function applyFitToVector(vec: number[], fit: PreprocessFit): number[] {
  return vec.map((v, j) => {
    if (fit.normalize === "minmax") {
      const d = fit.numericMaxs[j]! - fit.numericMins[j]! || 1;
      return (v - fit.numericMins[j]!) / d;
    }
    if (fit.normalize === "zscore") {
      return (v - fit.numericMeans[j]!) / (fit.numericStds[j] || 1);
    }
    return v;
  });
}


