/**
 * Image dataset import: decode files in the browser, resize them to the model's
 * input size, optionally augment the training split, and pack the result into the
 * same PreparedDataset structure every other dataset uses.
 *
 * The pixel maths (resize, augment, assembly) is deliberately DOM-free so it can
 * be exercised headlessly; only `readImageFiles` touches the browser APIs.
 */
import { RNG } from "../engine/rng";
import { Tensor } from "../engine/tensor";
import { pack, type DatasetMeta, type PreparedDataset } from "./datasets";

export const IMAGE_LIMITS = {
  /** Hard cap on files accepted in one import. */
  maxFiles: 4000,
  /** Refuse to rasterise images larger than this (width × height). */
  maxPixelsPerImage: 4_000_000,
  /** Cap on augmented training rows, to keep training time sane. */
  maxTrainRows: 60_000,
  /** Largest square/rectangular input the importer will target. */
  maxSide: 128,
};

export type ResampleMode = "bilinear" | "nearest";
export type LabelMode = "prefix" | "folder" | "all";

export interface ImageImportConfig {
  width: number;
  height: number;
  channels: 1 | 3;
  /** Bilinear looks better; nearest matches the resampling the exported runtimes use. */
  resample: ResampleMode;
  /** Value range the model will see. */
  scale: "01" | "pm1";
  labelMode: LabelMode;
  /** Number of augmented copies added to the training split (0 = none). */
  copies: number;
  flipH: boolean;
  flipV: boolean;
  rotate90: boolean;
  /** Maximum random rotation in degrees, applied to augmented copies. */
  rotate: number;
  /** Maximum translation in pixels, applied to augmented copies. */
  shift: number;
  brightness: number;
  contrast: number;
  noise: number;
  trainFrac: number;
  valFrac: number;
  testFrac: number;
  seed: number;
}

export const DEFAULT_IMAGE_CONFIG: ImageImportConfig = {
  width: 32,
  height: 32,
  channels: 1,
  resample: "bilinear",
  scale: "01",
  labelMode: "prefix",
  copies: 3,
  flipH: true,
  flipV: false,
  rotate90: true,
  rotate: 12,
  shift: 2,
  brightness: 0.15,
  contrast: 0.15,
  noise: 0.03,
  trainFrac: 0.7,
  valFrac: 0.15,
  testFrac: 0.15,
  seed: 7,
};

/** One decoded file, still at its natural size (channels = 3 RGB for our purposes). */
export interface ImageSource {
  name: string;
  folder?: string;
  width: number;
  height: number;
  channels: 3 | 4;
  /** Row-major HWC floats in [0, 1]. */
  pixels: Float32Array;
}

/* ------------------------------------------------------------------- pixels */

/** Inclusive integer range (engine RNG only exposes int(max)). */
function randInt(rng: RNG, lo: number, hi: number): number {
  return lo + rng.int(Math.max(1, hi - lo + 1));
}

export function toGrayscale(pixels: Float32Array, channels: number): Float32Array {
  const n = pixels.length / channels;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = pixels[i * channels]!;
    const g = channels >= 3 ? pixels[i * channels + 1]! : r;
    const b = channels >= 3 ? pixels[i * channels + 2]! : r;
    // Rec. 601 luma, same weights PIL/OpenCV use by default.
    out[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return out;
}

/** Bilinear or nearest resample of a HWC float image. */
export function resizePixels(
  src: Float32Array,
  sw: number,
  sh: number,
  channels: number,
  dw: number,
  dh: number,
  mode: ResampleMode = "bilinear",
): Float32Array {
  const out = new Float32Array(dw * dh * channels);
  if (sw === dw && sh === dh) return src.slice();
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      if (mode === "nearest") {
        const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
        const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
        for (let c = 0; c < channels; c++) out[(y * dw + x) * channels + c] = src[(sy * sw + sx) * channels + c]!;
        continue;
      }
      // Align pixel centres, then sample the four nearest source pixels.
      const fx = ((x + 0.5) * sw) / dw - 0.5;
      const fy = ((y + 0.5) * sh) / dh - 0.5;
      const x0 = Math.max(0, Math.min(sw - 1, Math.floor(fx)));
      const y0 = Math.max(0, Math.min(sh - 1, Math.floor(fy)));
      const x1 = Math.max(0, Math.min(sw - 1, x0 + 1));
      const y1 = Math.max(0, Math.min(sh - 1, y0 + 1));
      const tx = Math.max(0, Math.min(1, fx - x0));
      const ty = Math.max(0, Math.min(1, fy - y0));
      for (let c = 0; c < channels; c++) {
        const p00 = src[(y0 * sw + x0) * channels + c]!;
        const p10 = src[(y0 * sw + x1) * channels + c]!;
        const p01 = src[(y1 * sw + x0) * channels + c]!;
        const p11 = src[(y1 * sw + x1) * channels + c]!;
        const top = p00 + (p10 - p00) * tx;
        const bot = p01 + (p11 - p01) * tx;
        out[(y * dw + x) * channels + c] = top + (bot - top) * ty;
      }
    }
  }
  return out;
}

function rotatePixels(src: Float32Array, w: number, h: number, ch: number, degrees: number): Float32Array {
  const out = new Float32Array(src.length);
  const rad = (degrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const sx = cos * dx + sin * dy + cx;
      const sy = -sin * dx + cos * dy + cy;
      if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) continue; // black fill
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      const tx = sx - x0;
      const ty = sy - y0;
      for (let c = 0; c < ch; c++) {
        const p00 = src[(y0 * w + x0) * ch + c]!;
        const p10 = src[(y0 * w + x1) * ch + c]!;
        const p01 = src[(y1 * w + x0) * ch + c]!;
        const p11 = src[(y1 * w + x1) * ch + c]!;
        const top = p00 + (p10 - p00) * tx;
        const bot = p01 + (p11 - p01) * tx;
        out[(y * w + x) * ch + c] = top + (bot - top) * ty;
      }
    }
  }
  return out;
}

function shiftPixels(src: Float32Array, w: number, h: number, ch: number, dx: number, dy: number): Float32Array {
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const sy = y - dy;
    if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < w; x++) {
      const sx = x - dx;
      if (sx < 0 || sx >= w) continue;
      for (let c = 0; c < ch; c++) out[(y * w + x) * ch + c] = src[(sy * w + sx) * ch + c]!;
    }
  }
  return out;
}

export interface AugmentOps {
  flipH?: boolean;
  flipV?: boolean;
  rotate90?: boolean;
  rotate?: number;
  shift?: number;
  brightness?: number;
  contrast?: number;
  noise?: number;
}

/**
 * One random augmentation pass. Deterministic for a given RNG so a dataset built
 * from the same files, config and seed is bit-identical every time.
 */
export function augmentPixels(src: Float32Array, w: number, h: number, ch: number, ops: AugmentOps, rng: RNG): Float32Array {
  let px: Float32Array = src.slice();
  if (ops.flipH && rng.uniform(0, 1) < 0.5) {
    const next = new Float32Array(px.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) next[(y * w + x) * ch + c] = px[(y * w + (w - 1 - x)) * ch + c]!;
    px = next;
  }
  if (ops.flipV && rng.uniform(0, 1) < 0.5) {
    const next = new Float32Array(px.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) next[(y * w + x) * ch + c] = px[((h - 1 - y) * w + x) * ch + c]!;
    px = next;
  }
  const quarters = ops.rotate90 ? randInt(rng, 0, 3) : 0;
  if (quarters === 1) {
    // 90° needs a transpose; keep it exact by rolling the buffer.
    const next = new Float32Array(px.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) next[(x * h + (h - 1 - y)) * ch + c] = px[(y * w + x) * ch + c]!;
    // Only square images keep their shape; skip non-square rotations.
    if (w === h) px = next;
  } else if (quarters === 2) {
    const next = new Float32Array(px.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) next[(y * w + x) * ch + c] = px[((h - 1 - y) * w + (w - 1 - x)) * ch + c]!;
    px = next;
  } else if (quarters === 3 && w === h) {
    const next = new Float32Array(px.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < ch; c++) next[((w - 1 - x) * h + y) * ch + c] = px[(y * w + x) * ch + c]!;
    px = next;
  }
  const deg = ops.rotate ?? 0;
  if (deg > 0) {
    const jitter = rng.uniform(-deg, deg);
    if (Math.abs(jitter) > 0.01) px = rotatePixels(px, w, h, ch, jitter);
  }
  const shift = ops.shift ?? 0;
  if (shift > 0) {
    const dx = randInt(rng, -shift, shift);
    const dy = randInt(rng, -shift, shift);
    if (dx !== 0 || dy !== 0) px = shiftPixels(px, w, h, ch, dx, dy);
  }
  const brightness = ops.brightness ?? 0;
  const contrast = ops.contrast ?? 0;
  const noise = ops.noise ?? 0;
  if (brightness > 0 || contrast > 0 || noise > 0) {
    const b = brightness > 0 ? rng.uniform(1 - brightness, 1 + brightness) : 1;
    const cGain = contrast > 0 ? rng.uniform(1 - contrast, 1 + contrast) : 1;
    for (let i = 0; i < px.length; i++) {
      let v = px[i]! * b;
      if (cGain !== 1) v = (v - 0.5) * cGain + 0.5;
      if (noise > 0) v += rng.normal() * noise;
      px[i] = Math.max(0, Math.min(1, v));
    }
  }
  return px;
}

/* ------------------------------------------------------------------- labels */

export function labelFromName(name: string, mode: LabelMode, folder?: string): string {
  if (mode === "all") return "all";
  if (mode === "folder" && folder && folder.length > 0) return folder;
  const base = name.replace(/\.[^.\\/]+$/, "").replace(/^.*[/\\]/, "");
  const cut = base.split(/[_\-. ]/)[0];
  return (cut && cut.length > 0 ? cut : base).toLowerCase();
}

/* ---------------------------------------------------------------- assembly */

export interface ImageRow {
  pixels: Float32Array;
  label: string;
}

export function buildImageDataset(rows: ImageRow[], cfg: ImageImportConfig): PreparedDataset {
  if (rows.length < 4) throw new Error(`Need at least 4 images to build a dataset (have ${rows.length}).`);
  const w = Math.max(4, Math.min(IMAGE_LIMITS.maxSide, Math.round(cfg.width)));
  const h = Math.max(4, Math.min(IMAGE_LIMITS.maxSide, Math.round(cfg.height)));
  const ch = cfg.channels;
  const labels = [...new Set(rows.map((r) => r.label))].sort();
  const multi = labels.length > 2;
  const single = labels.length === 1;
  const labelIndex = new Map(labels.map((l, i) => [l, i]));
  const expected = w * h * ch;
  for (const row of rows) {
    if (row.pixels.length !== expected) {
      throw new Error(`Image rows must already be resized to ${w}×${h}×${ch} (expected ${expected} values, got ${row.pixels.length}).`);
    }
  }
  const rng = new RNG(cfg.seed + 17);

  // Deterministic shuffle, then split — the same split for any augmentation setting.
  const order = rng.shuffle(rows.map((_, i) => i));
  const nTrain = Math.max(1, Math.floor(rows.length * cfg.trainFrac));
  const nVal = Math.max(0, Math.floor(rows.length * cfg.valFrac));

  const toVec = (px: Float32Array): number[] => {
    const out = new Array(px.length);
    for (let i = 0; i < px.length; i++) {
      const v = px[i]!;
      out[i] = cfg.scale === "pm1" ? v * 2 - 1 : v;
    }
    return out;
  };
  const toTarget = (label: string): number[] => {
    if (single) return toVec(rows[0]!.pixels); // autoencoder: target = input
    if (multi) {
      const oh = new Array(labels.length).fill(0);
      oh[labelIndex.get(label)!] = 1;
      return oh;
    }
    return [labelIndex.get(label)!];
  };

  const trainX: number[][] = [];
  const trainY: number[][] = [];
  const valX: number[][] = [];
  const valY: number[][] = [];
  const testX: number[][] = [];
  const testY: number[][] = [];

  const copies = Math.max(0, Math.min(32, Math.round(cfg.copies)));
  const ops: AugmentOps = {
    flipH: cfg.flipH,
    flipV: cfg.flipV,
    rotate90: cfg.rotate90,
    rotate: Math.max(0, cfg.rotate),
    shift: Math.max(0, Math.round(cfg.shift)),
    brightness: Math.max(0, cfg.brightness),
    contrast: Math.max(0, cfg.contrast),
    noise: Math.max(0, cfg.noise),
  };

  for (let i = 0; i < rows.length; i++) {
    const j = order[i]!;
    const row = rows[j]!;
    const x = toVec(row.pixels);
    const y = toTarget(row.label);
    if (i < nTrain) {
      trainX.push(x);
      trainY.push(y);
      for (let c = 0; c < copies; c++) {
        const aug = augmentPixels(row.pixels, w, h, ch, ops, rng);
        trainX.push(toVec(aug));
        trainY.push(toTarget(row.label));
      }
    } else if (i < nTrain + nVal) {
      valX.push(x);
      valY.push(y);
    } else {
      testX.push(x);
      testY.push(y);
    }
  }

  const trim = (xs: number[][], ys: number[][]): [number[][], number[][]] => {
    if (xs.length <= IMAGE_LIMITS.maxTrainRows) return [xs, ys];
    return [xs.slice(0, IMAGE_LIMITS.maxTrainRows), ys.slice(0, IMAGE_LIMITS.maxTrainRows)];
  };
  const [trX, trY] = trim(trainX, trainY);
  if (valX.length === 0) {
    valX.push(trX[0]!.slice());
    valY.push(single ? trY[0]!.slice() : trY[0]!.slice());
  }
  if (testX.length === 0) {
    testX.push(valX[0]!.slice());
    testY.push(valY[0]!.slice());
  }

  const classNames = single ? undefined : labels;
  const meta: DatasetMeta = {
    id: `custom_images_${cfg.seed}_${rows.length}_${w}x${h}x${ch}`,
    name: `Images ${w}×${h}${ch === 1 ? " gray" : " RGB"}`,
    description: single
      ? `${rows.length} imported images resized to ${w}×${h}${ch === 1 ? " grayscale" : " RGB"}, ${cfg.scale === "pm1" ? "scaled to [-1, 1]" : "scaled to [0, 1]"}. Targets are the inputs, so this trains an autoencoder / reconstruction model.`
      : `${rows.length} imported images across ${labels.length} classes, resized to ${w}×${h}${ch === 1 ? " grayscale" : " RGB"}, ${cfg.scale === "pm1" ? "scaled to [-1, 1]" : "scaled to [0, 1]"}.${copies > 0 ? ` The training split adds ${copies} augmented copies per image (flips, rotations, shifts, brightness/contrast, noise); validation and test stay clean.` : ""}`,
    source: "custom",
    task: single ? "regression" : multi ? "multiclass" : "binary",
    featureNames: Array.from({ length: w * h * ch }, (_, i) => `px${i}`),
    targetNames: single ? Array.from({ length: w * h * ch }, (_, i) => `px${i}`) : multi ? labels : ["label"],
    classNames,
    recommendedLoss: single ? "mse" : multi ? "cce" : "bce",
    inputShape: [h, w, ch],
    outputUnits: single ? w * h * ch : multi ? labels.length : 1,
    reconstruction: single ? true : undefined,
    image: {
      height: h,
      width: w,
      channels: ch,
      layout: "HWC",
      range: cfg.scale === "pm1" ? [-1, 1] : [0, 1],
      description: `${w}×${h}×${ch} imported from ${rows.length} files, row-major HWC${cfg.resample === "nearest" ? ", nearest-neighbour resize" : ", bilinear resize"}.`,
    },
    attribution: "Imported locally in the browser — files never leave the page.",
  };
  return pack(meta, {
    train: [trX, trY],
    val: [valX, valY],
    test: [testX, testY],
  });
}

/* ------------------------------------------------------------------ decoding */

function rasterize(pixels: Float32Array, width: number, height: number, channels: number, cfg: ImageImportConfig): Float32Array {
  const w = Math.max(4, Math.min(IMAGE_LIMITS.maxSide, Math.round(cfg.width)));
  const h = Math.max(4, Math.min(IMAGE_LIMITS.maxSide, Math.round(cfg.height)));
  const resized = resizePixels(pixels, width, height, channels, w, h, cfg.resample);
  return cfg.channels === 1 ? toGrayscale(resized, channels) : (channels === 3 ? resized : toGrayscale(resized, channels));
}

async function decodeFile(file: File): Promise<{ pixels: Float32Array; width: number; height: number; channels: 3 | 4 }> {
  const hasBitmap = typeof createImageBitmap === "function";
  const source: ImageBitmap | HTMLImageElement = hasBitmap
    ? await createImageBitmap(file)
    : await new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Could not decode ${file.name}`));
        img.src = URL.createObjectURL(file);
      });
  const width = source.width;
  const height = source.height;
  if (width * height > IMAGE_LIMITS.maxPixelsPerImage) {
    throw new Error(`${file.name} is ${width}×${height} — larger than the ${IMAGE_LIMITS.maxPixelsPerImage.toLocaleString()} pixel import limit.`);
  }
  const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(width, height) : document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error("Could not create a 2D canvas context");
  ctx.drawImage(source as CanvasImageSource, 0, 0);
  const data = ctx.getImageData(0, 0, width, height).data;
  const pixels = new Float32Array(width * height * 4);
  for (let i = 0; i < pixels.length; i++) pixels[i] = data[i]! / 255;
  if (hasBitmap) (source as ImageBitmap).close?.();
  return { pixels, width, height, channels: 4 };
}

/**
 * Decode, resize and channel-convert a batch of files. Yields to the event loop
 * between images so the UI can show progress.
 */
export async function readImageFiles(
  files: File[],
  cfg: ImageImportConfig,
  onProgress?: (done: number, total: number, name: string) => void,
): Promise<ImageRow[]> {
  const list = files.slice(0, IMAGE_LIMITS.maxFiles).sort((a, b) => a.name.localeCompare(b.name));
  const rows: ImageRow[] = [];
  for (let i = 0; i < list.length; i++) {
    const file = list[i]!;
    onProgress?.(i, list.length, file.name);
    const decoded = await decodeFile(file);
    const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    const folder = relative.includes("/") ? relative.split("/").slice(-2, -1)[0] : undefined;
    rows.push({ pixels: rasterize(decoded.pixels, decoded.width, decoded.height, decoded.channels, cfg), label: labelFromName(file.name, cfg.labelMode, folder) });
    if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
  }
  onProgress?.(list.length, list.length, "");
  return rows;
}

/** Turn rows into the tensors `buildImageDataset` needs (exposed for tests). */
export function rowsToTensors(rows: ImageRow[], shape: number[]): { xs: Tensor; count: number } {
  const n = rows.length;
  const feat = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n * feat);
  for (let i = 0; i < n; i++) data.set(rows[i]!.pixels, i * feat);
  return { xs: new Tensor([n, ...shape], data), count: n };
}
