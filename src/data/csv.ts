/**
 * RFC-style CSV handling: quoted fields, escaped quotes, mixed newlines.
 *
 * `parseCsv` handles strings (tests, pasted data). `parseCsvFile` streams a File
 * in chunks so large exports never have to be held in memory twice, reports
 * progress, and stops cleanly at the row cap.
 */

export interface ParsedCsv {
  headers: string[];
  rows: string[][];
}

/** Rows kept in memory by default (a 250k-row table of 20 columns ≈ 5M cells). */
export const CSV_MAX_ROWS = 250_000;
export const CSV_MAX_COLUMNS = 1024;

/**
 * Incremental parser. Feed it decoded text chunks in order, then call finish().
 * The chunk boundaries may fall anywhere — including inside a quoted field or
 * between a carriage return and its line feed.
 */
export class CsvParser {
  private rows: string[][] = [];
  private row: string[] = [];
  private field = "";
  private inQuotes = false;
  /** A trailing "\r" that may be the first half of a "\r\n" pair. */
  private pendingCR = false;
  /** A trailing '"' inside a quoted field: closing quote or first half of '"'? */
  private pendingQuote = false;
  private atStart = true;
  private capped = false;

  constructor(private maxRows = CSV_MAX_ROWS) {}

  /** Number of data rows accepted so far (header excluded). */
  get dataRows(): number {
    return Math.max(0, this.rows.length - 1);
  }

  /** True once the cap was hit; further input is ignored. */
  get truncated(): boolean {
    return this.capped;
  }

  push(text: string): void {
    let s = text;
    if (this.atStart) {
      s = s.replace(/^\uFEFF/, "");
      this.atStart = false;
    }
    let i = 0;
    if (this.pendingCR) {
      this.pendingCR = false;
      if (s[0] === "\n") i = 1;
    }
    if (this.pendingQuote) {
      this.pendingQuote = false;
      // Mirror the whole-text rule: a quote directly followed by a quote is an
      // escaped quote; anything else means the field closed at the boundary.
      if (s[i] === '"') {
        this.field += '"';
        i += 1;
      } else {
        this.inQuotes = false;
      }
    }
    while (i < s.length) {
      if (this.rows.length - 1 >= this.maxRows) {
        this.capped = true;
        return; // cap reached: stop scanning, the caller is told via truncated
      }
      const ch = s[i]!;
      if (this.inQuotes) {
        if (ch === '"') {
          if (s[i + 1] === '"') {
            this.field += '"';
            i += 2;
            continue;
          }
          if (i === s.length - 1) {
            // Cannot tell a closing quote from the first half of an escaped pair
            // until the next chunk arrives.
            this.pendingQuote = true;
            return;
          }
          this.inQuotes = false;
          i++;
          continue;
        }
        this.field += ch;
        i++;
        continue;
      }
      if (ch === '"') {
        this.inQuotes = true;
        i++;
        continue;
      }
      if (ch === ",") {
        this.row.push(this.field);
        this.field = "";
        i++;
        continue;
      }
      if (ch === "\n" || ch === "\r") {
        if (ch === "\r") {
          if (s[i + 1] === "\n") i++;
          // A CR that closes the chunk ends the row here; a LF opening the next
          // chunk is then swallowed by the pendingCR handling above.
          else if (i === s.length - 1) this.pendingCR = true;
        }
        this.endRow();
        i++;
        continue;
      }
      this.field += ch;
      i++;
    }
  }

  private endRow(): void {
    this.row.push(this.field);
    this.field = "";
    if (this.row.some((c) => c.trim() !== "") || this.rows.length === 0) this.rows.push(this.row);
    this.row = [];
  }

  finish(): ParsedCsv {
    if (this.pendingQuote) {
      this.pendingQuote = false;
      this.inQuotes = false;
    }
    if (this.field !== "" || this.row.length > 0) {
      this.endRow();
    }
    if (this.rows.length === 0) return { headers: [], rows: [] };

    const width = Math.max(...this.rows.map((r) => r.length));
    const normalized = this.rows.map((r) => {
      const copy = r.slice();
      while (copy.length < width) copy.push("");
      return copy;
    });
    const headers = normalized[0]!.map((h, idx) => h.trim() || `col_${idx + 1}`);
    return { headers, rows: normalized.slice(1) };
  }
}

export function parseCsv(text: string, maxRows = CSV_MAX_ROWS): ParsedCsv {
  const parser = new CsvParser(maxRows);
  parser.push(text);
  return parser.finish();
}

export interface CsvStreamStats {
  bytes: number;
  totalBytes: number;
  rows: number;
  truncated: boolean;
}

export interface FileLike {
  size?: number;
  stream?: () => ReadableStream<Uint8Array>;
  text: () => Promise<string>;
}

/**
 * Stream a file through the incremental parser. Uses the File Streams API when
 * available (Chrome/Edge/Safari/Firefox) and falls back to reading the whole
 * text for older environments or for non-File inputs.
 */
export async function parseCsvFile(
  file: FileLike,
  opts: { maxRows?: number; onProgress?: (stats: CsvStreamStats) => void; signal?: { aborted: boolean } } = {},
): Promise<ParsedCsv & { truncated: boolean; bytes: number }> {
  const maxRows = opts.maxRows ?? CSV_MAX_ROWS;
  const parser = new CsvParser(maxRows);
  const totalBytes = file.size ?? 0;
  let bytes = 0;
  const report = (): void => opts.onProgress?.({ bytes, totalBytes, rows: parser.dataRows, truncated: parser.truncated });

  if (typeof file.stream === "function") {
    const reader = file.stream().getReader();
    const decoder = new TextDecoder("utf-8");
    for (;;) {
      if (opts.signal?.aborted) {
        await reader.cancel();
        break;
      }
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value ? value.byteLength : 0;
      parser.push(decoder.decode(value ?? new Uint8Array(), { stream: true }));
      report();
      if (parser.truncated) {
        await reader.cancel();
        break;
      }
    }
    parser.push(decoder.decode());
  } else {
    const text = await file.text();
    bytes = text.length;
    parser.push(text);
    report();
  }

  if (opts.signal?.aborted) throw new Error("Import cancelled.");
  report();
  const parsed = parser.finish();
  if (parsed.headers.length > CSV_MAX_COLUMNS) {
    throw new Error(`CSV has ${parsed.headers.length} columns — the importer supports up to ${CSV_MAX_COLUMNS}.`);
  }
  return { ...parsed, truncated: parser.truncated, bytes };
}

export function inferNumeric(values: string[]): boolean {
  let nums = 0;
  let nonempty = 0;
  for (const v of values) {
    const t = v.trim();
    if (t === "") continue;
    nonempty++;
    if (!Number.isNaN(Number(t))) nums++;
  }
  return nonempty > 0 && nums / nonempty > 0.8;
}

export interface CsvImportConfig {
  featureIdx: number[];
  targetIdx: number[];
  missing: "drop" | "zero" | "mean";
  categorical: "drop" | "onehot" | "label";
  normalize: "none" | "minmax" | "zscore";
  trainFrac: number;
  valFrac: number;
  testFrac: number;
  seed: number;
  shuffle: boolean;
}

export const DEFAULT_CSV_CONFIG: CsvImportConfig = {
  featureIdx: [],
  targetIdx: [],
  missing: "mean",
  categorical: "onehot",
  normalize: "zscore",
  trainFrac: 0.7,
  valFrac: 0.15,
  testFrac: 0.15,
  seed: 42,
  shuffle: true,
};
