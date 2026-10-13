/** RFC-style CSV parser that handles quoted fields, escaped quotes, and mixed newlines. */
export function parseCsv(text: string): { headers: string[]; rows: string[][] } {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let i = 0;
  let inQuotes = false;
  const s = text.replace(/^\uFEFF/, "");
  while (i < s.length) {
    const ch = s[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (s[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      field += ch;
      i++;
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (ch === ",") {
      row.push(field);
      field = "";
      i++;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && s[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.some((c) => c.trim() !== "") || rows.length === 0) rows.push(row);
      row = [];
      i++;
      continue;
    }
    field += ch;
    i++;
  }
  row.push(field);
  if (row.some((c) => c.trim() !== "")) rows.push(row);

  if (rows.length === 0) return { headers: [], rows: [] };

  const width = Math.max(...rows.map((r) => r.length));
  const normalized = rows.map((r) => {
    const copy = r.slice();
    while (copy.length < width) copy.push("");
    return copy;
  });
  const headers = normalized[0]!.map((h, idx) => h.trim() || `col_${idx + 1}`);
  return { headers, rows: normalized.slice(1) };
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
