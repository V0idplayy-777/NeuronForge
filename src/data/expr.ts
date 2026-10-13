/**
 * Safe arithmetic expression compiler. Recursive descent, produces a closure.
 * Supports: numbers, + - * / % ^, parentheses, variables (x, y, …),
 * functions sin cos tan asin acos atan sinh cosh tanh exp log log2 log10 sqrt abs
 * floor ceil round sign min max pow, constants pi and e.
 * No JavaScript is ever eval()'d — input is parsed into an AST of closures.
 */

export type ExprFn = (vars: Record<string, number>) => number;

const FUNCS: Record<string, (...a: number[]) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  exp: Math.exp,
  log: Math.log,
  log2: Math.log2,
  log10: Math.log10,
  sqrt: Math.sqrt,
  abs: Math.abs,
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  sign: Math.sign,
  min: Math.min,
  max: Math.max,
  pow: Math.pow,
};

const CONSTS: Record<string, number> = { pi: Math.PI, e: Math.E };

interface Tok {
  kind: "num" | "ident" | "op" | "lparen" | "rparen" | "comma";
  value: string;
}

function tokenize(src: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (/[0-9.]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9.eE]/.test(src[j]!)) {
        if ((src[j] === "e" || src[j] === "E") && (src[j + 1] === "+" || src[j + 1] === "-")) j++;
        j++;
      }
      const raw = src.slice(i, j);
      if (Number.isNaN(Number(raw))) throw new Error(`Invalid number "${raw}"`);
      toks.push({ kind: "num", value: raw });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[a-zA-Z0-9_]/.test(src[j]!)) j++;
      toks.push({ kind: "ident", value: src.slice(i, j) });
      i = j;
      continue;
    }
    if ("+-*/%^".includes(c)) {
      toks.push({ kind: "op", value: c });
      i++;
      continue;
    }
    if (c === "(") {
      toks.push({ kind: "lparen", value: c });
      i++;
      continue;
    }
    if (c === ")") {
      toks.push({ kind: "rparen", value: c });
      i++;
      continue;
    }
    if (c === ",") {
      toks.push({ kind: "comma", value: c });
      i++;
      continue;
    }
    throw new Error(`Unexpected character "${c}" at position ${i}`);
  }
  return toks;
}

export function compileExpr(src: string): { fn: ExprFn; vars: string[] } {
  const toks = tokenize(src);
  if (!toks.length) throw new Error("Empty expression");
  let pos = 0;
  const vars = new Set<string>();

  const peek = () => toks[pos];
  const next = () => toks[pos++];

  function parseExpr(): ExprFn {
    let left = parseTerm();
    while (peek()?.kind === "op" && (peek()!.value === "+" || peek()!.value === "-")) {
      const op = next()!.value;
      const right = parseTerm();
      const l = left;
      left = op === "+" ? (v) => l(v) + right(v) : (v) => l(v) - right(v);
    }
    return left;
  }

  function parseTerm(): ExprFn {
    let left = parseFactor();
    while (peek()?.kind === "op" && ["*", "/", "%"].includes(peek()!.value)) {
      const op = next()!.value;
      const right = parseFactor();
      const l = left;
      if (op === "*") left = (v) => l(v) * right(v);
      else if (op === "/") left = (v) => l(v) / right(v);
      else left = (v) => l(v) % right(v);
    }
    return left;
  }

  function parseFactor(): ExprFn {
    const base = parseUnary();
    if (peek()?.kind === "op" && peek()!.value === "^") {
      next();
      const exp = parseFactor(); // right-assoc
      return (v) => Math.pow(base(v), exp(v));
    }
    return base;
  }

  function parseUnary(): ExprFn {
    if (peek()?.kind === "op" && (peek()!.value === "-" || peek()!.value === "+")) {
      const op = next()!.value;
      const inner = parseUnary();
      return op === "-" ? (v) => -inner(v) : inner;
    }
    return parsePrimary();
  }

  function parsePrimary(): ExprFn {
    const t = next();
    if (!t) throw new Error("Unexpected end of expression");
    if (t.kind === "num") {
      const n = Number(t.value);
      return () => n;
    }
    if (t.kind === "lparen") {
      const inner = parseExpr();
      if (next()?.kind !== "rparen") throw new Error("Missing closing parenthesis");
      return inner;
    }
    if (t.kind === "ident") {
      const name = t.value.toLowerCase();
      if (peek()?.kind === "lparen") {
        next();
        const fn = FUNCS[name];
        if (!fn) throw new Error(`Unknown function "${t.value}"`);
        const args: ExprFn[] = [];
        if (peek()?.kind !== "rparen") {
          args.push(parseExpr());
          while (peek()?.kind === "comma") {
            next();
            args.push(parseExpr());
          }
        }
        if (next()?.kind !== "rparen") throw new Error(`Missing ) after ${t.value}(…`);
        return (v) => fn(...args.map((a) => a(v)));
      }
      if (name in CONSTS) {
        const c = CONSTS[name]!;
        return () => c;
      }
      vars.add(name);
      return (v) => {
        const val = v[name];
        if (val === undefined) throw new Error(`Variable "${name}" is not defined`);
        return val;
      };
    }
    throw new Error(`Unexpected token "${t.value}"`);
  }

  const fn = parseExpr();
  if (pos !== toks.length) throw new Error(`Unexpected trailing input "${toks[pos]?.value}"`);
  // Smoke-test with all vars = 0.5 so syntax errors surface immediately.
  const probe: Record<string, number> = {};
  for (const v of vars) probe[v] = 0.5;
  const test = fn(probe);
  if (typeof test !== "number") throw new Error("Expression did not evaluate to a number");
  return { fn, vars: [...vars] };
}
