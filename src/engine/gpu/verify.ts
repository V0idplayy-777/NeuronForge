/**
 * Device verification.
 *
 * Every operation the accelerator can issue is run on the candidate device and
 * on the CPU reference device with identical inputs, then diffed. This is the
 * honest part of the accelerated backend: nothing is trusted because it is
 * "supposed to" work — a device has to reproduce the CPU engine's numbers
 * before it is allowed to train a model, and if any check fails the app keeps
 * training on the CPU.
 *
 * On a machine without a GPU the same suite runs against `CpuOps` on both
 * sides, which still exercises the operation surface, the parameter packing and
 * the comparison logic itself.
 */

import { CpuOps } from "./cpu";
import { convGeomFor } from "./ops";
import type { ConvGeom, DevBuffer, GpuOps, PoolGeom, PoolMode } from "./ops";

export interface DeviceCheck {
  name: string;
  passed: boolean;
  maxDiff: number;
  tolerance: number;
  message: string;
}

/** Small deterministic PRNG so both devices see byte-identical inputs. */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rand(rnd: () => number, n: number, scale = 1): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rnd() * 2 - 1) * scale;
  return out;
}

interface DeviceCase {
  name: string;
  build: (ops: GpuOps, rnd: () => number) => Promise<DevBuffer>;
}

function poolGeom(B: number, H: number, W: number, C: number, pool: number, stride: number): PoolGeom {
  return { B, H, W, C, pool, stride, outH: Math.floor((H - pool) / stride) + 1, outW: Math.floor((W - pool) / stride) + 1 };
}

function conv(
  B: number,
  H: number,
  W: number,
  Cin: number,
  F: number,
  kernel: number,
  stride: number,
  padding: "same" | "valid",
): ConvGeom {
  return convGeomFor(B, H, W, Cin, F, { kernelSize: kernel, stride, padding });
}

/** Cases are built from the same generator on both devices, so inputs match exactly. */
function buildCases(): DeviceCase[] {
  const convCases: { name: string; g: ConvGeom }[] = [
    { name: "3×3 same s1", g: conv(2, 6, 6, 3, 4, 3, 1, "same") },
    { name: "3×3 same s2", g: conv(2, 7, 7, 2, 3, 3, 2, "same") },
    { name: "3×3 valid s1", g: conv(2, 7, 7, 2, 3, 3, 1, "valid") },
    { name: "1×1 same s1", g: conv(2, 4, 5, 3, 2, 1, 1, "same") },
    { name: "3×3 same B=1 odd", g: conv(1, 5, 5, 1, 2, 3, 1, "same") },
  ];
  const poolCases: { name: string; g: PoolGeom }[] = [
    { name: "2×2 s2", g: poolGeom(2, 6, 6, 3, 2, 2) },
    { name: "3×3 s2", g: poolGeom(2, 7, 7, 2, 3, 2) },
    { name: "2×2 s1 overlap", g: poolGeom(2, 5, 5, 2, 2, 1) },
  ];

  const cases: DeviceCase[] = [
    {
      name: "matmul [4,6]×[6,5]",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 24), [4, 6]), await ops.fromHost(rand(rnd, 30), [6, 5]), false, false),
    },
    {
      name: "matmul transpose A",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 24), [6, 4]), await ops.fromHost(rand(rnd, 30), [6, 5]), true, false),
    },
    {
      name: "matmul transpose B",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 24), [4, 6]), await ops.fromHost(rand(rnd, 30), [5, 6]), false, true),
    },
    {
      name: "matmul transpose A and B",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 24), [6, 4]), await ops.fromHost(rand(rnd, 30), [5, 6]), true, true),
    },
    {
      name: "matmul K=1",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 5), [5, 1]), await ops.fromHost(rand(rnd, 3), [1, 3]), false, false),
    },
    {
      name: "matmul 17×19×23 (multi-tile)",
      build: async (ops, rnd) =>
        ops.matmul(await ops.fromHost(rand(rnd, 17 * 19), [17, 19]), await ops.fromHost(rand(rnd, 19 * 23), [19, 23]), false, false),
    },
    {
      name: "matmul M=1 row vector",
      build: async (ops, rnd) => ops.matmul(await ops.fromHost(rand(rnd, 7), [1, 7]), await ops.fromHost(rand(rnd, 21), [7, 3]), false, false),
    },
    {
      name: "addBias [5,7]+[7]",
      build: async (ops, rnd) => ops.addBias(await ops.fromHost(rand(rnd, 35), [5, 7]), await ops.fromHost(rand(rnd, 7), [7])),
    },
    {
      name: "addBias [2,3,4,5]+[5]",
      build: async (ops, rnd) => ops.addBias(await ops.fromHost(rand(rnd, 120), [2, 3, 4, 5]), await ops.fromHost(rand(rnd, 5), [5])),
    },
    {
      name: "binary add [3,4]·[4]",
      build: async (ops, rnd) => ops.binary("add", await ops.fromHost(rand(rnd, 12), [3, 4]), await ops.fromHost(rand(rnd, 4), [4])),
    },
    {
      name: "binary mul [3,4]·[3,1]",
      build: async (ops, rnd) => ops.binary("mul", await ops.fromHost(rand(rnd, 12), [3, 4]), await ops.fromHost(rand(rnd, 3), [3, 1])),
    },
    {
      name: "binary sub scalar [5]-[1]",
      build: async (ops, rnd) => ops.binary("sub", await ops.fromHost(rand(rnd, 5), [5]), await ops.fromHost(rand(rnd, 1), [1])),
    },
    {
      name: "binary div [2,3,4]÷[4]",
      build: async (ops, rnd) =>
        ops.binary("div", await ops.fromHost(rand(rnd, 24), [2, 3, 4]), await ops.fromHost(rand(rnd, 4, 0.4).map((v) => v + 1.5), [4])),
    },
    {
      name: "unary neg/abs/square/scale/clip",
      build: async (ops, rnd) => {
        const x = await ops.fromHost(rand(rnd, 9, 2), [3, 3]);
        const a = await ops.unary("neg", x, 0, 0);
        const b = await ops.unary("abs", x, 0, 0);
        const c = await ops.unary("square", x, 0, 0);
        const d = await ops.unary("scale", x, 0.5, 0);
        const e = await ops.unary("clip", x, -0.4, 0.6);
        // Sum them so a single comparison covers every case.
        let acc = await ops.binary("add", a, b);
        acc = await ops.binary("add", acc, c);
        acc = await ops.binary("add", acc, d);
        return ops.binary("add", acc, e);
      },
    },
    {
      name: "activation fwd relu/leakyrelu/gelu/silu/sigmoid/tanh",
      build: async (ops, rnd) => {
        const x = await ops.fromHost(rand(rnd, 16, 1.6), [16]);
        let acc: DevBuffer | null = null;
        for (const name of ["relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh"]) {
          const y = await ops.activationFwd(name, x, 0.1);
          acc = acc ? await ops.binary("add", acc, y) : y;
        }
        return acc!;
      },
    },
    {
      name: "activation fwd softmax [7,9]",
      build: async (ops, rnd) => ops.softmaxLast(await ops.fromHost(rand(rnd, 63, 2), [7, 9])),
    },
    {
      name: "activation fwd softmax [3,130] (multi-lane row)",
      build: async (ops, rnd) => ops.softmaxLast(await ops.fromHost(rand(rnd, 390, 2), [3, 130])),
    },
    {
      name: "activation fwd softmax [4,1]",
      build: async (ops, rnd) => ops.softmaxLast(await ops.fromHost(rand(rnd, 4, 2), [4, 1])),
    },
    {
      name: "activation bwd relu/leakyrelu/gelu/silu/sigmoid/tanh",
      build: async (ops, rnd) => {
        const x = await ops.fromHost(rand(rnd, 16, 1.6), [16]);
        const dy = await ops.fromHost(rand(rnd, 16), [16]);
        let acc: DevBuffer | null = null;
        for (const name of ["relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh"]) {
          const y = await ops.activationFwd(name, x, 0.1);
          const dx = await ops.activationBwd(name, x, y, dy, 0.1);
          acc = acc ? await ops.binary("add", acc, dx) : dx;
        }
        return acc!;
      },
    },
    {
      name: "activation bwd softmax [5,6]",
      build: async (ops, rnd) => {
        const x = await ops.fromHost(rand(rnd, 30, 2), [5, 6]);
        const p = await ops.softmaxLast(x);
        const dy = await ops.fromHost(rand(rnd, 30), [5, 6]);
        return ops.softmaxBackward(p, dy);
      },
    },
    {
      name: "reduceOuter sum [6,4]",
      build: async (ops, rnd) => ops.reduceOuter("sum", await ops.fromHost(rand(rnd, 24), [6, 4])),
    },
    {
      name: "reduceOuter mean [2,3,4]",
      build: async (ops, rnd) => ops.reduceOuter("mean", await ops.fromHost(rand(rnd, 24), [2, 3, 4])),
    },
    {
      name: "reduceOuter mean grouped (gap2d) [2,3,3,4]",
      build: async (ops, rnd) => ops.reduceOuter("mean", await ops.fromHost(rand(rnd, 72), [2, 3, 3, 4]), 2),
    },
    {
      name: "broadcastPlanes [2,3]→[2,4,4,3]×1",
      build: async (ops, rnd) =>
        ops.broadcastPlanes(await ops.fromHost(rand(rnd, 6), [2, 3]), { B: 2, H: 4, W: 4, C: 3 }, 1),
    },
    {
      name: "broadcastPlanes [1,2]→[1,3,3,2]×0.25",
      build: async (ops, rnd) =>
        ops.broadcastPlanes(await ops.fromHost(rand(rnd, 2), [1, 2]), { B: 1, H: 3, W: 3, C: 2 }, 0.25),
    },
    {
      name: "reshapeView [2,3,4]→[6,4]",
      build: async (ops, rnd) => ops.reshapeView(await ops.fromHost(rand(rnd, 24), [2, 3, 4]), [6, 4]),
    },
    {
      name: "zeros [3,5]",
      build: async (ops, rnd) => {
        void rnd;
        return ops.zeros([3, 5]);
      },
    },
  ];

  for (const c of convCases) {
    const g = c.g;
    cases.push({
      name: `conv2d fwd ${c.name}`,
      build: async (ops, rnd) =>
        ops.conv2d(
          await ops.fromHost(rand(rnd, g.B * g.H * g.W * g.Cin), [g.B, g.H, g.W, g.Cin]),
          await ops.fromHost(rand(rnd, g.kH * g.kW * g.Cin * g.F), [g.kH, g.kW, g.Cin, g.F]),
          await ops.fromHost(rand(rnd, g.F), [g.F]),
          g,
        ),
    });
    cases.push({
      name: `conv2d fwd no bias ${c.name}`,
      build: async (ops, rnd) =>
        ops.conv2d(
          await ops.fromHost(rand(rnd, g.B * g.H * g.W * g.Cin), [g.B, g.H, g.W, g.Cin]),
          await ops.fromHost(rand(rnd, g.kH * g.kW * g.Cin * g.F), [g.kH, g.kW, g.Cin, g.F]),
          null,
          g,
        ),
    });
    cases.push({
      name: `conv2d dX ${c.name}`,
      build: async (ops, rnd) =>
        ops.conv2dInputGrad(
          await ops.fromHost(rand(rnd, g.B * g.outH * g.outW * g.F), [g.B, g.outH, g.outW, g.F]),
          await ops.fromHost(rand(rnd, g.kH * g.kW * g.Cin * g.F), [g.kH, g.kW, g.Cin, g.F]),
          g,
        ),
    });
    cases.push({
      name: `conv2d dK ${c.name}`,
      build: async (ops, rnd) =>
        ops.conv2dKernelGrad(
          await ops.fromHost(rand(rnd, g.B * g.H * g.W * g.Cin), [g.B, g.H, g.W, g.Cin]),
          await ops.fromHost(rand(rnd, g.B * g.outH * g.outW * g.F), [g.B, g.outH, g.outW, g.F]),
          g,
        ),
    });
  }

  for (const c of poolCases) {
    const g = c.g;
    for (const mode of ["max", "avg"] as PoolMode[]) {
      cases.push({
        name: `pool2d ${mode} fwd ${c.name}`,
        build: async (ops, rnd) => ops.pool2d(mode, await ops.fromHost(rand(rnd, g.B * g.H * g.W * g.C, 2), [g.B, g.H, g.W, g.C]), g),
      });
      cases.push({
        name: `pool2d ${mode} bwd ${c.name}`,
        build: async (ops, rnd) =>
          ops.pool2dBackward(
            mode,
            await ops.fromHost(rand(rnd, g.B * g.H * g.W * g.C, 2), [g.B, g.H, g.W, g.C]),
            await ops.fromHost(rand(rnd, g.B * g.outH * g.outW * g.C), [g.B, g.outH, g.outW, g.C]),
            g,
          ),
      });
    }
  }

  cases.push({
    name: "gatherRows [9,4] duplicate ids",
    build: async (ops, rnd) =>
      ops.gatherRows(await ops.fromHost(rand(rnd, 36), [9, 4]), await ops.fromHost(Float32Array.from([0, 8, 3, 3, 5, 1]), [6]), 4),
  });
  cases.push({
    name: "gatherRows [9,4] single dim",
    build: async (ops, rnd) =>
      ops.gatherRows(await ops.fromHost(rand(rnd, 9), [9, 1]), await ops.fromHost(Float32Array.from([7, 2]), [2]), 1),
  });
  cases.push({
    name: "scatterRows [6,4]→[9,4] duplicate ids",
    build: async (ops, rnd) =>
      ops.scatterRows(await ops.fromHost(rand(rnd, 24), [6, 4]), await ops.fromHost(Float32Array.from([2, 2, 0, 8, 5, 2]), [6]), 4, 9),
  });
  cases.push({
    name: "scatterRows [3,2]→[4,2]",
    build: async (ops, rnd) =>
      ops.scatterRows(await ops.fromHost(rand(rnd, 6), [3, 2]), await ops.fromHost(Float32Array.from([0, 3, 1]), [3]), 2, 4),
  });

  return cases;
}

/**
 * Runs the whole device suite. `reference` defaults to the CPU engine, which is
 * the definition of correct here: the accelerated path exists to be a faster
 * way of computing the same numbers, not a different model.
 */
export async function verifyDevice(device: GpuOps, reference: GpuOps = new CpuOps()): Promise<DeviceCheck[]> {
  const checks: DeviceCheck[] = [];
  const tolerance = 3e-4;

  for (const c of buildCases()) {
    let name = c.name;
    try {
      const expected = await reference.toHost(await c.build(reference, prng(0x51ed)));
      const actual = await device.toHost(await c.build(device, prng(0x51ed)));
      if (actual.length !== expected.length) {
        checks.push({
          name,
          passed: false,
          maxDiff: Infinity,
          tolerance: 0,
          message: `returned ${actual.length} values for ${expected.length} expected`,
        });
        continue;
      }
      let maxDiff = 0;
      let scale = 0;
      let badIndex = -1;
      for (let i = 0; i < expected.length; i++) {
        const e = expected[i]!;
        const a = actual[i]!;
        scale = Math.max(scale, Math.abs(e));
        const d = Math.abs(e - a);
        if (d > maxDiff) {
          maxDiff = d;
          badIndex = i;
        }
      }
      const limit = tolerance * Math.max(1, scale);
      const nan = actual.some((v) => !Number.isFinite(v));
      checks.push({
        name,
        passed: !nan && maxDiff <= limit,
        maxDiff,
        tolerance: limit,
        message: nan
          ? "non-finite values returned"
          : maxDiff <= limit
            ? `ok (worst |diff| ${maxDiff.toExponential(1)})`
            : `index ${badIndex}: CPU ${expected[badIndex]} vs device ${actual[badIndex]}`,
      });
    } catch (e) {
      checks.push({
        name,
        passed: false,
        maxDiff: Infinity,
        tolerance: 0,
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }

  // A device only earns trust if it also survives a full pass without erroring.
  try {
    await device.zeros([1]);
    checks.push({ name: "device stays healthy after the suite", passed: true, maxDiff: 0, tolerance: 0, message: "ok" });
  } catch (e) {
    checks.push({
      name: "device stays healthy after the suite",
      passed: false,
      maxDiff: Infinity,
      tolerance: 0,
      message: e instanceof Error ? e.message : String(e),
    });
  }

  return checks;
}

export function summarizeChecks(checks: DeviceCheck[]): { passed: number; failed: number; worst: number } {
  let passed = 0;
  let failed = 0;
  let worst = 0;
  for (const c of checks) {
    if (c.passed) passed++;
    else failed++;
    if (Number.isFinite(c.maxDiff)) worst = Math.max(worst, c.maxDiff);
  }
  return { passed, failed, worst };
}
