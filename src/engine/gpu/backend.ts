/**
 * Compute-backend selection.
 *
 * The CPU engine is always available and is the default. WebGPU is opt-in (or
 * "auto", which uses it when a device really comes up) and never a hard
 * requirement: if there is no `navigator.gpu`, no adapter, a blocked driver or a
 * lost device, the caller gets the CPU backend plus a status explaining why.
 */

import { CpuOps } from "./cpu";
import { WebGpuDevice } from "../webgpu/device";
import type { DeviceInfo, GpuOps } from "./ops";

export type BackendKind = "cpu" | "webgpu";
export type BackendPreference = "auto" | "cpu" | "webgpu";

export interface BackendStatus {
  kind: BackendKind;
  label: string;
  /** One-line explanation of why this backend is in use. */
  detail: string;
  /** The setting this came from. */
  preference: BackendPreference;
  /** True when this browser exposes WebGPU at all (regardless of the preference). */
  webgpuAvailable: boolean;
  /** Set when a WebGPU device was requested but could not be created. */
  error?: string;
}

export interface AcquiredBackend {
  ops: GpuOps;
  info: DeviceInfo;
  status: BackendStatus;
}

let cachedDevice: WebGpuDevice | null = null;
let creating: Promise<WebGpuDevice | null> | null = null;
let lastError: string | null = null;

/** Cheap feature check: does this browser expose WebGPU at all? */
export function hasWebGpu(): boolean {
  return WebGpuDevice.available();
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

async function acquireDevice(): Promise<WebGpuDevice | null> {
  if (cachedDevice) return cachedDevice;
  if (creating) return creating;
  creating = WebGpuDevice.create()
    .then((device) => {
      cachedDevice = device;
      lastError = null;
      return device;
    })
    .catch((e: unknown) => {
      lastError = message(e);
      return null;
    })
    .finally(() => {
      creating = null;
    });
  return creating;
}

/**
 * Picks the compute backend.
 *
 * "cpu" is the default and always works. "auto" and "webgpu" try to bring up a
 * WebGPU device; if that is impossible the caller gets the CPU backend plus a
 * status explaining what happened — never a hard failure.
 */
export async function acquireBackend(preference: BackendPreference): Promise<AcquiredBackend> {
  const available = hasWebGpu();
  const cpu = (detail: string, error?: string): AcquiredBackend => {
    const ops = new CpuOps();
    return {
      ops,
      info: ops.info,
      status: { kind: "cpu", label: ops.info.label, detail, preference, webgpuAvailable: available, error },
    };
  };

  if (preference === "cpu") {
    return cpu(
      available ? "CPU selected in Settings (WebGPU is available but not used)." : "CPU (this browser does not expose WebGPU).",
    );
  }

  if (!available) {
    const reason = "WebGPU is not available in this browser (navigator.gpu is missing).";
    if (preference === "webgpu") return cpu(reason, reason);
    return cpu("CPU (no WebGPU in this browser).");
  }

  const device = await acquireDevice();
  if (!device) {
    const reason = lastError ?? "Could not create a WebGPU device.";
    if (preference === "webgpu") return cpu(reason, reason);
    return cpu("CPU (WebGPU device could not be created).", reason);
  }

  return {
    ops: device,
    info: device.info,
    status: {
      kind: "webgpu",
      label: device.info.label,
      detail: preference === "webgpu" ? "WebGPU requested in Settings." : "WebGPU detected and used automatically.",
      preference,
      webgpuAvailable: true,
    },
  };
}

/** Drops the cached device, e.g. after a device loss or a preference change. */
export function releaseBackend(): void {
  if (cachedDevice) {
    try {
      cachedDevice.destroy();
    } catch {
      /* already gone */
    }
  }
  cachedDevice = null;
  creating = null;
}

/** The live device, if one has been acquired. */
export function currentDevice(): WebGpuDevice | null {
  return cachedDevice;
}

export { WebGpuDevice, CpuOps };
export type { DeviceInfo, GpuOps };
