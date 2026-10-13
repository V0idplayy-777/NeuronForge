import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { cn } from "../utils/cn";

export function Button({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "danger" | "accent" | "quiet"; size?: "sm" | "md" | "lg" }) {
  const v = {
    primary: "bg-cyan-500/90 text-slate-950 hover:bg-cyan-400 border-cyan-300/30",
    accent: "bg-violet-500/80 text-white hover:bg-violet-400 border-violet-300/20",
    ghost: "bg-white/5 text-slate-200 hover:bg-white/10 border-white/10",
    quiet: "bg-transparent text-slate-300 hover:bg-white/5 border-transparent",
    danger: "bg-rose-500/20 text-rose-200 hover:bg-rose-500/30 border-rose-500/30",
  }[variant];
  const s = { sm: "h-7 px-2 text-[11px] gap-1", md: "h-8 px-3 text-xs gap-1.5", lg: "h-10 px-4 text-sm gap-2" }[size];
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md border font-medium tracking-wide transition disabled:opacity-40 disabled:pointer-events-none",
        v,
        s,
        className,
      )}
      {...props}
    />
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="text-[10px] uppercase tracking-[0.14em] text-slate-500 font-medium">{label}</span>
      {children}
      {hint && <span className="text-[10px] leading-snug text-slate-500">{hint}</span>}
    </label>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "h-8 w-full rounded-md border border-white/10 bg-black/40 px-2 text-xs text-slate-100 outline-none focus:border-cyan-400/60 focus:ring-1 focus:ring-cyan-400/30",
        props.className,
      )}
    />
  );
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "h-8 w-full rounded-md border border-white/10 bg-black/40 px-2 text-xs text-slate-100 outline-none focus:border-cyan-400/60",
        props.className,
      )}
    />
  );
}

export function Panel({ title, actions, children, className }: { title?: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("rounded-lg border border-white/8 bg-panel/80 min-w-0", className)}>
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-white/8 px-3 py-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">{title}</h3>
          {actions}
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: string; accent?: string }) {
  return (
    <div className="rounded-md border border-white/8 bg-black/30 px-3 py-2 min-w-0">
      <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</div>
      <div className={cn("mt-0.5 font-mono text-sm text-slate-100 truncate", accent)}>{value}</div>
      {sub && <div className="text-[10px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "cyan" | "amber" | "rose" | "emerald" | "violet" }) {
  const t = {
    slate: "bg-white/8 text-slate-300",
    cyan: "bg-cyan-500/15 text-cyan-300",
    amber: "bg-amber-500/15 text-amber-300",
    rose: "bg-rose-500/15 text-rose-300",
    emerald: "bg-emerald-500/15 text-emerald-300",
    violet: "bg-violet-500/15 text-violet-300",
  }[tone];
  return <span className={cn("inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider", t)}>{children}</span>;
}

export function Empty({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center text-center px-8 py-16 gap-3">
      <div className="h-10 w-10 rounded-lg border border-dashed border-white/20" />
      <h3 className="text-sm font-medium text-slate-200">{title}</h3>
      <p className="text-xs text-slate-500 max-w-md leading-relaxed">{body}</p>
      {action}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 text-xs text-slate-300"
    >
      <span className={cn("h-4 w-7 rounded-full p-0.5 transition", checked ? "bg-cyan-500" : "bg-white/15")}>
        <span className={cn("block h-3 w-3 rounded-full bg-white transition", checked && "translate-x-3")} />
      </span>
      {label}
    </button>
  );
}
