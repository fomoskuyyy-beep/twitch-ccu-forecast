// gta6-twitch-ccu-forecast-01 entry: comparable-launch ratio model.
//
// The house baseline multiplies hour 1's print by a fixed random-walk spread and predicts no drift at all, which
// fails whenever release day ramps up or cools down. This entry instead learns the SHAPE of release day from the
// case's own `comparables` (38 past launches, each with hourly viewers across its release day):
//
//   1. Per comparable: r(h) = viewers(h) / viewers(hour 1 of that launch's own day_start), for h = 2..24, using only
//      rows whose timestamp sits exactly (h-1) hours after day_start; launches missing hour 1 or a row are skipped.
//   2. Per hour h, in log space, fit OLS of ln r(h) on ln(hour-1 viewers) across launches. Bigger launches ramp
//      less in ratio terms (saturating audience), so the slope is real signal: the case's own p1 is evaluated at
//      the fitted line, clamped to the observed ln-p1 range so a small or huge case never extrapolates.
//   3. mu(h) = 0.3 * cross-sectional median(ln r) + 0.7 * fitted value (regularizes the fit toward the median;
//      falls back to the median alone when fewer than 8 launches report the hour), then a light window-3 median
//      smoothing across neighbouring hours.
//   4. Quantiles keep the house's spread prior: q_tau(h) = p1 * exp(mu(h) + z_tau * 0.15 * sqrt(h - 1)), rounded to
//      2 decimals, clamped into (0, 1e9] and made non-decreasing.
//
// Everything is pure arithmetic over the case file: no network, no randomness, byte-identical across runs.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The spread of an hour's move, in natural-log units per square-root hour (the house's prior). */
const SIGMA = 0.15;
/** Weight of the cross-sectional median versus the fitted value when forming mu(h). */
const W_MED = 0.3;
/** Fewer comparable launches than this reporting an hour: no fit, median only. */
const MIN_PTS = 8;

interface HourlyRow {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyRow[];
}

interface Case {
  id?: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  comparables?: Comparable[];
}

/** x rounded to 2 decimals (half up), clamped into (0, 1e9]. */
const round2 = (x: number): number => {
  if (!Number.isFinite(x) || x <= 0) return 0.01;
  if (x >= 1e9) return 1e9;
  return Math.round(x * 100) / 100;
};

/** Arithmetic median of a copy of `values`. */
const median = (values: number[]): number => {
  const a = [...values].sort((u, v) => u - v);
  const n = a.length;
  if (n === 0) return 0;
  const m = n >> 1;
  return n % 2 === 1 ? a[m] : (a[m - 1] + a[m]) / 2;
};

/** OLS of y on x; null when the design is degenerate or there are too few points. */
const fit = (pts: Array<{ x: number; y: number }>): { a: number; b: number; xlo: number; xhi: number } | null => {
  const n = pts.length;
  if (n < MIN_PTS) return null;
  let mx = 0;
  let my = 0;
  for (const p of pts) {
    mx += p.x;
    my += p.y;
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  let xlo = Infinity;
  let xhi = -Infinity;
  for (const p of pts) {
    const dx = p.x - mx;
    sxx += dx * dx;
    sxy += dx * (p.y - my);
    if (p.x < xlo) xlo = p.x;
    if (p.x > xhi) xhi = p.x;
  }
  if (sxx < 1e-9) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b, xlo, xhi };
};

/** mu(h) for h = 2..24: blended fit + median of log-ratios, then window-3 median smoothing. */
const shapeMu = (comparables: Comparable[], p1: number): number[] => {
  // Re-parse into plain records so the loop below stays allocation-light and deterministic.
  const perHour: Array<Array<{ x: number; y: number }>> = [];
  for (let h = 2; h <= 24; h++) perHour.push([]);

  for (const comp of comparables) {
    const dayStart = Date.parse(comp.day_start);
    if (!Number.isFinite(dayStart)) continue;
    const byHour = new Map<number, number>();
    for (const row of comp.hourly ?? []) {
      const t = Date.parse(row.from);
      if (!Number.isFinite(t)) continue;
      const v = row.viewers;
      if (v === null || v === undefined || !Number.isFinite(v)) continue;
      const d = (t - dayStart) / 3600000;
      const hr = Math.round(d);
      if (Math.abs(d - hr) > 1e-6 || hr < 0 || hr > 23) continue;
      byHour.set(hr + 1, v);
    }
    const v1 = byHour.get(1);
    if (v1 === undefined || v1 <= 0) continue;
    const x = Math.log(Math.max(v1, 1));
    for (let h = 2; h <= 24; h++) {
      const v = byHour.get(h);
      if (v === undefined) continue;
      perHour[h - 2].push({ x, y: Math.log(Math.max(v, 1) / Math.max(v1, 1)) });
    }
  }

  const x0 = Math.log(Math.max(p1, 1));
  const mu: number[] = [];
  for (let i = 0; i < 23; i++) {
    const pts = perHour[i];
    const ys = pts.map((p) => p.y);
    const med = ys.length > 0 ? median(ys) : 0;
    const r = fit(pts);
    if (r === null) {
      mu.push(med);
    } else {
      const xc = Math.min(Math.max(x0, r.xlo), r.xhi);
      mu.push(W_MED * med + (1 - W_MED) * (r.a + r.b * xc));
    }
  }

  // Light window-3 median smoothing around each hour (edges use what exists).
  const sm: number[] = [];
  for (let i = 0; i < 23; i++) {
    const win = [mu[i]];
    if (i > 0) win.push(mu[i - 1]);
    if (i < 22) win.push(mu[i + 1]);
    sm.push(median(win));
  }
  return sm;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.levels) || c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} does not ask for levels ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const mu = shapeMu(c.comparables ?? [], p1);

  const hours: Array<{ hour: number; q: number[] }> = [];
  for (let i = 0; i < 23; i++) {
    const h = i + 2;
    const spread = SIGMA * Math.sqrt(h - 1);
    let prev = 0;
    const q: number[] = [];
    for (const z of Z) {
      let v = round2(p1 * Math.exp(mu[i] + z * spread));
      if (v < prev) v = prev;
      prev = v;
      q.push(v);
    }
    hours.push({ hour: h, q });
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
