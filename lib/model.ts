// Token Radar, model "radar-4": one snapshot of the protocol universe in, one ranked board out.
//
// The board ranks every rankable pre-token protocol: it records no token (no market cap, no
// CoinGecko id, no symbol) and is not a wrapped product. The 92 trading protocols that are not
// wrapped products are the comps: the market's own evidence for what TVL is worth.
//
// Implied value comes from a log-linear regression on the comps, fit by OLS with two Huber
// reweightings, category effects shrunk toward zero. The model was not picked by hand: ten
// candidates fight a tournament (leave-one-out, a time split and ten random splits, all scored by
// median absolute log error) and the lowest mean error wins. A 200-draw bootstrap puts a p10/p90
// band and a top-10 stability share on every row. Every fit on a subset — every fold, every
// holdout draw, every bootstrap refit — recomputes its own category dummies and median age from
// its own training pool, so no test protocol leaks into the fit it is scored against. Every
// number is computed at runtime from the snapshot; no protocol name is ever special-cased.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The model's version label: the dashboard names it and the rules quote it. */
export const MODEL_VERSION = 'radar-4';

/** How many live-token protocols the Live tokens view carries, by market cap. */
export const LIVE_ROWS = 24;

/** Seeded RNG seeds: the random splits, the blind holdout and the bootstrap are all reproducible. */
export const SEED_SPLITS = 7;
export const SEED_HOLDOUT = 20261114;
/** Blind-holdout draws: one seeded shuffle per draw, seeds 20261114..20261123. */
export const HOLDOUT_DRAWS = 10;
export const SEED_BOOTSTRAP = 20261114;

/** Bootstrap resamples. */
export const BOOTSTRAP_REPS = 200;

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

// ---------------------------------------------------------------- deterministic helpers

/** mulberry32: a small seeded PRNG. Every random draw in this program comes from one of these. */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher-Yates shuffle, in place, driven by `rnd`. Deterministic for a fixed seed. */
export function shuffle<T>(arr: T[], rnd: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = arr[i]!;
    arr[i] = arr[j]!;
    arr[j] = tmp;
  }
  return arr;
}

/** The median of a non-empty array (the array is not modified). */
export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** A quantile by linear interpolation on the sorted values, q in [0, 1]. */
export function quantile(values: number[], q: number): number {
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 1) return s[0]!;
  const at = q * (s.length - 1);
  const lo = Math.floor(at);
  const hi = Math.ceil(at);
  return lo === hi ? s[lo]! : s[lo]! * (hi - at) + s[hi]! * (at - lo);
}

const mean = (values: number[]): number => values.reduce((a, b) => a + b, 0) / values.length;

/** Sample standard deviation (n - 1 denominator), 0 for fewer than two values. */
function sd(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((a, b) => a + (b - m) * (b - m), 0) / (values.length - 1));
}

const dot = (a: number[], b: number[]): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

/**
 * Solve A x = b by Gaussian elimination with partial pivoting. Returns null when the system is
 * singular to working precision. A is square, b matches its size; neither is modified.
 */
export function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[pivot]![col]!)) pivot = r;
    if (Math.abs(M[pivot]![col]!) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = M[col]!;
      M[col] = M[pivot]!;
      M[pivot] = tmp;
    }
    const pv = M[col]![col]!;
    for (let r = col + 1; r < n; r++) {
      const f = M[r]![col]! / pv;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r]![c]! -= f * M[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= M[r]![c]! * x[c]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/** Weighted least squares for a design matrix X and target y, with a hair of ridge for stability. */
function wls(X: number[][], y: number[], w: number[]): number[] | null {
  const p = X[0]!.length;
  const A: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const b = new Array<number>(p).fill(0);
  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    const wi = w[i]!;
    for (let r = 0; r < p; r++) {
      b[r]! += wi * xi[r]! * y[i]!;
      for (let c = r; c < p; c++) A[r]![c]! += wi * xi[r]! * xi[c]!;
    }
  }
  for (let r = 0; r < p; r++) for (let c = 0; c < r; c++) A[r]![c] = A[c]![r]!;
  for (let r = 0; r < p; r++) A[r]![r]! += 1e-6;
  return solveLinear(A, b);
}

// ---------------------------------------------------------------- snapshot parsing

const DAY_MS = 86_400_000;

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

// ---------------------------------------------------------------- classification

/** Categories that mark a wrapped, staked or bridged product rather than a protocol of its own. */
const WRAPPED_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** A name ending in a version marker is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** A name carrying a whole word wrapped, staked or staking is a wrapped product. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** Exchange names: a protocol named after one is that exchange's product. */
const EXCHANGE_PREFIXES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];

/** Whether a protocol is a wrapped product of a token that already trades. */
export function isWrapped(p: SnapshotProtocol): boolean {
  if (p.category !== null && WRAPPED_CATEGORIES.has(p.category)) return true;
  if (VERSIONED.test(p.name)) return true;
  if (WRAPPED_WORD.test(p.name)) return true;
  return EXCHANGE_PREFIXES.some((x) => p.name.startsWith(x));
}

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol is one the board may rank: pre-token and not a wrapped product. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && !isWrapped(p);
}

/** The trading protocols the model learns from: a market cap, a TVL, not a wrapped product. */
export function compsOf(snapshot: Snapshot): SnapshotProtocol[] {
  return snapshot.protocols.filter((p) => (p.mcap_usd ?? 0) > 0 && p.tvl_usd > 0 && !isWrapped(p));
}

/**
 * The robust trim: drop comps whose log(mcap/TVL) sits beyond 3 robust SDs (1.4826 x MAD, floored
 * at 0.25) from the comps' median log multiple. The same rule trims every training set the model
 * ever fits, so a fold's training pool is trimmed the same way the full pool is.
 */
export function trimComps(pool: SnapshotProtocol[]): { kept: SnapshotProtocol[]; dropped: SnapshotProtocol[] } {
  const logs = pool.map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
  const med = median(logs);
  const scale = Math.max(1.4826 * median(logs.map((x) => Math.abs(x - med))), 0.25);
  const kept: SnapshotProtocol[] = [];
  const dropped: SnapshotProtocol[] = [];
  for (const p of pool) {
    if (Math.abs(Math.log(p.mcap_usd! / p.tvl_usd) - med) > 3 * scale) dropped.push(p);
    else kept.push(p);
  }
  return { kept, dropped };
}

// ---------------------------------------------------------------- the regression

/** What every fit shares: the imputed median age and the categories with enough comps for a dummy. */
export interface FitContext {
  asOf: string;
  medianAge: number;
  cats: string[];
}

/**
 * A FitContext for one training pool: the categories with at least five comps in THAT pool
 * (sorted, the same rule the full-board fit uses) and the median age of THAT pool's dated comps.
 * Every fit on a subset builds its own context from its own training pool, so a fold's dummy set
 * and median age never leak information from the protocols it is scored against.
 */
export function fitContextFor(pool: SnapshotProtocol[], asOf: string): FitContext {
  const ages = pool.map((p) => ageDays(asOf, p.listed_at)).filter((a): a is number => a !== null);
  return {
    asOf,
    medianAge: ages.length > 0 ? median(ages) : 0,
    cats: [...new Set(pool.map((p) => p.category ?? ''))]
      .filter((c) => pool.filter((p) => (p.category ?? '') === c).length >= 5)
      .sort(),
  };
}

/** 7-day growth as a fraction, clamped to [-0.5, 1]; unreported growth is 0. */
function growthOf(p: SnapshotProtocol): number {
  if (p.change_7d === null) return 0;
  return Math.min(100, Math.max(-50, p.change_7d)) / 100;
}

/** A protocol's age in days, or the comps' median age when DefiLlama reports none. */
function ageOf(p: SnapshotProtocol, ctx: FitContext): number {
  const a = ageDays(ctx.asOf, p.listed_at);
  return a === null ? ctx.medianAge : a;
}

/**
 * One protocol's feature row: [1, ln(TVL), growth, ln(age + 30), ln(chains)] plus one dummy per
 * category with at least five comps. `quad` adds (ln TVL)^2 / 100 after the five base features.
 */
export function features(p: SnapshotProtocol, ctx: FitContext, quad: boolean): number[] {
  const lnTvl = Math.log(p.tvl_usd);
  const row = [1, lnTvl, growthOf(p), Math.log(ageOf(p, ctx) + 30), Math.log(Math.max(1, p.chains.length))];
  if (quad) row.push((lnTvl * lnTvl) / 100);
  for (const c of ctx.cats) row.push(p.category === c ? 1 : 0);
  return row;
}

/** Feature names in the order `features` returns them, for the coefficients table. */
export function featureNames(ctx: FitContext, quad: boolean): string[] {
  const names = ['intercept', 'ln(TVL)', 'growth', 'ln(age+30)', 'ln(chains)'];
  if (quad) names.push('(ln TVL)^2/100');
  return [...names, ...ctx.cats.map((c) => `cat: ${c}`)];
}

/**
 * Fit log(mcap) on a pool of comps: OLS, then two Huber reweightings (k = 1.345 sigma, sigma the
 * residual MAD scaled by 1.4826, floored at 0.05). Category effects are shrunk by n/(n+5) so a
 * category with few comps cannot swing the board. Returns null when the system will not solve.
 */
export function fitOLS(pool: SnapshotProtocol[], ctx: FitContext, quad: boolean, weights?: number[]): number[] | null {
  const X = pool.map((p) => features(p, ctx, quad));
  const y = pool.map((p) => Math.log(p.mcap_usd!));
  const base = weights ?? pool.map(() => 1);
  let beta = wls(X, y, base);
  if (beta === null) return null;
  for (let iter = 0; iter < 2; iter++) {
    const resid = y.map((yi, i) => yi - dot(X[i]!, beta!));
    const med = median(resid);
    const sigma = Math.max(1.4826 * median(resid.map((r) => Math.abs(r - med))), 0.05);
    const k = 1.345 * sigma;
    const w2 = base.map((wi, i) => {
      const a = Math.abs(resid[i]!);
      return wi * (a <= k ? 1 : k / a);
    });
    const next = wls(X, y, w2);
    if (next === null) return null;
    beta = next;
  }
  const counts = new Map<string, number>();
  for (const p of pool) counts.set(p.category ?? '', (counts.get(p.category ?? '') ?? 0) + 1);
  const catBase = quad ? 6 : 5;
  for (let i = 0; i < ctx.cats.length; i++) {
    const n = counts.get(ctx.cats[i]!) ?? 0;
    beta[catBase + i]! *= n / (n + 5);
  }
  return beta;
}

/** The fitted log value of one protocol under coefficients `beta`. */
export function predictLog(beta: number[], p: SnapshotProtocol, ctx: FitContext, quad: boolean): number {
  return dot(beta, features(p, ctx, quad));
}

/** A point estimate from a fitted log value: exp, capped at 1e14, floored at 1, whole dollars. */
export function pointEstimate(logValue: number): number {
  return Math.max(1, Math.min(1e14, Math.round(Math.exp(logValue))));
}

/** The median mcap/TVL multiple of a pool of comps. */
function medianMultiple(pool: SnapshotProtocol[]): number {
  return median(pool.map((p) => p.mcap_usd! / p.tvl_usd));
}

// ---------------------------------------------------------------- candidates

/** A model that predicts log(mcap) for any protocol, fit on one pool of comps. */
export type Predictor = (p: SnapshotProtocol) => number;

export interface CandidateDef {
  id: string;
  name: string;
  kind: 'ols' | 'quad' | 'wols' | 'knn' | 'band';
  blend?: number;
  k?: number;
}

/** The ten tournament candidates, in the order the table shows them. */
export const CANDIDATES: CandidateDef[] = [
  { id: 'ols0', name: 'OLS, blend 0', kind: 'ols', blend: 0 },
  { id: 'ols25', name: 'OLS, blend 0.25', kind: 'ols', blend: 0.25 },
  { id: 'ols50', name: 'OLS, blend 0.5', kind: 'ols', blend: 0.5 },
  { id: 'quad0', name: 'OLS + (ln TVL)^2, blend 0', kind: 'quad', blend: 0 },
  { id: 'quad25', name: 'OLS + (ln TVL)^2, blend 0.25', kind: 'quad', blend: 0.25 },
  { id: 'wols', name: 'OLS, TVL-weighted', kind: 'wols', blend: 0 },
  { id: 'knn5', name: 'kNN k=5', kind: 'knn', k: 5 },
  { id: 'knn10', name: 'kNN k=10', kind: 'knn', k: 10 },
  { id: 'knn15', name: 'kNN k=15', kind: 'knn', k: 15 },
  { id: 'band', name: 'Band curve', kind: 'band' },
];

/** The four numeric features kNN standardizes: [ln TVL, growth, ln(age+30), ln(chains)]. */
function knnFeatures(p: SnapshotProtocol, ctx: FitContext): number[] {
  return [Math.log(p.tvl_usd), growthOf(p), Math.log(ageOf(p, ctx) + 30), Math.log(Math.max(1, p.chains.length))];
}

/** TVL band edges for the band-curve candidate: <$20M, $20-100M, $100-500M, $500M-2B, >$2B. */
const BAND_EDGES = [20_000_000, 100_000_000, 500_000_000, 2_000_000_000];

function bandIndex(tvl: number): number {
  for (let i = 0; i < BAND_EDGES.length; i++) if (tvl < BAND_EDGES[i]!) return i;
  return 4;
}

/**
 * Fit one candidate on a pool of comps. Returns null when the candidate cannot be fit on the pool
 * (a singular regression; kNN and the band curve always fit). Blend candidates mix the regression
 * fit with the pool's median multiple: exp((1-b) * fit + b * ln(TVL x median multiple)).
 */
export function fitCandidate(def: CandidateDef, pool: SnapshotProtocol[], ctx: FitContext): Predictor | null {
  if (def.kind === 'ols' || def.kind === 'quad' || def.kind === 'wols') {
    const weights = def.kind === 'wols' ? pool.map((p) => Math.sqrt(p.tvl_usd) / 1e4) : undefined;
    const beta = fitOLS(pool, ctx, def.kind === 'quad', weights);
    if (beta === null) return null;
    const mm = medianMultiple(pool);
    const blend = def.blend ?? 0;
    const quad = def.kind === 'quad';
    return (p) => {
      const fit = predictLog(beta, p, ctx, quad);
      return blend === 0 ? fit : (1 - blend) * fit + blend * Math.log(p.tvl_usd * mm);
    };
  }
  if (def.kind === 'knn') {
    const F = pool.map((p) => knnFeatures(p, ctx));
    const mu = [0, 1, 2, 3].map((j) => mean(F.map((f) => f[j]!)));
    const sdv = [0, 1, 2, 3].map((j) => {
      const s = sd(F.map((f) => f[j]!));
      return s === 0 ? 1 : s;
    });
    const Z = F.map((f) => f.map((v, j) => (v - mu[j]!) / sdv[j]!));
    const k = def.k ?? 5;
    return (p) => {
      const z = knnFeatures(p, ctx).map((v, j) => (v - mu[j]!) / sdv[j]!);
      const dists = Z.map((row, i) => {
        let d = 0;
        for (let j = 0; j < 4; j++) d += (row[j]! - z[j]!) * (row[j]! - z[j]!);
        d = Math.sqrt(d);
        if (pool[i]!.category === p.category) d *= 0.5;
        return { d, i };
      });
      dists.sort((a, b) => a.d - b.d || (pool[a.i]!.slug < pool[b.i]!.slug ? -1 : 1));
      const mults = dists.slice(0, k).map(({ i }) => pool[i]!.mcap_usd! / pool[i]!.tvl_usd);
      return Math.log(p.tvl_usd * median(mults));
    };
  }
  // Band curve: the median multiple of each TVL band, smoothed with its neighbours.
  const bands: number[][] = [[], [], [], [], []];
  for (const p of pool) bands[bandIndex(p.tvl_usd)]!.push(p.mcap_usd! / p.tvl_usd);
  const global = medianMultiple(pool);
  const smooth = bands.map((_, i) => {
    const vals: number[] = [];
    for (const j of [i - 1, i, i + 1]) if (j >= 0 && j < 5) vals.push(...bands[j]!);
    return vals.length > 0 ? median(vals) : global;
  });
  return (p) => Math.log(p.tvl_usd * smooth[bandIndex(p.tvl_usd)]!);
}

// ---------------------------------------------------------------- evaluation

/** Out-of-sample scores for one test set: median absolute log error, median APE, within-2x share. */
export interface EvalMetrics {
  n: number;
  male: number;
  medape: number;
  within2: number;
}

/** Score (true log mcap, predicted log mcap) pairs. within-2x means |pred - true| / true <= 1. */
export function evalMetrics(pairs: Array<[number, number]>): EvalMetrics {
  const errs = pairs.map(([yt, yp]) => yp - yt);
  const apes = errs.map((e) => Math.abs(Math.exp(e) - 1));
  return {
    n: pairs.length,
    male: median(errs.map((e) => Math.abs(e))),
    medape: median(apes),
    within2: apes.filter((a) => a <= 1).length / apes.length,
  };
}

/** Predict every test protocol with a candidate fit on the training pool (trimmed first). */
function evalCandidate(def: CandidateDef, trainPool: SnapshotProtocol[], testPool: SnapshotProtocol[], ctx: FitContext): Array<[number, number]> | null {
  const trimmed = trimComps(trainPool).kept;
  // The fold's own context: its dummy set and median age come from its training pool only, so
  // no test protocol leaks into the fit it is scored against.
  const localCtx = fitContextFor(trimmed, ctx.asOf);
  const fit = fitCandidate(def, trimmed, localCtx);
  if (fit === null) return null;
  return testPool.map((p) => [Math.log(p.mcap_usd!), fit(p)] as [number, number]);
}

/** One tournament row: a candidate's MALE on each of the three tests and their mean. */
export interface TournamentRow {
  id: string;
  name: string;
  loocv: number;
  timeSplit: number;
  randomSplit: number;
  mean: number;
  selected: boolean;
}

/**
 * The tournament: every candidate is scored by leave-one-out over the comps, a time split (fit on
 * the older-listed half of the dated comps, test on the newer half) and ten random 50/50 splits
 * (seed 7). The metric is median absolute log error per test; the winner has the lowest mean MALE.
 * Training pools are trimmed by the same rule everywhere; test pools are never trimmed. Each fold
 * fits with its own FitContext, built from its own training pool.
 */
export function runTournament(comps: SnapshotProtocol[], ctx: FitContext): TournamentRow[] {
  const trainPool = trimComps(comps).kept;
  const dated = comps
    .filter((p) => ageDays(ctx.asOf, p.listed_at) !== null)
    .sort((a, b) => ageDays(ctx.asOf, b.listed_at)! - ageDays(ctx.asOf, a.listed_at)! || (a.slug < b.slug ? -1 : 1));
  const half = Math.floor(dated.length / 2);
  const timeTrain = dated.slice(0, half);
  const timeTest = dated.slice(half);

  const rows: TournamentRow[] = [];
  for (const def of CANDIDATES) {
    // LOOCV: leave one comp out, fit on the trimmed pool without it, predict it.
    const loocvPairs: Array<[number, number]> = [];
    let loocvOk = true;
    for (const held of comps) {
      const pool = trainPool.filter((p) => p.slug !== held.slug);
      const fit = fitCandidate(def, pool, fitContextFor(pool, ctx.asOf));
      if (fit === null) {
        loocvOk = false;
        break;
      }
      loocvPairs.push([Math.log(held.mcap_usd!), fit(held)]);
    }
    if (!loocvOk) continue;
    const loocv = evalMetrics(loocvPairs).male;

    const timePairs = evalCandidate(def, timeTrain, timeTest, ctx);
    if (timePairs === null) continue;
    const timeSplit = evalMetrics(timePairs).male;

    const rnd = mulberry32(SEED_SPLITS);
    const repMales: number[] = [];
    let randomOk = true;
    for (let rep = 0; rep < 10; rep++) {
      const sh = shuffle([...comps], rnd);
      const h = Math.floor(sh.length / 2);
      const pairs = evalCandidate(def, sh.slice(0, h), sh.slice(h), ctx);
      if (pairs === null) {
        randomOk = false;
        break;
      }
      repMales.push(evalMetrics(pairs).male);
    }
    if (!randomOk) continue;
    const randomSplit = mean(repMales);
    rows.push({ id: def.id, name: def.name, loocv, timeSplit, randomSplit, mean: mean([loocv, timeSplit, randomSplit]), selected: false });
  }
  rows.sort((a, b) => a.mean - b.mean || (a.id < b.id ? -1 : 1));
  if (rows.length > 0) rows[0]!.selected = true;
  // Restore the candidates' own order for display; the winner is marked.
  const order = new Map(CANDIDATES.map((c, i) => [c.id, i]));
  rows.sort((a, b) => order.get(a.id)! - order.get(b.id)!);
  return rows;
}

/** The full validation report for the selected model. */
export interface ValidationReport {
  loocv: EvalMetrics;
  timeSplit: EvalMetrics;
  randomSplit: EvalMetrics;
  randomSplitMeanMale: number;
  /** The repeated blind holdout: the median across HOLDOUT_DRAWS draws (n is per draw). */
  holdout: EvalMetrics;
  holdoutDraws: number;
  /** The min-max range of the median error across the holdout draws: the noise, shown honestly. */
  holdoutMedapeMin: number;
  holdoutMedapeMax: number;
  /** The median trimmed training size across the holdout draws. */
  holdoutTrain: number;
  band80: { z: number; target: number; coverage: number };
  band50: { z: number; target: number; coverage: number };
  r2: number;
  residSd: number;
}

/**
 * Validate the selected model on every test the tournament used, plus a repeated blind holdout:
 * ten seeded shuffles (mulberry32 seeds 20261114..20261123) of the comps, 74 train / 18 test
 * each, the holdout never used in fitting or selection. One 18-protocol draw is too noisy to be
 * a headline number, so the report is the median across the ten draws, with the min-max range of
 * the median error across draws. Band calibration checks the LOOCV parametric band against the
 * comps.
 */
export function validateSelected(def: CandidateDef, comps: SnapshotProtocol[], ctx: FitContext, residSd: number): ValidationReport {
  const trainPool = trimComps(comps).kept;
  const loocvPairs: Array<[number, number]> = [];
  for (const held of comps) {
    const pool = trainPool.filter((p) => p.slug !== held.slug);
    const fit = fitCandidate(def, pool, fitContextFor(pool, ctx.asOf));
    if (fit !== null) loocvPairs.push([Math.log(held.mcap_usd!), fit(held)]);
  }
  const loocv = evalMetrics(loocvPairs);

  const dated = comps
    .filter((p) => ageDays(ctx.asOf, p.listed_at) !== null)
    .sort((a, b) => ageDays(ctx.asOf, b.listed_at)! - ageDays(ctx.asOf, a.listed_at)! || (a.slug < b.slug ? -1 : 1));
  const half = Math.floor(dated.length / 2);
  const timePairs = evalCandidate(def, dated.slice(0, half), dated.slice(half), ctx) ?? [];
  const timeSplit = evalMetrics(timePairs);

  const rnd = mulberry32(SEED_SPLITS);
  const allRandom: Array<[number, number]> = [];
  const repMales: number[] = [];
  for (let rep = 0; rep < 10; rep++) {
    const sh = shuffle([...comps], rnd);
    const h = Math.floor(sh.length / 2);
    const pairs = evalCandidate(def, sh.slice(0, h), sh.slice(h), ctx);
    if (pairs === null) continue;
    allRandom.push(...pairs);
    repMales.push(evalMetrics(pairs).male);
  }
  const randomSplit = evalMetrics(allRandom);

  const holdDraws: EvalMetrics[] = [];
  const holdTrains: number[] = [];
  for (let draw = 0; draw < HOLDOUT_DRAWS; draw++) {
    const rndH = mulberry32(SEED_HOLDOUT + draw);
    const shH = shuffle([...comps], rndH);
    const pairs = evalCandidate(def, shH.slice(0, 74), shH.slice(74), ctx);
    if (pairs === null) continue;
    holdDraws.push(evalMetrics(pairs));
    holdTrains.push(trimComps(shH.slice(0, 74)).kept.length);
  }
  const holdMedapes = holdDraws.map((d) => d.medape);
  const holdout: EvalMetrics =
    holdDraws.length > 0
      ? {
          n: 18,
          male: median(holdDraws.map((d) => d.male)),
          medape: median(holdMedapes),
          within2: median(holdDraws.map((d) => d.within2)),
        }
      : { n: 18, male: 0, medape: 0, within2: 0 };

  const coverage = (z: number): number => loocvPairs.filter(([yt, yp]) => Math.abs(yt - yp) <= z * residSd).length / loocvPairs.length;
  const beta = fitOLS(trainPool, ctx, false);
  let r2 = 0;
  if (beta !== null) {
    const y = comps.map((p) => Math.log(p.mcap_usd!));
    const yh = comps.map((p) => predictLog(beta, p, ctx, false));
    const m = mean(y);
    const ssRes = y.reduce((a, yi, i) => a + (yi - yh[i]!) ** 2, 0);
    const ssTot = y.reduce((a, yi) => a + (yi - m) ** 2, 0);
    r2 = 1 - ssRes / ssTot;
  }
  return {
    loocv,
    timeSplit,
    randomSplit,
    randomSplitMeanMale: mean(repMales),
    holdout,
    holdoutDraws: holdDraws.length,
    holdoutMedapeMin: holdDraws.length > 0 ? Math.min(...holdMedapes) : 0,
    holdoutMedapeMax: holdDraws.length > 0 ? Math.max(...holdMedapes) : 0,
    holdoutTrain: holdDraws.length > 0 ? Math.round(median(holdTrains)) : 0,
    band80: { z: 1.28, target: 0.8, coverage: coverage(1.28) },
    band50: { z: 0.67, target: 0.5, coverage: coverage(0.67) },
    r2,
    residSd,
  };
}

/** One row of the size curve: the median mcap/TVL multiple of comps in a TVL band. */
export interface SizeBand {
  label: string;
  n: number;
  medianMultiple: number;
  p25: number;
  p75: number;
}

/** The size curve: median multiple by TVL band over all comps. */
export function sizeCurve(comps: SnapshotProtocol[]): SizeBand[] {
  const bands: Array<[string, number, number]> = [
    ['$5-20M', 5_000_000, 20_000_000],
    ['$20-100M', 20_000_000, 100_000_000],
    ['$100-500M', 100_000_000, 500_000_000],
    ['$500M-2B', 500_000_000, 2_000_000_000],
    ['>$2B', 2_000_000_000, Number.POSITIVE_INFINITY],
  ];
  return bands.map(([label, lo, hi]) => {
    const mults = comps.filter((p) => p.tvl_usd >= lo && p.tvl_usd < hi).map((p) => p.mcap_usd! / p.tvl_usd);
    return {
      label,
      n: mults.length,
      medianMultiple: mults.length > 0 ? median(mults) : 0,
      p25: mults.length > 0 ? quantile(mults, 0.25) : 0,
      p75: mults.length > 0 ? quantile(mults, 0.75) : 0,
    };
  });
}

// ---------------------------------------------------------------- bootstrap

/** What the bootstrap says about one row: its value band, its rank band and its top-10 share. */
export interface BootstrapRow {
  p10: number;
  p90: number;
  rankP10: number;
  rankP90: number;
  top10Pct: number;
}

export interface BootstrapResult {
  rows: Map<string, BootstrapRow>;
  coefSd: number[];
  draws: number;
}

/**
 * The bootstrap: 200 resamples of the comps, drawn with replacement, each trimmed and refit the
 * same way the full model is, each refit with its own FitContext from its own resample. Failed
 * refits are skipped. Per row: the p10/p90 value band, the p10/p90 rank band and the share of
 * draws the row spent in the top 10. Also the coefficient SDs.
 */
export function bootstrap(
  comps: SnapshotProtocol[],
  rankable: SnapshotProtocol[],
  ctx: FitContext,
): BootstrapResult {
  const rnd = mulberry32(SEED_BOOTSTRAP);
  const drawValues = new Map<string, number[]>();
  const drawRanks = new Map<string, number[]>();
  const top10 = new Map<string, number>();
  for (const p of rankable) {
    drawValues.set(p.slug, []);
    drawRanks.set(p.slug, []);
    top10.set(p.slug, 0);
  }
  // Each draw keeps its own dummy set: a resample may drop a category below five comps, so the
  // coefficient vectors differ in length and the category SDs are matched by name, not by index.
  const coefs: Array<{ beta: number[]; cats: string[] }> = [];
  let draws = 0;
  for (let rep = 0; rep < BOOTSTRAP_REPS; rep++) {
    const sample: SnapshotProtocol[] = [];
    for (let i = 0; i < comps.length; i++) sample.push(comps[Math.floor(rnd() * comps.length)]!);
    const trimmed = trimComps(sample).kept;
    const localCtx = fitContextFor(trimmed, ctx.asOf);
    const beta = fitOLS(trimmed, localCtx, false);
    if (beta === null) continue;
    draws++;
    coefs.push({ beta, cats: localCtx.cats });
    const scored = rankable
      .map((p) => ({ slug: p.slug, value: Math.exp(predictLog(beta, p, localCtx, false)) }))
      .sort((a, b) => b.value - a.value || (a.slug < b.slug ? -1 : 1));
    for (let i = 0; i < scored.length; i++) {
      const s = scored[i]!;
      drawValues.get(s.slug)!.push(s.value);
      drawRanks.get(s.slug)!.push(i + 1);
      if (i < 10) top10.set(s.slug, top10.get(s.slug)! + 1);
    }
  }
  const rows = new Map<string, BootstrapRow>();
  for (const p of rankable) {
    const vals = drawValues.get(p.slug)!;
    const ranks = drawRanks.get(p.slug)!;
    rows.set(p.slug, {
      p10: pointEstimate(Math.log(quantile(vals, 0.1))),
      p90: pointEstimate(Math.log(quantile(vals, 0.9))),
      rankP10: Math.round(quantile(ranks, 0.1)),
      rankP90: Math.round(quantile(ranks, 0.9)),
      top10Pct: draws > 0 ? Math.round((1000 * top10.get(p.slug)!) / draws) / 10 : 0,
    });
  }
  const coefSd: number[] = [];
  if (coefs.length >= 2) {
    // The five base coefficients sit at the same index in every draw; a category's SD is taken
    // over the draws that carried its dummy, with 0 in the draws that did not.
    for (let j = 0; j < 5; j++) coefSd.push(sd(coefs.map((c) => c.beta[j]!)));
    for (const cat of ctx.cats) {
      coefSd.push(
        sd(
          coefs.map((c) => {
            const i = c.cats.indexOf(cat);
            return i >= 0 ? c.beta[5 + i]! : 0;
          }),
        ),
      );
    }
  }
  return { rows, coefSd, draws };
}

// ---------------------------------------------------------------- nearest comps

/** One of a row's five nearest comps: a trading protocol close in size and category. */
export interface NearestComp {
  slug: string;
  name: string;
  category: string | null;
  tvl_usd: number;
  mcap_usd: number;
  multiple: number;
  distance: number;
}

/**
 * The five nearest comps to a protocol: distance is |change in ln TVL| plus 0.75 when the category
 * differs. Ties break by slug, so the list is deterministic.
 */
export function nearestComps(p: SnapshotProtocol, comps: SnapshotProtocol[]): NearestComp[] {
  const lnTvl = Math.log(p.tvl_usd);
  return comps
    .map((c) => ({
      slug: c.slug,
      name: c.name,
      category: c.category,
      tvl_usd: c.tvl_usd,
      mcap_usd: c.mcap_usd!,
      multiple: c.mcap_usd! / c.tvl_usd,
      distance: Math.abs(Math.log(c.tvl_usd) - lnTvl) + (c.category === p.category ? 0 : 0.75),
    }))
    .sort((a, b) => a.distance - b.distance || (a.slug < b.slug ? -1 : 1))
    .slice(0, 5);
}

// ---------------------------------------------------------------- farm plans

/** Typical gas cost of one transaction on each chain, in US dollars. Unlisted chains cost $3. */
export const GAS_USD: Record<string, number> = {
  Ethereum: 8, Arbitrum: 1.5, Base: 0.5, Optimism: 1, zkSync: 1, 'zkSync Era': 1, Blast: 0.5,
  Linea: 0.5, Scroll: 0.5, Mantle: 0.3, Mode: 0.3, Solana: 0.05, BSC: 0.3, Binance: 0.3,
  Polygon: 0.1, Avalanche: 0.5, Berachain: 0.5, Hyperliquid: 0.1, 'Hyperliquid L1': 0.1, Sui: 0.05,
  Aptos: 0.05, Tron: 0.5, Fantom: 0.1, Gnosis: 0.1, Celo: 0.05, Kava: 0.1, Cronos: 0.1,
  Moonbeam: 0.1, Aurora: 0.1, Sei: 0.05, Injective: 0.05, Starknet: 0.5, TON: 0.1, BOB: 0.5,
  Plasma: 0.5, Hemi: 0.5, Katana: 0.5, MegaETH: 0.1, Monad: 0.1, Tempo: 0.5, Stable: 0.1,
  XDC: 0.05, CORE: 0.1, Merlin: 0.3, BSquared: 0.3, Corn: 0.3, Sonic: 0.1, Abstract: 0.3,
  Zora: 0.1, Unichain: 0.3, Ink: 0.3, Sophon: 0.3, TAC: 0.1, Eclipse: 0.05, Movement: 0.05,
  Superseed: 0.3, Lumia: 0.3, Arc: 0.5, Noble: 0.05, Hedera: 0.05, Stacks: 0.3, Tezos: 0.1,
  Near: 0.05, Cardano: 0.2, Algorand: 0.05, Flow: 0.05, MultiversX: 0.05, CosmosHub: 0.05,
  Osmosis: 0.05, dYdX: 0.05, Thorchain: 0.5, Bitcoin: 3, Stellar: 0.05, Flare: 0.1,
  Provenance: 0.05, 'Robinhood Chain': 0.3, Ripple: 0.05, RISE: 0.3, Bittensor: 0.3,
  'Bifrost Network': 0.1, Ronin: 0.1, Manta: 0.3, Proton: 0.05, Mezo: 0.3, Morph: 0.3,
  'Plume Mainnet': 0.3, Qubic: 0.05, Pharos: 0.3, 'World Chain': 0.3, 'Reya Network': 0.3,
  RSK: 0.1, Strato: 0.3, Klaytn: 0.1, Canton: 0.5, Fluent: 0.1, XION: 0.05,
};

/** The cheapest chain a protocol runs on, by the gas table; ties break by name. */
export function cheapestChain(p: SnapshotProtocol): { chain: string; gas: number } {
  if (p.chains.length === 0) return { chain: 'its main chain', gas: 3 };
  let best = p.chains[0]!;
  let bestGas = GAS_USD[best] ?? 3;
  for (const c of p.chains) {
    const g = GAS_USD[c] ?? 3;
    if (g < bestGas || (g === bestGas && c < best)) {
      best = c;
      bestGas = g;
    }
  }
  return { chain: best, gas: bestGas };
}

/** How much capital one farm position needs, from the protocol's own TVL. */
export function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 1_000_000_000) return 200;
  if (p.tvl_usd >= 100_000_000) return 120;
  return 60;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
export function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
function stepText(text: string, cost: string, time: string): string {
  const suffix = ` Cost: ${cost}. Time: ${time}.`;
  let t = text;
  if (t.length + suffix.length > 600) t = `${t.slice(0, 600 - suffix.length - 3)}...`;
  let s = `${t}${suffix}`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

const money = (n: number): string => (n >= 1000 ? `$${Math.round(n).toLocaleString('en-US')}` : `$${n.toFixed(2).replace(/\.00$/, '')}`);

/** The category's one action, in the words a farmer can follow. First match wins. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    return `Supply $${capital} of USDC to ${name} on ${chain} as collateral and keep the position open; borrowing a small amount against it can add activity, but never borrow near the liquidation line.`;
  }
  if (c.startsWith('dex') || c.includes(' dexs') || c === 'dexs') {
    return `Provide $${capital} of concentrated-range liquidity to ${name}'s main pool on ${chain}, set a range around the current price so the position stays in range, and keep it open.`;
  }
  if (c.includes('yield') || c.includes('leveraged farm')) {
    return `Deposit $${capital} into ${name}'s highest-TVL vault on ${chain} and keep the vault shares; the largest vault is usually the one the protocol's own points track.`;
  }
  if (c.includes('bridge') || c.includes('cross chain')) {
    return `Bridge $${capital} of USDC through ${name} to ${chain} and use the bridged funds inside its app, so the activity is more than a single transfer.`;
  }
  if (c.includes('stak')) {
    return `Stake $${capital} of the chain's staking asset through ${name} on ${chain}, keep the receipt token, and check whether the stake earns the protocol's own points.`;
  }
  if (c.includes('perpetual') || c.includes('derivative')) {
    return `Deposit $${capital} of USDC as margin on ${name} and open one small hedged position; keep leverage low and watch the funding rate, which is the real cost of holding.`;
  }
  if (c.includes('prediction')) {
    return `Fund $${capital} on ${name} and take one position in a market you have read, keeping the stake small enough that a wrong call costs little.`;
  }
  if (c.includes('curator') || c.includes('allocator') || c.includes('asset management') || c.includes('risk curator')) {
    return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares; the curator allocates the capital, so read which curator runs the vault before you pick one.`;
  }
  if (c.includes('stable') || c.includes('dual-token')) {
    return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain} and hold it; check the peg and the backing before you mint, and after.`;
  }
  if (c.includes('rwa')) {
    return `Mint $${capital} of ${name}'s yield asset on ${chain} and hold it; some RWA assets need identity checks, so confirm whether one applies before you deposit.`;
  }
  if (c.includes('insurance')) {
    return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy open; cover is the protocol's own product, so holding it is the qualifying activity.`;
  }
  if (c.includes('privacy')) {
    return `Deposit $${capital} into ${name}'s private pool on ${chain} and use it as the protocol intends; keep records of your own deposits in case you must prove them later.`;
  }
  if (c.includes('launchpad')) {
    return `Stake $${capital} through ${name} on ${chain} and join one sale it runs; the stake is usually what qualifies an account for allocations.`;
  }
  if (c.includes('chain')) {
    return `Bridge $${capital} into ${name}'s chain and use its native apps for a few weeks; activity on the chain itself is what a chain's own program tends to count.`;
  }
  if (c.includes('farm')) {
    return `Stake $${capital} of LP tokens in ${name}'s farm on ${chain} and keep the stake open through the reward periods.`;
  }
  if (c.includes('index')) {
    return `Buy $${capital} of ${name}'s index token on ${chain} and hold it; the token is the protocol's product, so holding it is the qualifying activity.`;
  }
  if (c.includes('basis')) {
    return `Open a ${name} delta-neutral position of $${capital} on ${chain} and keep it hedged; the yield comes from the basis, so check the hedge holds before you walk away.`;
  }
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/** The risks that belong to this category, in plain words. */
function risksFor(p: SnapshotProtocol): string {
  const c = (p.category ?? '').toLowerCase();
  let specific: string;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) {
    specific = 'liquidation if your collateral falls, and smart-contract risk in the lending pool';
  } else if (c.startsWith('dex') || c === 'dexs') {
    specific = 'impermanent loss if the price moves out of your range, and smart-contract risk in the pool';
  } else if (c.includes('yield') || c.includes('leveraged farm')) {
    specific = 'strategy risk in the vault, leverage risk where the vault borrows, and smart-contract risk';
  } else if (c.includes('bridge') || c.includes('cross chain')) {
    specific = 'bridge risk: a bridge exploit can strand or lose bridged funds';
  } else if (c.includes('stak')) {
    specific = 'slashing where the stake is slashable, and unstaking delays that lock your capital';
  } else if (c.includes('perpetual') || c.includes('derivative')) {
    specific = 'liquidation on a leveraged position, and funding payments that eat the position over time';
  } else if (c.includes('prediction')) {
    specific = 'losing the position outright when the market resolves against you';
  } else if (c.includes('curator') || c.includes('allocator') || c.includes('asset management') || c.includes('risk curator')) {
    specific = "curator risk: the vault follows one curator's allocation, which can be wrong";
  } else if (c.includes('stable') || c.includes('dual-token')) {
    specific = 'depeg risk if the stable asset loses its peg, and backing risk behind it';
  } else if (c.includes('rwa')) {
    specific = 'issuer and custody risk behind the real-world asset, and redemption delays';
  } else if (c.includes('insurance')) {
    specific = 'cover that pays out only on the terms the policy names, and smart-contract risk';
  } else if (c.includes('privacy')) {
    specific = 'regulatory risk around privacy tools, and smart-contract risk in the pool';
  } else if (c.includes('launchpad')) {
    specific = 'sale allocations that can be small or oversubscribed, and new-token risk';
  } else if (c.includes('chain')) {
    specific = 'chain risk: downtime, upgrades and bridge risk on a young chain';
  } else if (c.includes('farm')) {
    specific = 'reward dilution as more farmers join, and LP risk underneath the stake';
  } else if (c.includes('index')) {
    specific = 'index risk: the basket moves with its holdings, and rebalances can surprise';
  } else if (c.includes('basis')) {
    specific = 'basis risk if the spread collapses, and funding flips that turn the yield negative';
  } else {
    specific = 'smart-contract risk in the protocol itself';
  }
  return `Know the risks before you farm: ${specific}. Points can dilute as more farmers join, Sybil filters can exclude wallets that look farmed, and no airdrop is promised: the plan only puts you in the frame.`;
}

/** A verdict's one-line reason, from the numbers that decided it. */
function verdictReason(verdict: Verdict, band: number, p10: number, net: number, thin: boolean): string {
  if (verdict === 'farm') return `Verdict: farm. The low-end value clears $5M, the band is tight (x${band.toFixed(1)}), and gas is cheap, so the position is worth opening.`;
  if (verdict === 'skip') {
    if (thin && band > 6 && p10 < 5_000_000) return `Verdict: skip. Fewer than three comps back this category, the band is wide (x${band.toFixed(1)}) and the low end is under $5M, so the value is too uncertain to pay for.`;
    return `Verdict: skip. Gas of ${money(net)} is high for a low end under $10M, so the farm costs more than the floor-style case is worth.`;
  }
  return `Verdict: watch. The case is real but not strong enough to farm now: the low end, the band or the gas cost keeps it on the watchlist until one of them improves.`;
}

export type Verdict = 'farm' | 'watch' | 'skip';

/**
 * The verdict rule. Skip when the evidence is thin and the band wide with a low floor, or when gas
 * alone costs more than the floor case is worth; farm when the low end is solid, the band tight and
 * gas cheap; watch otherwise.
 */
export function verdictFor(thin: boolean, band: number, p10: number, net: number): Verdict {
  if ((thin && band > 6 && p10 < 5_000_000) || (net > 50 && p10 < 10_000_000)) return 'skip';
  if (p10 >= 5_000_000 && band <= 5 && net <= 25) return 'farm';
  return 'watch';
}

/** Everything a farm plan needs: the protocol, its age, its bootstrap band and its verdict inputs. */
export interface FarmInput {
  p10: number;
  p90: number;
  band: number;
  verdict: Verdict;
  thin: boolean;
}

/**
 * A protocol's farm plan: eight steps, each an action with a cost and a time. The plan farms on the
 * protocol's cheapest chain by gas, costs seven transactions (2 to fund, 4 weekly, 1 to claim) and
 * prices the whole farm in the last step, with a pro-rata expected value at the low end (p10) and
 * the verdict's reason.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null, input: FarmInput): string[] {
  const { chain, gas } = cheapestChain(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const net = 7 * gas;
  const ev = 0.1 * input.p10 * (capital / p.tvl_usd) - net;
  const action = actionFor(p, capital, chain);
  const gasEach = money(gas);
  const steps: Array<[string, string, string]> = [
    [
      'Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline on paper, never in a screenshot or a note on your phone.',
      '$0, no transaction yet',
      '5 minutes',
    ],
    [
      `Fund the wallet with $${capital} of capital plus gas on ${chain}, the cheapest chain this protocol runs on at about ${gasEach} a transaction.`,
      `$${capital} of capital plus about ${money(2 * gas)} of gas`,
      '10 minutes',
    ],
    [action, `about ${gasEach} of gas`, '15 minutes'],
    [
      'Make one qualifying action each week for four weeks, such as a deposit, a borrow, a swap or a vote, so the wallet shows steady activity rather than one burst.',
      `about ${money(4 * gas)} of gas over four weeks`,
      '10 minutes a week',
    ],
    [
      `Hold the position for at least ${hold} days so the activity lands in more than one snapshot; withdrawing early can leave the wallet out of the count entirely.`,
      `the $${capital} stays in the position`,
      `${hold} days`,
    ],
    [
      `Open ${p.name}'s own points or rewards page every week and confirm your activity counted; if a week shows nothing, make another qualifying action before the next snapshot.`,
      '$0 to check',
      '10 minutes a week',
    ],
    [risksFor(p), 'nothing to check', '10 minutes'],
    [
      `Net economics: 7 transactions on ${chain} cost ${money(net)} in gas and the $${capital} of capital returns on exit, so the break-even allocation is ${money(net)}. At the low-end value of ${money(input.p10)}, a 10% farmer share split pro-rata by capital gives an expected value of ${money(ev)}; a real airdrop is rarely purely pro-rata, so read this as a floor-style check, not a promise. ${verdictReason(input.verdict, input.band, input.p10, net, input.thin)}`,
      `${money(net)} of gas in all`,
      `${hold} days plus the claim`,
    ],
  ];
  return steps.map(([text, cost, time]) => stepText(text, cost, time));
}

// ---------------------------------------------------------------- the board

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  watchlist: boolean;
  implied_value_usd: number;
  p10_usd: number;
  p90_usd: number;
  band: number;
  top10_pct: number;
  rank_p10: number;
  rank_p90: number;
  verdict: Verdict;
  thin_comps: boolean;
  comps_in_category: number;
  net_gas_usd: number;
  capital_usd: number;
  ev_usd: number;
  cheapest_chain: string;
  hold_days: number;
  nearest_comps: string[];
  model: {
    base_multiple: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    breadth_multiplier: number;
    category_multiplier: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** One comp, as the dashboard's nearest-comp table reads it. */
export interface CompRow {
  slug: string;
  name: string;
  category: string | null;
  tvl_usd: number;
  mcap_usd: number;
  multiple: number;
}

/** One row of the category-support table: how much evidence backs each category. */
export interface CategorySupport {
  category: string;
  rankable: number;
  comps: number;
  thin: boolean;
}

/** Everything the dashboard needs: the board, the live set, the model and its validation. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  comps_count: number;
  trimmed_count: number;
  trimmed_slugs: string[];
  median_multiple: number;
  median_age_days: number;
  coefficients: Array<{ name: string; value: number; sd: number | null }>;
  tournament: TournamentRow[];
  selected_model: string;
  validation: ValidationReport;
  size_curve: SizeBand[];
  category_support: CategorySupport[];
  rows: BoardRow[];
  live: LiveRow[];
  comps: CompRow[];
}

/** The headline figures: how many watchlist slugs, how many rows, how many live protocols. */
export interface BoardSummary {
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const round2 = (n: number): number => Math.round(n * 100) / 100;

/** The fitted value's breakdown: base multiple times the growth, maturity, breadth and category multipliers. */
function breakdown(beta: number[], p: SnapshotProtocol, ctx: FitContext): BoardRow['model'] {
  const lnTvl = Math.log(p.tvl_usd);
  const age = ageOf(p, ctx);
  const catIdx = ctx.cats.indexOf(p.category ?? '');
  const catEffect = catIdx >= 0 ? beta[5 + catIdx]! : 0;
  return {
    base_multiple: round4(Math.exp(beta[0]! + (beta[1]! - 1) * lnTvl)),
    growth_multiplier: round4(Math.exp(beta[2]! * growthOf(p))),
    maturity_multiplier: round4(Math.exp(beta[3]! * Math.log(age + 30))),
    breadth_multiplier: round4(Math.exp(beta[4]! * Math.log(Math.max(1, p.chains.length)))),
    category_multiplier: round4(Math.exp(catEffect)),
    age_days: ageDays(ctx.asOf, p.listed_at),
  };
}

/** The live-token protocols, by market cap, for the dashboard's Live tokens view. */
export function liveRows(comps: SnapshotProtocol[]): LiveRow[] {
  return [...comps]
    .sort((a, b) => b.mcap_usd! - a.mcap_usd! || b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .slice(0, LIVE_ROWS)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const comps = compsOf(snapshot);
  const { kept: trainPool, dropped } = trimComps(comps);
  // The full-board fit's context, from the full trimmed pool: the board itself never changes.
  const ctx: FitContext = fitContextFor(trainPool, snapshot.as_of);

  const beta = fitOLS(trainPool, ctx, false);
  if (beta === null) throw new SnapshotError('the regression would not solve on the comps');
  const trainResid = trainPool.map((p) => Math.log(p.mcap_usd!) - predictLog(beta, p, ctx, false));
  const residSd = sd(trainResid);

  const tournament = runTournament(comps, ctx);
  const selected = tournament.find((r) => r.selected) ?? tournament[0]!;
  const selectedDef = CANDIDATES.find((c) => c.id === selected.id)!;
  const validation = validateSelected(selectedDef, comps, ctx, residSd);

  const rankable = snapshot.protocols.filter(isRankable);
  const boot = bootstrap(comps, rankable, ctx);

  const compCounts = new Map<string, number>();
  for (const p of comps) compCounts.set(p.category ?? '', (compCounts.get(p.category ?? '') ?? 0) + 1);

  const watchlist = new Set(snapshot.watchlist);
  const scored = rankable.map((p) => {
    const implied = pointEstimate(predictLog(beta, p, ctx, false));
    const b = boot.rows.get(p.slug)!;
    const band = b.p90 / b.p10;
    const { chain, gas } = cheapestChain(p);
    const net = 7 * gas;
    const capital = capitalFor(p);
    const nComps = compCounts.get(p.category ?? '') ?? 0;
    const thin = nComps < 3;
    const verdict = verdictFor(thin, band, b.p10, net);
    const ev = 0.1 * b.p10 * (capital / p.tvl_usd) - net;
    const age = ageDays(snapshot.as_of, p.listed_at);
    return {
      p,
      implied,
      b,
      band,
      net,
      capital,
      verdict,
      thin,
      nComps,
      ev,
      chain,
      age,
    };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : 1));
  const board: BoardRow[] = scored.map((s, i) => {
    const { p, implied, b, band, net, capital, verdict, thin, nComps, ev, chain, age } = s;
    return {
      rank: i + 1,
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      mcap_usd: p.mcap_usd,
      listed_at: p.listed_at,
      gecko_id: p.gecko_id,
      symbol: p.symbol,
      token_status: 'pre-token',
      age_days: age,
      watchlist: watchlist.has(p.slug),
      implied_value_usd: implied,
      p10_usd: b.p10,
      p90_usd: b.p90,
      band: round2(band),
      top10_pct: b.top10Pct,
      rank_p10: b.rankP10,
      rank_p90: b.rankP90,
      verdict,
      thin_comps: thin,
      comps_in_category: nComps,
      net_gas_usd: round2(net),
      capital_usd: capital,
      ev_usd: round2(ev),
      cheapest_chain: chain,
      hold_days: holdDaysFor(age),
      nearest_comps: nearestComps(p, comps).map((c) => c.slug),
      model: breakdown(beta, p, ctx),
      farm_steps: farmSteps(p, age, { p10: b.p10, p90: b.p90, band, verdict, thin }),
    };
  });

  const rankableCats = new Map<string, number>();
  for (const p of rankable) rankableCats.set(p.category ?? '(none)', (rankableCats.get(p.category ?? '(none)') ?? 0) + 1);
  const categorySupport: CategorySupport[] = [...rankableCats.entries()]
    .map(([category, rankableCount]) => {
      const n = compCounts.get(category === '(none)' ? '' : category) ?? 0;
      return { category, rankable: rankableCount, comps: n, thin: n < 3 };
    })
    .sort((a, b) => b.rankable - a.rankable || (a.category < b.category ? -1 : 1));

  const names = featureNames(ctx, false);
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: board.filter((r) => !r.watchlist).length,
    live_size: LIVE_ROWS,
    comps_count: comps.length,
    trimmed_count: dropped.length,
    trimmed_slugs: dropped.map((p) => p.slug).sort(),
    median_multiple: round4(medianMultiple(trainPool)),
    median_age_days: ctx.medianAge,
    coefficients: names.map((name, i) => ({ name, value: round4(beta[i]!), sd: boot.coefSd[i] !== undefined ? round4(boot.coefSd[i]!) : null })),
    tournament,
    selected_model: selectedDef.name,
    validation,
    size_curve: sizeCurve(comps),
    category_support: categorySupport,
    rows: board,
    live: liveRows(comps),
    comps: comps
      .map((p) => ({ slug: p.slug, name: p.name, category: p.category, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd!, multiple: round4(p.mcap_usd! / p.tvl_usd) }))
      .sort((a, b) => a.slug < b.slug ? -1 : 1),
  };
  return { board, radar };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) {
      const t = s.trim();
      if (t.length < 20 || t.length > 600) return `${r.slug} has a farm step of ${t.length} characters`;
    }
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
