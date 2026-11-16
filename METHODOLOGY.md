# Methodology — Token Radar, model radar-4

## The question

For every protocol in the snapshot that records no token, what would a token be worth if the
protocol's record stayed as it is? The board ranks the answer, and prices a farm plan for each
row. It does not predict whether or when a token launches.

## The universe

- **Comps (92)**: protocols with a market cap and a TVL that are not wrapped products. These
  are the market's own evidence for what TVL is worth.
- **Rankable (198)**: protocols that record no token (market cap, CoinGecko id and symbol all
  null) and are not wrapped products. All 40 watchlist protocols plus 158 early rows.
- **Wrapped product**: category in {Bridge, Canonical Bridge, Liquid Staking, Liquid Restaking,
  Restaked BTC, CEX}; or a name ending in a version marker (`V2`, `v2`, `2.0`, ...); or a name
  carrying a whole word `wrapped` / `staked` / `staking`; or a name starting with an exchange
  name (Binance, Coinbase, Bybit, Kraken, Bitget, OKX, Bitfinex, Gemini).

## The model

Implied value is a log-linear regression on the comps:

```
ln(mcap) = b0 + b1·ln(TVL) + b2·growth + b3·ln(age+30) + b4·ln(chains) + category effects
```

- `growth` = clamp(change_7d, −50, 100) / 100, unreported → 0.
- `age` = days from `listed_at` to the snapshot date, unreported → the comps' median age
  (859 days).
- `chains` = ln(max(1, number of chains)).
- Category dummies for categories with ≥ 5 comps: Lending, Dexs, RWA, Yield, Basis Trading.

Fit: OLS on log(mcap), then two Huber reweightings (k = 1.345σ, σ = 1.4826 × MAD of residuals,
floored at 0.05). Category effects shrunk by n/(n+5). Solved by Gaussian elimination with
partial pivoting, Node standard library only.

**Trim.** Comps whose log(mcap/TVL) sits beyond 3 robust SDs (1.4826 × MAD, floored at 0.25)
from the median log multiple are dropped from every training set. Exactly one is trimmed:
`rain`, at a 400× multiple. The training pool is 91 comps.

**Point estimate.** exp(fitted log value), capped at $10¹⁴, floored at $1. No median-multiple
blend: the tournament tested blends and the unblended fit won.

Fitted coefficients (this snapshot): intercept 9.18, ln(TVL) 0.39, growth 0.54, ln(age+30)
0.05, ln(chains) −0.19; Lending −0.82, Dexs 0.06, RWA 0.14, Yield −0.37, Basis Trading 0.42.
In-sample log R² 0.17 (all 92 comps), residual SD 1.76 (91-comp fit).

**Per-fold context.** The category dummy set and the median age are not fixed once for the
whole program: every fit on a subset — every tournament fold, every holdout draw, every
bootstrap refit — recomputes both from its own training pool. A fold's dummy set is decided
by its training comps only, so no test protocol leaks into the fit it is scored against.
Only the full-board fit uses the context from the full trimmed pool.

## The tournament

Ten candidates, scored live at runtime by leave-one-out over the 92 comps, a time split (fit
on the older-listed half of the 70 dated comps, test on the newer half) and ten random 50/50
splits (mulberry32 seed 7). Metric: median absolute log error (MALE) per test; the winner has
the lowest mean MALE. Training pools are trimmed by the same rule everywhere; test pools are
never trimmed. Each fold recomputes its category dummies and median age from its own training
pool.

| Candidate | LOOCV | Time | Random | Mean |
|---|---|---|---|---|
| OLS, blend 0 — **selected** | 1.242 | 0.941 | 1.244 | **1.142** |
| OLS, blend 0.25 | 1.211 | 0.967 | 1.287 | 1.155 |
| OLS + (ln TVL)², blend 0.25 | 1.221 | 0.992 | 1.305 | 1.173 |
| OLS + (ln TVL)², blend 0 | 1.316 | 0.954 | 1.255 | 1.175 |
| OLS, TVL-weighted | 1.330 | 0.878 | 1.348 | 1.186 |
| OLS, blend 0.5 | 1.348 | 0.996 | 1.330 | 1.225 |
| kNN k=10 | 1.356 | 1.302 | 1.411 | 1.356 |
| Band curve | 1.374 | 1.402 | 1.474 | 1.417 |
| kNN k=15 | 1.315 | 1.563 | 1.479 | 1.452 |
| kNN k=5 | 1.631 | 1.741 | 1.505 | 1.626 |

kNN standardizes [ln TVL, growth, ln age, ln chains], halves the distance for same-category
comps, and predicts the median multiple of the k nearest. The band curve takes the median
multiple per ln-TVL band, smoothed with its neighbours.

## Validation (selected model)

| Test | n | Median APE | MALE | Within 2× |
|---|---|---|---|---|
| Leave-one-out | 92 | 0.89 | 1.24 | 67% |
| Time split | 35 | 0.62 | 0.94 | 83% |
| Random splits (10 × 46) | 460 | 0.86 | 1.27 | 66% |
| Repeated blind holdout (median of 10 draws) | 10 × 18 | 0.86 | 1.16 | 72% |

"Within 2×" means |predicted − true| / true ≤ 1. The blind holdout is repeated: ten seeded
shuffles (mulberry32 seeds 20261114–20261123), 74 train / 18 test each, never used in fitting
or selection, and the table reports the median across the ten draws. A single 18-protocol
draw is too noisy to be a headline number — the median error across the ten draws ranged
0.55–3.63 — so the dashboard shows the median and the range, not any one draw.

**Band calibration.** The LOOCV parametric band exp(fit ± 1.28σ) covers 75% of comps (target
80%); ±0.67σ covers 49% (target 50%). The bands are close to honest, a touch narrow at 80%.

**Size curve.** Median mcap/TVL by TVL band: $5–20M 0.57 (n=42), $20–100M 0.24 (n=33),
$100–500M 0.13 (n=9), $500M–2B 0.02 (n=5), >$2B 0.02 (n=3). Small protocols trade at far
higher multiples — the regression's ln(TVL) coefficient below 1 is this curve in one number.

## The bootstrap

200 resamples of the comps, drawn with replacement (mulberry32 seed 20261114), each trimmed
and refit the same way the full model is, each refit with its own category dummies and median
age from its own resample; failed refits are skipped. Per row: the p10/p90 value band, the
p10/p90 rank band, and the share of draws the row spent in the top 10.
Rank 1 (Sentora Curator) is in the top 10 in 73.5% of draws. Coefficient SDs come from the
same refits, matched by coefficient name — a resample that drops a category below five comps
carries no dummy for it, so that draw counts as a zero for the category's SD.

## Nearest comps

Five per row. Distance = |Δ ln TVL| + 0.75 when the category differs. The 92-comp table is
stored once in `data.js`; each row carries five slugs.

## Farm plans

Eight steps per row, each 20–600 characters, each ending with a cost and a time.

- **Cheapest chain.** Every plan farms on the protocol's cheapest chain by the gas table
  (Solana $0.05, Polygon $0.10, Base $0.50, Ethereum $8; unlisted chains $3). The rule is
  stated on the page.
- **Net gas** = 7 transactions × cheapest-chain gas (2 to fund, 4 weekly qualifying actions,
  1 to claim). Median net gas across the board: $0.70.
- **Capital**: $200 at TVL ≥ $1B, $120 at ≥ $100M, else $60.
- **Hold**: 60 days if the protocol is under 180 days old, 45 under a year, else 30;
  45 when the age is unknown.
- **Step 3 is mechanism-aware**: the action is matched on the category (lending → supply,
  Dexs → concentrated-range LP, yield → highest-TVL vault, bridge → bridge and use, staking →
  stake, perps → hedged margin, prediction → market position, curator → vault shares,
  stablecoins → mint/deposit, RWA → mint yield asset, insurance → cover, privacy → private
  pool, launchpad → stake and join, chain → bridge in and use native dapps, farm → stake LP,
  indexes → buy the index token, basis trading → delta-neutral).
- **Step 7 names the category's risks** (liquidation, depeg, impermanent loss, bridge risk,
  points dilution, Sybil filters).
- **Step 8 prices the farm**: net gas, capital (returns on exit), break-even allocation
  (= net gas), and a pro-rata EV at the low end.

**Pro-rata EV** = 0.10 × p10 × (capital / TVL) − net gas. It assumes 10% of the low-end
implied value goes to farmers, split pro-rata by capital share, at the low end of the band.
Real airdrops are rarely purely pro-rata, so the EV is a floor-style check, not a promise.
EV is positive for 71 of 198 rows.

**Verdict rule.** Skip when (thin comps — fewer than 3 in the category — and band > 6× and
p10 < $5M) or (net gas > $50 and p10 < $10M). Otherwise farm when p10 ≥ $5M and band ≤ 5×
and net gas ≤ $25. Everything else is watch. Result: 30 farm / 101 watch / 67 skip. The
thin-comps flag shows on every thin row regardless of verdict. The counts moved from
36/101/61 when the bootstrap refits started recomputing their own contexts: a resample's
dummy set and median age now vary across draws, which widens the p10–p90 bands (median band
6.3× → 6.8×) and pushes fourteen rows across the band-5× and band-6× thresholds. The wider
bands are the honest ones — the old refits understated uncertainty by holding the dummy set
and median age fixed at their full-sample values.

## Determinism

Fixed seeds everywhere (mulberry32), no `Date.now`, no unseeded random, stable sort orders
with slug tie-breaks. Two runs write identical bytes. Runtime is about one second.

Output sizes on this snapshot: `board.json` 580,942 bytes, `data.js` 602,392 bytes, dashboard
source 33,428 bytes (the dossier's text cap is 65,536, so the judge reads the whole page
source), three screenshots of 278–286KB each, `dossier.json` 1,167,905 bytes (cap 1,572,864).

## Limitations

In-sample log R² is 0.17: TVL, growth, age and breadth explain little of why one token trades
at 0.02× TVL and another at 20×. Team, narrative, timing and unlock schedules are not in this
snapshot, and no regression on four features can price them. That is why every row carries a
band, not just a point: the band is the honest answer. The board does not predict whether or
when a token launches.
