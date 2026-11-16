# Token Radar — token-radar-01

A token launch radar: it ranks every pre-token protocol in the snapshot by an implied value —
what a token would be worth if the protocol's record stayed as it is — and gives each one a
costed farm plan. It does not predict whether or when a token launches.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>
```

Node 22.18+ (24 works), standard library only, no network. The program reads the snapshot,
writes `board.json`, `data.js`, `dashboard/` and `screenshots/` into the output directory, and
last writes `dossier.json`. Two runs write identical bytes.

## What it builds

- **198 rows**: all 40 watchlist protocols plus 158 early rows below the $50M floor — every
  rankable pre-token protocol in the snapshot (records no token, not a wrapped product).
- **The model, radar-4**: a log-linear regression on the 92 trading comps (ln market cap on
  ln TVL, 7-day growth, ln(age+30), ln chain count, plus shrunk category dummies), fit by OLS
  with two Huber reweightings. One 400× outlier trimmed. No median-multiple blend.
- **A tournament, run live**: 10 candidate models scored by leave-one-out, a time split and
  ten random splits; the lowest mean median-absolute-log-error wins. OLS blend 0 won
  (mean MALE 1.14). Every fold recomputes its category dummies and median age from its own
  training pool, so no test protocol leaks into the fit it is scored against.
- **A 200-draw bootstrap** puts a p10–p90 value band, a rank band and a top-10 stability share
  on every row.
- **Farm plans**: 8 steps each, farmed on the protocol's cheapest chain by gas, with net gas,
  capital, break-even allocation, a pro-rata EV at the low end, and a farm / watch / skip
  verdict from a stated rule.

Top of the board: Sentora Curator ≈ $49M, Steakhouse Financial ≈ $40M, USDD ≈ $35M,
Gauntlet ≈ $32M, OnRe ≈ $31M. Verdicts: 30 farm / 101 watch / 67 skip.

## Layout

- `main.ts` — entrypoint: read, score, write board + data + dashboard + dossier.
- `lib/model.ts` — the model, tournament, validation, bootstrap, farm plans, board.
- `vendor/makefun-board.ts` — make.fun's dossier library, byte-identical to the kit's.
- `dashboard/` — the static page (index.html, app.css, app.js); no network, reads `data.js`.
- `screenshots/` — the three submission screenshots.
- `METHODOLOGY.md` — the model, the validation and the farm economics, in full.

## Verification (16 Nov 2026, snapshot as_of 2026-11-14)

- Two runs into two directories: byte-identical outputs (`diff -r` clean). Runtime ~1s.
- Board: 198 rows, 40/40 watchlist, ranks 1–198 each once, implied values non-increasing,
  all 1,584 farm steps within 20–600 characters, top 3 within 1.3% of $49M / $40M / $35M.
  The per-fold-context fix left every implied value and rank unchanged (0 of 198 rows moved);
  only the bootstrap bands — and fourteen verdicts sitting at the band thresholds — moved.
- Dashboard source (index.html + app.css + app.js): 33,428 bytes — the dossier shows it whole,
  nothing cut. No network addresses anywhere in the dashboard source.
- Screenshots: 3 JPEGs, 1440×900, 278–286KB each. Dossier: 1,167,905 bytes (cap 1,572,864).
- The blind holdout is repeated, not a single draw: the median across 10 seeded 74/18 draws
  is 86% median error (MALE 1.16, within 2× 72%), with the median error across draws ranging
  55%–363%. One 18-protocol draw is too noisy to be a headline number, so the dashboard and
  METHODOLOGY.md show the median and the range rather than any one draw.

## Validation (computed at runtime, shown on the dashboard)

Leave-one-out median error 89% (MALE 1.24, within 2× 67%); time split median error 62%
(MALE 0.94, within 2× 83%); random splits median error 86% (MALE 1.27, within 2× 66%);
repeated blind holdout (median of 10 seeded 74/18 draws, never fit) median error 86%
(MALE 1.16, within 2× 72%), with the median error across draws ranging 55%–363%. The 80%
band covers 75% of comps, the 50% band 49%. In-sample log R² 0.17: launch pricing is noisy,
and the band is the honest answer.
