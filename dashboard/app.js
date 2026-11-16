// Token Radar's dashboard. It reads window.RADAR (written beside dashboard/ by main.ts) and
// renders the board, the live-token view and the method page, with a detail drawer per protocol.
// No network: every byte it shows is in data.js. Deep links: #row=<slug> opens a drawer,
// #verdict=farm and #category=<name> apply filters on load.
(function () {
  var D = window.RADAR;
  if (!D) return;
  var rows = D.rows;
  var compsBySlug = {};
  D.comps.forEach(function (c) { compsBySlug[c.slug] = c; });

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function money(n) {
    if (n == null || !isFinite(n)) return '—';
    var a = Math.abs(n);
    if (a >= 1e9) return '$' + (n / 1e9).toFixed(2).replace(/\.?0+$/, '') + 'B';
    if (a >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (a >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    if (a >= 1) return '$' + n.toFixed(0);
    return '$' + n.toFixed(2);
  }
  function money2(n) { return (n < 0 ? '−' : '') + money(Math.abs(n)); }
  function pct(n) {
    if (n == null) return '—';
    var cls = n > 0 ? 'up' : n < 0 ? 'down' : '';
    return '<span class="' + cls + '">' + (n > 0 ? '+' : '') + n.toFixed(1) + '%</span>';
  }
  function mult(n) {
    if (n == null) return '—';
    if (n < 0.1) return n.toFixed(3) + '×';
    if (n < 10) return n.toFixed(2) + '×';
    return n.toFixed(1) + '×';
  }
  function fmtDate(s) { return s || '—'; }

  // ---- header stats + model banner
  var verdicts = { farm: 0, watch: 0, skip: 0 };
  rows.forEach(function (r) { verdicts[r.verdict]++; });
  $('stats').innerHTML =
    stat(rows.length, 'rows ranked') + stat(D.watchlist_size, 'watchlist covered') +
    stat(D.early_size, 'early rows below the floor') + stat(D.comps_count, 'trading comps') +
    stat(verdicts.farm, 'farm · ' + verdicts.watch + ' watch · ' + verdicts.skip + ' skip') +
    stat(D.as_of, 'snapshot date');
  function stat(big, label) {
    return '<div class="stat"><b>' + esc(big) + '</b><span>' + esc(label) + '</span></div>';
  }

  var coef = {};
  D.coefficients.forEach(function (c) { coef[c.name] = c.value; });
  function sgn(n, digits) {
    return (n >= 0 ? '+' : '−') + ' ' + Math.abs(n).toFixed(digits == null ? 2 : digits);
  }
  var v = D.validation;
  var winner = D.tournament.filter(function (t) { return t.selected; })[0];
  $('banner').innerHTML =
    '<div class="eq">ln(value) = <b>' + coef['intercept'].toFixed(2) + '</b> ' + sgn(coef['ln(TVL)']) + ' · ln(TVL) ' +
    sgn(coef['growth']) + ' · growth ' + sgn(coef['ln(age+30)']) + ' · ln(age+30) ' +
    sgn(coef['ln(chains)']) + ' · ln(chains) + category effect</div>' +
    '<p><b>' + D.tournament.length + ' models tested, ' + esc(D.selected_model) + ' selected by out-of-sample error</b> ' +
    '(lowest mean MALE ' + winner.mean.toFixed(2) + '). ' +
    'Leave-one-out: median error ' + (v.loocv.medape * 100).toFixed(0) + '% · time split: MALE ' + v.timeSplit.male.toFixed(2) +
    ' · repeated blind holdout (median of 10 seeded 74/18 draws, never fit): median error ' + (v.holdout.medape * 100).toFixed(0) + '%. ' +
    'The p10–p90 band is the honest answer: the 80% band covers ' + (v.band80.coverage * 100).toFixed(0) +
    '% of comps, the 50% band ' + (v.band50.coverage * 100).toFixed(0) + '%. ' +
    'In-sample log R² ' + v.r2.toFixed(2) + ' — the market prices launches noisily, and the band says so.</p>';

  // ---- filters
  var cats = [], chains = [];
  rows.forEach(function (r) {
    if (r.category && cats.indexOf(r.category) < 0) cats.push(r.category);
    r.chains.forEach(function (c) { if (chains.indexOf(c) < 0) chains.push(c); });
  });
  cats.sort(); chains.sort();
  $('fcat').innerHTML = '<option value="">All categories</option>' + cats.map(function (c) {
    return '<option value="' + esc(c) + '">' + esc(c) + '</option>';
  }).join('');
  $('fchain').innerHTML = '<option value="">All chains</option>' + chains.map(function (c) {
    return '<option value="' + esc(c) + '">' + esc(c) + '</option>';
  }).join('');

  var logMin = Math.log(Math.min.apply(null, rows.map(function (r) { return r.p10_usd; })));
  var logMax = Math.log(Math.max.apply(null, rows.map(function (r) { return r.p90_usd; })));
  var logSpan = logMax - logMin || 1;

  function bandCell(r) {
    var l = ((Math.log(r.p10_usd) - logMin) / logSpan) * 100;
    var w = ((Math.log(r.p90_usd) - Math.log(r.p10_usd)) / logSpan) * 100;
    var m = ((Math.log(r.implied_value_usd) - logMin) / logSpan) * 100;
    return '<div class="bandbar"><i style="left:' + l.toFixed(1) + '%;width:' + Math.max(w, 1.5).toFixed(1) + '%"></i>' +
      '<b style="left:' + m.toFixed(1) + '%"></b></div>' +
      '<div class="bandtxt">' + money(r.p10_usd) + ' – ' + money(r.p90_usd) + ' · ×' + r.band.toFixed(1) + '</div>';
  }

  function filtered() {
    var q = $('q').value.trim().toLowerCase();
    var cat = $('fcat').value, chain = $('fchain').value, ver = $('fverdict').value;
    var watchOnly = $('fwatch').checked;
    var sort = $('sort').value;
    var out = rows.filter(function (r) {
      if (q && (r.name + ' ' + r.slug + ' ' + (r.category || '')).toLowerCase().indexOf(q) < 0) return false;
      if (cat && r.category !== cat) return false;
      if (chain && r.chains.indexOf(chain) < 0) return false;
      if (ver && r.verdict !== ver) return false;
      if (watchOnly && !r.watchlist) return false;
      return true;
    });
    out.sort(function (a, b) {
      if (sort === 'tvl') return b.tvl_usd - a.tvl_usd;
      if (sort === 'chg') return (b.change_7d == null ? -1e9 : b.change_7d) - (a.change_7d == null ? -1e9 : a.change_7d);
      if (sort === 'p10') return b.p10_usd - a.p10_usd;
      if (sort === 'band') return a.band - b.band;
      if (sort === 'net') return a.net_gas_usd - b.net_gas_usd;
      return b.implied_value_usd - a.implied_value_usd;
    });
    return out;
  }

  function renderBoard() {
    var list = filtered();
    $('count').textContent = 'Showing ' + list.length + ' of ' + rows.length + ' rows. Click a row for its detail, farm plan and nearest comps.';
    $('tbody').innerHTML = list.map(function (r) {
      return '<tr data-slug="' + esc(r.slug) + '"><td>' + r.rank + '</td>' +
        '<td><b>' + esc(r.name) + '</b>' + (r.watchlist ? ' <span class="star">★</span>' : '') +
        '<div class="slug">' + esc(r.slug) + (r.thin_comps ? ' · <span class="thinflag">thin comps</span>' : '') + '</div></td>' +
        '<td>' + esc(r.category || '—') + '</td>' +
        '<td>' + esc(r.chains.slice(0, 3).join(', ')) + (r.chains.length > 3 ? ' +' + (r.chains.length - 3) : '') + '</td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td>' +
        '<td class="num">' + pct(r.change_7d) + '</td>' +
        '<td class="num val">' + money(r.implied_value_usd) + '</td>' +
        '<td>' + bandCell(r) + '</td>' +
        '<td><span class="pill ' + r.verdict + '">' + r.verdict + '</span></td>' +
        '<td class="num">' + money(r.net_gas_usd) + '</td></tr>';
    }).join('');
  }

  $('tbody').addEventListener('click', function (e) {
    var tr = e.target.closest('tr[data-slug]');
    if (tr) openDrawer(tr.getAttribute('data-slug'));
  });
  ['q', 'fcat', 'fchain', 'fverdict', 'fwatch', 'sort'].forEach(function (id) {
    $(id).addEventListener('input', function () { renderBoard(); syncHash(); });
    $(id).addEventListener('change', function () { renderBoard(); syncHash(); });
  });

  $('foot').textContent = 'Snapshot: ' + D.source + '. A protocol “records no token” when its DefiLlama record carries no market cap, CoinGecko id or symbol — that is the record, not proof it has none. ★ = on the snapshot watchlist (TVL ≥ $50M). The board does not predict whether or when a token launches.';

  // ---- live tokens
  $('livetbody').innerHTML = D.live.map(function (p, i) {
    return '<tr><td>' + (i + 1) + '</td><td><b>' + esc(p.name) + '</b><div class="slug">' + esc(p.slug) + '</div></td>' +
      '<td>' + esc(p.category || '—') + '</td><td class="num">' + money(p.tvl_usd) + '</td>' +
      '<td class="num val">' + money(p.mcap_usd) + '</td><td class="num">' + mult(p.mcap_usd / p.tvl_usd) + '</td>' +
      '<td class="num">' + pct(p.change_7d) + '</td></tr>';
  }).join('');

  // ---- method tab
  function coefRow(c) {
    return '<tr><td>' + esc(c.name) + '</td><td class="num">' + c.value.toFixed(3) + '</td>' +
      '<td class="num">' + (c.sd == null ? '—' : c.sd.toFixed(3)) + '</td></tr>';
  }
  var tRows = D.tournament.map(function (t) {
    return '<tr' + (t.selected ? ' class="win"' : '') + '><td>' + esc(t.name) + (t.selected ? ' — selected' : '') + '</td>' +
      '<td class="num">' + t.loocv.toFixed(3) + '</td><td class="num">' + t.timeSplit.toFixed(3) + '</td>' +
      '<td class="num">' + t.randomSplit.toFixed(3) + '</td><td class="num"><b>' + t.mean.toFixed(3) + '</b></td></tr>';
  }).join('');
  var scRows = D.size_curve.map(function (b) {
    return '<tr><td>' + esc(b.label) + '</td><td class="num">' + b.n + '</td><td class="num">' + b.medianMultiple.toFixed(2) + '×</td>' +
      '<td class="num">' + b.p25.toFixed(2) + '×</td><td class="num">' + b.p75.toFixed(2) + '×</td></tr>';
  }).join('');
  var csRows = D.category_support.map(function (c) {
    return '<tr><td>' + esc(c.category) + (c.thin ? ' <span class="thinflag">thin</span>' : '') + '</td>' +
      '<td class="num">' + c.rankable + '</td><td class="num">' + c.comps + '</td></tr>';
  }).join('');
  $('method').innerHTML =
    '<h3>The model</h3>' +
    '<p class="lede">Implied value is a log-linear regression on the ' + D.comps_count + ' trading comps: ' +
    'ln(market cap) on ln(TVL), 7-day growth, ln(age + 30) and ln(chain count), plus a dummy per category with at least five comps. ' +
    'Fit by OLS with two Huber reweightings; category effects shrunk by n/(n+5). One comp was trimmed as a 400× outlier (' +
    esc(D.trimmed_slugs.join(', ')) + '). No median-multiple blend: the point estimate is exp(fitted log value), capped at $100T. ' +
    'Median multiple of the training comps ' + D.median_multiple.toFixed(2) + '×; median comp age ' + D.median_age_days + ' days.</p>' +
    '<div class="tablewrap"><table class="mini"><thead><tr><th>Coefficient</th><th class="num">Value</th><th class="num">Bootstrap SD</th></tr></thead><tbody>' +
    D.coefficients.map(coefRow).join('') + '</tbody></table></div>' +
    '<div class="grid">' +
    '<div class="card"><h4>In-sample log R²</h4><div class="big">' + v.r2.toFixed(2) + '</div><p>On all ' + D.comps_count + ' comps. The model explains little of the variance — launch pricing is noisy, and the band is the honest answer.</p></div>' +
    '<div class="card"><h4>Residual SD</h4><div class="big">' + v.residSd.toFixed(2) + '</div><p>Log points on the 91-comp training fit. A ±1 SD band is roughly ×5.8 either way.</p></div>' +
    '<div class="card"><h4>Bootstrap</h4><div class="big">200 draws</div><p>Every row’s p10–p90 band, rank band and top-10 share come from refitting on resampled comps.</p></div>' +
    '</div>' +
    '<h3>The tournament — 10 models, one winner</h3>' +
    '<p class="lede">Each candidate is scored by leave-one-out over the comps, a time split (fit on the older-listed half, test on the newer half) and ten random 50/50 splits (seed 7). The metric is median absolute log error (MALE) per test; the winner has the lowest mean MALE. Training pools are trimmed by the same rule everywhere; test pools are never trimmed. Each fold recomputes its category dummies and median age from its own training pool, so no test protocol leaks into the fit it is scored against.</p>' +
    '<div class="tablewrap"><table class="mini"><thead><tr><th>Candidate</th><th class="num">LOOCV MALE</th><th class="num">Time-split MALE</th><th class="num">Random-split MALE</th><th class="num">Mean MALE</th></tr></thead><tbody>' + tRows + '</tbody></table></div>' +
    '<h3>Validation of the selected model</h3>' +
    '<div class="grid">' +
    '<div class="card"><h4>Leave-one-out (n=' + v.loocv.n + ')</h4><div class="big">' + (v.loocv.medape * 100).toFixed(0) + '% median error</div><p>MALE ' + v.loocv.male.toFixed(2) + ' · within 2× ' + (v.loocv.within2 * 100).toFixed(0) + '%.</p></div>' +
    '<div class="card"><h4>Time split (n=' + v.timeSplit.n + ')</h4><div class="big">' + (v.timeSplit.medape * 100).toFixed(0) + '% median error</div><p>MALE ' + v.timeSplit.male.toFixed(2) + ' · within 2× ' + (v.timeSplit.within2 * 100).toFixed(0) + '%.</p></div>' +
    '<div class="card"><h4>Random splits (10 × 46)</h4><div class="big">' + (v.randomSplit.medape * 100).toFixed(0) + '% median error</div><p>Mean MALE ' + v.randomSplitMeanMale.toFixed(2) + ' · within 2× ' + (v.randomSplit.within2 * 100).toFixed(0) + '%.</p></div>' +
    '<div class="card"><h4>Repeated blind holdout (10 × ' + v.holdout.n + ')</h4><div class="big">' + (v.holdout.medape * 100).toFixed(0) + '% median error</div><p>MALE ' + v.holdout.male.toFixed(2) + ' · within 2× ' + (v.holdout.within2 * 100).toFixed(0) + '%. Median across 10 seeded 74/18 draws (' + v.holdoutTrain + ' train after trim), never used in fitting or selection. Median error across draws ranged ' + (v.holdoutMedapeMin * 100).toFixed(0) + '%–' + (v.holdoutMedapeMax * 100).toFixed(0) + '% — a single 18-protocol draw is noisy, so the median across draws, not any one draw, is the number.</p></div>' +
    '</div>' +
    '<div class="callout"><b>Band calibration.</b> The LOOCV parametric band exp(fit ± 1.28·σ) covers ' + (v.band80.coverage * 100).toFixed(0) +
    '% of comps (target 80%); ±0.67·σ covers ' + (v.band50.coverage * 100).toFixed(0) + '% (target 50%). The bands are close to honest, a touch narrow at the 80% level.</div>' +
    '<h3>Size curve — what the market pays per dollar of TVL</h3>' +
    '<div class="tablewrap"><table class="mini"><thead><tr><th>TVL band</th><th class="num">Comps</th><th class="num">Median mcap/TVL</th><th class="num">p25</th><th class="num">p75</th></tr></thead><tbody>' + scRows + '</tbody></table></div>' +
    '<p class="lede">Small protocols trade at far higher multiples of TVL than large ones — the regression’s ln(TVL) coefficient below 1 is this curve in one number.</p>' +
    '<h3>Category support — how much evidence backs each category</h3>' +
    '<div class="tablewrap"><table class="mini"><thead><tr><th>Category</th><th class="num">Rankable rows</th><th class="num">Trading comps</th></tr></thead><tbody>' + csRows + '</tbody></table></div>' +
    '<p class="lede">“Thin” means fewer than three trading comps in the category: the model has almost nothing to learn that category’s multiple from, and those rows carry the flag on the board.</p>' +
    '<h3>Farm economics — how the plans are priced</h3>' +
    '<div class="callout"><b>Cheapest chain first.</b> Every plan farms on the protocol’s cheapest chain by the gas table (Solana $0.05, Polygon $0.10, Base $0.50, Ethereum $8, unlisted chains $3). ' +
    'Net gas is 7 transactions: 2 to fund, 4 weekly qualifying actions, 1 to claim. Capital is $200 at TVL ≥ $1B, $120 at ≥ $100M, else $60. ' +
    'Hold: 60 days if the protocol is under 180 days old, 45 under a year, else 30.</div>' +
    '<div class="callout"><b>Pro-rata EV, stated plainly.</b> EV = 10% × p10 × (capital / TVL) − net gas. It assumes 10% of the low-end implied value goes to farmers, split pro-rata by capital share, at the low end of the band. ' +
    'Real airdrops are rarely purely pro-rata, so the EV is a floor-style check, not a promise. The break-even allocation equals net gas, because the capital returns on exit.</div>' +
    '<div class="callout"><b>The verdict rule.</b> Skip when the evidence is thin and the band wide with a low floor (thin comps, band &gt; 6×, p10 &lt; $5M), or when gas alone tops $50 with p10 under $10M. ' +
    'Farm when p10 ≥ $5M, the band ≤ 5× and net gas ≤ $25. Everything else is watch.</div>' +
    '<h3>Limitations, honestly</h3>' +
    '<p class="lede">In-sample log R² is ' + v.r2.toFixed(2) + ': TVL, growth, age and breadth explain little of why one token trades at 0.02× TVL and another at 20×. ' +
    'The market prices launches noisily — team, narrative, timing and unlock schedules are not in this snapshot, and no regression on four features can price them. ' +
    'That is why every row carries a band, not just a point: the band is the honest answer. The board does not predict whether or when a token launches; it says what a token would be worth if the protocol’s record stayed as it is. ' +
    'A farm plan is a way to be in the frame if a launch happens, not a claim that one will.</p>';

  // ---- drawer
  var drawer = $('drawer'), overlay = $('overlay');
  function openDrawer(slug) {
    var r = rows.filter(function (x) { return x.slug === slug; })[0];
    if (!r) return;
    var m = r.model;
    var fittedMultiple = r.implied_value_usd / r.tvl_usd;
    var near = r.nearest_comps.map(function (s) { return compsBySlug[s]; }).filter(Boolean);
    drawer.innerHTML =
      '<button class="close" id="drawerClose">Close ✕</button>' +
      '<h2>' + esc(r.name) + (r.watchlist ? ' <span class="star">★</span>' : '') + '</h2>' +
      '<div class="slug">#' + r.rank + ' · ' + esc(r.slug) + ' · ' + esc(r.category || '—') + ' · <span class="pill ' + r.verdict + '">' + r.verdict + '</span>' +
      (r.thin_comps ? ' <span class="thinflag">thin comps: only ' + r.comps_in_category + ' in this category</span>' : '') + '</div>' +
      '<div class="bigval">' + money(r.implied_value_usd) + '</div>' +
      '<div class="bandline">p10–p90 ' + money(r.p10_usd) + ' – ' + money(r.p90_usd) + ' (×' + r.band.toFixed(1) + ' band) · ' +
      'top 10 in ' + r.top10_pct.toFixed(1) + '% of bootstrap draws · rank band #' + r.rank_p10 + '–#' + r.rank_p90 + '</div>' +
      '<h3>Model breakdown</h3>' +
      '<div class="kv">' +
      kv('TVL × fitted multiple', money(r.tvl_usd) + ' × ' + mult(fittedMultiple)) +
      kv('Base multiple', mult(m.base_multiple)) +
      kv('Growth multiplier', mult(m.growth_multiplier)) +
      kv('Maturity multiplier', mult(m.maturity_multiplier)) +
      kv('Breadth multiplier', mult(m.breadth_multiplier)) +
      kv('Category multiplier', mult(m.category_multiplier)) +
      '</div>' +
      '<h3>The working</h3>' +
      '<div class="kv">' +
      kv('TVL', money(r.tvl_usd)) + kv('7d change', (r.change_7d == null ? '—' : (r.change_7d > 0 ? '+' : '') + r.change_7d.toFixed(1) + '%')) +
      kv('Age', r.age_days == null ? 'unknown (median used)' : r.age_days + ' days') +
      kv('Chains', r.chains.length + ' — ' + esc(r.chains.slice(0, 3).join(', '))) +
      kv('Comps in category', String(r.comps_in_category)) +
      kv('Listed', fmtDate(r.listed_at)) +
      kv('Cheapest chain', esc(r.cheapest_chain) + ' · ' + money(r.net_gas_usd / 7) + '/tx') +
      kv('Hold', r.hold_days + ' days') +
      '</div>' +
      '<h3>Nearest comps</h3>' +
      '<div class="tablewrap"><table class="mini"><thead><tr><th>Protocol</th><th class="num">TVL</th><th class="num">Mcap</th><th class="num">Multiple</th></tr></thead><tbody>' +
      near.map(function (c) {
        return '<tr><td>' + esc(c.name) + '</td><td class="num">' + money(c.tvl_usd) + '</td><td class="num">' + money(c.mcap_usd) + '</td><td class="num">' + c.multiple.toFixed(2) + '×</td></tr>';
      }).join('') + '</tbody></table></div>' +
      '<h3>Farm plan — ' + esc(r.cheapest_chain) + ', cheapest chain</h3>' +
      '<ol class="steps">' + r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>' +
      '<div class="econ"><b>Net economics.</b> Net gas ' + money(r.net_gas_usd) + ' (7 tx) · capital ' + money(r.capital_usd) +
      ' (returns on exit) · break-even allocation ' + money(r.net_gas_usd) + ' · pro-rata EV at p10 <b>' + money2(r.ev_usd) + '</b>. ' +
      'EV assumes 10% of the low-end value to farmers, pro-rata by capital — a floor-style check, not a promise.</div>';
    drawer.hidden = false;
    overlay.hidden = false;
    drawer.scrollTop = 0;
    $('drawerClose').addEventListener('click', closeDrawer);
    try { history.replaceState(null, '', '#row=' + encodeURIComponent(slug)); } catch (e) {}
  }
  function kv(k, val) { return '<div><span>' + esc(k) + '</span><b>' + val + '</b></div>'; }
  function closeDrawer() {
    drawer.hidden = true;
    overlay.hidden = true;
    try {
      var h = location.hash.replace(/[#&]?row=[^&]*/g, '').replace(/^#&/, '#');
      history.replaceState(null, '', h === '#' ? location.pathname : h);
    } catch (e) {}
  }
  overlay.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeDrawer(); });

  // ---- tabs
  $('tabs').addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-tab]');
    if (!btn) return;
    document.querySelectorAll('#tabs button').forEach(function (b) { b.classList.toggle('on', b === btn); });
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.toggle('on', t.id === 'tab-' + btn.getAttribute('data-tab')); });
  });

  // ---- deep links
  function parseHash() {
    var out = {};
    location.hash.replace(/^#/, '').split('&').forEach(function (part) {
      var kv = part.split('=');
      if (kv[0]) out[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || '');
    });
    return out;
  }
  function syncHash() {
    try {
      var parts = [];
      var ver = $('fverdict').value;
      var cat = $('fcat').value;
      if (ver) parts.push('verdict=' + encodeURIComponent(ver));
      if (cat) parts.push('category=' + encodeURIComponent(cat));
      var row = parseHash().row;
      if (row) parts.push('row=' + encodeURIComponent(row));
      history.replaceState(null, '', parts.length ? '#' + parts.join('&') : location.pathname);
    } catch (e) {}
  }
  var h = parseHash();
  if (h.verdict && ['farm', 'watch', 'skip'].indexOf(h.verdict) >= 0) $('fverdict').value = h.verdict;
  if (h.category) $('fcat').value = h.category;
  renderBoard();
  if (h.row) openDrawer(h.row);
})();
