"use strict";

// ===================================================================
// SN Ia target selection — nightly ranked candidates from the alert
// pipeline. Same-origin SPA, no dependencies (matches app.js).
// Endpoints consumed:
//   GET  /v1/selection?limit_nights=10  -> {nights:[...], persistence:[...]}
//   POST /login  (form-urlencoded; same session cookie as the dashboard)
// 401 -> login form; 403 -> friendly "Stubbs-group only" notice.
// ===================================================================

let DATA = null;        // {nights, persistence}
let NIGHT_IX = 0;       // index into DATA.nights (0 = newest)
let SHOW_ALL = false;
let OPEN_IXS = new Set();  // indices (into the night's candidates) of expanded rows
                           // (multiple allowed, so two sources can be compared)
const TOP_N = 20;

// Photometry band -> CSS color variable (defined in selection.html for both
// themes): ZTF g/r, Rubin/LSST ugrizy, ATLAS c/o; anything else neutral.
const BAND_COLORS = { u: "var(--band-u)", g: "var(--band-g)", r: "var(--band-r)",
                      i: "var(--band-i)", z: "var(--band-z)", y: "var(--band-y)",
                      c: "var(--band-c)", o: "var(--band-o)" };
const bandColor = (b) => BAND_COLORS[String(b || "").toLowerCase()] || "var(--slate)";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const fmt = (v, dp = 2) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(dp);

// ut20260818 -> "Aug 18 2026"
const monthAbbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function prettyStamp(stamp) {
  const m = /^ut(\d{4})(\d{2})(\d{2})$/.exec(stamp || "");
  if (!m) return stamp || "";
  return monthAbbr[+m[2] - 1] + " " + (+m[3]) + " " + m[1];
}

// ---- sexagesimal coordinates ------------------------------------------
function toHMS(raDeg) {
  let h = raDeg / 15;
  const H = Math.floor(h); h = (h - H) * 60;
  const M = Math.floor(h); const S = (h - M) * 60;
  const p2 = (n) => String(n).padStart(2, "0");
  return `${p2(H)}:${p2(M)}:${p2(Math.floor(S))}${(S % 1).toFixed(1).slice(1)}`;
}
function toDMS(decDeg) {
  const sign = decDeg < 0 ? "−" : "+";
  let d = Math.abs(decDeg);
  const D = Math.floor(d); d = (d - D) * 60;
  const M = Math.floor(d); const S = Math.round((d - M) * 60);
  const p2 = (n) => String(n).padStart(2, "0");
  return `${sign}${p2(D)}:${p2(M)}:${p2(S)}`;
}

// ---- object naming / linking ------------------------------------------
function objectCell(c) {
  const tns = c.tns_name;
  if (tns) {
    const bare = String(tns).replace(/^(SN|AT)\s+/i, "");
    const sub = c.ztf_oid || c.diaObjectId || "";
    return `<a class="objlink" href="https://www.wis-tns.org/object/${encodeURIComponent(bare)}"
      target="_blank" rel="noopener">${esc(tns)}</a>` +
      (sub ? `<span class="subid">${esc(sub)}</span>` : "");
  }
  const name = c.ztf_oid || c.diaObjectId || "?";
  const sub = (c.ztf_oid && c.diaObjectId && c.ztf_oid !== c.diaObjectId) ? c.diaObjectId : "";
  return `<span class="tname">${esc(name)}</span>` +
    (sub ? `<span class="subid">${esc(sub)}</span>` : "");
}

function badges(c) {
  const out = [];
  const t = c.tns_type;
  if (t === "SN Ia") out.push('<span class="badge specia">spec Ia</span>');
  else if (t) out.push(`<span class="badge spectype">spec ${esc(t)}</span>`);
  // g_info <= 0.1: already spec-typed AND spec-z'd -> the score's info-gain
  // factor marks it a "free" cosmology-sample entry (no follow-up slot needed)
  if (c.g_info != null && c.g_info <= 0.1)
    out.push('<span class="badge freesample" title="already spec-typed and spec-z’d — enters the cosmology sample without a slot">free sample</span>');
  if (c.offset_class === "nuclear") out.push('<span class="badge nuclear">nuclear</span>');
  if (c.n_points != null && c.n_points <= 8) out.push('<span class="badge sparse">sparse</span>');
  return out.join("");
}

// ===================================================================
// Boot / auth flow (mirrors app.js)
// ===================================================================
async function boot() {
  let res;
  try {
    res = await fetch("/v1/selection?limit_nights=10", { credentials: "include" });
  } catch (e) {
    $("loading").textContent = "Could not reach the server.";
    return;
  }
  if (res.status === 401) { showLogin(); return; }
  if (res.status === 403) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (e) { /* non-json */ }
    showDenied(detail);
    return;
  }
  if (!res.ok) {
    $("loading").textContent = "Selection error (" + res.status + ").";
    return;
  }
  try {
    DATA = await res.json();
  } catch (e) {
    $("loading").textContent = "Could not parse selection response.";
    return;
  }
  NIGHT_IX = 0;
  SHOW_ALL = false;
  renderAll();
}

function showDenied(detail) {
  $("loading").hidden = true;
  $("selview").hidden = true;
  $("loginview").hidden = true;
  $("deniedview").hidden = false;
  $("deniedmsg").innerHTML =
    "<b>This view is limited to the Stubbs group.</b><br>" +
    "The nightly SN Ia selection is the CfA-Stubbs pipeline's working list; " +
    "other programs see the shared queue instead." +
    (detail ? `<br><span style="font-size:.78rem;color:var(--faint)">(${esc(detail)})</span>` : "");
}

async function showLogin() {
  $("loading").hidden = true;
  $("selview").hidden = true;
  $("deniedview").hidden = true;
  $("loginview").hidden = false;
  let programs = ["CfA-Stubbs", "CfA-Villar", "UA"];
  try {
    const r = await fetch("/v1/programs", { credentials: "include" });
    if (r.ok) {
      const body = await r.json();
      if (Array.isArray(body.programs) && body.programs.length) programs = body.programs;
    }
  } catch (e) { /* keep fallback list */ }
  $("lg-prog").innerHTML = programs.map((p) => `<option>${esc(p)}</option>`).join("");
}

$("loginform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("lg-err");
  err.textContent = "";
  try {
    const res = await fetch("/login", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "program=" + encodeURIComponent($("lg-prog").value) +
            "&password=" + encodeURIComponent($("lg-pass").value),
    });
    if (!res.ok) {
      err.textContent = res.status === 401 ? "Wrong program or password." : "Login failed (" + res.status + ").";
      return;
    }
    $("loginview").hidden = true;
    $("loading").hidden = false;
    $("loading").textContent = "Loading the nightly selection…";
    boot();
  } catch (ex) {
    err.textContent = "Could not reach the server.";
  }
});

// ===================================================================
// Rendering
// ===================================================================
function renderAll() {
  $("loading").hidden = true;
  $("loginview").hidden = true;
  $("deniedview").hidden = true;
  $("selview").hidden = false;
  renderHeader();
  renderNightPills();
  renderSummary();
  renderCandidates();
  renderPersistence();
}

function night() { return (DATA.nights || [])[NIGHT_IX] || null; }

function renderHeader() {
  const nights = DATA.nights || [];
  const n = night();
  $("meta").textContent = n
    ? `${nights.length} night${nights.length === 1 ? "" : "s"} uploaded · viewing ${n.night_stamp}` +
      (n.mjd != null ? ` (MJD ${Math.round(n.mjd)})` : "") +
      (n.uploaded_at ? ` · uploaded ${n.uploaded_at}` : "")
    : "no nights uploaded yet";
  $("whoami-line").innerHTML = "pipeline output · <b>wide-sky mode</b>";
}

function selectNight(ix) {
  if (ix === NIGHT_IX) return;
  NIGHT_IX = ix;
  SHOW_ALL = false;
  OPEN_IXS.clear();
  renderNightPills();
  renderSummary();
  renderCandidates();
  renderHeader();
}

function renderNightPills() {
  const nights = DATA.nights || [];
  $("nightpills").innerHTML = nights.map((n, i) => {
    const on = i === NIGHT_IX;
    const nc = (n.summary && n.summary.n_candidates) != null ? n.summary.n_candidates
      : (n.candidates || []).length;
    const bf = n.summary && n.summary.backfilled;
    return `<button type="button" class="night${on ? " on" : ""}" data-ix="${i}">
      <span class="nd">${esc(prettyStamp(n.night_stamp))}${bf ? ' <span title="Retrospective re-run: fits use photometry and classifications as of upload time, not the night itself" style="color:var(--faint);font-size:.75em">↺</span>' : ""}</span>
      <span class="no">${esc(nc)} candidate${nc === 1 ? "" : "s"}${bf ? " · backfilled" : ""}</span>
    </button>`;
  }).join("") || '<div style="color:var(--faint);font-size:.85rem">No selection nights uploaded yet — run scripts/upload_selection_night.py after a pipeline night.</div>';
  $("nightpills").onclick = (e) => {
    const b = e.target.closest("button.night");
    if (b) selectNight(+b.dataset.ix);
  };
}

function renderSummary() {
  const n = night();
  if (!n) { $("nightsummary").innerHTML = ""; return; }
  const s = n.summary || {};
  const cands = n.candidates || [];
  const surv = s.surveys || {};
  // ZTF vs Rubin appearance counts (a candidate can be in both, e.g. "ZTF+Rubin")
  let ztf = 0, rubin = 0;
  Object.entries(surv).forEach(([k, v]) => {
    if (/ZTF/i.test(k)) ztf += v;
    if (/Rubin/i.test(k)) rubin += v;
  });
  const spec = s.n_spec_classified != null ? s.n_spec_classified
    : cands.filter((c) => c.tns_type).length;
  const specIa = s.n_spec_ia != null ? s.n_spec_ia
    : cands.filter((c) => c.tns_type === "SN Ia").length;
  // time cost of the ranking's head: sum exposure over the top-10 ranked rows
  const expTop = cands.slice(0, 10)
    .map((c) => c.exposure_minutes).filter((e) => e != null && isFinite(e));
  const stats = [
    [s.n_candidates != null ? s.n_candidates : cands.length, "candidates"],
    [ztf, "seen by ZTF"],
    [rubin, "seen by Rubin"],
    [`${specIa} / ${spec}`, "spec-Ia / spec-classified"],
    [s.median_z != null ? fmt(s.median_z, 3) : "—", "median z"],
    [s.median_n_points != null ? s.median_n_points : "—", "median LC points"],
  ];
  if (expTop.length) {
    stats.push([`≈ ${(expTop.reduce((a, b) => a + b, 0) / 60).toFixed(1)} h`,
                "top-10 exposure"]);
  }
  $("nightsummary").innerHTML = stats
    .map(([v, l]) => `<div class="stat"><span class="v">${esc(v)}</span><span class="l">${esc(l)}</span></div>`).join("")
    + (s.backfilled ? '<div class="stat" title="Retrospective re-run: fits use photometry and classifications as of upload time, not the night itself"><span class="v">↺</span><span class="l">backfilled</span></div>' : "");
}

// ===================================================================
// Candidate detail panel (row click): light curve + why-ranked + facts
// ===================================================================

// ---- light curve SVG (dependency-free, both-theme-safe) ---------------
function lcSvg(c, n) {
  const lc = c.lc || [];
  if (!lc.length) {
    return '<div class="lc-note">no photometry uploaded for this night</div>';
  }
  const nightMjd = (n && n.mjd != null) ? n.mjd
    : Math.max(...lc.map((p) => p[0]));         // fallback: last point
  const xs = lc.map((p) => p[0] - nightMjd);
  const mags = lc.map((p) => p[1]);
  const peakX = c.peak_mjd != null ? c.peak_mjd - nightMjd : null;

  let xmin = Math.min(...xs, peakX != null ? peakX : 0, 0);
  let xmax = Math.max(...xs, 0);
  const xpad = Math.max(1.5, (xmax - xmin) * 0.04);
  xmin -= xpad; xmax += xpad;
  let magMin = Math.min(...mags);               // brightest
  let magMax = Math.max(...mags);               // faintest
  if (c.peak_mag != null) magMin = Math.min(magMin, c.peak_mag);
  lc.forEach((p) => { if (p[2] != null) {
    magMin = Math.min(magMin, p[1] - p[2]); magMax = Math.max(magMax, p[1] + p[2]); } });
  magMin -= 0.25; magMax += 0.25;

  const W = 640, H = 260, ML = 42, MR = 14, MT = 12, MB = 34;
  const PW = W - ML - MR, PH = H - MT - MB;
  const ax = (x) => ML + (x - xmin) / (xmax - xmin) * PW;
  // magnitude axis INVERTED: brighter (smaller mag) at the top
  const ay = (m) => MT + (m - magMin) / (magMax - magMin) * PH;

  const axisFont = 'font:.6rem var(--mono);fill:var(--faint)';
  let s = `<svg class="lc-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"
    role="img" aria-label="light curve, magnitude vs days before the night">`;

  // y gridlines at ~4 round magnitudes
  const mstep = (magMax - magMin) > 3 ? 1.0 : 0.5;
  for (let m = Math.ceil(magMin / mstep) * mstep; m <= magMax; m += mstep) {
    s += `<line x1="${ML}" y1="${ay(m).toFixed(1)}" x2="${W - MR}" y2="${ay(m).toFixed(1)}"
      stroke="var(--hair2)"/>` +
      `<text x="${ML - 4}" y="${(ay(m) + 2.5).toFixed(1)}" text-anchor="end" style="${axisFont}">${m.toFixed(1)}</text>`;
  }
  // x ticks: round day steps, labeled as |days| under "days before night"
  const span = xmax - xmin;
  const dstep = span > 120 ? 30 : span > 60 ? 20 : span > 25 ? 10 : 5;
  for (let d = Math.ceil(xmin / dstep) * dstep; d <= xmax; d += dstep) {
    s += `<line x1="${ax(d).toFixed(1)}" y1="${MT + PH}" x2="${ax(d).toFixed(1)}" y2="${MT + PH + 4}"
      stroke="var(--faint)"/>` +
      `<text x="${ax(d).toFixed(1)}" y="${H - 18}" text-anchor="middle" style="${axisFont}">${Math.abs(d)}</text>`;
  }
  s += `<text x="${ML + PW / 2}" y="${H - 5}" text-anchor="middle" style="${axisFont}">days before night</text>`;
  s += `<text x="12" y="${MT + PH / 2}" text-anchor="middle" style="${axisFont}"
    transform="rotate(-90 12 ${MT + PH / 2})">mag</text>`;

  // overlays: night (x=0), peak epoch, peak magnitude
  s += `<line x1="${ax(0).toFixed(1)}" y1="${MT}" x2="${ax(0).toFixed(1)}" y2="${MT + PH}"
    stroke="var(--ink)" stroke-width="1.1" opacity=".55"/>` +
    `<text x="${(ax(0) + 3).toFixed(1)}" y="${MT + 10}" style="${axisFont}">night</text>`;
  if (peakX != null && peakX >= xmin && peakX <= xmax) {
    s += `<line x1="${ax(peakX).toFixed(1)}" y1="${MT}" x2="${ax(peakX).toFixed(1)}" y2="${MT + PH}"
      stroke="var(--copper)" stroke-dasharray="4 3" stroke-width="1.1"/>` +
      `<text x="${(ax(peakX) + 3).toFixed(1)}" y="${MT + 22}" style="font:.6rem var(--mono);fill:var(--copper)">peak</text>`;
  }
  if (c.peak_mag != null) {
    s += `<line x1="${ML}" y1="${ay(c.peak_mag).toFixed(1)}" x2="${W - MR}" y2="${ay(c.peak_mag).toFixed(1)}"
      stroke="var(--copper)" stroke-dasharray="2 4" opacity=".6"/>`;
  }

  // points + error bars, colored by band
  lc.forEach((p) => {
    const [mjd, mag, err, band] = p;
    const x = ax(mjd - nightMjd).toFixed(1), col = bandColor(band);
    if (err != null) {
      s += `<line x1="${x}" y1="${ay(mag - err).toFixed(1)}" x2="${x}" y2="${ay(mag + err).toFixed(1)}"
        stroke="${col}" stroke-width="1" opacity=".6"/>`;
    }
    s += `<circle cx="${x}" cy="${ay(mag).toFixed(1)}" r="3" fill="${col}" opacity=".85">
      <title>${esc(band)} ${mag.toFixed(2)}${err != null ? " ± " + err.toFixed(2) : ""} · MJD ${mjd.toFixed(2)}</title></circle>`;
  });
  s += "</svg>";

  // compact legend: only the bands present in THIS candidate's data
  const present = [...new Set(lc.map((p) => String(p[3])))];
  const order = ["u", "g", "r", "i", "z", "y", "c", "o"];
  present.sort((a, b) => (order.indexOf(a) + 99 * (order.indexOf(a) < 0)) -
                         (order.indexOf(b) + 99 * (order.indexOf(b) < 0)));
  const legend = `<div class="lc-legend">` + present.map((b) =>
    `<span><span class="sw" style="background:${bandColor(b)}"></span>${esc(b)}</span>`).join("") +
    `<span>${lc.length} points</span></div>`;
  return `<div class="lcwrap">${s}${legend}</div>`;
}

// ---- "why it's ranked here": the score chain as mini bars -------------
function scChainRow(sym, name, val, caption, sub) {
  const pct = val == null ? 0 : Math.max(1, Math.min(100, Math.round(val * 100)));
  return `<div class="sc-row">
    <span class="sym">${sym}</span>
    <span class="lbl">${name}</span>
    <span class="val">${fmt(val, 2)}</span>
    <span class="track"><span class="fill" style="width:${pct}%"></span></span>
    <span class="cap">${caption}</span></div>` +
    (sub ? `<div class="sc-sub">${sub}</div>` : "");
}

function gCaption(g) {
  if (g == null) return "";
  if (g <= 0.1) return "already spec-typed + spec-z'd — a spectrum adds little";
  // g_type_only = 0.15 (Chris 2026-08-18): typed objects without a spec-z are
  // barely worth a live slot — the host z arrives later in batch via MOS
  if (g <= 0.5) return "already typed — host z comes later via MOS; little gain";
  if (g <= 0.95) return "spec-z known but untyped — the type is the gain";
  return "no spectroscopic type or redshift yet — full information gain";
}

function uCaption(u) {
  if (u == null) return "";
  if (u >= 0.9) return "at/near peak — window open";
  if (u >= 0.3) return "past peak — window closing";
  return "well past peak — urgency nearly gone";
}

function scoreChain(c) {
  const hasScore = c.p_usable != null || c.v_z != null || c.score != null;
  if (!hasScore) {
    const rows = [
      scChainRow("w_time", "phase", c.w_time, "Gaussian phase weight (days from peak)"),
      scChainRow("w_mag", "brightness", c.w_mag, "followable-magnitude window"),
      scChainRow("w_prob", "P(Ia)", c.w_prob, "broker SN/Ia probability"),
      scChainRow("w_salt", "fit quality", c.w_salt, "SALT2 Ia-template fit quality"),
    ].join("");
    return `<div><h4>Why it's ranked here</h4>
      <div class="lc-note" style="margin-bottom:.35rem">legacy merit ranking — this night predates the score</div>
      <div class="scorechain">${rows}</div></div>`;
  }
  let pCap = "chance this is a cosmology-usable Ia";
  if (c.w_lcq != null && c.w_lcq < 0.7) pCap += " · light-curve shape still poorly constrained";
  const pSub = `= w_prob ${fmt(c.w_prob, 2)} × w_iaspec ${fmt(c.w_iaspec, 2)} × w_lcq ${fmt(c.w_lcq, 2)}`;
  const vCap = c.v_z == null ? "" : (c.v_z >= 0.5
    ? "under-sampled redshift bin — high Hubble-diagram value"
    : "well-populated redshift bin — lower marginal sample value");
  const exp = c.exposure_minutes;
  const expBar = exp == null ? null : Math.min(1, 45 / Math.max(exp, 5));
  const rows = [
    scChainRow("P", "usable Ia", c.p_usable, pCap, pSub),
    scChainRow("V(z)", "sample value", c.v_z, vCap),
    scChainRow("G", "info gain", c.g_info, gCaption(c.g_info)),
    scChainRow("U", "urgency", c.u_urgency, uCaption(c.u_urgency)),
    exp != null ? scChainRow("÷", "exposure", expBar,
      `estimated ${Math.round(exp)} min on target — divides score into score/hr`) : "",
  ].join("");
  return `<div><h4>Why it's ranked here</h4><div class="scorechain">${rows}
    <div class="sc-sub" style="margin-left:0">score ${fmt(c.score, 3)} × (45 min / ${exp != null ? Math.round(exp) + " min" : "exp"})^0.5 → score/hr ${fmt(c.score_rate, 3)}</div>
  </div></div>`;
}

// ---- fact grid ---------------------------------------------------------
function factGrid(c) {
  const f = [];
  const add = (k, v) => { if (v != null && v !== "") f.push([k, v]); };
  add("z", c.z != null
    ? `${fmt(c.z, 3)}${c.z_source ? " (" + esc(c.z_source) + ")" : ""}`
    : (c.salt_z != null
      ? `~${fmt(c.salt_z, 2)} (SALT2 LC fit${c.salt_z_railed ? ", railed" : ""})`
      : null));
  if (c.delta_t != null) {
    const rest = c.z != null ? ` · ${fmt(c.delta_t / (1 + c.z), 1)} rest` : "";
    add("phase (d from peak)", `${fmt(c.delta_t, 1)}${rest}`);
  }
  add("peak mag", c.peak_mag != null ? fmt(c.peak_mag, 2) : null);
  add("latest mag", c.latest_mag != null ? fmt(c.latest_mag, 2) : null);
  const lcBands = c.lc ? [...new Set(c.lc.map((p) => p[3]))].join("") : null;
  add("points", c.n_points != null ? `${c.n_points}${lcBands ? " (" + esc(lcBands) + ")" : ""}` : null);
  add("rise time (d)", c.rise_time != null ? fmt(c.rise_time, 1) : null);
  add("template", c.template_best
    ? `${esc(c.template_best)}${c.template_margin != null ? " (margin " + fmt(c.template_margin, 2) + ")" : ""}` : null);
  add("nuclear offset", c.nuclear_offset_arcsec != null
    ? `${fmt(c.nuclear_offset_arcsec, 2)}″${c.offset_class ? " · " + esc(c.offset_class) : ""}`
    : (c.offset_class ? esc(c.offset_class) : null));
  add("host morphology", c.host_morphology ? esc(c.host_morphology) : null);
  add("SALT x1", c.salt_x1 != null ? fmt(c.salt_x1, 2) : null);
  add("SALT c", c.salt_c != null
    ? `${fmt(c.salt_c, 3)}${c.salt_c_err != null ? " ± " + fmt(c.salt_c_err, 3) : ""}` : null);
  add("est. exposure", c.exposure_minutes != null ? `${Math.round(c.exposure_minutes)} min` : null);
  if (!f.length) return "";
  return `<div><h4>Details</h4><dl class="factgrid">` +
    f.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join("") + "</dl></div>";
}

// ---- external links ------------------------------------------------------
function linkRow(c) {
  const links = [];
  if (c.tns_name) {
    const bare = String(c.tns_name).replace(/^(SN|AT)\s+/i, "");
    links.push(`<a href="https://www.wis-tns.org/object/${encodeURIComponent(bare)}" target="_blank" rel="noopener">TNS</a>`);
  }
  if (c.ztf_oid) {
    links.push(`<a href="https://fink-portal.org/${encodeURIComponent(c.ztf_oid)}" target="_blank" rel="noopener">Fink portal</a>`);
    links.push(`<a href="https://alerce.online/object/${encodeURIComponent(c.ztf_oid)}" target="_blank" rel="noopener">ALeRCE</a>`);
  }
  return links.length ? `<div class="linkrow">${links.join("")}</div>` : "";
}

function detailRow(c, n, colspan) {
  return `<tr class="detail-tr"><td colspan="${colspan}"><div class="detail">
    ${lcSvg(c, n)}
    ${scoreChain(c)}
    ${factGrid(c)}
    ${linkRow(c)}
  </div></td></tr>`;
}

function renderCandidates() {
  const n = night();
  const cands = (n && n.candidates) || [];
  // Rank value: score_rate (PI-approved 2026-08-18 ordering) when this night
  // carries it, else the legacy merit — bars scale to the night's max.
  const useScore = cands.some((c) => c.score_rate != null);
  const valOf = (c) => (useScore ? c.score_rate : c.merit);
  $("rankval-th").textContent = useScore ? "score/hr" : "merit";
  const maxVal = Math.max(...cands.map((c) => valOf(c) || 0), 1e-9);
  const shown = SHOW_ALL ? cands : cands.slice(0, TOP_N);
  $("candbody").innerHTML = shown.map((c, i) => {
    const coords = (c.ra != null && c.dec != null)
      ? `${toHMS(c.ra)} ${toDMS(c.dec)}` : "—";
    const mag = c.latest_mag != null ? fmt(c.latest_mag, 1)
      : (c.peak_mag != null ? fmt(c.peak_mag, 1) + '<span class="am"> pk</span>' : "—");
    const zsrc = c.z_source ? ` title="z source: ${esc(c.z_source)}"` : "";
    const val = valOf(c);
    const pct = val != null ? Math.max(1, Math.round(100 * val / maxVal)) : 0;
    // factor breakdown tooltip on the score cell (P/V/G/U + w_lcq)
    const factors = useScore && (c.p_usable != null || c.v_z != null)
      ? ` title="score = P×V×G×U · P=${fmt(c.p_usable, 2)} V=${fmt(c.v_z, 2)} G=${fmt(c.g_info, 2)} U=${fmt(c.u_urgency, 2)}${c.w_lcq != null ? ` (w_lcq=${fmt(c.w_lcq, 2)})` : ""}${c.score != null ? ` → score=${fmt(c.score, 2)}` : ""}"`
      : "";
    const meritCell = val != null
      ? `<div class="meritbar"${factors}><span class="mv">${fmt(val, 2)}</span>
         <span class="track"><span class="fill" style="width:${pct}%"></span></span></div>`
      : "—";
    const open = OPEN_IXS.has(i);
    return `<tr class="cand-row${open ? " open" : ""}" data-ix="${i}"
        title="click for the light curve and ranking breakdown">
      <td class="num">${i + 1}</td>
      <td>${objectCell(c)}</td>
      <td class="num">${coords}</td>
      <td class="num">${mag}</td>
      <td class="num">${fmt(c.delta_t, 1)}</td>
      <td class="num"${zsrc}>${c.z != null ? fmt(c.z, 3)
        : (c.salt_z != null
          ? `<span style="color:var(--faint)" title="no spec/host redshift — SALT2 light-curve fit estimate${c.salt_z_railed ? " (fit railed at the z bound — treat as a limit)" : ""}">~${fmt(c.salt_z, 2)}</span>`
          : "—")}</td>
      <td class="num">${c.n_points == null ? "—" : esc(c.n_points)}</td>
      <td class="meritcell">${meritCell}</td>
      <td>${badges(c)}</td></tr>` +
      (open ? detailRow(c, n, 9) : "");
  }).join("") ||
    `<tr><td colspan="9" style="color:var(--faint)">no candidates for this night</td></tr>`;

  // row click toggles the inline detail panel (several may stay open so
  // sources can be compared); clicks on links (TNS etc.) pass through untouched
  $("candbody").onclick = (e) => {
    if (e.target.closest("a")) return;
    const tr = e.target.closest("tr.cand-row");
    if (!tr) return;
    const ix = +tr.dataset.ix;
    if (OPEN_IXS.has(ix)) OPEN_IXS.delete(ix); else OPEN_IXS.add(ix);
    renderCandidates();
  };

  const btn = $("showall");
  if (cands.length > TOP_N) {
    btn.hidden = false;
    btn.textContent = SHOW_ALL
      ? `Show top ${TOP_N} only`
      : `Show all ${cands.length} candidates`;
    btn.onclick = () => {
      SHOW_ALL = !SHOW_ALL;
      if (!SHOW_ALL) OPEN_IXS.forEach((v) => { if (v >= TOP_N) OPEN_IXS.delete(v); });
      renderCandidates();
    };
  } else {
    btn.hidden = true;
  }
}

function renderPersistence() {
  const nights = DATA.nights || [];
  const stampsOldest = nights.map((n) => n.night_stamp).slice().reverse();
  const rows = DATA.persistence || [];
  $("persistbody").innerHTML = rows.map((r) => {
    const byStamp = {};
    (r.appearances || []).forEach((a) => (byStamp[a.night_stamp] = a));
    const trail = stampsOldest.map((s) => {
      const a = byStamp[s];
      return a ? esc(a.rank) : '<span class="miss">·</span>';
    }).join(" → ");
    const label = r.tns_name
      ? `<a class="objlink" href="https://www.wis-tns.org/object/${encodeURIComponent(String(r.tns_name).replace(/^(SN|AT)\s+/i, ""))}"
          target="_blank" rel="noopener">${esc(r.tns_name)}</a>`
      : `<span class="tname">${esc(r.label)}</span>`;
    return `<tr><td>${label}${r.tns_name && r.label !== r.tns_name ? `<span class="subid">${esc(r.label)}</span>` : ""}</td>
      <td class="num">${(r.appearances || []).length}</td>
      <td class="ranktrail">${trail}</td>
      <td class="num">${fmt(r.latest_merit, 2)}</td></tr>`;
  }).join("") ||
    `<tr><td colspan="4" style="color:var(--faint)">no object has appeared in two or more of the recent nights yet</td></tr>`;
}

// footer
(function () {
  const f = $("foot");
  if (f) f.innerHTML =
    "Nightly ranked SN Ia candidates from the wide-sky alert pipeline " +
    "(<code>GET /v1/selection</code>, uploaded by <code>scripts/upload_selection_night.py</code>). " +
    "Rank = descending <code>score/hr</code> (score = P×V(z)×G×U, PI-approved 2026-08-18) " +
    "when a night carries it, else the legacy multiplicative merit; the persistence table " +
    "shows objects the pipeline kept re-selecting across nights.";
})();

boot();
