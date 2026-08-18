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
const TOP_N = 20;

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
  $("nightsummary").innerHTML = [
    [s.n_candidates != null ? s.n_candidates : cands.length, "candidates"],
    [ztf, "seen by ZTF"],
    [rubin, "seen by Rubin"],
    [`${specIa} / ${spec}`, "spec-Ia / spec-classified"],
    [s.median_z != null ? fmt(s.median_z, 3) : "—", "median z"],
    [s.median_n_points != null ? s.median_n_points : "—", "median LC points"],
  ].map(([v, l]) => `<div class="stat"><span class="v">${esc(v)}</span><span class="l">${esc(l)}</span></div>`).join("")
    + (s.backfilled ? '<div class="stat" title="Retrospective re-run: fits use photometry and classifications as of upload time, not the night itself"><span class="v">↺</span><span class="l">backfilled</span></div>' : "");
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
    return `<tr><td class="num">${i + 1}</td>
      <td>${objectCell(c)}</td>
      <td class="num">${coords}</td>
      <td class="num">${mag}</td>
      <td class="num">${fmt(c.delta_t, 1)}</td>
      <td class="num"${zsrc}>${fmt(c.z, 3)}</td>
      <td class="num">${c.n_points == null ? "—" : esc(c.n_points)}</td>
      <td class="meritcell">${meritCell}</td>
      <td>${badges(c)}</td></tr>`;
  }).join("") ||
    `<tr><td colspan="9" style="color:var(--faint)">no candidates for this night</td></tr>`;

  const btn = $("showall");
  if (cands.length > TOP_N) {
    btn.hidden = false;
    btn.textContent = SHOW_ALL
      ? `Show top ${TOP_N} only`
      : `Show all ${cands.length} candidates`;
    btn.onclick = () => { SHOW_ALL = !SHOW_ALL; renderCandidates(); };
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
