"use strict";

// ===================================================================
// RubinAlerts MAGNETS shared queue — live web interface
// Same-origin SPA. All fetches are relative + credentials:'include'.
// Endpoints consumed:
//   GET    /v1/dashboard?instrument=LDSS3   -> full dashboard payload
//   POST   /login   (form-urlencoded: program, password)
//   POST   /logout
//   POST   /v1/targets            body: [ {name,ra,dec,priority,instrument,exposure_minutes,mag} ]
//   PATCH  /v1/targets/{id}        body: { priority }
//   DELETE /v1/targets/{id}
// On total fetch failure (no backend during local dev) -> ./sample.json
// ===================================================================

// Currently-selected observing night. The initial value is only a fetch seed;
// boot() snaps to the next upcoming non-cancelled night from the calendar in
// the first payload (falling back to the latest past night when the season is
// over). Clicking a night in the schedule bar re-points and re-fetches.
let NIGHT = { date: "2026-08-13", instrument: "LDSS3" };
let NIGHT_SNAPPED = false;

// First non-cancelled calendar night on/after today (local clock), else the
// last non-cancelled night. Returns null if the payload carries no calendar.
function upcomingNight(nights) {
  const usable = (nights || []).filter((n) => n.status !== "cancelled" && n.date);
  if (!usable.length) return null;
  const today = new Date().toISOString().slice(0, 10);
  return usable.find((n) => n.date >= today) || usable[usable.length - 1];
}
const dashUrl = () =>
  "/v1/dashboard?date=" + encodeURIComponent(NIGHT.date) +
  "&instrument=" + encodeURIComponent(NIGHT.instrument);

const TIERS = ["P0", "P1", "P2", "P3", "P4", "P5"];
const order = { P0: 0, P1: 1, P2: 2, P3: 3, P4: 4, P5: 5 };
const TIERCOL = { P0: "#b3542e", P1: "#3a4650", P2: "#5c6570", P3: "#7a8590",
                 P4: "#9aa1a9", P5: "#c2c8ce" };
const RAWPAL = ["#6a4a86", "#2f7d6b", "#b3802e", "#3d6ea5", "#a5533d"];

let DATA = null;      // current dashboard payload
let RAW = {};         // program -> color
let PROGS = [];       // program names
let usingSample = false;

const $ = (id) => document.getElementById(id);
const chip = (t) =>
  `<span class="chip" style="background:${TIERCOL[t] || "#7a8590"};color:${["P3", "P4", "P5"].includes(t) ? "#2a2f34" : "#fff"}">${t}</span>`;
const esc = (s) =>
  String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// stable id for write endpoints: prefer explicit id, fall back to name
const idOf = (t) => (t.id != null ? t.id : t.name);
// observability-window flag: colour-codes whether a target is leaving early,
// rising late, or flexible — the "when do I actually have to do this" cue.
const WFLAGCOL = { early: "var(--copper)", late: "#3d6ea5", flexible: "var(--ok)", none: "var(--err)" };
const windowFlag = (flag, note) =>
  note ? `<span class="wflag" style="color:${WFLAGCOL[flag] || "var(--faint)"}">${esc(note)}</span>` : "";

// ---- toasts ----------------------------------------------------------
function toast(msg, kind = "ok") {
  const box = $("toasts");
  const el = document.createElement("div");
  el.className = "toast " + (kind === "err" ? "err" : "ok");
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ===================================================================
// Boot / auth flow
// ===================================================================
async function boot() {
  let res;
  try {
    res = await fetch(dashUrl(), { credentials: "include" });
  } catch (e) {
    // No backend at all -> local review fallback
    await loadSample();
    return;
  }
  if (res.status === 401) {
    showLogin();
    return;
  }
  if (!res.ok) {
    $("loading").textContent = "Dashboard error (" + res.status + ").";
    return;
  }
  try {
    DATA = await res.json();
  } catch (e) {
    $("loading").textContent = "Could not parse dashboard response.";
    return;
  }
  usingSample = false;
  // One-time snap: the hardcoded fetch seed goes stale, so re-point to the
  // next upcoming night from the calendar and re-fetch once.
  if (!NIGHT_SNAPPED) {
    NIGHT_SNAPPED = true;
    const up = upcomingNight(DATA && DATA.nights);
    if (up && (up.date !== NIGHT.date || up.instrument !== NIGHT.instrument)) {
      NIGHT = { date: up.date, instrument: up.instrument };
      await refresh();
      return;
    }
  }
  renderAll();
}

async function loadSample() {
  try {
    const r = await fetch("./sample.json");
    DATA = await r.json();
    usingSample = true;
    renderAll();
  } catch (e) {
    $("loading").textContent = "No backend and no local sample.json available.";
  }
}

async function refresh() {
  if (usingSample) {
    // In local-review mode there is no backend to re-fetch; just re-render.
    renderAll();
    return;
  }
  try {
    const res = await fetch(dashUrl(), { credentials: "include" });
    if (res.status === 401) { showLogin(); return; }
    if (!res.ok) { toast("Refresh failed (" + res.status + ")", "err"); return; }
    DATA = await res.json();
    renderAll();
  } catch (e) {
    toast("Network error refreshing dashboard", "err");
  }
}

// ---- login view ------------------------------------------------------
function programChoices() {
  // Prefer program list from a prior (sample) payload; else a sane default set
  // covering every group across both instruments (matches the demo GROUPS).
  if (DATA && DATA.programs) return Object.keys(DATA.programs);
  return ["CfA-Villar", "CfA-Stubbs", "UA"];
}

function fillProgramOptions(opts) {
  $("lg-prog").innerHTML = opts.map((p) => `<option>${esc(p)}</option>`).join("");
}

async function showLogin() {
  $("loading").hidden = true;
  $("dashview").hidden = true;
  $("loginview").hidden = false;
  // a fresh session may be a different program: re-hide + re-probe the
  // selection nav link after the next successful load
  selNavProbed = false;
  $("selnav").hidden = true;
  fillProgramOptions(programChoices());       // instant, from fallback/sample
  // Then refine from the server's actual configured groups (public endpoint),
  // so every configured program appears even before anyone signs in.
  try {
    const r = await fetch("/v1/programs", { credentials: "include" });
    if (r.ok) {
      const body = await r.json();
      if (Array.isArray(body.programs) && body.programs.length) {
        fillProgramOptions(body.programs);
      }
    }
  } catch (e) { /* offline / no backend: keep the fallback list */ }
}

$("loginform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("lg-err");
  err.textContent = "";
  const program = $("lg-prog").value;
  const password = $("lg-pass").value;
  try {
    const res = await fetch("/login", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body:
        "program=" + encodeURIComponent(program) +
        "&password=" + encodeURIComponent(password),
    });
    if (!res.ok) {
      err.textContent = res.status === 401 ? "Wrong program or password." : "Login failed (" + res.status + ").";
      return;
    }
    $("loginview").hidden = true;
    $("loading").hidden = false;
    $("loading").textContent = "Loading the shared queue…";
    boot();
  } catch (ex) {
    err.textContent = "Could not reach the server.";
  }
});

$("logout").addEventListener("click", async () => {
  try {
    await fetch("/logout", { method: "POST", credentials: "include" });
  } catch (e) { /* ignore */ }
  showLogin();
});

// ===================================================================
// Write actions
// ===================================================================
async function addTarget(ev) {
  ev.preventDefault();
  const name = $("t-name").value.trim();
  const raS = $("t-ra").value.trim();
  const decS = $("t-dec").value.trim();
  const expS = $("t-exp").value.trim();
  const magS = $("t-mag").value.trim();

  if (!name && !(raS && decS)) {
    toast("Provide a name or an RA + Dec pair.", "err");
    return;
  }
  // Exposure is the submitter's call — we don't auto-size someone else's
  // target (the ETC is tuned for our own Ia candidates).
  if (expS === "") {
    toast("Exposure (min) is required — set the integration you want.", "err");
    return;
  }
  const aminS = $("t-amin").value.trim();
  const amaxS = $("t-amax").value.trim();
  const item = {
    name: name || null,
    ra: raS === "" ? null : Number(raS),
    dec: decS === "" ? null : Number(decS),
    priority: $("t-pri").value,
    instrument: $("t-inst").value,
    exposure_minutes: expS === "" ? null : Number(expS),
    mag: magS === "" ? null : Number(magS),
    // optional airmass range: hard scheduling constraint; empty = default
    // behavior (observe at minimum airmass)
    airmass_min: aminS === "" ? null : Number(aminS),
    airmass_max: amaxS === "" ? null : Number(amaxS),
  };

  const btn = $("addbtn");
  btn.disabled = true;
  try {
    const res = await fetch("/v1/targets", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([item]),
    });
    if (res.status === 401) { showLogin(); return; }
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-json */ }
    reportItemResults(body, res.ok);
    if (res.ok) {
      $("addform").reset();
      $("t-pri").value = "P1";
      $("t-inst").value = "LDSS3";
      $("etc-note").hidden = true;
      etcSuggested = null;
    }
    await refresh();
  } catch (e) {
    toast("Network error adding target", "err");
  } finally {
    btn.disabled = false;
  }
}

// The POST returns an array of per-item results; surface ok/error for each.
function reportItemResults(body, httpOk) {
  const items = Array.isArray(body) ? body
    : (body && Array.isArray(body.results)) ? body.results
    : null;
  if (items) {
    items.forEach((r) => {
      const label = r.name || r.id || "target";
      if (r && (r.ok === false || r.error)) toast(label + ": " + (r.error || "rejected"), "err");
      else toast(label + ": added", "ok");
    });
    return;
  }
  toast(httpOk ? "Target submitted" : "Submission rejected", httpOk ? "ok" : "err");
}

async function changePriority(id, priority) {
  try {
    const res = await fetch("/v1/targets/" + encodeURIComponent(id), {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ priority }),
    });
    if (res.status === 401) { showLogin(); return; }
    if (!res.ok) { toast("Priority update failed (" + res.status + ")", "err"); }
    else { toast("Priority → " + priority, "ok"); }
  } catch (e) {
    toast("Network error updating priority", "err");
  }
  await refresh();
}

async function withdraw(id, name) {
  if (!window.confirm("Withdraw " + name + " from the queue?")) return;
  try {
    const res = await fetch("/v1/targets/" + encodeURIComponent(id), {
      method: "DELETE",
      credentials: "include",
    });
    if (res.status === 401) { showLogin(); return; }
    if (!res.ok) { toast("Withdraw failed (" + res.status + ")", "err"); }
    else { toast(name + " withdrawn", "ok"); }
  } catch (e) {
    toast("Network error withdrawing target", "err");
  }
  await refresh();
}

// ===================================================================
// Rendering (adapted from the reference dashboard)
// ===================================================================
function renderAll() {
  PROGS = Object.keys(DATA.programs || {});
  RAW = {};
  PROGS.forEach((p, i) => (RAW[p] = RAWPAL[i % RAWPAL.length]));

  $("loading").hidden = true;
  $("loginview").hidden = true;
  $("dashview").hidden = false;

  renderHeader();
  renderNights();
  renderSummary();
  renderRightNow();
  renderWriteControls();
  renderPlan();
  renderOverflow();
  renderPrograms();
  renderAllocations();
  renderQueue();
  renderFoot();
  probeSelectionNav();
  loadObserved();
}

// ---- observed repository (item F): what actually went on sky ----------
let OBS_ROWS = [];          // this night's ingested observation rows
let OBSERVED_TARGETS = {};  // target_id (string) -> latest observed night
let obsNightLoaded = null;  // avoid refetch loops for the same night

async function loadObserved() {
  if (usingSample) return;
  const key = NIGHT.date;
  if (obsNightLoaded === key) { renderObserved(); return; }
  try {
    const [rNight, rAll] = await Promise.all([
      fetch("/v1/observations?night=" + encodeURIComponent(NIGHT.date),
            { credentials: "include" }),
      fetch("/v1/observations", { credentials: "include" }),
    ]);
    if (!rNight.ok || !rAll.ok) return;      // 401 etc: leave section hidden
    OBS_ROWS = (await rNight.json()).observations || [];
    OBSERVED_TARGETS = (await rAll.json()).observed_targets || {};
    obsNightLoaded = key;
    renderObserved();
    renderQueue();                            // adds the observed ✓ markers
  } catch (e) { /* offline: section stays hidden */ }
}

// ut20260816 -> "Aug 16 2026" for the queue marker
function prettyStamp(stamp) {
  const m = /^ut(\d{4})(\d{2})(\d{2})$/.exec(stamp || "");
  if (!m) return stamp || "";
  return monthAbbr[+m[2] - 1] + " " + (+m[3]) + " " + m[1];
}

function renderObserved() {
  const has = OBS_ROWS.length > 0;
  $("obs-h").hidden = !has;
  $("obs-section").hidden = !has;
  if (!has) return;
  // aggregate per target (unassociated rows grouped by their raw name)
  const groups = {};
  OBS_ROWS.forEach((o) => {
    const key = o.target_name || (o.object_name_raw || "?") + " (raw)";
    const g = groups[key] = groups[key] || {
      name: o.target_name || o.object_name_raw || "?",
      program: o.program, method: o.assoc_method,
      n: 0, sec: 0, ams: [] };
    g.n += 1;
    g.sec += o.exptime_s || 0;
    if (o.airmass != null) g.ams.push(o.airmass);
  });
  const unassoc = OBS_ROWS.filter((o) => o.assoc_method === "unassociated");
  $("obs-callout").innerHTML = unassoc.length
    ? `<div class="banner">⚠ ${unassoc.length} exposure${unassoc.length === 1 ? "" : "s"} could not be
       associated with any queue target (pointing &gt;1&prime; from everything, no name match) —
       ${unassoc.map((o) => esc(o.object_name_raw || o.filename)).join(", ")}. Unassociated time is not charged.</div>`
    : "";
  $("obsbody").innerHTML = Object.values(groups)
    .sort((a, b) => b.sec - a.sec)
    .map((g) => {
      const am = g.ams.length
        ? (g.ams.reduce((x, y) => x + y, 0) / g.ams.length).toFixed(2) : "—";
      const method = g.method === "unassociated"
        ? '<span style="color:var(--err)">unassociated</span>'
        : esc(g.method);
      return `<tr><td><span class="tname">${esc(g.name)}</span></td>
        <td>${g.program ? `<span class="dot" style="background:${RAW[g.program] || "var(--slate)"}"></span>${esc(g.program)}` : "—"}</td>
        <td class="num">${g.n}</td>
        <td class="num">${Math.round(g.sec / 60)}m</td>
        <td>${method}</td>
        <td class="num">${am}</td></tr>`;
    }).join("");
}

// Show the "SN Ia target selection" nav link only to programs the selection
// endpoint admits (a cheap authorized probe; 403/401 leave it hidden).
let selNavProbed = false;
async function probeSelectionNav() {
  if (selNavProbed || usingSample) return;
  selNavProbed = true;
  try {
    const r = await fetch("/v1/selection?limit_nights=1", { credentials: "include" });
    if (r.ok) $("selnav").hidden = false;
  } catch (e) { /* leave hidden */ }
}

function renderHeader() {
  const p = DATA.plan;
  $("title").firstChild.textContent = "Queue & observing plan — " + p.date;
  let window = "dark window n/a";
  if (p.twilight_start && p.twilight_end) {
    const local = (p.twilight_start_local && p.twilight_end_local)
      ? ` (${p.twilight_start_local}–${p.twilight_end_local} Chile ${p.tz_offset || ""})` : "";
    window = `dark ${p.twilight_start}–${p.twilight_end} UT${local} · ${p.dark_hours ?? 0} h`;
  }
  $("meta").textContent = `${p.instrument} · ${p.moon} time · ${window}`;

  const caller = DATA.caller_program;
  const who = $("whoami-line");
  if (caller) {
    who.innerHTML = `signed in as <b>${esc(caller)}</b>` + (usingSample ? " <span style='color:var(--faint)'>(local sample)</span>" : "");
    $("logout").hidden = usingSample; // no session to end in sample mode
  } else {
    who.innerHTML = usingSample ? "<span style='color:var(--faint)'>local sample</span>" : "read-only";
    $("logout").hidden = usingSample;
  }

  const banner = $("callerbanner");
  if (usingSample) {
    banner.innerHTML = `<div class="banner">No backend reached — rendering <code>./sample.json</code> for local review. Write actions require the live API.</div>`;
  } else {
    banner.innerHTML = "";
  }
}

// ---- 2026B observing schedule ----------------------------------------
const monthAbbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function prettyDate(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
  if (!m) return iso || "";
  return monthAbbr[+m[2] - 1] + " " + (+m[3]) + " " + m[1];
}

function selectNight(date, instrument) {
  // Storm-cancelled nights are not selectable.
  const n = ((DATA && DATA.nights) || []).find(
    (x) => x.date === date && x.instrument === instrument);
  if (n && n.status === "cancelled") {
    toast(`${prettyDate(date)} ${instrument} — cancelled (${n.note || "no observations"})`, "err");
    return;
  }
  if (NIGHT.date === date && NIGHT.instrument === instrument) return;
  NIGHT = { date, instrument };
  if (usingSample) { renderNights(); return; }  // no backend to re-fetch
  $("nights").classList.add("loading");
  refresh().finally(() => $("nights").classList.remove("loading"));
}

function renderNights() {
  const box = $("nights");
  const nights = (DATA && DATA.nights) || [];
  if (!nights.length) { box.innerHTML = ""; return; }
  const cur = (DATA.selected_night) || NIGHT;
  box.innerHTML = nights.map((n) => {
    const on = n.date === cur.date && n.instrument === cur.instrument;
    const cancelled = n.status === "cancelled";
    const past = !cancelled && n.date < new Date().toISOString().slice(0, 10);
    const half = (n.length === "half") ? `<span class="nl">½</span>` : "";
    const cls = "night" + (on ? " on" : "") + (cancelled ? " cancelled" : "")
      + (past ? " past" : "");
    const title = esc((cancelled ? (n.note || "cancelled")
                       : (past ? "observed · " : "") + (n.observer || ""))
                      + (n.program ? " · " + n.program : ""));
    const mark = cancelled ? `<span class="nx" aria-label="cancelled">✕</span>` : "";
    return `<button type="button" class="${cls}"${cancelled ? " disabled" : ""}
        onclick="selectNight('${esc(n.date)}','${esc(n.instrument)}')" title="${title}">
      ${mark}
      <span class="nd">${esc(prettyDate(n.date))}</span>
      <span class="ni ni-${esc((n.instrument || "").toLowerCase())}">${esc(n.instrument)}</span>${half}
      <span class="no">${esc(cancelled ? "cancelled" : (past ? "done · " : "") + (n.observer || ""))}</span>
    </button>`;
  }).join("");
}

function renderSummary() {
  const p = DATA.plan;
  const overflowCount = (p.overflow && p.overflow.length) || 0;
  // Groups actually populating tonight's queue — distinct programs among the
  // targets for this night/instrument, not the count of all configured groups.
  const activeProgs = new Set((DATA.targets || []).map((t) => t.program)).size;
  $("summary").innerHTML = [
    [DATA.targets.length, "targets in the queue"],
    [p.n_scheduled ?? 0, "fit tonight"],
    [overflowCount, "in overflow"],
    [(p.scheduled_science_hours ?? 0) + " h", "science planned"],
    [activeProgs, activeProgs === 1 ? "group sharing the night" : "groups sharing the night"],
  ].map(([v, l]) => `<div class="stat"><span class="v">${esc(v)}</span><span class="l">${esc(l)}</span></div>`).join("");
}

// ---- right now: airmass + eligibility -------------------------------
let clockInited = false;
function renderRightNow() {
  const GRID = DATA.grid, LIMIT = DATA.airmass_limit, N = GRID.length;
  const AMW = 900, AMH = 240, ML = 34, MR = 14, MT = 10, MB = 26;
  const PW = AMW - ML - MR, PH = AMH - MT - MB, YMAX = 1.9;
  const ax = (i) => ML + (N > 1 ? i / (N - 1) : 0) * PW;
  const ay = (a) => MT + (Math.min(YMAX, Math.max(1, a)) - 1) / (YMAX - 1) * PH;

  function tracePath(am) {
    let d = "", pen = false;
    (am || []).forEach((v, i) => {
      if (v == null || v > YMAX) { pen = false; return; }
      d += (pen ? "L" : "M") + ax(i).toFixed(1) + " " + ay(v).toFixed(1) + " ";
      pen = true;
    });
    return d.trim();
  }

  let svg = `<svg class="am-svg" viewBox="0 0 ${AMW} ${AMH}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="airmass through the night">`;
  [1, 1.2, 1.4, 1.6, 1.8].forEach((a) => {
    svg += `<line class="gl" x1="${ML}" y1="${ay(a)}" x2="${AMW - MR}" y2="${ay(a)}"/>` +
      `<text x="3" y="${ay(a) + 3}" style="font:.62rem var(--mono);fill:var(--faint)">${a.toFixed(1)}</text>`;
  });
  for (let i = 0; i < N; i += 12) {
    svg += `<text x="${ax(i)}" y="${AMH - 8}" text-anchor="middle" style="font:.62rem var(--mono);fill:var(--faint)">${esc(GRID[i])}</text>`;
  }
  svg += `<line class="limit" x1="${ML}" y1="${ay(LIMIT)}" x2="${AMW - MR}" y2="${ay(LIMIT)}"/>` +
    `<text x="${AMW - MR}" y="${ay(LIMIT) - 3}" text-anchor="end" style="font:.62rem var(--mono);fill:var(--err,#b3402e)">airmass ${LIMIT} limit</text>`;
  DATA.targets.forEach((t) => {
    const d = tracePath(t.airmass);
    if (d) svg += `<path class="trace" data-name="${esc(t.name)}" stroke="${RAW[t.program]}" d="${d}"></path>`;
  });
  svg += `<line class="cursor" id="cursor" x1="${ax(0)}" y1="${MT}" x2="${ax(0)}" y2="${MT + PH}"></line>`;
  svg += `<circle id="marker" class="am-marker" r="5.5" cx="0" cy="0" style="display:none"></circle></svg>`;
  $("amplot").innerHTML = svg;

  const amplot = $("amplot");
  const clock = $("clock");
  let pinned = null;

  function markerAt(name) {
    const t = DATA.targets.find((x) => x.name === name);
    const i = +clock.value, mk = $("marker");
    if (t && t.airmass && t.airmass[i] != null && t.airmass[i] <= YMAX) {
      mk.setAttribute("cx", ax(i).toFixed(1));
      mk.setAttribute("cy", ay(t.airmass[i]).toFixed(1));
      mk.style.display = "";
    } else {
      mk.style.display = "none";
    }
    amplot.querySelectorAll("path.trace").forEach((p) => p.classList.toggle("hot", p.dataset.name === name));
  }
  function clearMark() {
    const mk = $("marker");
    if (mk) mk.style.display = "none";
    amplot.querySelectorAll("path.trace.hot").forEach((p) => p.classList.remove("hot"));
  }

  clock.max = N - 1;
  if (!clockInited) { clock.value = Math.floor(N / 2); clockInited = true; }
  if (+clock.value > N - 1) clock.value = N - 1;

  function renderNow() {
    const i = +clock.value;
    $("clocklab").textContent = GRID[i] + " UT";
    const cur = $("cursor"), x = ax(i);
    cur.setAttribute("x1", x); cur.setAttribute("x2", x);
    const elig = DATA.targets
      .map((t) => ({ t, am: t.airmass ? t.airmass[i] : null }))
      .filter((o) => o.am != null && o.am <= LIMIT)
      .sort((a, b) => order[a.t.tier] - order[b.t.tier] || (a.t.exp_est - b.t.exp_est) || a.am - b.am);
    const eset = new Set(elig.map((o) => o.t.name));
    const top = new Set(elig.slice(0, 3).map((o) => o.t.name));
    amplot.querySelectorAll("path.trace").forEach((p) => {
      const n = p.dataset.name;
      p.classList.toggle("hi", top.has(n));
      p.classList.toggle("up", eset.has(n) && !top.has(n));
    });
    $("rnhead").innerHTML =
      `<b>${elig.length}</b> targets observable at <b>${esc(GRID[i])} UT</b>, ranked by priority then how quickly they can be done. "In plan" = already in tonight's sequence.`;
    $("rnpanel").innerHTML = elig.length
      ? elig.map((o, k) => {
          const t = o.t;
          const inplan = t.sched_utc
            ? `<span style="color:var(--ok);font-weight:600">in plan · ${esc(t.sched_utc)}</span>`
            : "not in plan";
          return `<div class="rn-card ${k < 3 ? "top" : ""} ${t.name === pinned ? "pinned" : ""}" data-name="${esc(t.name)}" style="border-left-color:${RAW[t.program]}">
            <span class="rn-rank">${k + 1}</span>${chip(t.tier)}
            <div class="rn-main"><span class="nm">${esc(t.name)}</span>
              <div class="sub"><span>${esc(t.program)}</span><span>airmass ${o.am.toFixed(2)}</span><span>r ${t.mag == null ? "—" : esc(t.mag)}</span><span>~${esc(t.exp_est)}m</span><span>${inplan}</span></div>
            </div></div>`;
        }).join("")
      : '<div class="rn-head">Nothing above the airmass limit at this moment.</div>';
    if (pinned) markerAt(pinned); else clearMark();
  }

  clock.oninput = renderNow;
  const rnpanel = $("rnpanel");
  rnpanel.onmouseover = (e) => {
    const c = e.target.closest(".rn-card");
    if (c && c.dataset.name !== pinned) markerAt(c.dataset.name);
  };
  rnpanel.onmouseout = (e) => {
    const c = e.target.closest(".rn-card");
    if (!c) return;
    if (pinned) markerAt(pinned); else clearMark();
  };
  rnpanel.onclick = (e) => {
    const c = e.target.closest(".rn-card");
    if (!c) return;
    pinned = pinned === c.dataset.name ? null : c.dataset.name;
    renderNow();
  };
  renderNow();
}

// ---- write controls: show add-form only when we know the caller -----
function renderWriteControls() {
  const canWrite = !!DATA.caller_program && !usingSample;
  const showForm = !!DATA.caller_program; // form visible in sample too, submits are just no-op fallbacks
  $("addhead").hidden = !showForm;
  $("addform").hidden = !showForm;
}

// ---- observing plan --------------------------------------------------
function renderPlan() {
  // Export links carry the session cookie automatically (same-origin GET).
  const base = "/v1/plan/export?instrument=" + encodeURIComponent(DATA.plan.instrument) +
    "&date=" + encodeURIComponent(DATA.plan.date) + "&fmt=";
  $("exp-cat").href = base + "catalog";
  $("exp-csv").href = base + "csv";
  $("exp-txt").href = base + "text";
  // Join each scheduled row to its target record for the observability window.
  const byName = {};
  (DATA.targets || []).forEach((t) => (byName[t.name] = t));
  $("planbody").innerHTML = DATA.plan.timeline.map((e, i) => {
    const t = byName[e.target] || {};
    const win = (t.obs_start && t.obs_end) ? `${esc(t.obs_start)}–${esc(t.obs_end)}` : "—";
    const best = t.obs_best
      ? `${esc(t.obs_best)}${t.min_airmass == null ? "" : ` <span class="am">airmass ${esc(t.min_airmass)}</span>`}` : "—";
    const nominal = e.utc ? `<span class="nominal">nominal ≈ ${esc(e.utc)}</span>` : "";
    return `<tr><td class="num">${i + 1}</td>
      <td><span class="tname">${esc(e.target)}</span>${nominal}</td>
      <td><span class="dot" style="background:${RAW[e.program]}"></span>${esc(e.program)}</td>
      <td>${chip(e.tier)}</td>
      <td class="num">${win}</td><td class="num">${best}</td>
      <td class="num">${e.mag == null ? "—" : esc(e.mag)}</td>
      <td class="num">${e.exp_min == null ? "—" : Math.round(e.exp_min) + "m"}</td>
      <td>${windowFlag(t.window_flag, t.window_note)}</td></tr>`;
  }).join("") ||
    `<tr><td colspan="9" style="color:var(--faint)">nothing scheduled for this instrument yet</td></tr>`;
}

// ---- overflow bench --------------------------------------------------
function renderOverflow() {
  const overRows = DATA.targets.filter((t) => t.status === "overflow")
    .sort((a, b) => order[a.tier] - order[b.tier]);
  $("overbody").innerHTML = overRows.map((t, i) =>
    `<tr><td class="num">${i + 1}</td><td class="num">—</td><td>${esc(t.name)}</td>
     <td><span class="dot" style="background:${RAW[t.program]}"></span>${esc(t.program)}</td><td>${chip(t.tier)}</td>
     <td class="num">${esc(t.ra)}</td><td class="num">${esc(t.dec)}</td><td class="num">${t.mag == null ? "—" : esc(t.mag)}</td>
     <td class="num">~${esc(t.exp_est)}m</td><td class="num">—</td></tr>`
  ).join("") || `<tr><td colspan="10" style="color:var(--faint)">nothing overflowed</td></tr>`;
}

// ---- by program ------------------------------------------------------
function renderPrograms() {
  const req = DATA.plan.requested_hours || {}, sch = DATA.plan.scheduled_hours || {};
  const counts = {};
  DATA.targets.forEach((t) => {
    (counts[t.program] = counts[t.program] || { P0: 0, P1: 0, P2: 0, P3: 0 })[t.tier]++;
  });
  $("progs").innerHTML = PROGS.map((p) => {
    const c = counts[p] || {};
    const tc = TIERS.filter((t) => c[t]).map((t) => `${t}×${c[t]}`).join(" · ");
    const mine = p === DATA.caller_program ? " mine" : "";
    return `<div class="card${mine}" style="border-top-color:${RAW[p]}">
      <h3><span class="dot" style="background:${RAW[p]}"></span>${esc(p)}</h3>
      <div class="sci">${esc((DATA.programs[p] || {}).science || "")}</div>
      <div class="nums"><span>scheduled ${(sch[p] || 0).toFixed(1)} h</span><span>requested ~${esc(req[p] || 0)} h</span></div>
      <div class="split">${tc}</div></div>`;
  }).join("");
}

// ---- time allocations (season budget per group, both instruments) ----
function renderAllocations() {
  const a = DATA.allocations;
  if (!a || !a.programs || !Object.keys(a.programs).length) {
    $("allocs").innerHTML = ""; return;
  }
  const insts = a.instruments || ["LLAMAS", "LDSS3"];
  // `tracked` is false until a post-night reconciliation ledger populates
  // `used`; until then we show the allocation but label usage as not-yet-live
  // rather than presenting a hardcoded 0 as a real figure.
  const tracked = a.tracked;
  $("allocs").innerHTML = Object.keys(a.programs).sort().map((p) => {
    const col = RAW[p] || "var(--slate)";
    const rows = insts.map((inst) => {
      const d = a.programs[p][inst];
      if (!d) {
        return `<div class="alloc-row"><span class="ai ai-${esc(inst.toLowerCase())}">${esc(inst)}</span>
          <span class="bar"></span><span class="al none">no allocation</span></div>`;
      }
      if (!tracked) {
        return `<div class="alloc-row"><span class="ai ai-${esc(inst.toLowerCase())}">${esc(inst)}</span>
          <span class="bar"><span class="fill untracked" style="width:0%"></span></span>
          <span class="al">${esc(d.initial)} h allocated <span class="rem">· usage not tracked yet</span></span></div>`;
      }
      const pct = d.initial > 0 ? Math.round(100 * d.used / d.initial) : 0;
      return `<div class="alloc-row"><span class="ai ai-${esc(inst.toLowerCase())}">${esc(inst)}</span>
        <span class="bar"><span class="fill" style="width:${pct}%"></span></span>
        <span class="al">${esc(d.remaining)} h left <span class="rem">/ ${esc(d.initial)} h${d.used ? " · " + esc(d.used) + " used" : ""}</span></span></div>`;
    }).join("");
    return `<div class="card${p === DATA.caller_program ? " mine" : ""}" style="border-top-color:${col}">
      <h3><span class="dot" style="background:${col}"></span>${esc(p)}</h3>${rows}</div>`;
  }).join("");
}

// ---- the queue (by priority; caller rows are editable) --------------
let queueFilter = "All";
function queueRow(t, caller, brk) {
  const st = t.status === "scheduled"
    ? `<span class="st st-sched"><span class="d" style="background:var(--ok)"></span>scheduled</span>`
    : t.status === "overflow"
      ? `<span class="st st-over"><span class="d" style="background:var(--faint)"></span>overflow</span>`
      : `<span class="st st-over"><span class="d" style="background:var(--faint)"></span>${esc(t.status || "queued")}</span>`;
  const mine = caller && t.program === caller;
  const manage = mine ? editControls(t) : `<span class="readonly-note">read-only</span>`;
  // observed marker (item F): this target has ingested on-sky history
  const obsNight = OBSERVED_TARGETS[String(t.id)];
  const obsMark = obsNight
    ? ` <span style="color:var(--ok);font-size:.72rem;white-space:nowrap"
         title="ingested observation(s); latest ${esc(prettyStamp(obsNight))}">observed ✓ ${esc(prettyStamp(obsNight))}</span>`
    : "";
  return `<tr class="q${brk ? " tierbreak" : ""}${mine ? " mine" : ""}" data-prog="${esc(t.program)}">
    <td>${chip(t.tier)}</td><td>${esc(t.name)}${obsMark}</td>
    <td><span class="dot" style="background:${RAW[t.program] || "var(--slate)"}"></span>${esc(t.program)}</td>
    <td class="num">${t.mag == null ? "—" : esc(t.mag)}</td><td>${st}</td>
    <td>${manage}</td></tr>`;
}

function renderQueue() {
  const caller = DATA.caller_program;
  const all = DATA.queue_targets || DATA.targets || [];
  // Split by instrument — LLAMAS and LDSS3 are parallel systems.
  const insts = ["LLAMAS", "LDSS3", "EITHER"].filter((i) => all.some((t) => t.instrument === i));
  $("qgroups").innerHTML = insts.map((inst) => {
    const rows = all.filter((t) => t.instrument === inst)
      .sort((a, b) => order[a.tier] - order[b.tier] || a.program.localeCompare(b.program));
    let last = null;
    const body = rows.map((t) => {
      const brk = t.tier !== last; last = t.tier;
      return queueRow(t, caller, brk);
    }).join("") || `<tr><td colspan="6" style="color:var(--faint)">no targets on this instrument</td></tr>`;
    return `<div class="qgroup">
      <h3 class="qgroup-h"><span class="ai ai-${esc(inst.toLowerCase())}">${esc(inst)}</span>
        <span class="qcount">${rows.length} target${rows.length === 1 ? "" : "s"}</span></h3>
      <div class="tablewrap"><table>
        <thead><tr><th>Tier</th><th>Target</th><th>Program</th><th>r</th><th>Status</th><th>Manage</th></tr></thead>
        <tbody>${body}</tbody></table></div></div>`;
  }).join("") || `<div style="color:var(--faint)">queue is empty</div>`;

  // filters (by program)
  const fbar = $("filters");
  fbar.innerHTML = ["All", ...PROGS].map((p) =>
    `<button aria-pressed="${p === queueFilter}" data-f="${esc(p)}">${p === "All" ? "All (shared)" : esc(p)}</button>`
  ).join("");
  fbar.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    queueFilter = b.dataset.f;
    [...fbar.children].forEach((x) => x.setAttribute("aria-pressed", x === b));
    applyFilter();
  };
  applyFilter();

  // per-row edit controls (event delegation on the groups container)
  const groups = $("qgroups");
  groups.onchange = (e) => {
    const sel = e.target.closest("select.pri-select");
    if (sel) changePriority(sel.dataset.id, sel.value);
  };
  groups.onclick = (e) => {
    const btn = e.target.closest("button.withdraw");
    if (btn) withdraw(btn.dataset.id, btn.dataset.name);
  };
}

function editControls(t) {
  const id = esc(idOf(t));
  const opts = TIERS.map((p) => `<option value="${p}"${p === t.tier ? " selected" : ""}>${p}</option>`).join("");
  const disabled = usingSample ? "disabled" : "";
  return `<span class="rowedit">
    <select class="pri-select" data-id="${id}" aria-label="priority for ${esc(t.name)}" ${disabled}>${opts}</select>
    <button class="btn danger withdraw" data-id="${id}" data-name="${esc(t.name)}" ${disabled}>Withdraw</button>
  </span>`;
}

function applyFilter() {
  $("qgroups").querySelectorAll("tr[data-prog]").forEach((tr) =>
    tr.classList.toggle("hide", queueFilter !== "All" && tr.dataset.prog !== queueFilter)
  );
}

function renderFoot() {
  $("foot").innerHTML =
    "Live MAGNETS shared queue for " + esc(DATA.plan.date) + " (" + esc(DATA.plan.instrument) + "). " +
    "Groups edit their own targets through the API " +
    "(<code>GET /v1/dashboard</code>, <code>POST/PATCH/DELETE /v1/targets</code>); the scheduler cross-hashes by " +
    "priority and observability against each group's derived " + esc(DATA.plan.instrument) + " budget.";
}

// ---- wire the add form + go ------------------------------------------
$("addform").addEventListener("submit", addTarget);

// ---- ETC pre-fill (stamped #2) -----------------------------------------
// Typing an anticipated magnitude pre-fills the exposure from the LLAMAS
// S/N ETC as an EDITABLE suggestion (canonical 3-sub-exposure CR triplet).
// Exposure ownership is respected: only an empty field — or our own previous
// suggestion — is ever (re)filled; anything the user typed stays untouched.
let etcSuggested = null;   // the last value WE wrote into t-exp
async function suggestExposure() {
  const magS = $("t-mag").value.trim();
  const note = $("etc-note");
  if (magS === "" || isNaN(Number(magS))) return;
  const expEl = $("t-exp");
  const cur = expEl.value.trim();
  if (cur !== "" && cur !== etcSuggested) return;   // user owns this value
  try {
    const r = await fetch("/v1/etc?mag=" + encodeURIComponent(magS),
                          { credentials: "include" });
    if (!r.ok) { note.hidden = true; return; }      // 401/503/422: stay quiet
    const b = await r.json();
    etcSuggested = String(Math.round(b.minutes));
    expEl.value = etcSuggested;
    note.textContent = `suggested for mag ${magS} (editable): `
      + `${b.n_exposures} × ${Math.round(b.exposure_seconds)} s ≈ ${b.minutes} min `
      + `(LLAMAS ETC, binned S/N ${b.snr})`
      + (b.extrapolated ? " — outside the calibrated range, treat as rough" : "");
    note.hidden = false;
  } catch (e) { /* offline / no backend: leave the field alone */ }
}
$("t-mag").addEventListener("change", suggestExposure);
boot();
