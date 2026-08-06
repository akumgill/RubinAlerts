# Deployment Plan — MAGNETS Queue Website

**Goal.** A live site where collaboration members can, for the shared observing
queue:

1. **Submit programmatically** — POST targets to the API with a per-group key
   (for groups that run their own pipelines / scripts), and
2. **Log in through the browser** and add / remove / reprioritize their own
   targets by hand,

plus a **shared read view** of the queue and tonight's observing plan.

This is a small, trusted service (~6 groups, ~a dozen people). The plan is
sized accordingly — no OAuth, no microservices, no Kubernetes. Fastest path to
something real that people can read *and* write.

---

## 1. Architecture — one small service

A single FastAPI process does everything:

```
                         ┌─────────────────────────────────────┐
  group's script ─POST──▶│  FastAPI (api/app.py)                │
  (bearer key)           │   /v1/targets   submit/list/patch/…  │──▶ SQLite file
                         │   /v1/queue     shared read           │    (persistent disk)
  person's browser ─────▶│   /v1/plan/preview  runs orchestrator │
  (login cookie)         │   /            the dashboard (static) │──▶ orchestrator
                         │   /login /logout  browser auth        │    (in-process)
                         └─────────────────────────────────────┘
```

- **No separate frontend/backend, no DB server.** The dashboard is static
  HTML/JS served by the same app; the queue lives in one SQLite file; the
  scheduler (orchestrator) runs in-process for plan previews (a few seconds —
  fine at this scale).

## 2. Stack — cheap, easy, fast

| Piece | Choice | Why |
|---|---|---|
| Host | **Render** (or Fly.io) | connect the GitHub repo → auto-deploy on push, free HTTPS + URL, persistent disk for SQLite, no server admin |
| Cost | **$0** free tier (cold-starts when idle) or **~$7/mo** always-on | a domain is optional (~$12/yr) |
| Backend | FastAPI + uvicorn | already written (`api/app.py`) |
| Storage | SQLite (stdlib / SQLModel) | one file; more than enough for 6 groups |
| Frontend | the existing dashboard, served static, `fetch()`-ing the API | render code already done |
| Packaging | a **Dockerfile** pinning deps (astropy, sncosmo, …) | makes the host interchangeable — migrate to an institutional VM later with zero code change |

## 3. Auth — two doors, both lightweight

Everything resolves to a **program** (a group). Two ways to prove who you are:

1. **Programmatic (machines):** `Authorization: Bearer <group-key>`. Already
   built and tested. Issue 6 keys, stored as env secrets.
2. **Browser (humans):** a `/login` page — pick your group, enter a password.
   The server checks it against a small table (6 rows), sets a signed session
   cookie carrying the program identity. The dashboard's write buttons then call
   the API with the cookie (`credentials: 'include'`); no key handling in JS.

Write endpoints accept **either** a valid bearer key **or** a valid session, and
both map to a program. **Writes are scoped to your own group** (you can only
edit your own targets); **reads** (queue + plan) are open to any logged-in user.

## 4. Build steps (all mine to do)

| # | Step | Effort |
|---|---|---|
| 1 | **SQLite persistence** — swap the in-memory store in `TargetQueueService` for SQLite; the service interface is unchanged. + a few DB-backed tests. | ~0.5 d |
| 2 | **Browser auth** — a `/login`/`/logout` + signed-cookie session; make the API's key check accept "bearer **or** session". A tiny group→password table. | ~0.5 d |
| 3 | **Interactive UI** — extend the dashboard: an "add target" form, a priority dropdown and a withdraw button on *your own* rows; each calls `POST`/`PATCH`/`DELETE` and re-fetches. Read view unchanged for everyone. | ~1 d |
| 4 | **Serve the UI live** — mount the dashboard static from FastAPI; swap its embedded snapshot for `fetch('/v1/queue')` + `fetch('/v1/plan/preview')`. | ~0.5 d |
| 5 | **Dockerfile + `render.yaml`** — pin deps, declare the persistent disk for SQLite. | ~0.5 d |
| 6 | **Deploy + wire secrets** — push, point Render at the repo, set the 6 keys/passwords + real allocations as env, smoke-test with 2 groups. | ~0.5 d |

**Total ≈ 3–3.5 focused days.**

## 5. Milestones (fastest useful thing first)

- **M1 — durable, authed API live (~2 days):** steps 1 + 5 + 6, minus the fancy
  write UI. Groups with scripts can already `POST` (programmatic write), and
  everyone can view. This is the fastest path to a real, writable service.
- **M2 — browser edit (~+1.5 days):** steps 2 + 3 + 4. Humans log in and
  add/remove/reprioritize by hand.
- **Later:** night tabs for multiple evenings, per-instrument views, the
  ORACLE/Villar feed adapter, post-night reconcile UI.

## 6. Decisions needed (not code)

- **Whose account** hosts it. Your own Render account is fine to start and fully
  migratable (that's what the Dockerfile buys). An institutional VM is a later,
  free, no-code-change move.
- **Login granularity:** one shared password per group (6 logins, fastest) vs
  individual accounts. Recommend per-group passwords now; individuals later only
  if needed.
- **Reads public or login-gated:** recommend login-gated (collaboration-internal).
- **Real allocations + keys:** issue the 6 keys and set each group's per-
  instrument budgets (from the schedule; total-per-instrument model).

## 7. Security posture (right-sized)

HTTPS (platform-provided), bearer keys + `httpOnly`/`secure` session cookies,
secrets in env not the repo. That is proportionate for a trusted ~dozen-person
collaboration — no rate-limiting, OAuth, or pen-testing warranted.

## 8. Why this is small — most of it exists

Already built and tested: the queue service (submit/upsert/list/patch/withdraw/
queue/preview), the scheduler with instrument routing and total-per-instrument
budgets, the dashboard render code, and the FastAPI app skeleton. Deployment is
really just **persistence + login + wiring the UI's write actions + Docker +
push** — the four small steps above, not a rebuild.
