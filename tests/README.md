# Tests

Two buckets, for two audiences.

## `tests/pipeline/` — alert follow-up → Type Ia candidate generation
Everything that turns broker alerts into ranked candidates: broker clients
(Fink, ALeRCE), multi-broker aggregation, light-curve fitting (SALT2, the
multi-type template tournament, Villar, parabola), merit scoring, and the
wide-sky selection funnel. This is how the **Ia (Stubbs) program** produces its
target list — only that group needs to care about it.

```
pytest tests/pipeline
```

## `tests/orchestrator/` — the scheduler, given a set of inputs
Everything that turns a queue of targets (from any group, any source, either
instrument) into an observing plan: the composite-score prioritizer, per-program
budgets and the fairness/parity band, the cross-night integration ledger,
standard stars, mandatory/ToO reservations, normalization of manual + pipeline
inputs, and the submission API / queue service. This is the
**collaboration-facing** half — source- and instrument-agnostic.

```
pytest tests/orchestrator
```

## Shared
`tests/conftest.py` holds the in-memory fixtures (hierarchical, so both buckets
inherit them). `tests/test_smoke.py` is a cross-cutting import/fixture check.
No test contacts a live broker/DB/API; `@pytest.mark.live` tests are skipped
unless you pass `--run-live`.
