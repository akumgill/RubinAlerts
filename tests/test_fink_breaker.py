"""Unit tests for the pure FinkBreaker pause-and-resume circuit breaker.

No sleeping, no network — the helper is pure decision logic.
"""

from core.fink_breaker import (
    FinkBreaker, ACTION_FETCH, ACTION_COOLDOWN, ACTION_PROCEED,
)


def test_empty_ok_results_never_trip_breaker():
    """A long run of empty-OK results (record_success) never trips the
    breaker — it always says 'fetch'."""
    b = FinkBreaker(threshold=5, max_cooldowns=3)
    for _ in range(50):
        assert b.decide() == ACTION_FETCH
        b.record_success()
    assert b.consecutive_failures == 0
    assert b.cooldowns_used == 0


def test_n_consecutive_failures_triggers_cooldown_and_reset():
    """N consecutive transport failures triggers a cooldown (not abort);
    recording the cooldown resets the streak so processing resumes."""
    b = FinkBreaker(threshold=5, max_cooldowns=3)
    for _ in range(5):
        assert b.decide() == ACTION_FETCH
        b.record_failure()

    # Threshold reached, cooldown budget available -> cooldown.
    assert b.decide() == ACTION_COOLDOWN
    b.record_cooldown()

    # Streak reset, one cooldown spent, processing resumes.
    assert b.consecutive_failures == 0
    assert b.cooldowns_used == 1
    assert b.decide() == ACTION_FETCH


def test_after_cooldown_cap_proceeds_without_aborting():
    """Once the cooldown budget is exhausted, the breaker says 'proceed'
    (no sleep) — it never aborts the run."""
    b = FinkBreaker(threshold=5, max_cooldowns=2)

    # Trip and spend both cooldowns.
    for _ in range(2):
        for _ in range(5):
            b.record_failure()
        assert b.decide() == ACTION_COOLDOWN
        b.record_cooldown()

    assert b.cooldowns_used == 2

    # Trip again: budget exhausted -> proceed, not cooldown, not abort.
    for _ in range(5):
        b.record_failure()
    assert b.decide() == ACTION_PROCEED


def test_failure_then_success_resets_streak():
    """A success in the middle of a failure run resets the streak so the
    breaker keeps fetching."""
    b = FinkBreaker(threshold=3, max_cooldowns=3)
    b.record_failure()
    b.record_failure()
    assert b.decide() == ACTION_FETCH
    b.record_success()
    assert b.consecutive_failures == 0
    assert b.decide() == ACTION_FETCH
