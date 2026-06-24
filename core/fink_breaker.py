"""Pure, unit-testable circuit-breaker for the Fink photometry pass.

The breaker PAUSES (cooldown) and RESUMES rather than aborting the night.
Its only job is the decision logic; the actual sleeping is kept OUT of the
helper so tests never wait.

Decision model
--------------
The loop, before fetching each candidate, asks the breaker what to do given
how many consecutive *transport* failures it has seen and how many cooldowns
it has already spent:

    'fetch'    -> below threshold; go fetch this candidate normally.
    'cooldown' -> threshold reached and cooldown budget remains; the caller
                  should sleep ``cooldown_seconds`` (a recovery pause), then
                  call ``record_cooldown()`` to reset the consecutive counter
                  and spend one cooldown, then CONTINUE processing.
    'proceed'  -> threshold reached but the cooldown budget is exhausted; the
                  caller should NOT sleep, must reset the consecutive counter,
                  and continues processing. The night is NEVER aborted.

Only a transport error (``get_light_curve`` returns ``None``) counts as a
failure. An empty DataFrame (object queried OK but has no photometry) is NOT a
failure — callers should skip it without calling ``record_failure``.
"""

from dataclasses import dataclass

FINK_MAX_CONSECUTIVE_FAILURES = 5
FINK_MAX_COOLDOWNS = 3
FINK_COOLDOWN_SECONDS = 30

# Action constants returned by FinkBreaker.decide().
ACTION_FETCH = "fetch"
ACTION_COOLDOWN = "cooldown"
ACTION_PROCEED = "proceed"


@dataclass
class FinkBreaker:
    """Pause-and-resume circuit breaker for the Fink photometry loop.

    Parameters
    ----------
    threshold : int
        Consecutive transport failures that trip a cooldown.
    max_cooldowns : int
        Maximum number of cooldown pauses before the breaker stops sleeping
        (but still keeps processing).
    cooldown_seconds : float
        How long the caller should sleep on a cooldown. The breaker never
        sleeps itself; this is advisory for the caller.
    """

    threshold: int = FINK_MAX_CONSECUTIVE_FAILURES
    max_cooldowns: int = FINK_MAX_COOLDOWNS
    cooldown_seconds: float = FINK_COOLDOWN_SECONDS
    consecutive_failures: int = 0
    cooldowns_used: int = 0

    def decide(self) -> str:
        """Return the action for the current state (pure; no side effects)."""
        if self.consecutive_failures < self.threshold:
            return ACTION_FETCH
        if self.cooldowns_used < self.max_cooldowns:
            return ACTION_COOLDOWN
        return ACTION_PROCEED

    def record_success(self) -> None:
        """A candidate yielded data (or was empty-OK): reset the streak."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """A candidate hit a transport error (get_light_curve returned None)."""
        self.consecutive_failures += 1

    def record_cooldown(self) -> None:
        """Spend one cooldown and reset the consecutive-failure streak."""
        self.cooldowns_used += 1
        self.consecutive_failures = 0
