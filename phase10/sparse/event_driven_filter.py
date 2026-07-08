"""
EventDrivenFilter  —  Step 29 (technique 2)
--------------------------------------------
Suppresses redundant updates — only propagates signals
when the state has changed meaningfully.

Three event types:
    1. Spike events    : only forward if spikes occurred
    2. Value events    : only propagate if |dV| > threshold
    3. Reward events   : only trigger plasticity on reward

Biological analog: event-driven neural computation.
In silicon: this directly maps to Intel Loihi / IBM TrueNorth
spike-event queues that only wake up on activity.
"""

import numpy as np
from collections import deque


class EventDrivenFilter:

    def __init__(self,
                 value_threshold   : float = 0.01,
                 reward_threshold  : float = 0.05,
                 dt                : float = 0.1e-3):

        self.value_threshold  = value_threshold
        self.reward_threshold = reward_threshold
        self.dt               = dt

        # Previous values for delta comparison
        self._prev_V    = None
        self._prev_U    = None
        self._prev_DA   = None

        # Event counters
        self.spike_events   = 0
        self.value_events   = 0
        self.reward_events  = 0
        self.total_steps    = 0
        self.skipped_steps  = 0

        # History
        self.event_rate_history = deque(maxlen=1000)

    def reset(self) -> None:
        self._prev_V   = None
        self._prev_U   = None
        self._prev_DA  = None
        self.spike_events  = 0
        self.value_events  = 0
        self.reward_events = 0
        self.total_steps   = 0
        self.skipped_steps = 0
        self.event_rate_history.clear()

    def check_spike_event(self,
                           spikes: np.ndarray) -> bool:
        """
        Returns True (fire event) only if any spike occurred.
        Saves computation on silent timesteps.
        """
        self.total_steps += 1
        fired = bool(np.any(np.asarray(spikes, dtype=bool)))
        if fired:
            self.spike_events += 1
        else:
            self.skipped_steps += 1
        return fired

    def check_value_event(self,
                           V_combined: np.ndarray,
                           U        : float) -> bool:
        """
        Returns True only if belief or uncertainty changed
        by more than value_threshold.
        """
        V = np.asarray(V_combined, dtype=float)

        changed = False
        if self._prev_V is None:
            changed = True
        else:
            dV = float(np.abs(V - self._prev_V).max())
            dU = abs(float(U) - float(self._prev_U or 0.0))
            changed = (dV > self.value_threshold
                       or dU > self.value_threshold)

        if changed:
            self._prev_V = V.copy()
            self._prev_U = float(U)
            self.value_events += 1

        return changed

    def check_reward_event(self,
                            delta_prime: float,
                            DA         : float) -> bool:
        """
        Returns True only if dopamine signal exceeds threshold.
        Plasticity updates are only triggered on reward events.
        """
        significant = (abs(float(delta_prime)) > self.reward_threshold
                       or abs(float(DA) - float(
                           self._prev_DA or 0.5)) > self.reward_threshold)
        if significant:
            self._prev_DA = float(DA)
            self.reward_events += 1
        return significant

    def event_rate(self) -> dict:
        """
        Returns fraction of steps that triggered each event type.
        """
        n = max(self.total_steps, 1)
        return {
            "spike_event_rate"  : self.spike_events  / n,
            "value_event_rate"  : self.value_events  / n,
            "reward_event_rate" : self.reward_events / n,
            "skip_rate"         : self.skipped_steps / n,
            "total_steps"       : self.total_steps,
        }