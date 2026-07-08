"""
ConditionalGates  —  Step 30
------------------------------
Implements four resource-aware activation gates:

    Gate 1  STN gate       : only activate when U > threshold
                             (uncertainty-triggered)
    Gate 2  Action gate    : only release when C > threshold
                             (confidence-triggered)
    Gate 3  Neuromod gate  : only update when delta > threshold
                             (event-driven neuromodulation)
    Gate 4  Pathway gate   : recruit indirect/hyperdirect
                             only when needed
                             (conditional pathway recruitment)

Energy model:
    Each component has an energy cost per activation.
    Gating saves energy by skipping unnecessary activations.

    Component costs (relative units):
        STN activation       : 3.0
        Indirect pathway     : 2.0
        Hyperdirect pathway  : 2.5
        Neuromodulator update: 1.5
        Direct pathway       : 1.0  (always active, baseline)
        Action release       : 0.5
"""

import numpy as np
from collections import deque


# Relative energy costs per component (normalised)
COMPONENT_COSTS = {
    "direct"       : 1.0,
    "indirect"     : 2.0,
    "hyperdirect"  : 2.5,
    "stn"          : 3.0,
    "neuromod"     : 1.5,
    "action_release": 0.5,
    "plasticity"   : 2.0,
    "reasoning"    : 1.0,
}


class ConditionalGates:

    def __init__(self,
                 stn_U_threshold    : float = 0.55,
                 action_C_threshold : float = 0.55,
                 neuromod_threshold : float = 0.05,
                 conflict_threshold : float = 0.25,
                 dt                 : float = 0.1e-3):
        """
        stn_U_threshold   : activate STN only when U > this
        action_C_threshold: release action only when C > this
        neuromod_threshold: update neuromod only when |delta| > this
        conflict_threshold: activate indirect/hyperdirect if
                            conflict > this
        """
        self.stn_U_thr      = stn_U_threshold
        self.action_C_thr   = action_C_threshold
        self.nm_thr         = neuromod_threshold
        self.conflict_thr   = conflict_threshold
        self.dt             = dt

        # Gate states (True = open / active)
        self.stn_gate       = False
        self.action_gate    = False
        self.neuromod_gate  = False
        self.indirect_gate  = False
        self.hyperdirect_gate = False

        # Energy accounting
        self.energy_used    = 0.0
        self.energy_saved   = 0.0
        self.step_count     = 0

        # Per-gate activation counts
        self.gate_counts    = {
            "stn"        : 0,
            "action"     : 0,
            "neuromod"   : 0,
            "indirect"   : 0,
            "hyperdirect": 0,
        }

        # History
        self.gate_history   = deque(maxlen=2000)
        self.energy_history = deque(maxlen=2000)

    def reset(self) -> None:
        self.stn_gate         = False
        self.action_gate      = False
        self.neuromod_gate    = False
        self.indirect_gate    = False
        self.hyperdirect_gate = False
        self.energy_used      = 0.0
        self.energy_saved     = 0.0
        self.step_count       = 0
        for k in self.gate_counts:
            self.gate_counts[k] = 0
        self.gate_history.clear()
        self.energy_history.clear()

    def evaluate(self,
                  U              : float,
                  C              : float,
                  delta_prime    : float,
                  conflict_score : float,
                  stn_trigger    : float = 0.0) -> dict:
        """
        Evaluates all four gates.

        Gate 1 — STN:
            Activate if U > stn_U_threshold
            OR conflict > conflict_threshold
            OR external stn_trigger > 0.5

        Gate 2 — Action release:
            Open if C > action_C_threshold
            AND NOT STN suppressing

        Gate 3 — Neuromodulation:
            Fire if |delta_prime| > neuromod_threshold

        Gate 4 — Pathway recruitment:
            Indirect  : activate if conflict > threshold
            Hyperdirect: activate if U > threshold AND conflict high

        Returns gate state dict and energy cost estimate.
        """
        self.step_count += 1

        # ── Gate 1: STN (uncertainty-triggered) ───────────────
        stn_by_U        = float(U)        > self.stn_U_thr
        stn_by_conflict = float(conflict_score) > self.conflict_thr
        stn_by_trigger  = float(stn_trigger) > 0.5
        self.stn_gate   = stn_by_U or stn_by_conflict or stn_by_trigger

        if self.stn_gate:
            self.gate_counts["stn"] += 1

        # ── Gate 2: Action release (confidence-triggered) ─────
        self.action_gate = (float(C) > self.action_C_thr
                            and not self.stn_gate)
        if self.action_gate:
            self.gate_counts["action"] += 1

        # ── Gate 3: Neuromodulation (event-driven) ────────────
        self.neuromod_gate = (abs(float(delta_prime))
                              > self.nm_thr)
        if self.neuromod_gate:
            self.gate_counts["neuromod"] += 1

        # ── Gate 4: Pathway recruitment ───────────────────────
        self.indirect_gate = (float(conflict_score) > self.conflict_thr
                               or float(U) > 0.4)
        self.hyperdirect_gate = (float(U) > self.stn_U_thr
                                  and float(conflict_score)
                                  > self.conflict_thr * 0.8)

        if self.indirect_gate:
            self.gate_counts["indirect"] += 1
        if self.hyperdirect_gate:
            self.gate_counts["hyperdirect"] += 1

        # ── Energy accounting ──────────────────────────────────
        # Compute energy used vs hypothetical always-on energy
        always_on = sum(COMPONENT_COSTS.values())

        used = (COMPONENT_COSTS["direct"]                    # always on
                + (COMPONENT_COSTS["stn"]        if self.stn_gate        else 0)
                + (COMPONENT_COSTS["action_release"] if self.action_gate else 0)
                + (COMPONENT_COSTS["neuromod"]   if self.neuromod_gate   else 0)
                + (COMPONENT_COSTS["indirect"]   if self.indirect_gate   else 0)
                + (COMPONENT_COSTS["hyperdirect"]if self.hyperdirect_gate else 0)
                + COMPONENT_COSTS["plasticity"]  * float(self.neuromod_gate)
                + COMPONENT_COSTS["reasoning"]   * 0.5)  # always partial

        saved = always_on - used
        self.energy_used  += used
        self.energy_saved += max(saved, 0.0)

        gate_state = {
            "stn_gate"        : bool(self.stn_gate),
            "action_gate"     : bool(self.action_gate),
            "neuromod_gate"   : bool(self.neuromod_gate),
            "indirect_gate"   : bool(self.indirect_gate),
            "hyperdirect_gate": bool(self.hyperdirect_gate),
            "energy_step"     : float(used),
            "energy_saved_step": float(max(saved, 0.0)),
            "always_on_cost"  : float(always_on),
        }

        self.gate_history.append(gate_state.copy())
        self.energy_history.append(float(used))

        return gate_state

    def cumulative_efficiency(self) -> float:
        """
        Fraction of potential energy saved vs always-on baseline.
        """
        potential = (self.step_count
                     * sum(COMPONENT_COSTS.values()))
        return float(self.energy_saved
                     / max(potential, 1e-9))

    def activation_rates(self) -> dict:
        n = max(self.step_count, 1)
        return {name: cnt / n
                for name, cnt in self.gate_counts.items()}

    def gate_summary(self) -> dict:
        return {
            "step_count"       : self.step_count,
            "cumulative_eff"   : self.cumulative_efficiency(),
            "activation_rates" : self.activation_rates(),
            "total_energy_used": float(self.energy_used),
            "total_energy_saved":float(self.energy_saved),
            "gate_counts"      : dict(self.gate_counts),
        }