"""
ThalamocorticalRelay  —  Step 15
----------------------------------
Once the GPi gate opens for action a:
  1. Thalamus relay neurons for channel a are disinhibited
  2. Thalamus drives motor cortex — action is executed
  3. Competing action channels remain suppressed
  4. Lateral inhibition prevents simultaneous multi-action release

Biological basis:
  GPi inhibits thalamic VL/VA nuclei tonically.
  When GPi_a drops below threshold:
    - VL/VA neurons for action a increase firing
    - Motor cortex receives thalamic excitation
    - Supplementary motor area prepares the motor command
    - Primary motor cortex executes it

Additional features:
  - Refractory period after release (biological 100–200 ms)
  - Suppression of non-winning channels
  - Thalamic burst mode on first disinhibition
  - Release confidence score
"""

import numpy as np
from collections import deque


class ThalamocorticalRelay:

    def __init__(self,
                 n_actions         : int,
                 refractory_ms     : float = 150.0,
                 thal_excit        : float = 2.0,
                 suppression_factor: float = 0.8,
                 dt                : float = 0.1e-3):
        """
        n_actions          : number of competing action channels
        refractory_ms      : post-release refractory period (ms)
        thal_excit         : thalamic excitation strength on release
        suppression_factor : how strongly non-winning channels suppressed
        dt                 : simulation timestep
        """
        self.n_actions          = n_actions
        self.refractory_steps   = int(refractory_ms * 1e-3 / dt)
        self.thal_excit         = thal_excit
        self.suppression_factor = suppression_factor
        self.dt                 = dt

        # Thalamic state per channel
        self.thal_activity      = np.zeros(n_actions)

        # Refractory counters per channel
        self.refractory_count   = np.zeros(n_actions, dtype=int)

        # Current released action (None if all suppressed)
        self.released_action    = None
        self.release_confidence = 0.0

        # Motor cortex output signal (scalar or per-action)
        self.motor_output       = np.zeros(n_actions)

        # Full decision log for explainability (Step 15 requirement)
        self.decision_log       = []

        # Running statistics
        self.total_releases     = 0
        self.release_history    = deque(maxlen=2000)

    def reset(self):
        self.thal_activity      = np.zeros(self.n_actions)
        self.refractory_count   = np.zeros(self.n_actions, dtype=int)
        self.released_action    = None
        self.release_confidence = 0.0
        self.motor_output       = np.zeros(self.n_actions)
        self.decision_log.clear()
        self.total_releases     = 0
        self.release_history.clear()

    def step(self,
             gpi_activity      : np.ndarray,
             threshold         : float,
             action_probs      : np.ndarray,
             U                 : float,
             C                 : float,
             conflict_score    : float,
             pathway_contribs  : dict,
             t_ms              : float = 0.0) -> dict:
        """
        One thalamocortical relay step.

        gpi_activity    : (A,) current GPi activity from Step 13
        threshold       : current adaptive threshold from Step 14
        action_probs    : (A,) P(a|s) from Phase 3 Bayesian pipeline
        U, C            : uncertainty and confidence from Phase 3
        conflict_score  : from Phase 4 hyperdirect pathway
        pathway_contribs: dict from GPiGateEngine.pathway_contributions()
        t_ms            : current simulation time (ms)

        Returns decision dict (also appended to decision_log).
        """
        gpi  = np.asarray(gpi_activity, dtype=float)
        prob = np.asarray(action_probs,  dtype=float)

        # Count down refractory periods
        self.refractory_count = np.maximum(
            self.refractory_count - 1, 0)

        # Identify which channels are below threshold AND not refractory
        below_thresh = gpi < threshold
        not_refrac   = self.refractory_count == 0
        eligible     = below_thresh & not_refrac

        # Thalamic activity update
        # Eligible channels: disinhibited (activity rises)
        # Ineligible channels: suppressed (activity falls)
        target_thal         = np.where(eligible,
                                        self.thal_excit, 0.0)
        self.thal_activity  = 0.7 * self.thal_activity + 0.3 * target_thal

        # Select winning action
        if eligible.any():
            # Winner = eligible channel with lowest GPi
            candidates          = np.where(eligible)[0]
            winner              = int(candidates[
                np.argmin(gpi[candidates])])
            self.released_action = winner

            # Compute release confidence
            gate_margin         = float(threshold - gpi[winner])
            prob_winner         = float(prob[winner]) if len(prob) > winner else 0.5
            self.release_confidence = float(
                np.clip(0.5 * gate_margin / threshold
                        + 0.5 * prob_winner, 0.0, 1.0))

            # Motor output: winner gets full excitation,
            # others get suppressed
            self.motor_output              = np.zeros(self.n_actions)
            self.motor_output[winner]      = self.thal_activity[winner]
            suppress_mask                  = np.ones(self.n_actions, dtype=bool)
            suppress_mask[winner]          = False
            self.motor_output[suppress_mask] *= (
                1.0 - self.suppression_factor)

            # Start refractory period for winner
            self.refractory_count[winner]  = self.refractory_steps

            self.total_releases += 1
            action_released = True

        else:
            self.released_action    = None
            self.release_confidence = 0.0
            self.motor_output       = np.zeros(self.n_actions)
            action_released         = False

        # Build complete decision record (Step 15 logging requirement)
        record = self._build_decision_record(
            gpi, threshold, action_released, prob,
            U, C, conflict_score, pathway_contribs, t_ms)

        self.decision_log.append(record)
        self.release_history.append(self.released_action)
        return record

    def _build_decision_record(self,
                                gpi, threshold,
                                action_released,
                                prob, U, C,
                                conflict_score,
                                pathway_contribs,
                                t_ms) -> dict:
        """
        Builds the full explainability record for this timestep.
        This is the data structure consumed by Phase 9 (XAI layer).
        """
        pc = pathway_contribs if pathway_contribs else {}
        return {
            # ── Core decision ────────────────────────────────────
            "t_ms"              : float(t_ms),
            "released_action"   : self.released_action,
            "action_released"   : bool(action_released),
            "release_confidence": float(self.release_confidence),

            # ── Gate state ───────────────────────────────────────
            "gpi_activity"      : gpi.copy(),
            "threshold"         : float(threshold),
            "gate_margins"      : (threshold - gpi).tolist(),
            "thal_activity"     : self.thal_activity.copy(),
            "motor_output"      : self.motor_output.copy(),

            # ── Uncertainty and confidence ───────────────────────
            "U"                 : float(U),
            "C"                 : float(C),

            # ── Pathway contributions (explainability) ───────────
            "pathway_go"        : pc.get("go",   np.zeros(self.n_actions)).tolist(),
            "pathway_nogo"      : pc.get("nogo", np.zeros(self.n_actions)).tolist(),
            "pathway_stn"       : pc.get("stn",  np.zeros(self.n_actions)).tolist(),
            "pathway_base"      : pc.get("base", np.zeros(self.n_actions)).tolist(),

            # ── Conflict ─────────────────────────────────────────
            "conflict_score"    : float(conflict_score),

            # ── Action probabilities ─────────────────────────────
            "action_probs"      : prob.tolist(),

            # ── Refractory state ─────────────────────────────────
            "refractory_ms"     : (self.refractory_count
                                   * self.dt * 1000).tolist(),
        }

    def suppressed_actions(self, threshold: float,
                            gpi: np.ndarray) -> list:
        """Returns list of actions that were suppressed (GPi ≥ θ)."""
        return [a for a in range(self.n_actions)
                if gpi[a] >= threshold]

    def relay_summary(self) -> dict:
        recent = list(self.release_history)
        releases = [a for a in recent if a is not None]
        return {
            "total_releases"    : self.total_releases,
            "release_rate"      : len(releases) / max(len(recent), 1),
            "most_selected"     : (int(max(set(releases),
                                           key=releases.count))
                                   if releases else None),
            "last_action"       : self.released_action,
            "last_confidence"   : float(self.release_confidence),
        }