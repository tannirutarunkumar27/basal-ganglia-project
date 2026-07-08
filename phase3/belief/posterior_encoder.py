"""
PosteriorBeliefEncoder
-----------------------
Implements Step 5:  Va = log P(s|a) + log P(a)

Combines:
  - log-likelihood  log P(s|a)  from SpikeEvidenceExtractor
  - log-prior       log P(a)    from PriorTracker

Returns the posterior-like belief score Va for each action.
This is the central quantity that drives all downstream
reasoning, action selection, and uncertainty estimation.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from belief.spike_evidence import SpikeEvidenceExtractor
from belief.prior_tracker  import PriorTracker


class PosteriorBeliefEncoder:

    def __init__(self,
                 n_actions      : int,
                 n_neurons_total: int,
                 window_steps   : int   = 100,
                 alpha_prior    : float = 0.05,
                 dt             : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        self.evidence_extractor = SpikeEvidenceExtractor(
            n_actions       = n_actions,
            n_neurons_total = n_neurons_total,
            window_steps    = window_steps,
            dt              = dt,
        )

        self.prior_tracker = PriorTracker(
            n_actions   = n_actions,
            alpha_prior = alpha_prior,
        )

        # Current belief scores Va — shape (A,)
        self.V = np.zeros(n_actions)

        # History for temporal updating (Step 6)
        self.V_history = []

    def reset(self):
        self.evidence_extractor.reset()
        self.prior_tracker.reset()
        self.V         = np.zeros(self.n_actions)
        self.V_history = []

    def encode(self, spike_vector: np.ndarray) -> np.ndarray:
        """
        Step 5 core computation:
            Va = log P(s|a) + log P(a)

        spike_vector: bool/float array from cortex or bayesian_layer
        Returns V (A,) — belief score per action.
        """
        # Update spike evidence window
        self.evidence_extractor.update(spike_vector)

        # log P(s|a) from spike counts
        log_likelihood = self.evidence_extractor.log_likelihood()

        # log P(a) from learned prior
        log_prior = self.prior_tracker.log_prior()

        # Posterior-like belief score
        self.V = log_likelihood + log_prior

        # Store for temporal reasoning
        self.V_history.append(self.V.copy())

        return self.V.copy()

    def update_prior(self, action: int, reward: float):
        """Propagate reward signal to update P(a)."""
        self.prior_tracker.update(action, reward)

    def belief_summary(self) -> dict:
        return {
            "V"              : self.V.copy(),
            "log_likelihood" : self.evidence_extractor.log_likelihood(),
            "log_prior"      : self.prior_tracker.log_prior(),
            "prior"          : self.prior_tracker.prior.copy(),
            "likelihood"     : self.evidence_extractor.likelihood(),
            "firing_rates_hz": self.evidence_extractor.firing_rate_hz(),
        }