"""
TrainingConfig — centralises all hyperparameters for Phase 11.
Single source of truth for the complete training loop.
"""

from dataclasses import dataclass, field
from typing      import List


@dataclass
class TrainingConfig:

    # ── Simulation ─────────────────────────────────────────────
    dt            : float = 0.1e-3    # timestep (s)
    episode_steps : int   = 10000     # steps per episode (1 second)
    n_episodes    : int   = 10        # training episodes
    seed          : int   = 42

    # ── Network ────────────────────────────────────────────────
    n_actions     : int   = 4
    state_dim     : int   = 12        # 2*n_actions + 4
    n_d1_per_action: int  = 20
    n_d2_per_action: int  = 20

    # ── Phase 3: Bayesian reasoning ────────────────────────────
    belief_window : int   = 100       # spike window (steps)
    belief_lam    : float = 0.85      # temporal smoothing
    alpha_prior   : float = 0.05      # prior learning rate

    # ── Phase 4: BG pathways ───────────────────────────────────
    conflict_eps  : float = 0.3

    # ── Phase 5: Action gating ─────────────────────────────────
    theta_0       : float = 0.5
    gate_beta     : float = 0.4
    gate_kappa    : float = 0.3
    refractory_ms : float = 150.0

    # ── Phase 6: RL ────────────────────────────────────────────
    gamma         : float = 0.99
    alpha_rl      : float = 0.05

    # ── Phase 7: STDE plasticity ───────────────────────────────
    base_alpha    : float = 0.05
    min_delta     : float = 0.03
    prune_every   : int   = 2000

    # ── Phase 8: Neuromodulators ───────────────────────────────
    alpha_0       : float = 0.05
    eta           : float = 0.10
    omega_d       : float = 0.50
    omega_s       : float = 0.30
    omega_n       : float = 0.20

    # ── Phase 9: Reasoning ─────────────────────────────────────
    action_names  : List[str] = field(default_factory=lambda:
                    ["reach_left","reach_right",
                     "press_button","wait"])
    explain_every : int   = 1000      # steps between full explanations

    # ── Phase 10: Energy ───────────────────────────────────────
    target_sparsity   : float = 0.08
    stn_U_threshold   : float = 0.55
    action_C_threshold: float = 0.55
    episode_budget_nJ : float = 5000.0

    # ── Training loop ──────────────────────────────────────────
    correct_action    : int   = 2     # rewarded action for demo task
    reward_correct    : float = 1.0
    reward_wrong      : float = -0.1
    log_every         : int   = 500   # steps between console logs
    save_every        : int   = 5000  # steps between checkpoints
    results_dir       : str   = "results"
    checkpoint_dir    : str   = "checkpoints"

    def validate(self) -> None:
        assert self.dt > 0
        assert self.n_actions > 0
        assert self.n_episodes > 0
        assert self.episode_steps > 0
        assert 0 < self.belief_lam < 1
        assert 0 < self.alpha_0 <= 1
        assert self.state_dim == 2 * self.n_actions + 4
        print("  [CONFIG] All parameters validated.")