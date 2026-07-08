"""
RLEngine — Phase 6 integration.
Combines Steps 16-19 into one update cycle.
"""

"""
RLEngine — Phase 6 integration.
Combines Steps 16-19 into one update cycle.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from actor.striatal_actor          import StriatalActor
from critics.multi_critic          import MultiCriticSystem
from risk.risk_module              import RiskModule
from dopamine.predictive_dopamine  import PredictiveDopamineModule


class RLEngine:

    def __init__(self,
                 n_actions      : int,
                 state_dim      : int,
                 n_d1_per_action: int   = 20,
                 gamma          : float = 0.99,
                 dt             : float = 0.1e-3):

        self.n_actions = n_actions
        self.state_dim = state_dim
        self.dt        = dt

        self.actor = StriatalActor(
            n_actions, n_d1_per_action, dt=dt)
        self.multi_critic = MultiCriticSystem(
            state_dim, n_actions, dt=dt)
        self.risk = RiskModule(n_actions, dt=dt)
        self.dopamine = PredictiveDopamineModule(n_actions, dt=dt)

        self.current_state  = np.zeros(state_dim)
        self.step_count     = 0
        self.episode_reward = 0.0
        self.rl_log         = []

    def reset(self):
        self.actor.reset()
        self.multi_critic.reset()
        self.risk.reset()
        self.dopamine.reset()
        self.current_state  = np.zeros(self.state_dim)
        self.step_count     = 0
        self.episode_reward = 0.0
        self.rl_log.clear()

    def build_state(self,
                     d1_spikes      : np.ndarray,
                     belief_scores  : np.ndarray,
                     U              : float,
                     C              : float,
                     dopamine_level : float) -> np.ndarray:
        """Constructs the state vector for the critics."""
        d1_rate = self.actor.encode_d1_activity(d1_spikes, dopamine_level)

        V = np.asarray(belief_scores, dtype=float)
        v_min   = V.min()
        v_range = V.max() - v_min          # replaces ptp()
        V_norm  = (V - v_min) / (v_range + 1e-8)
        V_norm  = (V_norm[:self.n_actions]
                   if len(V_norm) >= self.n_actions
                   else np.pad(V_norm,
                               (0, self.n_actions - len(V_norm))))

        state = np.concatenate([
            d1_rate,
            V_norm,
            [float(U), float(C), float(dopamine_level),
             float(self.step_count / 10000.0)]
        ])

        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]

        self.current_state = state
        return state

    def step(self,
             d1_spikes      : np.ndarray,
             belief_scores  : np.ndarray,
             raw_reward     : float,
             action         : int,
             U              : float,
             C              : float,
             conflict_score : float,
             stn_burst      : bool,
             dopamine_level : float = 1.0,
             done           : bool  = False) -> dict:
        """Full Phase 6 RL update step."""
        self.step_count     += 1
        self.episode_reward += raw_reward

        state      = self.build_state(d1_spikes, belief_scores,
                                       U, C, dopamine_level)
        next_state = state + np.random.randn(self.state_dim) * 0.05

        # Step 19a: predict dopamine before computing delta
        D_pred = self.dopamine.predict(belief_scores, U, C)

        # Step 18: risk statistics and rho
        self.risk.update_statistics(action, raw_reward)
        rho    = self.risk.adapt_rho(U, conflict_score,
                                      recent_loss=min(raw_reward, 0.0))
        Q_risk = self.risk.compute_q_risk()

        # Meta learning rate (Phase 8 preview)
        alpha_t = 0.05 * (1.0 + 2.0 * U)

        # Step 17: multi-critic TD errors
        mc_out = self.multi_critic.step(
            raw_reward     = raw_reward,
            state          = state,
            next_state     = next_state,
            action         = action,
            U              = U,
            C              = C,
            conflict_score = conflict_score,
            stn_burst      = stn_burst,
            rho            = rho,
            done           = done,
            alpha_override = alpha_t)

        delta_total = mc_out["delta_total"]

        # Step 19b: combine observed + predicted dopamine
        delta_prime = self.dopamine.combine_signals(delta_total)
        self.dopamine.update_predictor(raw_reward, belief_scores, U, C)

        # Step 16: actor update
        self.actor.compute_policy(
            d1_rate       = self.actor.d1_rate,
            belief_scores = belief_scores)
        self.actor.update(delta_prime, action)
        self.actor.adapt_temperature(U)

        tonic_da = self.dopamine.tonic_dopamine_level()

        result = {
            # Core RL outputs
            "delta_total"   : float(delta_total),
            "delta_prime"   : float(delta_prime),
            "D_pred"        : float(D_pred),
            "alpha_t"       : float(alpha_t),
            "rho"           : float(rho),

            # Individual critic deltas (explicit — no **spread)
            "delta_reward"  : mc_out["delta_reward"],
            "delta_risk"    : mc_out["delta_risk"],
            "delta_unc"     : mc_out["delta_unc"],
            "delta_conflict": mc_out["delta_conflict"],
            "delta_habit"   : mc_out["delta_habit"],

            # Transformed rewards
            "r_reward"      : mc_out["r_reward"],
            "r_risk"        : mc_out["r_risk"],
            "r_unc"         : mc_out["r_unc"],
            "r_conf"        : mc_out["r_conf"],
            "r_habit"       : mc_out["r_habit"],

            # Actor state
            "pi"            : self.actor.pi.copy(),
            "action_pref"   : self.actor.action_preference.copy(),
            "actor_entropy" : self.actor.actor_summary()["entropy"],

            # Risk and dopamine
            "Q_risk"        : Q_risk,
            "tonic_da"      : float(tonic_da),
            "da_response"   : self.dopamine.dopamine_response_type(
                delta_prime),
        }

        self.rl_log.append({
            "step"        : self.step_count,
            "delta_total" : float(delta_total),
            "delta_prime" : float(delta_prime),
            "reward"      : float(raw_reward),
        })

        return result

    def rl_summary(self) -> dict:
        return {
            "step_count"     : self.step_count,
            "episode_reward" : float(self.episode_reward),
            "actor"          : self.actor.actor_summary(),
            "multi_critic"   : self.multi_critic.system_summary(),
            "risk"           : self.risk.risk_summary(),
            "dopamine"       : self.dopamine.dopamine_summary(),
        }