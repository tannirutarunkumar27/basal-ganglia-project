"""
MultiCriticSystem  —  Step 17
-------------------------------
Five critics, each evaluating a different aspect of the decision:

  k=1  RewardCritic     : standard discounted return
  k=2  RiskCritic       : variance-penalised value
  k=3  UncertaintyCritic: rewards exploration under high Ut
  k=4  ConflictCritic   : penalises STN-conflict states
  k=5  HabitGoalCritic  : blends habitual and goal-directed value

Combined signal:
  delta_total = sum_k lambda_k * delta_k

Advanced innovation: multi-critic RL lets the agent evaluate
actions from five simultaneous perspectives, each with its own
temporal difference error and eligibility trace.
"""

"""
MultiCriticSystem  —  Step 17
Five critics: reward, risk, uncertainty, conflict, habit/goal.
delta_total = sum_k lambda_k * delta_k
"""

import numpy as np
from critics.base_critic import BaseCritic


class RewardCritic(BaseCritic):
    def __init__(self, state_dim, n_actions, dt=0.1e-3):
        super().__init__(
            name="reward", state_dim=state_dim,
            n_actions=n_actions, gamma=0.99,
            alpha=0.05, lam_weight=0.40, dt=dt)

    def compute_reward(self, raw_reward: float) -> float:
        return self.normalise_reward(raw_reward)


class RiskCritic(BaseCritic):
    def __init__(self, state_dim, n_actions, dt=0.1e-3):
        super().__init__(
            name="risk", state_dim=state_dim,
            n_actions=n_actions, gamma=0.95,
            alpha=0.03, lam_weight=0.20, dt=dt)
        self._r_history = {a: [] for a in range(n_actions)}

    def compute_risk_reward(self, raw_reward: float,
                             action: int,
                             rho: float = 0.5) -> float:
        self._r_history[action].append(raw_reward)
        hist  = self._r_history[action][-50:]
        E_r   = float(np.mean(hist))
        Var_r = float(np.var(hist)) if len(hist) > 1 else 0.0
        return E_r - rho * Var_r


class UncertaintyCritic(BaseCritic):
    def __init__(self, state_dim, n_actions, dt=0.1e-3):
        super().__init__(
            name="uncertainty", state_dim=state_dim,
            n_actions=n_actions, gamma=0.90,
            alpha=0.04, lam_weight=0.15, dt=dt)
        self.explore_bonus = 0.3

    def compute_unc_reward(self, raw_reward: float,
                            U: float, C: float) -> float:
        bonus   = self.explore_bonus * U
        penalty = 0.1 * (1.0 - C)
        return float(raw_reward + bonus - penalty)


class ConflictCritic(BaseCritic):
    def __init__(self, state_dim, n_actions, dt=0.1e-3):
        super().__init__(
            name="conflict", state_dim=state_dim,
            n_actions=n_actions, gamma=0.95,
            alpha=0.03, lam_weight=0.10, dt=dt)
        self.conflict_penalty = 0.4

    def compute_conf_reward(self, raw_reward: float,
                             conflict_score: float,
                             stn_burst: bool) -> float:
        burst_penalty = 0.2 if stn_burst else 0.0
        conf_norm     = float(np.clip(conflict_score / 2.0, 0.0, 1.0))
        return float(raw_reward
                     - self.conflict_penalty * conf_norm
                     - burst_penalty)


class HabitGoalCritic(BaseCritic):
    def __init__(self, state_dim, n_actions, dt=0.1e-3):
        super().__init__(
            name="habit_goal", state_dim=state_dim,
            n_actions=n_actions, gamma=0.97,
            alpha=0.02, lam_weight=0.15, dt=dt)
        self.habit_value  = np.zeros(n_actions)
        self.alpha_habit  = 0.01
        self.alpha_blend  = 0.5

    def update_habit(self, action: int, reward: float) -> None:
        self.habit_value[action] = (
            (1 - self.alpha_habit) * self.habit_value[action]
            + self.alpha_habit * reward)

    def compute_habit_reward(self, raw_reward: float,
                              action: int) -> float:
        h = float(self.habit_value[action])
        return float(self.alpha_blend * h
                     + (1 - self.alpha_blend) * raw_reward)


class MultiCriticSystem:
    """
    Manages all five critics and computes delta_total.
    delta_total = sum_k lambda_k * delta_k
    """

    def __init__(self,
                 state_dim : int,
                 n_actions : int,
                 dt        : float = 0.1e-3):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.dt        = dt

        self.reward   = RewardCritic(state_dim, n_actions, dt)
        self.risk     = RiskCritic(state_dim, n_actions, dt)
        self.unc      = UncertaintyCritic(state_dim, n_actions, dt)
        self.conflict = ConflictCritic(state_dim, n_actions, dt)
        self.habit    = HabitGoalCritic(state_dim, n_actions, dt)

        self.critics  = [self.reward, self.risk,
                         self.unc, self.conflict, self.habit]

        self.delta_total_history  = []
        self.critic_delta_history = {c.name: [] for c in self.critics}

    def reset(self):
        for c in self.critics:
            c.reset()
        self.delta_total_history.clear()
        for name in self.critic_delta_history:
            self.critic_delta_history[name].clear()

    def step(self,
             raw_reward    : float,
             state         : np.ndarray,
             next_state    : np.ndarray,
             action        : int,
             U             : float,
             C             : float,
             conflict_score: float,
             stn_burst     : bool,
             rho           : float,
             done          : bool  = False,
             alpha_override: float = None) -> dict:
        """
        Step 17 core:
          For each critic k:
              r_k     = transform(raw_reward, ...)
              delta_k = r_k + gamma*V_k(s') - V_k(s)
          delta_total = sum_k lambda_k * delta_k
        """
        # Per-critic reward transforms
        r_reward = self.reward.compute_reward(raw_reward)
        r_risk   = self.risk.compute_risk_reward(raw_reward, action, rho)
        r_unc    = self.unc.compute_unc_reward(raw_reward, U, C)
        r_conf   = self.conflict.compute_conf_reward(
            raw_reward, conflict_score, stn_burst)
        r_habit  = self.habit.compute_habit_reward(raw_reward, action)

        # TD errors
        d_reward  = self.reward.compute_delta(r_reward,  state, next_state, done)
        d_risk    = self.risk.compute_delta(r_risk,      state, next_state, done)
        d_unc     = self.unc.compute_delta(r_unc,        state, next_state, done)
        d_conf    = self.conflict.compute_delta(r_conf,  state, next_state, done)
        d_habit   = self.habit.compute_delta(r_habit,    state, next_state, done)

        # Update all critics
        for critic, delta in [
            (self.reward,   d_reward),
            (self.risk,     d_risk),
            (self.unc,      d_unc),
            (self.conflict, d_conf),
            (self.habit,    d_habit),
        ]:
            critic.delta = delta
            critic.update(state, action, alpha_override)

        self.habit.update_habit(action, raw_reward)

        # Combined delta_total
        deltas      = [d_reward, d_risk, d_unc, d_conf, d_habit]
        lams        = [c.lam_weight for c in self.critics]
        delta_total = float(sum(l * d for l, d in zip(lams, deltas)))

        self.delta_total_history.append(delta_total)
        for c, d in zip(self.critics, deltas):
            self.critic_delta_history[c.name].append(d)

        return {
            # Combined signal
            "delta_total"   : delta_total,

            # Individual deltas — keys used by run_step17 and RLEngine
            "delta_reward"  : float(d_reward),
            "delta_risk"    : float(d_risk),
            "delta_unc"     : float(d_unc),
            "delta_conflict": float(d_conf),
            "delta_habit"   : float(d_habit),   # consistent key name

            # Per-critic transformed rewards
            "r_reward"      : float(r_reward),
            "r_risk"        : float(r_risk),
            "r_unc"         : float(r_unc),
            "r_conf"        : float(r_conf),
            "r_habit"       : float(r_habit),
        }

    def get_value_estimate(self, state: np.ndarray) -> float:
        return float(sum(c.lam_weight * c.value(state)
                         for c in self.critics))

    def system_summary(self) -> dict:
        hist = self.delta_total_history
        return {
            "delta_total_mean": float(np.mean(hist)) if hist else 0.0,
            "delta_total_std" : float(np.std(hist))  if hist else 0.0,
            "critics"         : [c.critic_summary() for c in self.critics],
        }