"""
TrainingMetrics — tracks all training statistics across episodes.
Implements the evaluation metrics from the methodology document:

Behavioral : accuracy, reward_rate, convergence_speed, regret
Learning   : delta_total, alpha_t, weight_change
Reasoning  : explanation_conf, n_rules_fired
Neural     : spike_sparsity, population_rates
Energy     : spikes_per_step, energy_nJ, efficiency_score
"""

import numpy as np
from collections import deque
import json
import os


class TrainingMetrics:

    def __init__(self, n_actions: int,
                 window: int = 500,
                 results_dir: str = "results"):

        self.n_actions   = n_actions
        self.window      = window
        self.results_dir = results_dir

        # Per-step buffers (rolling window)
        self.reward_buf       = deque(maxlen=window)
        self.accuracy_buf     = deque(maxlen=window)
        self.delta_buf        = deque(maxlen=window)
        self.alpha_buf        = deque(maxlen=window)
        self.U_buf            = deque(maxlen=window)
        self.spike_buf        = deque(maxlen=window)
        self.energy_buf       = deque(maxlen=window)
        self.expl_conf_buf    = deque(maxlen=window)
        self.n_rules_buf      = deque(maxlen=window)

        # Per-episode aggregates
        self.episode_rewards  = []
        self.episode_accuracy = []
        self.episode_energy   = []
        self.episode_eff      = []

        # Full step log (every step)
        self.step_log         = []
        self.episode_count    = 0
        self.global_step      = 0

        # Running totals
        self.total_correct    = 0
        self.total_steps      = 0
        self.cum_reward       = 0.0

    def reset_episode(self) -> None:
        self.reward_buf.clear()
        self.accuracy_buf.clear()

    def record_step(self,
                     action        : int,
                     correct_action: int,
                     reward        : float,
                     delta_total   : float,
                     alpha_t       : float,
                     U             : float,
                     n_spikes      : int,
                     energy_pJ     : float,
                     expl_conf     : float,
                     n_rules       : int) -> dict:
        """
        Records one training step. Returns computed metrics dict.
        """
        self.global_step  += 1
        self.total_steps  += 1
        is_correct = int(action == correct_action)
        self.total_correct += is_correct
        self.cum_reward    += reward

        self.reward_buf.append(reward)
        self.accuracy_buf.append(is_correct)
        self.delta_buf.append(delta_total)
        self.alpha_buf.append(alpha_t)
        self.U_buf.append(U)
        self.spike_buf.append(n_spikes)
        self.energy_buf.append(energy_pJ)
        self.expl_conf_buf.append(expl_conf)
        self.n_rules_buf.append(n_rules)

        metrics = {
            "step"            : self.global_step,
            "action"          : action,
            "correct"         : is_correct,
            "reward"          : float(reward),
            "mean_reward"     : float(np.mean(list(self.reward_buf))),
            "accuracy"        : float(np.mean(list(self.accuracy_buf))),
            "delta_total"     : float(delta_total),
            "alpha_t"         : float(alpha_t),
            "U"               : float(U),
            "n_spikes"        : int(n_spikes),
            "energy_pJ"       : float(energy_pJ),
            "expl_conf"       : float(expl_conf),
            "n_rules"         : int(n_rules),
            "cum_reward"      : float(self.cum_reward),
            "global_accuracy" : float(
                self.total_correct / max(self.total_steps, 1)),
        }
        self.step_log.append({
            k: v for k, v in metrics.items()
            if k in ["step","reward","accuracy","delta_total",
                     "alpha_t","U","energy_pJ","expl_conf"]
        })
        return metrics

    def end_episode(self, episode: int,
                     task_accuracy  : float,
                     total_energy_nJ: float,
                     efficiency_score: float) -> dict:
        """Aggregates stats at episode end."""
        self.episode_count    += 1
        self.episode_rewards.append(
            float(np.mean(list(self.reward_buf))))
        self.episode_accuracy.append(float(task_accuracy))
        self.episode_energy.append(float(total_energy_nJ))
        self.episode_eff.append(float(efficiency_score))

        return {
            "episode"         : episode,
            "mean_reward"     : self.episode_rewards[-1],
            "accuracy"        : task_accuracy,
            "total_energy_nJ" : total_energy_nJ,
            "efficiency_score": efficiency_score,
            "mean_delta"      : float(np.mean(list(self.delta_buf))),
            "mean_alpha_t"    : float(np.mean(list(self.alpha_buf))),
            "mean_U"          : float(np.mean(list(self.U_buf))),
            "mean_sparsity"   : float(
                np.mean(list(self.spike_buf))),
            "mean_expl_conf"  : float(
                np.mean(list(self.expl_conf_buf))),
        }

    def convergence_step(self,
                          threshold: float = 0.75,
                          window   : int   = 200) -> int:
        """
        Returns step at which accuracy first exceeded threshold
        for `window` consecutive steps. -1 if not yet converged.
        """
        acc_series = [r["accuracy"] for r in self.step_log
                      if "accuracy" in r]
        for i in range(len(acc_series) - window):
            if all(a >= threshold
                   for a in acc_series[i:i + window]):
                return i
        return -1

    def regret(self, optimal_reward: float = 1.0) -> float:
        """Cumulative regret vs always-optimal policy."""
        rewards = [r["reward"] for r in self.step_log
                   if "reward" in r]
        return float(sum(optimal_reward - r for r in rewards))

    def save(self, filename: str = "training_metrics.json") -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, filename)

        summary = {
            "total_steps"    : self.total_steps,
            "total_correct"  : self.total_correct,
            "global_accuracy": float(
                self.total_correct / max(self.total_steps, 1)),
            "cum_reward"     : float(self.cum_reward),
            "convergence_step": self.convergence_step(),
            "regret"         : self.regret(),
            "episode_rewards" : self.episode_rewards,
            "episode_accuracy": self.episode_accuracy,
            "episode_energy"  : self.episode_energy,
            "recent_steps"    : self.step_log[-500:],
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def print_episode_report(self, ep_metrics: dict) -> None:
        print(f"\n  {'─'*56}")
        print(f"  Episode {ep_metrics['episode']:3d}  |  "
              f"accuracy={ep_metrics['accuracy']*100:5.1f}%  |  "
              f"mean_reward={ep_metrics['mean_reward']:+.3f}")
        print(f"  delta={ep_metrics['mean_delta']:+.4f}  |  "
              f"alpha_t={ep_metrics['mean_alpha_t']:.5f}  |  "
              f"U={ep_metrics['mean_U']:.3f}")
        print(f"  energy={ep_metrics['total_energy_nJ']:.1f}nJ  |  "
              f"eff_score={ep_metrics['efficiency_score']:.4f}  |  "
              f"expl_conf={ep_metrics['mean_expl_conf']:.3f}")
        print(f"  {'─'*56}")