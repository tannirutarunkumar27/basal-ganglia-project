"""
ValidationHarness  —  Step 32
-------------------------------
Central runner that executes the full SNN-BG pipeline on
each benchmark task and collects structured result records.

For every task it:
  1. Resets all pipeline components
  2. Runs N_STEPS of the 19-step algorithm
  3. Collects per-step metrics
  4. Computes aggregate capability scores
  5. Stores results for report generation

The harness is task-agnostic — any object that implements
the BaseTask interface (from phase1) is accepted.
"""

import numpy as np
import time
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskResult:
    """Complete result record for one task evaluation."""
    task_name          : str
    n_steps            : int
    accuracy           : float
    mean_reward        : float
    cumulative_reward  : float
    convergence_step   : int           # -1 = never converged
    regret             : float
    mean_U             : float
    mean_delta_total   : float
    mean_alpha_t       : float
    mean_expl_conf     : float
    mean_n_rules       : float
    mean_energy_pJ     : float
    total_energy_nJ    : float
    spike_reduction_pct: float
    capability_scores  : dict = field(default_factory=dict)
    step_log           : list = field(default_factory=list)
    elapsed_s          : float = 0.0
    extra              : dict = field(default_factory=dict)


class ValidationHarness:

    def __init__(self,
                 net         : object,
                 pipeline    : object,
                 p4_ctrl     : object,
                 p5_ctrl     : object,
                 rl_engine   : object,
                 plast       : object,
                 nm_ctrl     : object,
                 reasoning   : object,
                 optimizer   : object,
                 results_dir : str   = "results",
                 dt          : float = 0.1e-3):

        self.net       = net
        self.pipeline  = pipeline
        self.p4_ctrl   = p4_ctrl
        self.p5_ctrl   = p5_ctrl
        self.rl        = rl_engine
        self.plast     = plast
        self.nm_ctrl   = nm_ctrl
        self.reasoning = reasoning
        self.optimizer = optimizer
        self.results_dir = results_dir
        self.dt        = dt

        os.makedirs(results_dir, exist_ok=True)

        # All results keyed by task name
        self.results : dict[str, TaskResult] = {}

    def reset_all(self) -> None:
        """Resets all pipeline components for a fresh task run."""
        self.net.reset()
        self.pipeline.reset()
        self.p4_ctrl.reset()
        self.p5_ctrl.reset()
        self.rl.reset()
        self.plast.reset()
        self.nm_ctrl.reset()
        self.reasoning.reset()
        self.optimizer.reset()

    def _run_one_step(self, t: float, t_ms: float,
                       task_state    : np.ndarray,
                       prev_action   : Optional[int],
                       prev_reward   : Optional[float],
                       explain       : bool = False) -> dict:
        """
        Executes the full 19-step algorithm for one timestep.
        task_state is encoded as the cortical input amplitude.
        """
        # State encoding: map task_state vector to cortical drive
        state_norm = np.asarray(task_state, dtype=float)
        if state_norm.max() - state_norm.min() > 1e-8:
            state_norm = ((state_norm - state_norm.min())
                          / (state_norm.max() - state_norm.min()))
        state_scalar = float(state_norm.mean())

        ctx_amp = (0.3e-9 + 0.4e-9 * state_scalar
                   + 0.2e-9 * np.sin(2 * np.pi * 5 * t))
        ctx_in  = np.full(self.net.pops["cortex"].N, ctx_amp)

        # Phase 2 + Phase 10 sparse optimization
        raw_spks = self.net.step(
            cortex_input=ctx_in, dopamine_signal=0.0)
        membrane_V = {n: p.V.copy()
                      for n, p in self.net.pops.items()}
        n_raw = sum(int(np.asarray(sp).sum())
                    for sp in raw_spks.values())

        p10 = self.optimizer.full_step(
            spike_dict     = raw_spks,
            membrane_V     = membrane_V,
            U              = 0.5,
            C              = 0.5,
            delta_prime    = float(prev_reward or 0.0),
            conflict_score = 0.1,
            n_reasoning    = int(explain))
        opt_spks = p10["optimized_spikes"]
        n_opt    = sum(int(np.asarray(sp).sum())
                       for sp in opt_spks.values())

        # Phase 3
        p3 = self.pipeline.step(
            spike_vector = opt_spks.get(
                "bayesian_layer",
                np.zeros(self.net.pops["bayesian_layer"].N)),
            reward      = prev_reward,
            prev_action = prev_action)

        # Phase 8 neuromodulators (pre)
        snc_rate = self.net.pops["snc"].population_rate(50)
        da_level = float(np.clip(
            1.0 + (snc_rate - 4.0) / 20.0, 0.1, 3.0))

        # Phase 4
        p4 = self.p4_ctrl.step(
            cortex_spikes = opt_spks.get(
                "cortex", np.zeros(self.net.pops["cortex"].N)),
            d1_spikes     = opt_spks.get(
                "d1_msn", np.zeros(self.net.pops["d1_msn"].N)),
            d2_spikes     = opt_spks.get(
                "d2_msn", np.zeros(self.net.pops["d2_msn"].N)),
            belief_scores = p3["V_combined"],
            U=p3["U"], C=p3["C"],
            dopamine_level=da_level)

        # Phase 5
        p5 = self.p5_ctrl.step(
            direct_inh=p4["direct_inh"],
            indirect_exc=p4["indirect_exc"],
            stn_global=p4["stn_global"],
            w_go=p4["w_go"], w_nogo=p4["w_nogo"],
            w_stn=p4["w_stn"],
            U=p3["U"], C=p3["C"],
            action_probs=p3["prob"],
            belief_scores=p3["V_combined"],
            conflict_score=p4["conflict_score"],
            t_ms=t_ms)

        action = (p5["released_action"]
                  if p5["action_released"] else p3["action"])

        # Phase 6
        rl_out = self.rl.step(
            d1_spikes=opt_spks.get(
                "d1_msn", np.zeros(self.net.pops["d1_msn"].N)),
            belief_scores=p3["V_combined"],
            raw_reward=float(prev_reward or 0.0),
            action=action,
            U=p3["U"], C=p3["C"],
            conflict_score=p4["conflict_score"],
            stn_burst=p4["stn_burst"],
            dopamine_level=da_level,
            done=False)

        # Phase 8
        nm_out = self.nm_ctrl.step(
            delta_prime=rl_out["delta_prime"],
            U=p3["U"], C=p3["C"],
            reward=float(prev_reward or 0.0),
            rho=rl_out["rho"],
            conflict_score=p4["conflict_score"],
            snc_rate_hz=snc_rate)

        # Phase 7
        if p10.get("is_reward_event", False):
            self.plast.step(
                opt_spks,
                rl_out["delta_prime"] * nm_out["Mt"],
                nm_out["alpha_t"])

        # Phase 9
        gm = p5.get("gate_margins", np.zeros(4))
        if not isinstance(gm, np.ndarray):
            gm = np.zeros(4)
        p9 = self.reasoning.step(
            V_combined=p3["V_combined"],
            U=p3["U"], C=p3["C"],
            Q_risk=rl_out.get("Q_risk", np.zeros(4)),
            conflict_score=p4["conflict_score"],
            stn_burst=p4["stn_burst"],
            direct_inh=p4["direct_inh"],
            indirect_exc=p4["indirect_exc"],
            stn_global=p4["stn_global"],
            gate_margins=gm,
            DA=nm_out["DA"], ht5=nm_out["5HT"], NE=nm_out["NE"],
            alpha_t=nm_out["alpha_t"], Mt=nm_out["Mt"],
            rho=rl_out["rho"], reward=prev_reward,
            gate_action=action, t_ms=t_ms, explain=explain)

        # Feedback
        self.net.step(ctx_in,
                      rl_out["delta_prime"] * nm_out["DA"])
        self.pipeline.inject_dopamine_signal(
            rl_out["delta_prime"] * nm_out["Mt"])
        self.p4_ctrl.apply_reward(
            float(prev_reward or 0.0), action)

        # Energy
        energy_pJ = self.optimizer.budget.record_step(
            n_opt, int(n_opt * 15),
            20 if p10.get("is_reward_event") else 0,
            int(explain))

        return {
            "action"      : action,
            "U"           : float(p3["U"]),
            "C"           : float(p3["C"]),
            "delta_total" : float(rl_out["delta_total"]),
            "delta_prime" : float(rl_out["delta_prime"]),
            "alpha_t"     : float(nm_out["alpha_t"]),
            "expl_conf"   : float(p9["explanation_conf"]),
            "n_rules"     : int(p9["n_rules_fired"]),
            "n_spikes_raw": n_raw,
            "n_spikes_opt": n_opt,
            "energy_pJ"   : float(energy_pJ),
            "stn_active"  : bool(p4["stn_burst"]),
            "gate_open"   : bool(p5["action_released"]),
        }

    def run_task(self,
                  task          : object,
                  n_steps       : int   = 3000,
                  explain_every : int   = 500,
                  task_name     : str   = None) -> TaskResult:
        """
        Runs one complete task evaluation.

        task       : any Phase 1 BaseTask subclass
        n_steps    : number of simulation steps
        explain_every: generate full explanations every N steps
        task_name  : override task display name
        """
        name = task_name or getattr(task, "name",
                                     type(task).__name__)
        print(f"\n  Running task: {name} ({n_steps} steps)...")

        self.reset_all()
        state = task.reset()

        prev_action = None
        prev_reward = None

        rewards    = []
        step_log   = []
        t0         = time.time()

        for step in range(n_steps):
            t    = step * self.dt
            t_ms = t * 1000
            do_explain = (step % explain_every == 0)

            out = self._run_one_step(
                t, t_ms, state,
                prev_action, prev_reward, do_explain)

            action = out["action"]

            # Step the task
            if not task.done:
                next_state, reward, done, info = task.step(action)
                state = next_state
            else:
                reward = 0.0
                done   = True
                info   = {}

            rewards.append(float(reward))
            prev_action = action
            prev_reward = reward

            step_log.append({
                "step"       : step,
                "action"     : action,
                "reward"     : float(reward),
                "U"          : out["U"],
                "delta_total": out["delta_total"],
                "alpha_t"    : out["alpha_t"],
                "expl_conf"  : out["expl_conf"],
                "n_rules"    : out["n_rules"],
                "energy_pJ"  : out["energy_pJ"],
                "stn_active" : out["stn_active"],
            })

            if done and step < n_steps - 1:
                state = task.reset()

        elapsed = time.time() - t0
        n_raw_mean = float(np.mean(
            [s.get("n_spikes_raw", 0) for s in step_log]))

        result = self._aggregate(
            name, step_log, rewards,
            elapsed, n_steps, n_raw_mean)
        self.results[name] = result
        self._print_task_summary(result)
        return result

    def _aggregate(self, name, step_log, rewards,
                    elapsed, n_steps, n_raw_mean) -> TaskResult:
        rewards_arr = np.array(rewards)
        acc         = float(np.mean(rewards_arr > 0))
        mean_rew    = float(np.mean(rewards_arr))
        cum_rew     = float(np.sum(rewards_arr))

        # Convergence: first window of 200 steps where acc > 0.65
        conv_step   = -1
        window = 200
        for i in range(len(rewards) - window):
            if np.mean(rewards_arr[i:i+window] > 0) >= 0.65:
                conv_step = i
                break

        regret = float(np.sum(
            np.maximum(rewards_arr.max() - rewards_arr, 0)))

        mean_U    = float(np.mean([s["U"] for s in step_log]))
        mean_dt   = float(np.mean([s["delta_total"] for s in step_log]))
        mean_alp  = float(np.mean([s["alpha_t"]     for s in step_log]))
        mean_ec   = float(np.mean([s["expl_conf"]   for s in step_log]))
        mean_nr   = float(np.mean([s["n_rules"]     for s in step_log]))
        mean_epj  = float(np.mean([s["energy_pJ"]   for s in step_log]))
        total_nJ  = float(self.optimizer.budget.total_energy_nJ())

        n_opt_mean = float(np.mean(
            [s.get("energy_pJ", 0) / 23.0
             for s in step_log]))
        spike_red = float(np.clip(
            1.0 - n_opt_mean / max(n_raw_mean, 1), 0, 1)) * 100

        return TaskResult(
            task_name           = name,
            n_steps             = n_steps,
            accuracy            = acc,
            mean_reward         = mean_rew,
            cumulative_reward   = cum_rew,
            convergence_step    = conv_step,
            regret              = regret,
            mean_U              = mean_U,
            mean_delta_total    = mean_dt,
            mean_alpha_t        = mean_alp,
            mean_expl_conf      = mean_ec,
            mean_n_rules        = mean_nr,
            mean_energy_pJ      = mean_epj,
            total_energy_nJ     = total_nJ,
            spike_reduction_pct = spike_red,
            step_log            = step_log,
            elapsed_s           = elapsed,
        )

    def _print_task_summary(self, r: TaskResult) -> None:
        conv = (str(r.convergence_step)
                if r.convergence_step >= 0 else "never")
        print(f"    accuracy={r.accuracy*100:5.1f}%  |  "
              f"mean_reward={r.mean_reward:+.3f}  |  "
              f"convergence={conv}  |  "
              f"regret={r.regret:.1f}")
        print(f"    mean_U={r.mean_U:.3f}  |  "
              f"expl_conf={r.mean_expl_conf:.3f}  |  "
              f"rules/step={r.mean_n_rules:.1f}  |  "
              f"energy={r.total_energy_nJ:.1f}nJ  |  "
              f"spike_red={r.spike_reduction_pct:.0f}%")