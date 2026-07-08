"""
TrainingLoop  —  Step 31
--------------------------
Executes the complete 19-step algorithm for each timestep.

Step  1: encode input state into cortical spikes
Step  2: update temporal belief memory
Step  3: compute posterior evidence
Step  4: estimate uncertainty and confidence
Step  5: compute risk-sensitive utility
Step  6: apply neuromodulator fusion
Step  7: activate D1/D2 competition
Step  8: dynamically weight BG pathways
Step  9: trigger STN if conflict is high
Step 10: compute GPi gating output
Step 11: release selected action
Step 12: receive reward
Step 13: compute multi-critic errors
Step 14: update predictive dopamine
Step 15: update meta-dopamine learning rate
Step 16: update eligibility traces
Step 17: apply STDE synaptic updates
Step 18: generate explanation and counterfactuals
Step 19: continue to next state
"""

import numpy as np
import sys
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingLoop:

    def __init__(self,
                 net        : object,
                 pipeline   : object,
                 p4_ctrl    : object,
                 p5_ctrl    : object,
                 rl_engine  : object,
                 plast      : object,
                 nm_ctrl    : object,
                 reasoning  : object,
                 optimizer  : object,
                 metrics    : object,
                 ckpt_mgr   : object,
                 config     : object):

        self.net       = net
        self.pipeline  = pipeline
        self.p4_ctrl   = p4_ctrl
        self.p5_ctrl   = p5_ctrl
        self.rl        = rl_engine
        self.plast     = plast
        self.nm_ctrl   = nm_ctrl
        self.reasoning = reasoning
        self.optimizer = optimizer
        self.metrics   = metrics
        self.ckpt_mgr  = ckpt_mgr
        self.cfg       = config

        self.global_step  = 0
        self.episode      = 0

        # State carried between steps
        self._prev_action = None
        self._prev_reward = None

    def reset_episode(self) -> None:
        """Resets all components for a new episode."""
        self.net.reset()
        self.pipeline.reset()
        self.p4_ctrl.reset()
        self.p5_ctrl.reset()
        self.rl.reset()
        self.plast.reset()
        self.nm_ctrl.reset()
        self.reasoning.reset()
        self.optimizer.reset()
        self.metrics.reset_episode()
        self._prev_action = None
        self._prev_reward = None

    def _build_ctx_input(self, t: float) -> np.ndarray:
        """
        Step 1 helper: encode current time/state as cortical drive.
        In a real task this would encode the task state vector.
        Here we use a 5 Hz sinusoidal drive as a placeholder.
        """
        amp = 0.5e-9 * (np.sin(2 * np.pi * 5 * t) + 1.0)
        return np.full(self.net.pops["cortex"].N, amp)

    def execute_step(self, t: float, t_ms: float) -> dict:
        """
        Executes all 19 algorithm steps for one timestep.
        Returns a metrics dict for this step.
        """
        cfg = self.cfg

        # ══════════════════════════════════════════════════════
        # STEP 1: Encode input state into cortical spikes
        # ══════════════════════════════════════════════════════
        ctx_in = self._build_ctx_input(t)
        raw_spks = self.net.step(
            cortex_input    = ctx_in,
            dopamine_signal = 0.0)

        # ── Phase 10: sparse coding BEFORE downstream processing
        membrane_V = {name: pop.V.copy()
                      for name, pop in self.net.pops.items()}
        n_raw = sum(int(np.asarray(sp).sum())
                    for sp in raw_spks.values())

        p10_early = self.optimizer.full_step(
            spike_dict     = raw_spks,
            membrane_V     = membrane_V,
            U              = 0.5,
            C              = 0.5,
            delta_prime    = float(self._prev_reward or 0.0),
            conflict_score = 0.1,
            n_reasoning    = 0)

        opt_spks = p10_early["optimized_spikes"]

        # ══════════════════════════════════════════════════════
        # STEP 2: Update temporal belief memory
        # STEP 3: Compute posterior evidence
        # STEP 4: Estimate uncertainty and confidence
        # (all handled inside BayesianReasoningPipeline.step())
        # ══════════════════════════════════════════════════════
        p3 = self.pipeline.step(
            spike_vector = opt_spks.get(
                "bayesian_layer",
                np.zeros(self.net.pops["bayesian_layer"].N)),
            reward      = self._prev_reward,
            prev_action = self._prev_action)

        U  = p3["U"]
        C  = p3["C"]

        # ══════════════════════════════════════════════════════
        # STEP 5: Compute risk-sensitive utility
        # (handled inside RLEngine — Q_risk from RiskModule)
        # ══════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════
        # STEP 6: Apply neuromodulator fusion
        # ══════════════════════════════════════════════════════
        snc_rate = self.net.pops["snc"].population_rate(50)
        da_level = float(np.clip(
            1.0 + (snc_rate - 4.0) / 20.0, 0.1, 3.0))

        nm_pre = self.nm_ctrl.step(
            delta_prime    = float(self._prev_reward or 0.0),
            U              = U,
            C              = C,
            reward         = float(self._prev_reward or 0.0),
            rho            = 0.5,
            conflict_score = 0.1,
            snc_rate_hz    = snc_rate)

        # ══════════════════════════════════════════════════════
        # STEP 7: Activate D1/D2 competition
        # STEP 8: Dynamically weight BG pathways
        # STEP 9: Trigger STN if conflict is high
        # STEP 10: Compute GPi gating output
        # (all handled inside BGPathwayController.step())
        # ══════════════════════════════════════════════════════
        p4 = self.p4_ctrl.step(
            cortex_spikes = opt_spks.get(
                "cortex", np.zeros(self.net.pops["cortex"].N)),
            d1_spikes     = opt_spks.get(
                "d1_msn", np.zeros(self.net.pops["d1_msn"].N)),
            d2_spikes     = opt_spks.get(
                "d2_msn", np.zeros(self.net.pops["d2_msn"].N)),
            belief_scores = p3["V_combined"],
            U             = U,
            C             = C,
            dopamine_level= da_level)

        # ══════════════════════════════════════════════════════
        # STEP 11: Release selected action
        # (thalamocortical relay in ActionGatingController)
        # ══════════════════════════════════════════════════════
        p5 = self.p5_ctrl.step(
            direct_inh    = p4["direct_inh"],
            indirect_exc  = p4["indirect_exc"],
            stn_global    = p4["stn_global"],
            w_go          = p4["w_go"],
            w_nogo        = p4["w_nogo"],
            w_stn         = p4["w_stn"],
            U             = U,
            C             = C,
            action_probs  = p3["prob"],
            belief_scores = p3["V_combined"],
            conflict_score= p4["conflict_score"],
            t_ms          = t_ms)

        action = (p5["released_action"]
                  if p5["action_released"]
                  else p3["action"])

        # ══════════════════════════════════════════════════════
        # STEP 12: Receive reward
        # ══════════════════════════════════════════════════════
        reward = (cfg.reward_correct
                  if action == cfg.correct_action
                  else cfg.reward_wrong)

        # ══════════════════════════════════════════════════════
        # STEP 13: Compute multi-critic errors
        # STEP 14: Update predictive dopamine
        # (both inside RLEngine.step())
        # ══════════════════════════════════════════════════════
        rl_out = self.rl.step(
            d1_spikes     = opt_spks.get(
                "d1_msn", np.zeros(self.net.pops["d1_msn"].N)),
            belief_scores = p3["V_combined"],
            raw_reward    = reward,
            action        = action,
            U             = U,
            C             = C,
            conflict_score= p4["conflict_score"],
            stn_burst     = p4["stn_burst"],
            dopamine_level= da_level,
            done          = False)

        delta_prime = rl_out["delta_prime"]
        delta_total = rl_out["delta_total"]

        # ══════════════════════════════════════════════════════
        # STEP 15: Update meta-dopamine learning rate
        # (MetaDopamine inside NeuromodulatorController)
        # ══════════════════════════════════════════════════════
        nm_out = self.nm_ctrl.step(
            delta_prime    = delta_prime,
            U              = U,
            C              = C,
            reward         = reward,
            rho            = rl_out["rho"],
            conflict_score = p4["conflict_score"],
            snc_rate_hz    = snc_rate)

        alpha_t = nm_out["alpha_t"]
        Mt      = nm_out["Mt"]

        # ══════════════════════════════════════════════════════
        # STEP 16: Update eligibility traces
        # STEP 17: Apply STDE synaptic updates
        # (PlasticityManager handles both via STDP + STDE)
        # ══════════════════════════════════════════════════════
        if p10_early.get("is_reward_event", True):
            self.plast.step(
                spike_dict  = opt_spks,
                delta_prime = delta_prime * Mt,
                alpha_t     = alpha_t)

        # ══════════════════════════════════════════════════════
        # STEP 18: Generate explanation and counterfactuals
        # (ReasoningPipeline — only full text every N steps)
        # ══════════════════════════════════════════════════════
        gm = p5.get("gate_margins", np.zeros(cfg.n_actions))
        if not isinstance(gm, np.ndarray):
            gm = np.zeros(cfg.n_actions)

        do_explain = (self.global_step % cfg.explain_every == 0)

        p9 = self.reasoning.step(
            V_combined     = p3["V_combined"],
            U              = U,
            C              = C,
            Q_risk         = rl_out.get(
                "Q_risk", np.zeros(cfg.n_actions)),
            conflict_score = p4["conflict_score"],
            stn_burst      = p4["stn_burst"],
            direct_inh     = p4["direct_inh"],
            indirect_exc   = p4["indirect_exc"],
            stn_global     = p4["stn_global"],
            gate_margins   = gm,
            DA             = nm_out["DA"],
            ht5            = nm_out["5HT"],
            NE             = nm_out["NE"],
            alpha_t        = alpha_t,
            Mt             = Mt,
            rho            = rl_out["rho"],
            reward         = reward,
            gate_action    = action,
            t_ms           = t_ms,
            explain        = do_explain)

        # ── Feed enriched dopamine back into Phase 2 ──────────
        self.net.step(
            cortex_input    = ctx_in,
            dopamine_signal = delta_prime * nm_out["DA"])

        # Update Phase 3 prior with fused dopamine signal
        self.pipeline.inject_dopamine_signal(delta_prime * Mt)

        # Update Phase 4 synaptic weights with reward
        self.p4_ctrl.apply_reward(reward, action)

        # Phase 10 energy recording (full step)
        n_opt = sum(int(np.asarray(sp).sum())
                    for sp in opt_spks.values())
        energy_pJ = self.optimizer.budget.record_step(
            n_spikes           = n_opt,
            n_synapse_events   = int(n_opt * 15),
            n_weight_updates   = 20 if p10_early.get(
                "is_reward_event", False) else 0,
            n_reasoning_calls  = int(do_explain))

        # ══════════════════════════════════════════════════════
        # STEP 19: Continue to next state
        # (cache state for next call)
        # ══════════════════════════════════════════════════════
        self._prev_action = action
        self._prev_reward = reward
        self.global_step += 1

        return {
            "action"      : action,
            "reward"      : reward,
            "delta_total" : float(delta_total),
            "delta_prime" : float(delta_prime),
            "alpha_t"     : float(alpha_t),
            "Mt"          : float(Mt),
            "U"           : float(U),
            "C"           : float(C),
            "n_spikes_opt": n_opt,
            "energy_pJ"   : float(energy_pJ),
            "expl_conf"   : float(p9["explanation_conf"]),
            "n_rules"     : int(p9["n_rules_fired"]),
            "stn_active"  : bool(p4["stn_burst"]),
            "gate_open"   : bool(p5["action_released"]),
        }

    def run_episode(self, episode: int) -> dict:
        """
        Runs one complete training episode.
        Returns episode metrics dict.
        """
        self.episode = episode
        self.reset_episode()

        cfg     = self.cfg
        correct = 0
        t0      = time.time()

        step_log_ep = []

        for step in range(cfg.episode_steps):
            t    = step * cfg.dt
            t_ms = t * 1000

            step_out = self.execute_step(t, t_ms)
            correct += int(step_out["action"] == cfg.correct_action)

            step_metrics = self.metrics.record_step(
                action         = step_out["action"],
                correct_action = cfg.correct_action,
                reward         = step_out["reward"],
                delta_total    = step_out["delta_total"],
                alpha_t        = step_out["alpha_t"],
                U              = step_out["U"],
                n_spikes       = step_out["n_spikes_opt"],
                energy_pJ      = step_out["energy_pJ"],
                expl_conf      = step_out["expl_conf"],
                n_rules        = step_out["n_rules"])

            step_log_ep.append(step_out)

            # Console progress log
            if step % cfg.log_every == 0 and step > 0:
                acc = correct / (step + 1) * 100
                print(f"    step {step:5d}/{cfg.episode_steps}  |  "
                      f"acc={acc:5.1f}%  |  "
                      f"U={step_out['U']:.2f}  |  "
                      f"alpha={step_out['alpha_t']:.4f}  |  "
                      f"energy={self.optimizer.budget.total_energy_nJ():.0f}nJ")

            # Periodic checkpoint
            if (step > 0 and step % cfg.save_every == 0):
                self.ckpt_mgr.save(
                    step          = self.global_step,
                    episode       = episode,
                    actor         = self.rl.actor,
                    multi_critic  = self.rl.multi_critic,
                    pipeline_p3   = self.pipeline,
                    nm_ctrl       = self.nm_ctrl,
                    plast         = self.plast,
                    metrics_summary = {
                        "accuracy": correct/(step+1),
                        "step"    : step
                    })

        accuracy = correct / cfg.episode_steps
        energy   = self.optimizer.budget.total_energy_nJ()
        eff      = self.optimizer.budget.compute_efficiency_score(
            accuracy)

        ep_metrics = self.metrics.end_episode(
            episode        = episode,
            task_accuracy  = accuracy,
            total_energy_nJ= energy,
            efficiency_score= eff)

        elapsed = time.time() - t0
        ep_metrics["elapsed_s"]   = float(elapsed)
        ep_metrics["steps_per_s"] = float(
            cfg.episode_steps / elapsed)

        self.metrics.print_episode_report(ep_metrics)
        return ep_metrics, step_log_ep

    def run_training(self) -> list:
        """
        Runs the full multi-episode training loop.
        Returns list of episode metrics.
        """
        cfg = self.cfg
        all_ep_metrics = []

        print("\n" + "="*60)
        print("  Phase 11 — Step 31: Full Online Training Loop")
        print(f"  {cfg.n_episodes} episodes × "
              f"{cfg.episode_steps} steps = "
              f"{cfg.n_episodes*cfg.episode_steps:,} total steps")
        print("="*60)

        for ep in range(1, cfg.n_episodes + 1):
            print(f"\n  Episode {ep}/{cfg.n_episodes}  "
                  f"(global step {self.global_step:,})")
            ep_metrics, _ = self.run_episode(ep)
            all_ep_metrics.append(ep_metrics)

        # Final checkpoint
        self.ckpt_mgr.save(
            step         = self.global_step,
            episode      = cfg.n_episodes,
            actor        = self.rl.actor,
            multi_critic = self.rl.multi_critic,
            pipeline_p3  = self.pipeline,
            nm_ctrl      = self.nm_ctrl,
            plast        = self.plast,
            metrics_summary = {
                "final_accuracy": (self.metrics.total_correct
                                   / max(self.metrics.total_steps,1)),
                "total_steps"   : self.metrics.total_steps,
            })

        metrics_path = self.metrics.save("phase11_metrics.json")
        print(f"\n  Metrics saved: {metrics_path}")
        return all_ep_metrics