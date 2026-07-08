"""
MetricAggregator  —  Step 33
------------------------------
Combines all seven metric categories into one structured
report for a single task evaluation run.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from behavioral.behavioral_metrics       import BehavioralMetrics
from learning.learning_metrics           import LearningMetrics
from reasoning.reasoning_metrics         import ReasoningMetrics
from neural.neural_metrics               import NeuralMetrics
from meta.meta_learning_metrics          import MetaLearningMetrics
from neuromod.neuromodulation_metrics    import NeuromodulationMetrics
from energy.energy_metrics               import EnergyMetrics


class MetricAggregator:

    def __init__(self, n_actions: int,
                 correct_action: int,
                 dt: float = 0.1e-3):

        self.n_actions     = n_actions
        self.correct_action= correct_action
        self.dt            = dt

        self.behavioral  = BehavioralMetrics(n_actions)
        self.learning    = LearningMetrics()
        self.reasoning   = ReasoningMetrics(n_actions)
        self.neural      = NeuralMetrics(dt=dt)
        self.meta        = MetaLearningMetrics()
        self.neuromod    = NeuromodulationMetrics()
        self.energy      = EnergyMetrics(dt=dt)

        self.step_count  = 0

    def reset(self) -> None:
        for m in [self.behavioral, self.learning, self.reasoning,
                  self.neural, self.meta, self.neuromod, self.energy]:
            m.reset()
        self.step_count = 0

    def record(self,
                action          : int,
                reward          : float,
                V_combined      : "np.ndarray",
                U               : float,
                C               : float,
                delta_total     : float,
                alpha_t         : float,
                DA              : float,
                ht5             : float,
                NE              : float,
                rho             : float,
                expl_conf       : float,
                n_rules         : int,
                n_spikes        : int,
                n_neurons_total : int,
                direct_inh      : "np.ndarray",
                indirect_exc    : "np.ndarray",
                trace_mag       : float,
                ahp_mag         : float,
                mean_weight_mag : float,
                n_syn_events    : int,
                n_weight_updates: int,
                gate_open       : bool,
                cf_delta        : float = 0.0,
                is_reversal     : bool  = False,
                new_stimulus    : bool  = False) -> None:
        """
        Records all signals for one timestep across all seven
        metric categories simultaneously.
        """
        s = self.step_count
        ca = self.correct_action
        correct = int(action == ca)

        self.behavioral.record(action, reward, ca)

        self.learning.record_step(reward, delta_total, mean_weight_mag)

        self.reasoning.record_step(
            V_combined, action, ca, U, expl_conf, n_rules, cf_delta)

        self.neural.record_step(
            n_spikes, n_neurons_total, DA, reward,
            direct_inh, indirect_exc, trace_mag, ahp_mag, s)

        self.meta.record_step(
            U, alpha_t, reward, correct, is_reversal, s)

        self.neuromod.record_step(
            DA, ht5, NE, U, rho, expl_conf,
            risky_avoided=int(rho > 0.6 and correct))

        self.energy.record_step(
            n_spikes, n_syn_events, n_weight_updates,
            n_reasoning=1, step=s,
            gate_open=gate_open, new_stimulus=new_stimulus)

        self.step_count += 1

    def record_episode(self, accuracy: float) -> None:
        self.learning.record_episode(accuracy)

    def record_weights(self, weight_dict: dict) -> None:
        self.energy.record_weights(weight_dict)

    def compute_all(self) -> dict:
        acc = self.behavioral.accuracy()
        return {
            "behavioral"    : self.behavioral.compute_all(),
            "learning"      : self.learning.compute_all(),
            "reasoning"     : self.reasoning.compute_all(),
            "neural"        : self.neural.compute_all(),
            "meta_learning" : self.meta.compute_all(),
            "neuromodulation": self.neuromod.compute_all(),
            "energy"        : self.energy.compute_all(acc),
            "step_count"    : self.step_count,
        }