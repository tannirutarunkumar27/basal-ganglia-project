"""
HiddenRuleTask — reward rule changes silently every ~50 steps.
Agent must infer the current rule from reward feedback alone.
Tests reasoning under uncertainty.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "../../phase1"))

from tasks.base_task import BaseTask


class HiddenRuleTask(BaseTask):

    def __init__(self, n_actions=4, max_steps=3000,
                 rule_duration=50):
        super().__init__("hidden_rule", n_actions, max_steps)
        self.rule_duration = rule_duration
        self.current_rule  = 0
        self.n_rules       = n_actions
        self.steps_in_rule = 0
        self.rule_rewards  = None

    def reset(self):
        self.current_rule = np.random.randint(0, self.n_rules)
        self._generate_rule_rewards()
        self.step_count    = 0
        self.done          = False
        self.steps_in_rule = 0
        return self._get_obs()

    def _generate_rule_rewards(self):
        self.rule_rewards = np.random.rand(self.n_actions)
        self.rule_rewards[self.current_rule] = 0.9

    def _get_obs(self):
        # Noisy observation — agent cannot directly see the rule
        obs = np.random.randn(self.n_actions) * 0.3
        obs[self.current_rule] += 0.5
        return obs

    def step(self, action: int):
        self.steps_in_rule += 1
        if self.steps_in_rule >= self.rule_duration:
            self.current_rule  = np.random.randint(0, self.n_rules)
            self._generate_rule_rewards()
            self.steps_in_rule = 0

        reward = float(
            np.random.rand() < self.rule_rewards[action])
        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        return self._get_obs(), reward, self.done, {
            "rule": self.current_rule}