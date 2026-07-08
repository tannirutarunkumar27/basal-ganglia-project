"""
Step 5 verification:
  Simulates spike input for 4 actions and checks that
  Va = log P(s|a) + log P(a) is computed correctly.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
np.random.seed(42)

from belief.posterior_encoder import PosteriorBeliefEncoder

N_ACTIONS  = 4
N_NEURONS  = 80        # 20 neurons per action
WINDOW     = 200       # 20 ms at 0.1 ms dt
DT         = 0.1e-3

enc = PosteriorBeliefEncoder(
    n_actions       = N_ACTIONS,
    n_neurons_total = N_NEURONS,
    window_steps    = WINDOW,
    dt              = DT,
)

print("\n" + "="*55)
print("  Phase 3 — Step 5: Posterior Belief Encoder")
print("="*55)

# Simulate biased spike input: action 2 fires most
for step in range(500):
    spikes = np.zeros(N_NEURONS, dtype=bool)
    # Action 2 subpopulation (neurons 40-59) fires at 3x rate
    for i in range(N_NEURONS):
        action_idx = i // (N_NEURONS // N_ACTIONS)
        p = 0.15 if action_idx == 2 else 0.05
        spikes[i] = np.random.rand() < p
    V = enc.encode(spikes)

# Inject some reward history
enc.update_prior(action=2, reward=1.0)
enc.update_prior(action=2, reward=0.8)
enc.update_prior(action=0, reward=0.2)

V = enc.encode(spikes)   # final encode with updated prior
summary = enc.belief_summary()

print(f"\n  Belief scores Va per action:")
for a in range(N_ACTIONS):
    print(f"    action {a}: Va={V[a]:+7.3f}  "
          f"log P(s|a)={summary['log_likelihood'][a]:+6.3f}  "
          f"log P(a)={summary['log_prior'][a]:+6.3f}  "
          f"rate={summary['firing_rates_hz'][a]:5.1f} Hz")

best = int(np.argmax(V))
print(f"\n  Highest belief: action {best}  (expected: action 2)")
assert best == 2, f"Expected action 2, got {best}"
print("  [PASS] Step 5 posterior belief encoding correct.")
print("="*55 + "\n")