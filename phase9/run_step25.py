"""
Step 25 verification — neuro-symbolic reasoning layer.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from symbolic.neuro_symbolic_reasoner import NeuroSymbolicReasoner

np.random.seed(0)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 200
CORRECT   = 2

reasoner = NeuroSymbolicReasoner(N_ACTIONS, dt=DT)

print("\n" + "="*60)
print("  Phase 9 — Step 25: Neuro-Symbolic Reasoning Layer")
print("="*60)

for step in range(N_STEPS):
    progress = step / N_STEPS
    V = np.array([-0.3, -0.1, 0.5 + progress*0.8, -0.2]) \
        + np.random.randn(N_ACTIONS) * 0.05
    U = max(0.1, 0.6 - progress * 0.5)
    C = 1.0 - U
    Q = np.array([0.4, 0.3, 0.7 + progress*0.3, -0.2])
    direct_inh   = np.array([0.05, 0.06, 0.10 + progress*0.4, 0.04])
    indirect_exc = np.array([0.10, 0.12, 0.06, 0.11])
    gate_margins = np.array([-0.3, -0.2, 0.05 + progress*0.3, -0.4])
    reward_hist  = [float(np.random.rand() > 0.4 - progress*0.3)
                    for _ in range(10)]

    out = reasoner.reason(
        V_combined    = V,
        U             = U,
        C             = C,
        Q_risk        = Q,
        conflict_score= 0.1,
        stn_burst     = False,
        direct_inh    = direct_inh,
        indirect_exc  = indirect_exc,
        DA            = 0.6 + progress * 0.2,
        ht5           = 0.4,
        NE            = 0.3,
        reward_history= reward_hist,
        gate_margins  = gate_margins)

print(f"\n  Final reasoning output:")
print(f"    Selected action    : {out['selected_action']}"
      f"  (expected: {CORRECT})")
print(f"    Explanation conf   : {out['explanation_conf']:.3f}")
print(f"    Rules fired        : {out['n_rules_fired']}")
print(f"    Blocked actions    : {out['blocked_actions']}")
print(f"    Alternative ranking: "
      f"{[(a, round(s,3)) for a,s in out['alternative_ranking']]}")
print(f"\n  Primary rationale:")
for line in out["rationale"].split(" | "):
    print(f"    - {line}")
print(f"\n  All conclusions:")
for c in out["conclusions"]:
    print(f"    [{c['verdict']:<10s}] {c['rule']:<30s}"
          f" strength={c['strength']:.3f}")

assert out["selected_action"] == CORRECT, \
    f"Expected action {CORRECT}, got {out['selected_action']}"
print(f"\n  [PASS] Neuro-symbolic reasoner selects correct action.")
print("="*60 + "\n")