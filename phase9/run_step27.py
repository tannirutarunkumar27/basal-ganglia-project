"""
Step 27 verification — counterfactual reasoning.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from counterfactual.counterfactual_engine import CounterfactualEngine

np.random.seed(2)

N_ACTIONS = 4
CORRECT   = 2
NAMES     = ["reach_left", "reach_right", "press_button", "wait"]

cf_eng = CounterfactualEngine(N_ACTIONS, action_names=NAMES)

print("\n" + "="*60)
print("  Phase 9 — Step 27: Counterfactual Reasoning")
print("="*60)

V  = np.array([-0.3, -0.1, 1.2, -0.4])
Q  = np.array([0.3,  0.25, 0.8, -0.15])
di = np.array([0.04, 0.05, 0.45, 0.03])
gm = np.array([-0.3, -0.2, 0.25, -0.45])

counterfactuals = cf_eng.generate(
    selected_action = CORRECT,
    V_combined      = V,
    Q_risk          = Q,
    direct_inh      = di,
    U               = 0.3,
    C               = 0.7,
    conflict_score  = 0.1,
    stn_burst       = False,
    gate_margins    = gm,
    rho             = 0.5,
    DA              = 0.65,
    ht5             = 0.35,
    NE              = 0.30)

print(f"\n  Selected action: {NAMES[CORRECT]}")
print(f"\n  Generated {len(counterfactuals)} counterfactuals:")
for cf in counterfactuals:
    print(f"\n  Rejected: {NAMES[cf.rejected_action]}")
    print(f"    Explanation   : {cf.explanation}")
    print(f"    Delta Q_risk  : {cf.delta_Q_risk:+.3f}")
    print(f"    Pathway gap   : {cf.pathway_gap:+.4f}")
    print(f"    STN would block: {cf.stn_would_block}")
    print(f"    Reason codes  : {cf.reason_codes}")
    print(f"    CF confidence : {cf.confidence:.3f}")

print(f"\n  Formatted output:\n")
print(cf_eng.format_counterfactuals(counterfactuals))

assert len(counterfactuals) == N_ACTIONS - 1
assert all(cf.rejected_action != CORRECT for cf in counterfactuals)
print("\n  [PASS] Counterfactual reasoning generates correct outputs.")
print("="*60 + "\n")