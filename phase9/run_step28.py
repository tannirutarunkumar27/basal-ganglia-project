"""
Step 28 verification — full explanation output.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from symbolic.neuro_symbolic_reasoner      import NeuroSymbolicReasoner
from attention.attention_module            import AttentionModule
from counterfactual.counterfactual_engine  import CounterfactualEngine
from explanation.explanation_composer      import ExplanationComposer

np.random.seed(3)

N_ACTIONS = 4
CORRECT   = 2
NAMES     = ["reach_left", "reach_right", "press_button", "wait"]

reasoner  = NeuroSymbolicReasoner(N_ACTIONS, dt=0.1e-3)
attn_mod  = AttentionModule(N_ACTIONS, dt=0.1e-3)
cf_eng    = CounterfactualEngine(N_ACTIONS, action_names=NAMES)
composer  = ExplanationComposer(
    N_ACTIONS, action_names=NAMES,
    log_dir=os.path.join(HERE, "results"))

print("\n" + "="*65)
print("  Phase 9 — Step 28: Full Explanation Output")
print("="*65)

V  = np.array([-0.3, -0.1, 1.2, -0.4])
Q  = np.array([0.3,  0.25, 0.8, -0.15])
di = np.array([0.04, 0.05, 0.45, 0.03])
ie = np.array([0.10, 0.12, 0.06, 0.11])
gm = np.array([-0.3, -0.2, 0.25, -0.45])
rw = [0.0, 1.0, 0.8, 1.0, 0.9, 1.0, 0.7, 1.0, 0.9, 1.0]

U, C = 0.3, 0.7
DA, ht5, NE = 0.65, 0.35, 0.30
alpha_t = 0.063
Mt      = 0.52
rho     = 0.5
conflict = 0.1
stn_burst = False

# Step 25
r_out = reasoner.reason(
    V_combined=V, U=U, C=C, Q_risk=Q, conflict_score=conflict,
    stn_burst=stn_burst, direct_inh=di, indirect_exc=ie,
    DA=DA, ht5=ht5, NE=NE, reward_history=rw,
    gate_margins=gm, gate_action=CORRECT)

# Step 26
a_out = attn_mod.compute(
    V_combined=V, Q_risk=Q, direct_inh=di, indirect_exc=ie,
    stn_global=0.05, reward_history=rw,
    V_history=np.random.randn(10, N_ACTIONS) * 0.2,
    DA=DA, ht5=ht5, NE=NE)
attn_mod.update_weights(1.0, CORRECT)

# Step 27
cfs = cf_eng.generate(
    selected_action=CORRECT, V_combined=V, Q_risk=Q,
    direct_inh=di, U=U, C=C, conflict_score=conflict,
    stn_burst=stn_burst, gate_margins=gm,
    rho=rho, DA=DA, ht5=ht5, NE=NE)

# Step 28
out = composer.compose(
    reasoning_out=r_out, attention_out=a_out, counterfactuals=cfs,
    U=U, C=C, DA=DA, ht5=ht5, NE=NE, alpha_t=alpha_t,
    Mt=Mt, rho=rho, conflict_score=conflict,
    gate_margin=float(gm[CORRECT]), t_ms=0.0)

print(f"\n{'='*65}")
print(out["full_explanation"])
print(f"{'='*65}")

log_path = composer.save_log("step28_explanation.json", last_n=10)
print(f"\n  Log saved: {log_path}")

assert out["chosen_action"] == CORRECT
assert 0.0 <= out["confidence"] <= 1.0
assert len(out["rejected_texts"]) > 0
print("\n  [PASS] Full explanation record generated correctly.")
print("="*65 + "\n")