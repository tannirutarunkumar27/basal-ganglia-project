"""
Step 18 verification — risk-sensitive decision modeling.
Configures two actions with same E[R] but different variance.
Confirms risk-sensitive selection prefers the lower-variance option.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from risk.risk_module import RiskModule

np.random.seed(2)

N_ACTIONS = 4
DT        = 0.1e-3
N_STEPS   = 500

risk = RiskModule(N_ACTIONS, rho_base=0.5, dt=DT)

q_risk_log = []
rho_log    = []
e_r_log    = []
var_r_log  = []

print("\n" + "="*55)
print("  Phase 6 — Step 18: Risk-Sensitive Decision Modeling")
print("="*55)
print("\n  Action design:")
print("    action 0: E[R]=0.5, Var=0.01  (safe)")
print("    action 1: E[R]=0.5, Var=0.80  (risky, same mean)")
print("    action 2: E[R]=0.8, Var=0.05  (best risk-adjusted)")
print("    action 3: E[R]=0.3, Var=0.20  (bad)")

# Define reward distributions per action
reward_dists = {
    0: lambda: 0.5 + np.random.randn() * 0.1,    # safe
    1: lambda: 0.5 + np.random.randn() * 0.9,    # risky
    2: lambda: 0.8 + np.random.randn() * 0.22,   # best
    3: lambda: 0.3 + np.random.randn() * 0.45,   # bad
}

for step in range(N_STEPS):
    # Sample rewards for all actions this step
    for a in range(N_ACTIONS):
        r = float(reward_dists[a]())
        risk.update_statistics(a, r)

    # Adapt rho
    U = max(0.1, 0.6 - step / N_STEPS)
    rho = risk.adapt_rho(U=U, conflict_score=0.1,
                          recent_loss=-0.05 * (1 - step/N_STEPS))

    # Compute risk-adjusted Q values
    Q_risk = risk.compute_q_risk()

    q_risk_log.append(Q_risk.copy())
    rho_log.append(rho)
    e_r_log.append(risk.E_r.copy())
    var_r_log.append(risk.Var_r.copy())

q_risk_arr = np.array(q_risk_log)
e_r_arr    = np.array(e_r_log)
var_r_arr  = np.array(var_r_log)

final_q = q_risk_arr[-50:].mean(axis=0)
best_q  = int(np.argmax(final_q))
summary = risk.risk_summary()

print(f"\n  Final risk-adjusted Q values:")
for a in range(N_ACTIONS):
    print(f"    action {a}: Q_risk={final_q[a]:+.4f}  "
          f"E[R]={summary['E_r'][a]:.3f}  "
          f"Var={summary['Var_r'][a]:.3f}")

print(f"\n  Best Q_risk action : {best_q}  (expected: 2)")
print(f"  Safest action      : {summary['safest_action']}")
print(f"  Final rho          : {summary['rho']:.3f}")
assert best_q == 2, f"Expected action 2 (best Q_risk), got {best_q}"
print("  [PASS] Risk-sensitive utility correct.")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
t_ms   = np.arange(N_STEPS) * DT * 1000
colors = ["steelblue", "coral", "forestgreen", "gold"]

for a in range(N_ACTIONS):
    axes[0].plot(t_ms, q_risk_arr[:, a],
                 color=colors[a], lw=1.2,
                 label=f"Q_risk a{a}")
axes[0].set_title("Risk-adjusted utility  Q_a = E[R_a] - rho*Var(R_a)")
axes[0].set_ylabel("Q_risk")
axes[0].legend(fontsize=8)
axes[0].axhline(0, color="gray", ls="--", lw=0.5)

for a in range(N_ACTIONS):
    axes[1].plot(t_ms, var_r_arr[:, a],
                 color=colors[a], lw=1, label=f"Var a{a}")
axes[1].set_title("Reward variance per action (rolling window)")
axes[1].set_ylabel("Var(R_a)")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms, rho_log, color="slateblue", lw=1.2)
axes[2].set_title("Dynamic risk aversion rho (adapts to uncertainty)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("rho")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "step18_risk.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("  Plot saved: results/step18_risk.png")
print("="*55 + "\n")