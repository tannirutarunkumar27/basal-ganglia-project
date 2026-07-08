"""
AttentionModule  —  Step 26
-----------------------------
Computes attention weights over all input signals to
identify which contributed most to the current decision.

Input signal groups:
    - state features    : V_combined (belief scores)
    - action beliefs    : Q_risk per action
    - pathway signals   : direct_inh, indirect_exc, stn_global
    - reward traces     : recent reward history
    - memory states     : temporal belief trajectory

Attention mechanism:
    score_i = W_q . x_i                  (query projection)
    alpha_i = softmax(score / sqrt(d))   (scaled dot-product)
    context = sum_i alpha_i * x_i        (weighted sum)

Attention weights identify the most decision-relevant signals.
Advanced innovation: these weights feed directly into the
explanation text, naming the dominant factors.
"""

"""
AttentionModule  —  Step 26
Attention weights over input signal groups.
"""

import numpy as np
from collections import deque


class AttentionModule:

    def __init__(self,
                 n_actions  : int,
                 feat_dim   : int   = 8,
                 temperature: float = 1.0,
                 dt         : float = 0.1e-3):

        self.n_actions   = n_actions
        self.feat_dim    = feat_dim
        self.temperature = temperature
        self.dt          = dt

        # Learnable query vector (one per signal group)
        self.W_q = np.random.randn(feat_dim) * 0.1
        self.lr  = 0.01

        # Named signal groups — matches _encode_signal output order
        self.signal_names = [
            "belief_Va",
            "risk_Q",
            "pathway_Go",
            "pathway_NoGo",
            "pathway_STN",
            "reward_history",
            "memory_fast",
            "neuromodulators",
        ]
        self.n_signals = len(self.signal_names)

        # Attention weights from last step
        self.alpha         = np.ones(self.n_signals) / self.n_signals
        self.alpha_history = deque(maxlen=2000)
        self.step_count    = 0

    def reset(self) -> None:
        self.alpha = np.ones(self.n_signals) / self.n_signals
        self.alpha_history.clear()
        self.step_count = 0

    def _encode_signal(self,
                        V_combined    : np.ndarray,
                        Q_risk        : np.ndarray,
                        direct_inh    : np.ndarray,
                        indirect_exc  : np.ndarray,
                        stn_global    : float,
                        reward_history: list,
                        V_history     : np.ndarray,
                        DA            : float,
                        ht5           : float,
                        NE            : float) -> np.ndarray:
        """
        Encodes each signal group into a scalar magnitude.
        Returns feature vector of shape (n_signals,).
        """
        V  = np.asarray(V_combined,   dtype=float)
        Q  = np.asarray(Q_risk,       dtype=float)
        di = np.asarray(direct_inh,   dtype=float)
        ie = np.asarray(indirect_exc, dtype=float)

        # Scalar summary per signal group
        v_range  = float(V.max() - V.min()) if len(V) else 0.0
        q_range  = float(Q.max() - Q.min()) if len(Q) else 0.0
        go_max   = float(di.max())           if len(di) else 0.0
        nogo_max = float(ie.max())           if len(ie) else 0.0
        stn_s    = float(np.clip(stn_global, 0, 1))

        r_hist = list(reward_history)
        r_var  = (float(np.var(r_hist[-10:]))
                  if len(r_hist) >= 2 else 0.0)

        if V_history is not None and len(V_history) > 1:
            mem_mag = float(
                np.abs(np.diff(V_history[-10:], axis=0)).mean())
        else:
            mem_mag = 0.0

        nm_mag = float((DA + ht5 + NE) / 3.0)

        return np.array([
            v_range, q_range, go_max, nogo_max,
            stn_s, r_var, mem_mag, nm_mag
        ])

    def compute(self,
                 V_combined    : np.ndarray,
                 Q_risk        : np.ndarray,
                 direct_inh    : np.ndarray,
                 indirect_exc  : np.ndarray,
                 stn_global    : float,
                 reward_history: list,
                 V_history     : np.ndarray,
                 DA            : float,
                 ht5           : float,
                 NE            : float) -> dict:
        """
        Step 26 core:
            score_i = W_q[i] * feature_i
            alpha_i = softmax(score / tau)

        Returns attention dict with weights, names, top signals.
        """
        self.step_count += 1
        features = self._encode_signal(
            V_combined, Q_risk, direct_inh, indirect_exc,
            stn_global, reward_history, V_history, DA, ht5, NE)

        # Scaled dot-product attention
        scores  = features * self.W_q
        tau     = max(self.temperature, 1e-3)
        shifted = (scores - scores.max()) / (
            tau * np.sqrt(self.n_signals))
        exp_s   = np.exp(np.clip(shifted, -20, 20))
        self.alpha = exp_s / (exp_s.sum() + 1e-12)

        self.alpha_history.append(self.alpha.copy())

        # Top-3 most attended signals
        top_idx = np.argsort(self.alpha)[::-1][:3]
        top_signals = [
            {
                "name"  : self.signal_names[i],
                "weight": float(self.alpha[i]),
                "value" : float(features[i]),
            }
            for i in top_idx
        ]

        return {
            "attention_weights" : self.alpha.tolist(),
            "signal_names"      : self.signal_names,   # always included
            "top_signals"       : top_signals,
            "features"          : features.tolist(),
            "dominant_signal"   : self.signal_names[
                int(np.argmax(self.alpha))],
        }

    def update_weights(self, reward: float,
                        selected_action: int) -> None:
        """
        Reward-modulated update: signals that predicted
        reward receive higher query weight.
        """
        sign     = float(np.sign(reward))
        grad     = sign * self.alpha
        self.W_q += self.lr * grad
        self.W_q  = np.clip(self.W_q, -2.0, 2.0)

    def attention_explanation(self) -> str:
        """Returns a sentence naming the dominant attended signals."""
        if not self.alpha_history:
            return "No attention computed yet."
        top3 = np.argsort(self.alpha)[::-1][:3]
        names_weights = [
            f"{self.signal_names[i]} ({self.alpha[i]:.2f})"
            for i in top3
        ]
        return ("Decision was most influenced by: "
                + ", ".join(names_weights) + ".")

    def attention_summary(self) -> dict:
        """
        Returns summary dict including signal_names so callers
        can zip names with weights without a KeyError.
        """
        hist = np.array(list(self.alpha_history))
        return {
            "attention_weights" : self.alpha.tolist(),
            "signal_names"      : self.signal_names,       # FIXED — was missing
            "dominant_signal"   : self.signal_names[
                int(np.argmax(self.alpha))],
            "mean_weights"      : (hist.mean(axis=0).tolist()
                                   if len(hist) else self.alpha.tolist()),
            "explanation"       : self.attention_explanation(),
            "step_count"        : self.step_count,
        }