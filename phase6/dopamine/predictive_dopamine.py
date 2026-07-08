"""
PredictiveDopamineModule  —  Step 19
--------------------------------------
Instead of reacting only after reward, estimate future reward:

    D_t^pred = g(V_t, U_t, C_t, history)

Then combine predicted and observed signals:

    delta_t' = omega_1 * delta_t + omega_2 * D_t^pred

Advanced innovation:
  Anticipatory neuromodulation — the dopamine system begins
  updating weights BEFORE reward arrives based on the agent's
  belief about upcoming reward. This accelerates learning
  and produces more human-like dopamine response profiles:
    - Burst on unexpected reward
    - Dip on unexpected omission
    - Reduced burst on fully predicted reward (habituation)

Components:
  1. Reward predictor: linear predictor from belief + uncertainty
  2. Prediction error: delta_pred = actual - predicted
  3. Temporal discounting: older predictions decay
  4. Omega adaptation: weights omega_1/omega_2 adapt to accuracy
"""

import numpy as np
from collections import deque


class PredictiveDopamineModule:

    def __init__(self,
                 n_actions    : int,
                 omega_1      : float = 0.6,
                 omega_2      : float = 0.4,
                 pred_lr      : float = 0.05,
                 tau_pred     : float = 200e-3,
                 dt           : float = 0.1e-3):
        """
        n_actions : number of action channels
        omega_1   : weight on observed delta_t
        omega_2   : weight on predicted dopamine D_t^pred
        pred_lr   : learning rate for predictor weights
        tau_pred  : decay time of prediction trace (s)
        dt        : simulation timestep
        """
        self.n_actions  = n_actions
        self.omega_1    = omega_1
        self.omega_2    = omega_2
        self.pred_lr    = pred_lr
        self.tau_pred   = tau_pred
        self.dt         = dt

        # Predictor: D_pred = W_pred . [V_t, U_t, C_t, hist_mean]
        self.feature_dim = 4 + n_actions  # V(A), U, C, hist_mean, hist_std
        self.W_pred      = np.zeros(self.feature_dim)

        # Running reward history for predictor features
        self.reward_hist    = deque(maxlen=100)

        # Prediction trace — exponentially decaying prediction
        self.pred_trace     = 0.0

        # Last prediction
        self.D_pred         = 0.0

        # Prediction accuracy tracking
        self.pred_error_hist = deque(maxlen=200)

        # Adaptive omega weights
        self._omega_1_adapt = omega_1
        self._omega_2_adapt = omega_2

        # Full history
        self.D_pred_history   = []
        self.delta_prime_hist = []
        self.omega_history    = []

    def reset(self):
        self.W_pred       = np.zeros(self.feature_dim)
        self.reward_hist.clear()
        self.pred_trace   = 0.0
        self.D_pred       = 0.0
        self.pred_error_hist.clear()
        self._omega_1_adapt = self.omega_1
        self._omega_2_adapt = self.omega_2
        self.D_pred_history.clear()
        self.delta_prime_hist.clear()
        self.omega_history.clear()

    def build_features(self,
                        V_combined   : np.ndarray,
                        U            : float,
                        C            : float) -> np.ndarray:
        """
        Constructs the feature vector for reward prediction:
          [V_0..V_{A-1}, U, C, hist_mean, hist_std]
        """
        V = np.asarray(V_combined, dtype=float)
        V = V[:self.n_actions] if len(V) >= self.n_actions \
            else np.pad(V, (0, self.n_actions - len(V)))

        hist = list(self.reward_hist)
        h_mean = float(np.mean(hist)) if hist else 0.0
        h_std  = float(np.std(hist))  if hist else 0.0

        features = np.concatenate([
            V,
            [float(U), float(C), h_mean, h_std]
        ])

        # Normalise
        norm = np.linalg.norm(features)
        return features / (norm + 1e-8)

    def predict(self, V_combined: np.ndarray,
                 U: float, C: float) -> float:
        """
        Computes D_t^pred = g(V_t, U_t, C_t, history)
        using a linear predictor with sigmoid squashing.

        Returns D_pred ∈ [-1, 1].
        """
        features   = self.build_features(V_combined, U, C)
        raw_pred   = float(np.dot(self.W_pred, features))

        # Sigmoid squash to [-1, 1]
        self.D_pred = float(np.tanh(raw_pred))

        # Update exponentially decaying prediction trace
        decay          = np.exp(-self.dt / self.tau_pred)
        self.pred_trace = decay * self.pred_trace + (1 - decay) * self.D_pred

        self.D_pred_history.append(self.D_pred)
        return float(self.D_pred)

    def update_predictor(self, actual_reward: float,
                          V_combined: np.ndarray,
                          U: float, C: float) -> float:
        """
        Updates predictor weights using prediction error:
            pred_error = actual_reward - D_pred
            W_pred += lr * pred_error * features
        """
        self.reward_hist.append(actual_reward)
        features   = self.build_features(V_combined, U, C)
        pred_error = float(actual_reward - self.D_pred)

        self.W_pred += self.pred_lr * pred_error * features
        self.W_pred  = np.clip(self.W_pred, -5.0, 5.0)

        self.pred_error_hist.append(abs(pred_error))
        return pred_error

    def combine_signals(self, delta_t: float) -> float:
        """
        Core Step 19:
            delta_t' = omega_1 * delta_t + omega_2 * D_t^pred

        Combines the observed TD error with the predictive signal.
        Adapts omega weights based on predictor accuracy.

        Returns delta_prime (the enriched dopamine signal).
        """
        self._adapt_omegas()

        delta_prime = (self._omega_1_adapt * delta_t
                       + self._omega_2_adapt * self.D_pred)

        self.delta_prime_hist.append(delta_prime)
        self.omega_history.append(
            (self._omega_1_adapt, self._omega_2_adapt))

        return float(delta_prime)

    def _adapt_omegas(self) -> None:
        """
        Adaptive omega: if predictor is accurate, increase omega_2.
        Accuracy = 1 - mean(abs(pred_error)) over recent history.
        """
        errors = list(self.pred_error_hist)
        if not errors:
            return

        mean_err = float(np.mean(errors[-20:]))
        # accuracy ∈ [0,1]: 0=bad predictor, 1=perfect
        accuracy = float(np.clip(1.0 - mean_err, 0.0, 1.0))

        # Shift weight toward prediction when predictor is accurate
        target_w2 = self.omega_2 * (0.5 + 0.5 * accuracy)
        target_w1 = 1.0 - target_w2

        self._omega_2_adapt = (0.95 * self._omega_2_adapt
                                + 0.05 * target_w2)
        self._omega_1_adapt = (0.95 * self._omega_1_adapt
                                + 0.05 * target_w1)

        # Normalise so omega_1 + omega_2 = 1
        total = self._omega_1_adapt + self._omega_2_adapt + 1e-8
        self._omega_1_adapt /= total
        self._omega_2_adapt /= total

    def tonic_dopamine_level(self) -> float:
        """
        Estimates current tonic DA level from prediction trace.
        1.0 = baseline, > 1 = elevated (predicted reward incoming).
        """
        return float(1.0 + 0.5 * self.pred_trace)

    def dopamine_response_type(self,
                                delta_prime: float,
                                threshold: float = 0.1) -> str:
        """
        Classifies dopamine response:
          burst : unexpected reward
          dip   : unexpected omission
          flat  : fully predicted outcome
        """
        if delta_prime > threshold:
            return "burst"
        elif delta_prime < -threshold:
            return "dip"
        else:
            return "flat"

    def dopamine_summary(self) -> dict:
        hist = list(self.delta_prime_hist)
        w_hist = list(self.omega_history)
        return {
            "D_pred"         : float(self.D_pred),
            "pred_trace"     : float(self.pred_trace),
            "tonic_level"    : self.tonic_dopamine_level(),
            "omega_1"        : float(self._omega_1_adapt),
            "omega_2"        : float(self._omega_2_adapt),
            "mean_pred_error": (float(np.mean(
                list(self.pred_error_hist)[-50:]))
                if self.pred_error_hist else 0.0),
            "mean_delta_prime": (float(np.mean(hist))
                                  if hist else 0.0),
        }