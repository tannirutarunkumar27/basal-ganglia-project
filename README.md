# Spiking Neural Network Basal Ganglia (SNN-BG) Project

A biologically plausible, neuromorphic agent model simulating the **Basal Ganglia (BG)** pathways and cortical integration. Built using **Spiking Neural Networks (SNNs)** with Adaptive Exponential (AdEx) Integrate-and-Fire neurons, the project integrates Bayesian belief tracking, reinforcement learning (Actor-Critic), multi-neuromodulatory control, neuro-symbolic reasoning, and sparse neuromorphic optimizations.

---

## 🌌 Architectural Overview

The project is structured into **13 progressive phases**, starting from objective formulation up to unified experimental validation and multi-dimensional metric reporting:

```
                  ┌──────────────────────────────────────────┐
                  │       Phase 1: Objectives & Constraints  │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 2: Network & Calibration     │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 3: Bayesian Belief Pipeline  │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 4: BG Pathways (Go/NoGo/STN) │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 5: Thalamocortical Gating    │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 6: RL Engine & Critics       │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 7: Plasticity (STDP/STDE)   │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 8: Multi-Neuromodulation     │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 9: Neuro-symbolic Reasoning  │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 10: Neuromorphic Optimizer   │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 11: Training Loop & Checkpts │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 12: Experimental Validation  │
                  └────────────────────┬─────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │       Phase 13: Metric Evaluation        │
                  └──────────────────────────────────────────┘
```

---

## 🛠️ Phase-by-Phase Breakdown

### 🎯 Phase 1: Objectives & Constraints
* **Unified Objective**: Formulates the agent's discount-return maximization objective subject to biological constraints (no backpropagation, spike-based communication, strict energy budgets).
* **Constraint Validator**: Automatically tests and reports compliance of agent configurations with biological limits.

### 🧠 Phase 2: Network Topology & Calibration
* **AdEx Neurons**: Populations of Adaptive Exponential Integrate-and-Fire neurons representing cortical and subcortical regions.
* **Connectivity & Synapses**: Custom synaptic wiring and topology validation.
* **Calibration**: Ensures target baseline spike-rates and provides real-time energy estimation (in nanojoules).

### ⏳ Phase 3: Bayesian Belief & Temporal Tracking
* **Belief Accumulation**: Translates incoming cortical spike evidence into probabilistic belief states.
* **Temporal Integration**: Tracks memory traces and calculates entropy/uncertainty metrics of the current state representation.

### ⚡ Phase 4: Basal Ganglia Pathways
* **Direct Pathway**: Facilitates action selection (Go) via D1 MSNs.
* **Indirect Pathway**: Inhibits actions (NoGo) via D2 MSNs and GPe.
* **Hyperdirect Pathway**: Intercepts actions (Veto/Stop) through STN excitation of GPi.
* **Conflict Weighting**: Adjusts pathway weights dynamically based on uncertainty and action conflicts.

### ⛩️ Phase 5: Thalamocortical Gating & Relay
* **Gpi Gating**: Integrates pathway signals to release/gate motor actions.
* **Adaptive Thresholds**: Modulates the thalamic decision boundary.
* **Explainability Logger**: Logs gate status and decision-making margins.

### 📈 Phase 6: Reinforcement Learning & Critics
* **Striatal Actor-Critic**: A biologically plausible RL model driven by Dopamine-mediated Reward Prediction Error (RPE).
* **Multi-Critic**: Tracks task values and safety metrics to support risk-sensitive choices.

### 🧬 Phase 7: Synaptic Plasticity
* **STDP & STDE**: Spiking Three-Factor Eligibility (STDE) plasticity managers that combine pre/post-synaptic spike timing with neuromodulatory signals.
* **Multi-Timescale Traces**: Maintains eligibility traces operating over short and long timescales.

### 🧪 Phase 8: Multi-Neuromodulation
* **Dopamine (DA)**: Drives reinforcement learning and exploitation.
* **Serotonin (5-HT)**: Drives risk assessment and safety valuation.
* **Norepinephrine (NE)**: Tracks environmental volatility and modulates exploration/vigilance.

### 🔮 Phase 9: Neuro-Symbolic Explanation
* **Attention**: Focuses reasoning on salient behavioral events.
* **Explanation Composer**: Translates SNN activity and neuromodulator ratios into human-readable logical statements using symbolic rules.
* **Counterfactuals**: Simulates alternative choice outcomes to explain "why not" another action.

### ⚡ Phase 10: Neuromorphic Optimization
* **Sparsity Optimization**: Dynamic pruning of inactive synapses to maintain target sparsity.
* **AHP Rate Limiting**: Mitigates runaway excitation by adapting After-Hyperpolarization (AHP) dynamics.

### 🔄 Phase 11: System Training Loop
* **Orchestrator**: Handles episodic execution, check-pointing, model saving/loading, and parameter configurations.

### 🧪 Phase 12: Experimental Validation
* Runs the integrated pipeline across standard and advanced behavioral benchmarks to score capabilities (e.g., adaptability, learning speed, safety, and explanation quality).

### 📊 Phase 13: Metric Evaluation
* Computes and visualizes comprehensive metrics: Behavioral accuracy, learning speed, neuromodulation profiles, neural synchronization, explanation confidence, and energy efficiency.

---

## 🏃 Getting Started

### Prerequisites
* Python 3.8+
* Install dependencies within a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Validation Suite
To validate SNN-BG performance across all 9 benchmark tasks:
```bash
python phase12/run_phase12_validation.py
```
This saves capability radar charts, task comparisons, and validation JSON reports in `phase12/results/`.

### Run Detailed Metric Evaluation
To run detailed metric profiling and generate timeseries logs:
```bash
python phase13/run_phase13_evaluation.py
```
This saves detailed evaluations and time-series plots under `phase13/results/`.

---

## 🎮 Benchmark Task Suite

The validation suite includes both standard and complex task environments:
1. **Probabilistic Bandit**: K-armed bandit with noisy rewards.
2. **Reversal Learning**: Rule reversal where optimal choice changes mid-run.
3. **Stop-Signal Task**: Tests response cancellation / hyperdirect pathway veto.
4. **Sequential Decision Task**: Two-stage decision tasks requiring planning.
5. **Grid World**: Navigation under sparse reward signals.
6. **Risk-Sensitive Choice**: Choices between safe low-reward and risky high-reward options.
7. **Volatile Reward Bandit**: Rapid reward probability fluctuations.
8. **Hidden Rule Task**: Unannounced rules that the model must infer.
9. **Counterfactual Choice**: Tasks requiring valuation of unchosen alternatives.
