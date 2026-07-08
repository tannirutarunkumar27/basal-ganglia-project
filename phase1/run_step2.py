"""
Runs one episode on every task with a random agent.
Verifies all tasks are functional before connecting the SNN.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tasks.task_registry import ALL_TASKS, get_task

print("\n" + "="*60)
print("  Phase 1 — Step 2: Benchmark Task Suite Verification")
print("="*60)

for task_name in ALL_TASKS:
    task  = get_task(task_name)
    state = task.reset()
    total_reward = 0.0
    steps        = 0

    while not task.done:
        action = np.random.randint(0, task.n_actions)
        state, reward, done, info = task.step(action)
        total_reward += reward
        steps        += 1

    tag = "[STANDARD]" if task_name in \
        ["probabilistic_bandit","reversal_learning","stop_signal",
         "sequential_decision","grid_world"] else "[ADVANCED]"

    print(f"  {tag:12s} {task_name:<30s} "
          f"steps={steps:4d}  total_reward={total_reward:+7.2f}")

print("="*60)
print("  All tasks completed successfully.")
print("="*60 + "\n")