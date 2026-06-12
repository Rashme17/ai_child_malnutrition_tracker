# test_qtable.py - Run this to verify Q-table structure
from q_learning_agent import agent
import json

# Simulate a few episodes
for _ in range(100):
    import random
    start = random.choice(['severe', 'moderate', 'mild'])
    agent.simulate_episode(start)

# Get the data
policy = agent.get_policy()
q_table = agent.get_q_table()
states = agent.states
actions = agent.actions

print("=" * 60)
print("POLICY:")
print(json.dumps(policy, indent=2))

print("\n" + "=" * 60)
print("STATES:")
print(states)

print("\n" + "=" * 60)
print("ACTIONS:")
print(actions)

print("\n" + "=" * 60)
print("Q-TABLE STRUCTURE:")
print(f"Type: {type(q_table)}")
print(f"Shape: {len(q_table)} rows x {len(q_table[0]) if q_table else 0} columns")

print("\n" + "=" * 60)
print("Q-TABLE VALUES:")
for i, state in enumerate(states):
    if state != 'healthy':
        print(f"\n{state.upper()}:")
        for j, action in enumerate(actions):
            print(f"  {action}: {q_table[i][j]:.2f}")
        best_idx = q_table[i].index(max(q_table[i]))
        print(f"  → BEST: {actions[best_idx]}")

print("\n" + "=" * 60)
print("JSON SERIALIZATION TEST:")
test_data = {
    "policy": policy,
    "q_table": q_table,
    "states": states,
    "actions": actions
}
try:
    json_str = json.dumps(test_data, indent=2)
    print("✓ JSON serialization successful")
    print(f"Size: {len(json_str)} bytes")
except Exception as e:
    print(f"✗ JSON serialization failed: {e}")