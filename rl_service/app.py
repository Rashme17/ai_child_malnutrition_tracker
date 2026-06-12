from flask import Flask, request, jsonify
import random, numpy as np
from q_learning_agent import agent

app = Flask(__name__)

@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(silent=True) or {}
    episodes = int(data.get("episodes", 500))
    rewards = []
    for _ in range(episodes):
        start_state = random.choice(['severe','moderate','mild'])
        _, reward = agent.simulate_episode(start_state)
        rewards.append(reward)
    
    # Get Q-table and verify it's a proper list
    q_table_list = agent.get_q_table()
    print(f"RL Service - Q-table type: {type(q_table_list)}")
    print(f"RL Service - Q-table length: {len(q_table_list)}")
    print(f"RL Service - First row: {q_table_list[0]}")
    
    # Return Q-table along with policy
    response = {
        "avg_reward": float(np.mean(rewards)), 
        "policy": agent.get_policy(),
        "q_table": q_table_list,  # This should be a 2D list
        "states": agent.states,  # List of state names
        "actions": agent.actions  # List of action names
    }
    
    print(f"RL Service - Response keys: {response.keys()}")
    return jsonify(response)

@app.route("/update_feedback", methods=["POST"])
def update_feedback():
    data = request.get_json(silent=True) or {}
    agent.update_from_feedback(data["state"], data["action"], data["feedback_type"])
    
    # Return updated Q-table
    return jsonify({
        "policy": agent.get_policy(),
        "q_table": agent.get_q_table(),
        "states": agent.states,
        "actions": agent.actions
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)