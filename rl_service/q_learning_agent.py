import numpy as np
import random

class QLearningAgent:
    def __init__(self, states, actions, rewards, alpha=0.5, gamma=0.9, epsilon=0.1):  # Increased alpha to 0.5, epsilon to 0.1 for better learning
        self.states = states
        self.actions = actions
        self.rewards = rewards  # Dict: (state, action) -> (next_state, reward)
        self.alpha = alpha  # Learning rate (increased for stronger updates)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate (increased for more exploration initially)
        self.q_table = np.zeros((len(states), len(actions)))

    def get_state_index(self, state):
        return self.states.index(state)

    def get_action_index(self, action):
        return self.actions.index(action)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            state_idx = self.get_state_index(state)
            action_idx = np.argmax(self.q_table[state_idx])
            return self.actions[action_idx]  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        state_idx = self.get_state_index(state)
        action_idx = self.get_action_index(action)
        next_state_idx = self.get_state_index(next_state)
        best_next = np.max(self.q_table[next_state_idx])
        old_q = self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * (reward + self.gamma * best_next - old_q)
        # Debug print (remove after testing)
        print(f"Update: {state} + {action} -> {next_state}, Reward: {reward}, Old Q: {old_q:.2f}, New Q: {self.q_table[state_idx, action_idx]:.2f}")

    def update_from_feedback(self, state, action, feedback_type):
        # Update Q-value based on user feedback
        if feedback_type == 'approve':
            reward = 5  # Positive feedback
        elif feedback_type == 'disapprove':
            reward = -5  # Negative feedback
        else:
            return  # Invalid feedback
        
        # Assume next state is 'healthy' for simplicity (or use original rewards)
        next_state = self.rewards.get((state, action), ('healthy', 0))[0]
        self.update_q_table(state, action, reward, next_state)
        print(f"Feedback Update: {state} + {action} -> {feedback_type.upper()}, Adjusted Q-value.")

    def simulate_episode(self, start_state):
        state = start_state
        total_reward = 0
        path = []
        while state != 'healthy':
            action = self.choose_action(state)
            next_state, reward = self.rewards.get((state, action), ('healthy', 0))
            self.update_q_table(state, action, reward, next_state)
            total_reward += reward
            path.append((state, action, next_state, reward))
            state = next_state
        return path, total_reward

    def get_policy(self):
        policy = {}
        for state in self.states[:-1]:  # Exclude healthy
            idx = np.argmax(self.q_table[self.get_state_index(state)])
            policy[state] = self.actions[idx]
        return policy

    def get_q_table(self):
        return self.q_table.tolist()

    def reset_q_table(self):  # New method to reset Q-table
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        print("Q-table reset to zeros.")

# Define states, actions, rewards (from your example)
states = ['severe', 'moderate', 'mild', 'healthy']
actions = ['home_remedy', 'balanced_diet', 'micronutrient_supplementation', 'counseling', 'therapeutic_feeding', 'hospital_treatment']
rewards = {
    ('severe', 'home_remedy'): ('severe', -15),
    ('severe', 'balanced_diet'): ('severe', -10),
    ('severe', 'micronutrient_supplementation'): ('moderate', 2),
    ('severe', 'counseling'): ('moderate', 5),
    ('severe', 'therapeutic_feeding'): ('mild', 12),
    ('severe', 'hospital_treatment'): ('moderate', 15),
    ('moderate', 'home_remedy'): ('moderate', -5),
    ('moderate', 'balanced_diet'): ('mild', 3),
    ('moderate', 'micronutrient_supplementation'): ('mild', 8),
    ('moderate', 'counseling'): ('mild', 8),
    ('moderate', 'therapeutic_feeding'): ('mild', 10),
    ('moderate', 'hospital_treatment'): ('mild', 8),
    ('mild', 'home_remedy'): ('healthy', 2),
    ('mild', 'balanced_diet'): ('healthy', 5),
    ('mild', 'micronutrient_supplementation'): ('healthy', 10),
    ('mild', 'counseling'): ('healthy', 8),
    ('mild', 'therapeutic_feeding'): ('healthy', 6),
    ('mild', 'hospital_treatment'): ('healthy', 2),
}

# Add healthy state rewards (terminal state, no change)
for action in actions:
    rewards[('healthy', action)] = ('healthy', 0)

agent = QLearningAgent(states, actions, rewards)