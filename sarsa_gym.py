import numpy as np
import random
from gym_environment import Environment

class SARSAAgent:
    def __init__(self, num_actions, num_bins, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Utilizando menos características e menos bins
        self.q_table = np.zeros([num_bins] * 4 + [num_actions])
        self.bins = [np.linspace(0, 10, num_bins + 1)[:-1] for _ in range(4)]  # Ajuste este intervalo conforme necessário

    def discretize(self, state):
        state_idx = tuple(np.digitize([state[i]], self.bins[i])[0] - 1 for i in range(len(self.bins)))
        return state_idx

    def choose_action(self, state):
        state_idx = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state, next_action):
        current_state_idx = self.discretize(state)
        next_state_idx = self.discretize(next_state)
        td_target = reward + self.gamma * self.q_table[next_state_idx][next_action]
        td_error = td_target - self.q_table[current_state_idx][action]
        self.q_table[current_state_idx][action] += self.alpha * td_error

def process_lidar_points(points):
    min_val = np.min(points)
    max_val = np.max(points)
    mean_val = np.mean(points)
    return [min_val, max_val, mean_val]

env = Environment()
num_actions = 3
num_bins = 5  # Usando apenas 5 bins

agent = SARSAAgent(num_actions, num_bins)

num_episodes = 1000
max_steps_per_episode = 400

for episode in range(num_episodes):
    initial_data = env.reset()
    points = initial_data[0] if isinstance(initial_data, tuple) else initial_data
    distance_to_goal = initial_data[1]['distance'] if isinstance(initial_data, tuple) else 0

    features = process_lidar_points(points)
    features.append(distance_to_goal)
    state = tuple(features)

    action = agent.choose_action(state)
    total_reward = 0
    done = False

    while not done:
        next_data, reward, done, _, info = env.step(action)
        next_features = process_lidar_points(next_data)
        next_distance = info['distance']
        next_features.append(next_distance)
        next_state = tuple(next_features)

        next_action = agent.choose_action(next_state)
        agent.update_q_table(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}: Total reward = {total_reward}")
