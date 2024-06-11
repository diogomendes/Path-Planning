import numpy as np
import random
import pickle
import os
import pandas as pd
from environment import Environment

class QLearningAgent:
    def __init__(self, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bins = np.linspace(0, 3, 5)  # Cria 5 bins de 0 a 3
        self.q_table = np.zeros((4,) * 7 + (num_actions,))  # 7 dimensões de estado, 1 de ação

    def discretize(self, state):
        if len(state) != 7:
            raise ValueError("State should have exactly 7 dimensions based on the lidar input specified.")
        state_idx = [np.digitize(state[i], self.bins) - 1 for i in range(len(state))]
        state_idx = tuple(max(0, min(x, len(self.bins) - 2)) for x in state_idx)
        return state_idx

    def choose_action(self, state):
        state_idx = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state):
        current_state_idx = self.discretize(state)
        next_state_idx = self.discretize(next_state)
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action]
        td_error = td_target - self.q_table[current_state_idx][action]
        self.q_table[current_state_idx][action] += self.alpha * td_error
        return td_error

def process_lidar_points(lidar_data):
    lidar_points = lidar_data[:-1]
    num_groups = 7
    points_per_group = len(lidar_points) // num_groups
    reduced_lidar_points = [np.mean(lidar_points[i:i + points_per_group]) for i in range(0, len(lidar_points), points_per_group)]
    return reduced_lidar_points

def save_model(agent, episodes, reward, loss):
    name_model = 'QLearning' + str(episodes) + '.pkl'
    name_lr = os.path.join('QLearning_lr.csv')
    if not os.path.isfile(name_lr):
        df = pd.DataFrame(columns=['Episodes', 'Loss', 'Reward'])
        df.to_csv(name_lr, index=False)
    df = pd.read_csv(name_lr)
    df.loc[len(df)] = [episodes, reward, loss]
    df.to_csv(name_lr, index=False)

    with open(name_model, 'wb') as f:
        pickle.dump(agent.q_table, f)

if __name__ == "__main__":
    env = Environment()
    num_actions = 3
    agent = QLearningAgent(num_actions)

    num_episodes = 200
    for e in range(num_episodes):
        raw_state, info = env.reset()
        state = process_lidar_points(raw_state)
        state = agent.discretize(state)

        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_raw_state, reward, done, _, info = env.step(action)
            next_state = process_lidar_points(next_raw_state)
            next_state = agent.discretize(next_state)

            error = agent.update_q_table(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        if e % 10 == 0:
            print(f"Episode {e}: Total reward = {total_reward}")
            save_model(agent, e, total_reward, error)
