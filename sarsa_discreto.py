import numpy as np
import random
import pickle
import os
import pandas as pd
from gym_environment import Environment

class SARSAAgent:
    def __init__(self, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Garantindo que todos os valores sejam capturados pelos bins
        self.bins = np.linspace(0, 3, 6)  # Isso cria 5 bins de 0 a 3
        self.q_table = np.zeros((5,) * 7 + (num_actions,))  # 7 grupos de leituras, 5 bins cada

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
        return td_error


def process_lidar_points(points):
    lidar_points = points[:-1]  # Assuming the last value is not a LIDAR reading
    grouped_points = [np.mean(lidar_points[i:i+6]) for i in range(0, len(lidar_points), 6)]
    return grouped_points

def save_model(agent, episodes, reward, loss):
    name_model = f'Sarsa_{episodes}.pkl'
    name_lr = os.path.join('Sarsa_lr.csv')
    if not os.path.isfile(name_lr):
        df = pd.DataFrame(columns=['Episodes', 'Loss', 'Reward'])
        df.to_csv(name_lr, index=False)
    df = pd.read_csv(name_lr)
    df.loc[len(df)] = [episodes, reward, loss]
    df.to_csv(name_lr, index=False)
    with open(name_model, 'wb') as f:
        pickle.dump(agent.q_table, f)

