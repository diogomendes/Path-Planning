import numpy as np
import random
import pickle
from gym_environment import Environment
import os
import pandas as pd

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
        return td_error

def process_lidar_points(points):
    min_val = np.min(points)
    max_val = np.max(points)
    mean_val = np.mean(points)
    return [min_val, max_val, mean_val]

def save_model(agent, episodes, reward, loss):
    name_model = 'Sarsa' + str(episodes) + '.pkl'
    name_lr = os.path.join('Sarsa_lr.csv')
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
    num_bins = 5  # Usando apenas 5 bins

    agent = SARSAAgent(num_actions, num_bins)

    num_episodes = 10 #Este numero é multiplicdo por 500 episodios
    max_steps_per_episode = 400
    for e in range(num_episodes):
        re = 0 # total reward so para graficos/analise
        i = 0 #iteraçoes para calculo de media
        loss = 0 #total loss para graficos/analise
        for episode in range(500):
            initial_data = env.reset()
            points = initial_data[0] if isinstance(initial_data, tuple) else initial_data
            distance_to_goal = initial_data[1]['distance'] if isinstance(initial_data, tuple) else 0

            features = process_lidar_points(points)
            features.append(distance_to_goal)
            state = tuple(features)

            action = agent.choose_action(state)
            total_reward = 0
            done = False
            error_t = 0
            t = 0

            while not done:
                next_data, reward, done, _, info = env.step(action)
                next_features = process_lidar_points(next_data)
                next_distance = info['distance']
                next_features.append(next_distance)
                next_state = tuple(next_features)

                next_action = agent.choose_action(next_state)
                error = agent.update_q_table(state, action, reward, next_state, next_action)
                error_t += error
                t += 1

                state = next_state
                action = next_action

            re += reward
            loss += (error_t/t)
            if episode % 100 == 0:
                print(f"Episode {500*e+episode}: Total reward = {reward}")

        save_model(agent, 500*(e+1), re/500, loss/500)