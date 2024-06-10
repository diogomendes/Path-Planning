from sarsa_gym import SARSAAgent, process_lidar_points
import pickle
from gym_environment import Environment
import numpy as np


# Carregar a tabela Q salva
with open('Sarsa0.pkl', 'rb') as f:
    q_table = pickle.load(f)


env = Environment()
def evaluate(q_table, env):
    num_actions = 3
    num_bins = 5
    agent = SARSAAgent(num_actions, num_bins)
    agent.q_table = q_table
    rewards = []
    for episode in range(5):
        initial_data = env.reset()
        points = initial_data[0] if isinstance(initial_data, tuple) else initial_data
        distance_to_goal = initial_data[1]['distance'] if isinstance(initial_data, tuple) else 0

        features = process_lidar_points(points)
        features.append(distance_to_goal)
        state = tuple(features)

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_data, reward, done, _, info = env.step(action)
            next_features = process_lidar_points(next_data)
            next_distance = info['distance']
            next_features.append(next_distance)
            next_state = tuple(next_features)

            state = next_state

        rewards.append(reward)

    return np.mean(rewards), np.std(rewards)

print(evaluate(q_table,env)) #Reward Media e Desvio Padrao para 5 episodios