from sarsa_discreto import SARSAAgent, process_lidar_points
import pickle
from environment_test import Environment
import numpy as np
import time
import pandas as pd
import os

######################SARSA
path = 'Sarsa40000.pkl'
with open(path, 'rb') as f:
    q_table = pickle.load(f)
alg = 'Sarsa_teste'
######################

env = Environment()

def evaluate(alg,q_table,env,n):
    num_actions = 3
    agent = SARSAAgent(num_actions)
    agent.q_table = q_table
    rewards = []
    for e in range(n):
        start_time = time.time()
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

            state = next_state
            total_reward += reward
            if done:
                end_time = time.time()
                elapsed_time = end_time - start_time
                done = True
                goal = False
                a = alg + '.csv'
                name_lr = os.path.join(a)
                if not os.path.isfile(name_lr):
                    df = pd.DataFrame(columns=['Episodes', 'Reward', 'Distance_Goal', 'Goal', 'Time', 'Distance_close_object','Min_Goal_Distance'])
                    df.to_csv(name_lr, index=False)
                df = pd.read_csv(name_lr)
                if info['distance'] <= 0.125:
                    goal = True
                df.loc[len(df)] = [e, reward, info['distance'], goal, elapsed_time, info['mean_obs'], info['min_distance']]
                df.to_csv(name_lr, index=False)

evaluate(alg,q_table,env,25)