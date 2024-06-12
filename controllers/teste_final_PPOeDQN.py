import gymnasium as gym
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO,DQN
from environment_test import Environment  # Importa a classe do ambiente
import pandas as pd
import time

# Crie uma instância do ambiente diretamente
env = Environment()

####################### PPO
#path = os.path.join('PPO', '5000000') #Path PPO
#model = PPO.load(path, env) #PPO teste
#alg = 'PPO_teste'
#######################

####################### DQN
path = os.path.join('DQN', '5000000') #Path DQN
model = DQN.load(path, env) #DQN teste
alg = 'DQN_teste'
#######################



def teste_model(alg,model,env,n): #alg= nome para o csv, model = modelo a testar, path(ficheiro do modelo), env=Environment, n=numero de testes
    for i in range(n):
        start_time = time.time()
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                end_time = time.time()
                elapsed_time = end_time - start_time
                done = True
                goal = False
                a = alg + '.csv'
                name_lr = os.path.join(a)
                if not os.path.isfile(name_lr):
                    df = pd.DataFrame(columns=['Episodes', 'Reward', 'Distance_Goal', 'Goal', 'Time', 'Distance_close_object', 'Min_Goal_Distance'])
                    df.to_csv(name_lr, index=False)
                df = pd.read_csv(name_lr)
                if info['distance'] <= 0.125:
                    goal = True
                df.loc[len(df)] = [i, reward, info['distance'], goal, elapsed_time, info['mean_obs'], info['min_distance']]
                df.to_csv(name_lr, index=False)

teste_model(alg,model,env,25)
