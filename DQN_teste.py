from environment import Environment
from stable_baselines3 import DQN

env = Environment()

model = DQN.load("dqn")#Substituir

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()