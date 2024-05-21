from stable_baselines3 import PPO
from gym_environment import Environment  # Importa a classe do ambiente

# Crie uma instância do ambiente diretamente
env = Environment()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)

model.save("ppo_robot_navigation")

