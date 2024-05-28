from stable_baselines3 import PPO
from gym_environment import Environment  # Importa a classe do ambiente

# Crie uma instância do ambiente diretamente
env = Environment()

model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=2000)

model.save("ppo_robot_navigation")

#model = PPO.load(ppo_path, env) #load model ja treinado

TIMESTEPS = 10000
iters = 0
#n=2
#while iters<n:
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,progress_bar=True, tb_log_name='PPO')
    model.save(f"{'Training/PPO'}/{TIMESTEPS*iters}") #Colocar + numero de timesteps anteriores para não gravar no sitio errado