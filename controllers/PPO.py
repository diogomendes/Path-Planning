from stable_baselines3 import PPO
from environment import Environment


env = Environment()

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log='logdir')

#model = PPO.load('Training\\PPO\\2500000', env) #load model ja treinado


model.save("ppo_robot_navigation")

TIMESTEPS = 10000
iters = 0

while iters<250:

    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,progress_bar=True, tb_log_name='PPO')
    model.save(f"{'Training/PPO'}/{TIMESTEPS*iters}") #Colocar + numero de timesteps anteriores para não gravar no sitio errado