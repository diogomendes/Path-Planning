from stable_baselines3 import DQN
from environment import Environment


env = Environment()

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='logdir')
#model = DQN.load('Training\\DQN\\5000000', env) #load model ja treinado

model.save("dqn_robot_navigation")



TIMESTEPS = 10000
iters = 0

while iters<250:

    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,progress_bar=True, tb_log_name='DQN')

    model.save(f"{'Training/DQN'}/{TIMESTEPS*iters}") #Colocar + numero de timesteps anteriores para não gravar no sitio errado

