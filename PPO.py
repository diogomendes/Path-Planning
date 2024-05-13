import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel,warp_robot
import random

class Environment(gym.Env):
    def __init__(self):
        """
               Inicializa o ambiente e seus componentes.

               Retorna:
               - None
               """
        self.robot: Supervisor = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.lidar: Lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.max_episodes = 100
        self.max_steps_per_episode = 1000
        self.init_pos = np.array([0.17, 0.12])
        self.final_position = np.array([2.45, 0])  # Definir a posição final
        self.action_size = 3  # Definir o tamanho do espaço de ações
        self.max_speed = 1.5

        self.observation_space=spaces.Box(low=0, high=10, shape=(self.lidar.getHorizontalResolution(),), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

    def reset(self):
        print("initial position", self.init_pos)
        warp_robot(self.robot, "EPUCK", self.init_pos)

    def calculate_distance_to_goal(self):
        """
        Calcula a distância atual até a posição final.

        Retorna:
        - float: Distância até o objetivo.
        """
        current_position = np.array(self.gps.getValues()[:2])
        # print("current position", current_position)
        distance_to_goal = np.linalg.norm(current_position - self.final_position)
        return distance_to_goal

    def detect_collision(self, lidar_point_cloud):
        """
        Detecta colisão usando dados do sensor LIDAR.

        Parâmetros:
        - lidar_point_cloud (array): Array de pontos LIDAR.
        - collision_threshold (float): Threshold para detectar colisão.

        Retorna:
        - bool: True se colisão detectada, False caso contrário.
        """
        for point in lidar_point_cloud:
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2)
            #print("distance_to_robot = ", distance_to_robot)
            if distance_to_robot == float('inf'):
                return True  # Colisão detectada
        return False  # Nenhuma colisão detectada

    def detect_obstacle_proximity(self, lidar_point_cloud):
        """
        Detecta a proximidade de obstáculos usando dados do sensor LIDAR.

        Parâmetros:
        - lidar_point_cloud (array): Array de pontos LIDAR.

        Retorna:
        - float: Recompensa negativa se estiver próximo a um obstáculo, 0 caso contrário.
        """
        obstacle_proximity_threshold = 0.1  # Definir um limite para considerar a proximidade de um obstáculo
        min_distance = float('inf')

        for point in lidar_point_cloud:
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2)
            if distance_to_robot < min_distance:
                min_distance = distance_to_robot

        if min_distance < obstacle_proximity_threshold:
            # print("entrou")
            return -10  # Recompensa negativa se estiver próximo a um obstáculo
        else:
            return 0  # Nenhuma obstáculo próximo, então retornar 0

    def get_state(self):
        """
        Obtém o estado atual do ambiente.

        Retorna:
        - numpy.ndarray: Estado atual do ambiente.
        """
        lidar_point_cloud = self.lidar.getPointCloud()
        distance_to_goal = self.calculate_distance_to_goal()
        state = [point.x for point in lidar_point_cloud] + [point.y for point in lidar_point_cloud] + [distance_to_goal]
        return np.array(state)

    def get_reward(self):
        """
        Calcula a recompensa com base na distância até o objetivo e detecção de colisão.

        Retorna:
        - float: Recompensa atual.
        """
        distance_to_goal = self.calculate_distance_to_goal()
        collision = self.detect_collision(self.lidar.getPointCloud())
        # print("Collision detected: ", collision)
        obstacle_proximity_reward = self.detect_obstacle_proximity(self.lidar.getPointCloud())
        # print("Obstacle proximity", obstacle_proximity_reward)

        if collision:
            return -500

        if obstacle_proximity_reward < 0:
            return obstacle_proximity_reward

        if distance_to_goal < 0.06:
            return 150
        elif distance_to_goal < 5:
            return 1.5 * (1 - np.exp(-1 / distance_to_goal))
        elif distance_to_goal < 15:
            return 0.5 * (1 - np.exp(-1 / distance_to_goal))
        else:
            return -1

    def apply_action(self, action):
        """
        Aplica a ação ao robô.

        Parâmetros:
        - action (int): Ação a ser aplicada.

        Retorna:
        - None
        """
        linear_vel = 0
        angular_vel = 0

        if action == 0:
            linear_vel = self.max_speed
        elif action == 1:
            angular_vel = self.max_speed
        elif action == 2:
            angular_vel = -self.max_speed
        self.robot.step(200)

        # Aplicar velocidades ao robô
        cmd_vel(self.robot, linear_vel, angular_vel)

    def step(self, action):
        """
        Aplica a ação ao ambiente e retorna o novo estado.

        Parâmetros:
        - action (int): Ação a ser aplicada.

        Retorna:
        - numpy.ndarray: Novo estado após a aplicação da ação.
        """
        self.apply_action(action)  # Aplica a ação ao ambiente
        # self.robot.step(self.timestep)  # Avança a simulação para atualizar o estado do LiDAR
        state = self.get_state()  # Obtém o novo estado com base nos dados atualizados do LiDAR
        self.robot.step(self.timestep)
        return state


def main():
    env=Environment()

    env=DummyVecEnv([lambda: env])

    model=PPO("MlpPolicy", env,verbose=1)

    model.learn(total_timesteps=10000)

    model.save("ppo_model")


if __name__ == "__main__":
    main()