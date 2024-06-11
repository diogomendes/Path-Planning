import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel, warp_robot
import math
import random


class Environment(gym.Env):
    def __init__(self):
        """
        Inicializa o ambiente e seus componentes.

        Retorna:
        - None
        """
        super(Environment, self).__init__()

        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.touch = self.robot.getDevice('touch sensor')
        self.touch.enable(self.timestep)
        self.max_episodes = 100
        self.max_steps_per_episode = 400
        self.init_pos = np.array([0.17, 0.12])
        #self.final_position = np.array([2.45, 0])  # Definir a posição final
        self.flag = self.robot.getFromDef('Flag')  # Flag visual do destino
        # Escolha random de flag
        self.posicoes = [ #[0.05,0.05]
            [1.75, 1.38],
            [1.75, 0.37],
            [0.25, 0.75],
            [0.3, 1.50],
        ]
        #self.maps = ['worlds/Project_1.wbt','worlds/Project_2.wbt','worlds/Project_3.wbt',]
        #self.map_choice = random.choice(self.maps)
        #self.robot.loadWorld(self.map_choice)
        self.last_move = None
        self.reward = 0
        self.final_position = np.array(random.choice(self.posicoes)) #Posiçao final
        self.flag.getField('translation').setSFVec3f(list(self.final_position)+[0]) #Colocar flag na posiçao final
        #print(len(list(self.final_position)+[0]))
        self.action_size = 3  # Definir o tamanho do espaço de ações
        self.max_speed = 1
        self.steps_done = 0
        self.distance_before = np.linalg.norm(self.init_pos - self.final_position)
        self.bate=False

        # Definindo o espaço de ação
        self.action_space = spaces.Discrete(self.action_size)

        # Definindo o espaço de observação
        # O estado inclui as coordenadas dos pontos LIDAR e a distância até o objetivo
        lidar_points_count = self.lidar.getNumberOfPoints()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * lidar_points_count + 1,),
                                            dtype=np.float32)
        #self.observation_space = lidar_points_count

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para o estado inicial.

        Retorna:
        - numpy.ndarray: Estado inicial do ambiente.
        """
        #definir posicao final
        #self.map_choice = random.choice(self.maps)
        #self.robot.loadWorld(self.map_choice)
        self.reward = 0
        self.last_move = None
        self.bate = False
        self.final_position = np.array(random.choice(self.posicoes))  # Posiçao final
        self.flag.getField('translation').setSFVec3f(list(self.final_position)+[0])  # Colocar flag na posiçao final
        #print(self.flag.getField('translation'))
        warp_robot(self.robot, "EPUCK", self.init_pos)
        self.steps_done = 0
        info = self.calculate_distance_to_goal()
        n = {"distance": info}
        return self.get_state(),n

    def calculate_distance_to_goal(self):
        """
        Calcula a distância atual até a posição final.

        Retorna:
        - float: Distância até o objetivo.
        """

        current_position = np.array(self.gps.getValues()[:2])
        distance_to_goal = np.linalg.norm(current_position - self.final_position)
        if np.isnan(distance_to_goal):
            distance_to_goal = np.linalg.norm(self.init_pos - self.final_position)
        return distance_to_goal

    def detect_collision(self):
        """
        Detecta colisão usando dados do sensor LIDAR.

        Parâmetros:
        - lidar_point_cloud (array): Array de pontos LIDAR.

        Retorna:
        - bool: True se colisão detectada, False caso contrário.
        """
        if self.touch.getValue() == 1.0:
            return True
        return False


    def detect_obstacle_proximity(self, lidar_point_cloud):
        """
        Detecta a proximidade de obstáculos usando dados do sensor LIDAR.

        Parâmetros:
        - lidar_point_cloud (array): Array de pontos LIDAR.

        Retorna:
        - float: Recompensa negativa se estiver próximo a um obstáculo, 0 caso contrário.
        """
        obstacle_proximity_threshold = 0.05  # Definir um limite para considerar a proximidade de um obstáculo
        min_distance = float('inf')

        for point in lidar_point_cloud:
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2)
            if distance_to_robot < min_distance:
                min_distance = distance_to_robot

        if min_distance < obstacle_proximity_threshold:
         
