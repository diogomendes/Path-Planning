import numpy as np
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel

class Environment:
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
        self.final_position = np.array([1.8, 1.8])  # Definir a posição final
        self.state_size = self.lidar.getHorizontalResolution() * self.lidar.getVerticalResolution() + 1
        self.action_size = 4  # Definir o tamanho do espaço de ações
        self.max_speed = 1

    def reset(self):
        """
        Reseta o ambiente para o estado inicial.

        Retorna:
        - None
        """
        self.robot.simulationReset()
        self.robot.simulationResetPhysics()
        self.robot.step(self.timestep)

    def calculate_distance_to_goal(self):
        """
        Calcula a distância atual até a posição final.

        Retorna:
        - float: Distância até o objetivo.
        """
        current_position = np.array(self.gps.getValues()[:2])
        distance_to_goal = np.linalg.norm(current_position - self.final_position)
        return distance_to_goal

    def detect_collision(self, lidar_point_cloud):
        """
        Detecta colisão usando dados do sensor LIDAR.

        Parâmetros:
        - lidar_point_cloud (array): Array de pontos LIDAR.

        Retorna:
        - bool: True se colisão detectada, False caso contrário.
        """
        for point in lidar_point_cloud:
            if np.any(np.isinf([point.x, point.y, point.z])):
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
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            if distance_to_robot < min_distance:
                min_distance = distance_to_robot

        if min_distance < obstacle_proximity_threshold:
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
        state = [point.x for point in lidar_point_cloud] + [point.y for point in lidar_point_cloud] + [point.z for point in lidar_point_cloud] + [distance_to_goal]
        return np.array(state)

    def get_reward(self):
        """
        Calcula a recompensa com base na distância até o objetivo e detecção de colisão.

        Retorna:
        - float: Recompensa atual.
        """
        distance_to_goal = self.calculate_distance_to_goal()
        collision = self.detect_collision(self.lidar.getPointCloud())
        obstacle_proximity_reward = self.detect_obstacle_proximity(self.lidar.getPointCloud())

        if distance_to_goal < 0.06:
            reward = 25
        elif distance_to_goal < 42:
            reward = 2.5 * (1 - np.exp(-1 / distance_to_goal))
        elif collision:
            reward = -100
        elif obstacle_proximity_reward < 0:
            reward = obstacle_proximity_reward
        else:
            reward = -distance_to_goal / 100

        return reward

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
        self.apply_action(action)
        lidar_points = self.lidar.getPointCloud()
        state = self.get_state(lidar_points)
        return state

