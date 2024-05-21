import gymnasium as gym
from gym import spaces
import numpy as np
import numpy as np
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel, warp_robot


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
        self.max_episodes = 100
        self.max_steps_per_episode = 1000
        self.init_pos = np.array([0.17, 0.12])
        self.final_position = np.array([2.45, 0])  # Definir a posição final
        self.action_size = 3  # Definir o tamanho do espaço de ações
        self.max_speed = 1.5

        # Definindo o espaço de ação
        self.action_space = spaces.Discrete(self.action_size)

        # Definindo o espaço de observação
        # O estado inclui as coordenadas dos pontos LIDAR e a distância até o objetivo
        lidar_points_count = self.lidar.getNumberOfPoints()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * lidar_points_count + 1,),
                                            dtype=np.float32)
        #self.observation_space = lidar_points_count

    def reset(self):
        """
        Reinicia o ambiente para o estado inicial.

        Retorna:
        - numpy.ndarray: Estado inicial do ambiente.
        """
        warp_robot(self.robot, "EPUCK", self.init_pos)
        return self.get_state()

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
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2)
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
        obstacle_proximity_threshold = 0.03  # Definir um limite para considerar a proximidade de um obstáculo
        min_distance = float('inf')

        for point in lidar_point_cloud:
            distance_to_robot = np.sqrt(point.x ** 2 + point.y ** 2)
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
        state = [point.x for point in lidar_point_cloud] + [point.y for point in lidar_point_cloud] + [distance_to_goal]
        return np.array(state, dtype=np.float32)

    def get_reward(self):
        """
        Calcula a recompensa com base na distância até o objetivo e detecção de colisão.

        Retorna:
        - float: Recompensa atual.
        """
        distance_to_goal = self.calculate_distance_to_goal()
        collision = self.detect_collision(self.lidar.getPointCloud())
        obstacle_proximity_reward = self.detect_obstacle_proximity(self.lidar.getPointCloud())

        if collision:
            return -500

        if obstacle_proximity_reward < 0:
            return obstacle_proximity_reward

        if distance_to_goal < 0.06:
            return 300
        elif distance_to_goal < 0.8:
            return 20
        elif distance_to_goal < 1.8:
            return 10
        else:
            return 1

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
        - tuple: Novo estado após a aplicação da ação, recompensa, se está finalizado e informações adicionais.
        """
        self.apply_action(action)  # Aplica a ação ao ambiente
        self.robot.step(self.timestep)  # Avança a simulação para atualizar o estado do LiDAR
        state = self.get_state()  # Obtém o novo estado com base nos dados atualizados do LiDAR
        reward = self.get_reward()  # Calcula a recompensa com base no novo estado
        done = reward == -500 or self.calculate_distance_to_goal() < 0.06
        info = {}  # Pode ser usado para informações adicionais

        return state, reward, done, info

    def render(self, mode='human'):
        """
        Renderiza o ambiente (opcional).

        Parâmetros:
        - mode (str): Modo de renderização (não usado).

        Retorna:
        - None
        """
        pass  # Por enquanto, não fazemos nada aqui. No futuro, poderíamos adicionar código para visualizar o ambiente.

    def close(self):
        """
        Fecha o ambiente e libera recursos (opcional).

        Retorna:
        - None
        """
        self.lidar.disable()
        self.gps.disable()
        pass  # Aqui você pode adicionar qualquer outra limpeza necessária.
