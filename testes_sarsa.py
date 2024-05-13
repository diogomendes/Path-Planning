import numpy as np
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel,warp_robot
import random

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
        self.init_pos=np.array([0.17, 0.12])
        self.final_position = np.array([2.45, 0])  # Definir a posição final
        self.action_size = 3  # Definir o tamanho do espaço de ações
        self.max_speed = 1.5

    def reset(self):

        #print("initial position", self.init_pos)
        warp_robot(self.robot, "EPUCK", self.init_pos)




    def calculate_distance_to_goal(self):
        """
        Calcula a distância atual até a posição final.

        Retorna:
        - float: Distância até o objetivo.
        """
        current_position = np.array(self.gps.getValues()[:2])
        #print("current position", current_position)
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
        #print("min_distance", min_distance)
        if min_distance < obstacle_proximity_threshold:
            #print("entrou")
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
        #print("Collision detected: ", collision)
        obstacle_proximity_reward = self.detect_obstacle_proximity(self.lidar.getPointCloud())
        #print("Obstacle proximity", obstacle_proximity_reward)

        if collision:
            return -500

        if obstacle_proximity_reward < 0:
            return obstacle_proximity_reward

        if distance_to_goal < 0.06:
            return 150
        elif distance_to_goal < 5:
            return 1.5 * (1 - np.exp(-1 / distance_to_goal))
        elif distance_to_goal <15:
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
        #self.robot.step(self.timestep)  # Avança a simulação para atualizar o estado do LiDAR
        state = self.get_state()  # Obtém o novo estado com base nos dados atualizados do LiDAR
        self.robot.step(self.timestep)
        return state


import pickle


class SarsaAgent:
    def __init__(self, action_size, learning_rate=0.1, gamma=0.99, epsilon=0.1, q_table_file=None):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        if q_table_file:
            self.q_table = self.load_q_table(q_table_file)
        else:
            self.q_table = {}

    def choose_action(self, state):
        state_key = tuple(state)  # Convertendo o estado numpy em uma tupla
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)  # Escolhe uma ação aleatória
        else:
            return np.argmax(
                self.q_table.get(state_key, np.zeros(self.action_size)))  # Escolhe a melhor ação baseada nos valores Q

    def update_q_table(self, state, action, reward, next_state,next_action):
        state_key = tuple(state)  # Convertendo o estado numpy em uma tupla
        next_state_key = tuple(next_state)  # Convertendo o próximo estado numpy em uma tupla
        current_q = self.q_table.get(state_key, np.zeros(self.action_size))
        next_q = self.q_table.get(next_state_key, np.zeros(self.action_size))
        td_target = reward + self.gamma * next_q[next_action]
        td_error = td_target - current_q[action]
        current_q[action] += self.learning_rate * td_error
        self.q_table[state_key] = current_q

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


# Criar uma instância do ambiente
env = Environment()

# Criar um agente Q-learning
agent = SarsaAgent(action_size=env.action_size)

# Treinar o agente ou carregar treinamento existente
train_new_agent = True
q_table_file = 'sarsa_table.pkl'
if not train_new_agent:
    agent.q_table = agent.load_q_table(q_table_file)

for episode in range(env.max_episodes):
    state = env.reset()
    total_reward = 0
    for step in range(env.max_steps_per_episode):
        # Obter o estado inicial após reset
        if step == 0:
            state = env.get_state()
        action = agent.choose_action(state)
        next_state = env.step(action)
        reward = env.get_reward()
        next_action=agent.choose_action(next_state)
        agent.update_q_table(state, action, reward, next_state,next_action)
        state = next_state
        total_reward += reward
        if env.calculate_distance_to_goal() < 0.06:  # Se o robô alcançar o objetivo, interrompe o episódio
            break
        print("total reward", total_reward)
        if total_reward < -300:  # Se a recompensa for muito negativa, interrompe o episódio
            break
    print("episode", episode)

# Salvar a tabela Q após o treinamento
agent.save_q_table(q_table_file)

# Testar o agente treinado
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state = env.step(action)
    # Implementar condição de parada (e.g., quando o objetivo é alcançado)
