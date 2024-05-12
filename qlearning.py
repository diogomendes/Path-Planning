import math
import numpy as np
from controller import Lidar, GPS, LidarPoint, Robot, Supervisor
from controllers.utils import cmd_vel

#supervisor: Supervisor = Supervisor()


class Environment:
    def __init__(self):
        self.robot : Supervisor = Supervisor()
        #self.timestep = 100
        self.timestep = int(self.robot.getBasicTimeStep())
        self.lidar : Lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.max_episodes = 100
        self.max_steps_per_episode = 1000
        self.final_position = np.array([1.8, 1.8])  # Definir a posição final
        self.num_sectors = 8
        self.state_size = self.num_sectors * 2 + 2  # Definir o tamanho do espaço de estados
        self.action_size = 4  # Definir o tamanho do espaço de ações
        self.max_speed=1
    def reset(self):
        # Resetar o ambiente para o estado inicial
        self.robot.simulationReset()
        self.robot.simulationResetPhysics()
        self.robot.step(self.timestep)

    def calculate_distance_to_goal(self):
        # Calcular a distância atual até a posição final
        current_position = np.array(self.gps.getValues()[:2])
        distance_to_goal = np.linalg.norm(current_position - self.final_position)
        return distance_to_goal

    def get_reward(self):
        # Calcular a recompensa com base na distância até o objetivo e outras condições
        distance_to_goal = self.calculate_distance_to_goal()

        # Definir as recompensas conforme necessário
        if distance_to_goal < 0.06:  # Se a posição final foi alcançada
            reward = 25
        elif distance_to_goal < 42:  # Se estiver perto do objetivo
            reward = 2.5 * (1 - np.exp(-1 / distance_to_goal))
        else:  # Outras condições (ajuste conforme necessário)
            reward = -distance_to_goal / 100

        return reward

    def apply_action(self, action):
        # Aplicar a ação ao robô
        linear_vel = 0
        angular_vel = 0

        if action == 0:  # Move forward
            linear_vel = self.max_speed
        elif action == 1:  # Move backward
            linear_vel = -self.max_speed
        elif action == 2:  # Turn left
            angular_vel = self.max_speed
        elif action == 3:  # Turn right
            angular_vel = -self.max_speed

        cmd_vel(self.robot, linear_vel, angular_vel)

    def step(self, action):
        # Aplica a ação ao ambiente
        self.apply_action(action)

        # Obtém o novo estado após aplicar a ação
        lidar_points = self.lidar.getPointCloud()
        state = self.get_state_features(lidar_points)

        # Retorna o estado após aplicar a ação
        return state

    def get_state_features(self, lidar_points: [LidarPoint]):
        sector_width = 2 * math.pi / self.num_sectors
        distances = [float('inf')] * self.num_sectors
        num_obstacles = [0] * self.num_sectors

        for point in lidar_points:
            angle = math.atan2(point.y, point.x)
            sector_index = int((angle + math.pi) / sector_width)
            distance = math.sqrt(point.x ** 2 + point.y ** 2)
            distances[sector_index] = min(distances[sector_index], distance)
            num_obstacles[sector_index] += 1

        min_distance = min(distances)
        num_obstacles_total = sum(num_obstacles)

        return np.concatenate((distances, num_obstacles, [min_distance, num_obstacles_total]))


class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename)


def main():
    env = Environment()
    agent = QLearningAgent(env.state_size, env.action_size)  # Usar o tamanho correto do espaço de estados e ações

    # Treinamento do agente
    for episode in range(env.max_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            state = env.step(action)
            reward = env.get_reward()
            total_reward += reward

            next_state = state
            agent.update_q_table(state, action, reward, next_state)

            if env.calculate_distance_to_goal() < 0.06 or env.max_steps_per_episode < 0:
                done = True

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Salvar a tabela Q após o treinamento
    agent.save_q_table('q_table.npy')

if __name__ == "__main__":
    main()
