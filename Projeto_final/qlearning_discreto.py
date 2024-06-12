import numpy as np
import random
import pickle
import os
import pandas as pd
from environment import Environment

class QLearningAgent:
    def __init__(self, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bins = np.linspace(0, 3, 5)  # Cria 5 bins de 0 a 3
        self.q_table = np.zeros((4,) * 7 + (num_actions,))  # 7 dimensões de estado, 1 de ação

    def discretize(self, state):
        if len(state) != 7:
            raise ValueError("State should have exactly 7 dimensions.")
        state_idx = [np.digitize(state[i], self.bins) - 1 for i in range(len(state))]
        state_idx = tuple(max(0, min(x, len(self.bins) - 2)) for x in state_idx)
        return state_idx

    def choose_action(self, state):
        state_idx = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state):
        current_state_idx = self.discretize(state)
        next_state_idx = self.discretize(next_state)
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx + (best_next_action,)]
        td_error = td_target - self.q_table[current_state_idx + (action,)]
        self.q_table[current_state_idx + (action,)] += self.alpha * td_error
        return td_error

def process_lidar_points(lidar_data):
    # Verifica se lidar_data tem 43 elementos (21 pontos com x e y mais a distância ao obstáculo)
    if len(lidar_data) != 43:
        raise ValueError(
            "Expected 43 elements in lidar_data: 42 for points (x and y for each) plus one for the distance to the obstacle.")

    num_groups = 3  # Quer reduzir para 3 pontos
    points_per_group = 14  # 7 pontos por grupo, cada ponto com x e y (7 * 2 = 14 elementos)
    reduced_lidar_points = []

    # Processa cada grupo para calcular a média dos componentes x e y
    for i in range(0, 42, points_per_group):  # Usa 42 para evitar incluir a distância no processamento de pontos
        group_x = lidar_data[i:i + points_per_group:2]  # Pega os elementos x do grupo
        group_y = lidar_data[i + 1:i + points_per_group:2]  # Pega os elementos y do grupo

        # Calcula a média de x e de y
        avg_x = np.mean(group_x)
        avg_y = np.mean(group_y)

        # Adiciona as médias à lista reduzida
        reduced_lidar_points.extend([avg_x, avg_y])

    # Adiciona a distância ao obstáculo ao final da lista reduzida
    distance_to_obstacle = lidar_data[-1]
    reduced_lidar_points.append(distance_to_obstacle)

    return reduced_lidar_points


def save_model(agent, episodes, reward, loss):
    name_model = 'Qlearning' + str(episodes) + '.pkl'
    name_lr = os.path.join('Qlearning_lr.csv')
    if not os.path.isfile(name_lr):
        df = pd.DataFrame(columns=['Episodes', 'Loss', 'Reward'])
        df.to_csv(name_lr, index=False)
    df = pd.read_csv(name_lr)
    df.loc[len(df)] = [episodes, reward, loss]
    df.to_csv(name_lr, index=False)

    with open(name_model, 'wb') as f:
        pickle.dump(agent.q_table, f)


if __name__ == "__main__":
    env = Environment()
    num_actions = 3
    agent = QLearningAgent(num_actions)
    re = 0  # total reward para gráficos/análise
    loss = 0  # total loss para gráficos/análise

    num_episodes = 40000
    for e in range(num_episodes):
        raw_state, info = env.reset()
        state = process_lidar_points(raw_state)
        state = agent.discretize(state)

        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_raw_state, reward, done, _, info = env.step(action)
            next_state = process_lidar_points(next_raw_state)
            next_state = agent.discretize(next_state)

            error = agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        re += total_reward
        if e % 5000 == 0:
            print(f"Episode {e}: Total reward = {total_reward}")
            save_model(agent, e, re / 5000, loss / 5000)
            re = 0
            loss = 0  # Reset após salvar para que o monitoramento seja por blocos de 250 episódios
