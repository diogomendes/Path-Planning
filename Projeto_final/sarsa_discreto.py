import numpy as np
import random
import pickle
import os
import pandas as pd
from environment import Environment

class SARSAAgent:
    def __init__(self, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bins = np.linspace(0, 3, 5)  # Cria 5 bins de 0 a 3
        self.q_table = np.zeros((4,) * 7 + (num_actions,))  # 7 dimensões de estado, 1 de ação

    def discretize(self, state):
        # Assumindo que `state` tem um número adequado de características de acordo com o espaço de observação
        if len(state) != 7:
            raise ValueError("State should have exactly 7 dimensions based on the lidar input specified.")
        state_idx = [np.digitize(state[i], self.bins) - 1 for i in
                     range(len(state))]  # Ajuste o range conforme necessário
        state_idx = tuple(max(0, min(x, len(self.bins) - 2)) for x in state_idx)  # Evita índices fora dos limites
        return state_idx

    def choose_action(self, state):
        state_idx = self.discretize(state)  # Deve gerar apenas índices para estados
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            try:
                return np.argmax(self.q_table[state_idx])  # Apenas usa índices de estado
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"State index provided: {state_idx}")
                print(f"Q-table shape: {self.q_table.shape}")
                raise

    def update_q_table(self, state, action, reward, next_state, next_action):
        current_state_idx = self.discretize(state)
        next_state_idx = self.discretize(next_state)
        td_target = reward + self.gamma * self.q_table[next_state_idx + (next_action,)]
        td_error = td_target - self.q_table[current_state_idx + (action,)]
        self.q_table[current_state_idx + (action,)] += self.alpha * td_error
        return td_error

def process_lidar_points(lidar_data):
    # Supondo que lidar_data inclui todos os dados do LIDAR e a distância até o objetivo no final
    # Removendo a distância do objetivo para simplificar o exemplo
    lidar_points = lidar_data[:-1]  # Remove a última entrada (distância ao objetivo)
    num_groups = 7  # Número de dimensões desejadas
    points_per_group = len(lidar_points) // num_groups
    # Calcula a média de cada grupo para reduzir dimensões
    reduced_lidar_points = [np.mean(lidar_points[i:i + points_per_group]) for i in range(0, len(lidar_points), points_per_group)]
    return reduced_lidar_points

def save_model(agent, episodes, reward, loss):
    name_model = 'Sarsa' + str(episodes) + '.pkl'
    name_lr = os.path.join('Sarsa_lr.csv')
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
    agent = SARSAAgent(num_actions)

    num_episodes = 10000  # Total de episódios
    for e in range(num_episodes):
        raw_state, info = env.reset()
        state = process_lidar_points(raw_state)  # Reduz as dimensões dos dados do LIDAR
        state = agent.discretize(state)

        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_raw_state, reward, done, _, info = env.step(action)
            next_state = process_lidar_points(next_raw_state)
            next_state = agent.discretize(next_state)

            next_action = agent.choose_action(next_state)
            error = agent.update_q_table(state, action, reward, next_state, next_action)

            state = next_state
            total_reward += reward

        if e % 250 == 0:
            print(f"Episode {e}: Total reward = {total_reward}")
            save_model(agent, e, total_reward, error)




