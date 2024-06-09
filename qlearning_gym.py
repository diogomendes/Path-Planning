import numpy as np
import random
from gym_environment import Environment

class QLearningAgent:
    def __init__(self, num_actions, state_limits, num_bins, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.num_actions = num_actions
        self.state_limits = state_limits
        self.num_bins = num_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(num_bins + [num_actions])
        self.bins = [np.linspace(lim[0], lim[1], n+1)[:-1] for lim, n in zip(state_limits, num_bins)]

    def discretize(self, state):
        """ Discretiza o estado contínuo para os índices de estado na tabela Q. """
        state_idx = tuple(
            int(np.digitize(val, self.bins[i]) - 1)  # Subtrai 1 para fazer o índice base-0
            for i, val in enumerate(state)
        )
        # Ajustar índices para garantir que eles estejam dentro dos limites
        state_idx = tuple(max(0, min(idx, len(self.bins[i]) - 1)) for i, idx in enumerate(state_idx))
        return state_idx

    def choose_action(self, state):
        """ Escolhe uma ação usando a política epsilon-greedy. """
        state_idx = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state):
        """ Atualiza a tabela Q com base na experiência. """
        current_state_idx = self.discretize(state)
        next_state_idx = self.discretize(next_state)
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action]
        td_error = td_target - self.q_table[current_state_idx][action]
        self.q_table[current_state_idx][action] += self.alpha * td_error

# Inicialize o ambiente e o agente
env = Environment()
num_actions = 3
state_limits = [(0, 5), (0, 5), (0, 5)]  # Assumindo que todas as características estão em uma escala similar
num_bins = [10, 10, 10]

agent = QLearningAgent(num_actions, state_limits, num_bins)

num_episodes = 1000
max_steps_per_episode = 400

for episode in range(num_episodes):
    raw_state = env.reset()
    lidar_data = raw_state[0]
    state_info = raw_state[1]['distance']

    # Processar o estado inicial
    min_lidar = np.min(lidar_data)
    mean_lidar = np.mean(lidar_data)
    goal_distance = state_info
    state = (min_lidar, mean_lidar, goal_distance)

    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        raw_next_state, reward, done, _, info = env.step(action)

        next_lidar_data = raw_next_state
        next_goal_distance = info['distance']

        # Processar o próximo estado
        next_min_lidar = np.min(next_lidar_data)
        next_mean_lidar = np.mean(next_lidar_data)
        next_state = (next_min_lidar, next_mean_lidar, next_goal_distance)

        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}: Total reward = {total_reward}")

