import numpy as np
from controller import Robot, Lidar, LidarPoint
from controllers.utils import cmd_vel
class Environment(Supervisor):
    """The robot's environment in Webots."""

    def __init__(self):
        super().__init__()

        # General environment parameters
        self.max_speed = 1  # Maximum Angular speed in rad/s
        self.destination_coordinate = np.array([2.45, 0])  # Target (Goal) position
        self.reach_threshold = 0.06  # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1  # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])

        # Activate Devices
        self.Lidar : Lidar=robot.getDevice("lidar")
        self.lidar.enable(timestep)
        self.lidar.enablePointCloud()

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200)  # take some dummy steps in environment for initialization

        # Create dictionary from all available distance sensors and keep min and max of from total values


    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_distance_to_goal(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.

        Returns:
        - numpy.ndarray: Normalized distance vector.
        """

        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)

        return normalizied_coordinate_vector

    def get_sensor_data(self):
        """
        Retrieves and normalizes data from LiDAR sensors.

        Returns:
        - numpy.ndarray: Normalized LiDAR data.
        """
        lidar_data = self.laser.getRangeImage()
        # Normalize the LiDAR data (optional)
        normalized_lidar_data = lidar_data / self.laser.getMaxRange()
        return normalized_lidar_data

    def get_observations(self):
        """
        Obtains and returns the normalized sensor data and current distance to the goal.

        Returns:
        - numpy.ndarray: State vector representing distance to goal and distance sensors value.
        """

        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalizied_current_coordinate = np.array([self.get_distance_to_goal()], dtype=np.float32)

        state_vector = np.concatenate([normalizied_current_coordinate, normalized_sensor_data], dtype=np.float32)

        return state_vector

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observations.

        Returns:
        - numpy.ndarray: Initial state vector.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations()

    def step(self, action, max_steps):
        """
        Takes a step in the environment based on the given action.

        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """

        self.apply_action(action)
        step_reward, done = self.get_reward()

        state = self.get_observations()  # New state

        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            done = True

        return state, step_reward, done

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.

        Returns:
        - The reward and done flag.
        """

        done = False
        reward = 0

        normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = self.get_distance_to_goal()

        normalized_current_distance *= 100  # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100

        # (1) Reward according to distance
        if normalized_current_distance < 42:
            if normalized_current_distance < 10:
                growth_factor = 5
                A = 2.5
            elif normalized_current_distance < 25:
                growth_factor = 4
                A = 1.5
            elif normalized_current_distance < 37:
                growth_factor = 2.5
                A = 1.2
            else:
                growth_factor = 1.2
                A = 0.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))

        else:
            reward += -normalized_current_distance / 100

        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        if normalized_current_distance < reach_threshold:
            # Reward for finishing the task
            done = True
            reward += 25
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            done = True
            reward -= 5


        # (3) Punish if close to obstacles
        elif np.any(normalized_sensor_data[normalized_sensor_data > self.obstacle_threshold]):
            reward -= 0.001

        return reward, done

    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.

        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        if action == 0:
            linear_vel=self.max_speed
        if action == 1:
            angular_vel = self.max_speed
        if action == 3:
            angular_vel = -self.max_speed
        if action == 4:
            linear_vel = -self.max_speed

        cmd_vel(self.robot,linear_vel,angular_vel)

    def get_point_cloud_data(self):
        """
        Retrieves and normalizes data from Lidar's Point Cloud.

        Returns:
        - numpy.ndarray: Normalized Point Cloud data.
        """
        # Get Point Cloud data
        point_cloud_data = self.lidar.getRangeImage()
        # Normalize the Point Cloud data (if needed)
        # You might need to adjust this normalization based on your specific LiDAR sensor
        normalized_point_cloud_data = point_cloud_data / self.lidar.getMaxRange()
        return normalized_point_cloud_data
class QLearningAgent():
    """Agente Q-learning para planejamento de caminho."""

    def __init__(self, num_episodes, max_steps, learning_rate, gamma, epsilon_decay):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.q_values = {}

    def select_action(self, state, epsilon):
        """Seleciona uma ação usando uma política epsilon-greedy."""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(3)  # Ação aleatória
        else:
            return np.argmax(self.q_values.get(state, np.zeros(3)))  # Ação com maior valor Q

    def update_q_values(self, state, action, reward, next_state):
        """Atualiza os valores Q com base na regra de atualização do Q-learning."""
        # Atualização dos valores Q usando a fórmula do Q-learning
        current_q_value = self.q_values.get(state, np.zeros(3))[action]
        max_next_q_value = np.max(self.q_values.get(next_state, np.zeros(3)))
        new_q_value = current_q_value + self.learning_rate * (reward + self.gamma * max_next_q_value - current_q_value)
        self.q_values[state] = np.array([new_q_value if i == action else val for i, val in enumerate(self.q_values.get(state, np.zeros(3)))])

    def train(self, env):
        """Treina o agente usando o algoritmo Q-learning."""
        epsilon = 1.0  # Valor inicial de epsilon
        for episode in range(1, self.num_episodes + 1):
            state = env.get_state()
            episode_reward = 0
            for step in range(self.max_steps):
                action = self.select_action(state, epsilon)
                next_state, reward, done = env.step(action)
                self.update_q_values(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
                if done:
                    break
            epsilon *= self.epsilon_decay  # Decaimento de epsilon

def main():
    # Parâmetros do agente e ambiente
    num_episodes = 1000
    max_steps = 100
    learning_rate = 0.1
    gamma = 0.9
    epsilon_decay = 0.99

    # Inicialização do ambiente e do agente
    env = Environment()
    agent = QLearningAgent(num_episodes, max_steps, learning_rate, gamma, epsilon_decay)

    # Treinamento do agente
    agent.train(env)

if __name__ == '__main__':
    main()

