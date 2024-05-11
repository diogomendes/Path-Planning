import math
import numpy as np
from controller import Robot, Lidar, Compass, GPS, LidarPoint
from controllers.TP4.occupancy_grid import OccupancyGrid
from controllers.transformations import create_tf_matrix, get_translation
from controllers.utils import cmd_vel, bresenham
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error


class DeterministicOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        super().__init__(origin, dimensions, resolution)
        self.occupancy_grid: np.ndarray = np.full(dimensions, 0.5, dtype=np.float32)

    def update_map(self, robot_tf: np.ndarray, lidar_points: [LidarPoint]) -> None:
        robot_coord: (int, int) = self.real_to_grid_coords(get_translation(robot_tf)[0:2])

        for point in lidar_points:
            coord: (int, int) = self.real_to_grid_coords(np.dot(robot_tf, [point.x, point.y, point.z, 1.0])[0:2])
            self.update_cell(coord, True)

    def update_cell(self, coords: (int, int), is_occupied: bool) -> None:
        if self.are_grid_coords_in_bounds(coords):
            self.occupancy_grid[coords] = 1 if is_occupied else 0


def extract_features(lidar_points: [LidarPoint]):
    num_sectors = 8
    sector_width = 2 * math.pi / num_sectors
    distances = [float('inf')] * num_sectors
    num_obstacles = [0] * num_sectors

    for point in lidar_points:
        angle = math.atan2(point.y, point.x)
        sector_index = int((angle + math.pi) / sector_width)
        distance = math.sqrt(point.x ** 2 + point.y ** 2)
        distances[sector_index] = min(distances[sector_index], distance)
        num_obstacles[sector_index] += 1

    min_distance = min(distances)
    num_obstacles_total = sum(num_obstacles)

    return distances, num_obstacles, min_distance, num_obstacles_total


def calculate_sector_distances(lidar_points):
    num_sectors = 8  # Número de setores ao redor do robô
    sector_angles = np.linspace(0, 2 * np.pi, num_sectors + 1)[:-1]  # Ângulos dos setores
    sector_distances = np.zeros(num_sectors)  # Distâncias médias aos obstáculos em cada setor

    for point in lidar_points:
        # Calcular o ângulo do ponto em relação ao robô
        point_angle = math.atan2(point.y, point.x)
        if point_angle < 0:
            point_angle += 2 * np.pi

        # Determinar o setor ao qual o ponto pertence
        sector_index = int(np.digitize(point_angle, sector_angles)) - 1

        # Calcular a distância do ponto ao robô
        distance = math.sqrt(point.x ** 2 + point.y ** 2)

        # Atualizar a distância média no setor correspondente
        sector_distances[sector_index] += distance

    # Calcular a distância média em cada setor
    sector_distances /= len(lidar_points)

    return sector_distances


def main():
    robot: Robot = Robot()
    timestep: int = 100
    map: DeterministicOccupancyGrid = DeterministicOccupancyGrid([0.0, 0.0], [200, 200], 0.01)
    lidar: Lidar = robot.getDevice('Lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    compass: Compass = robot.getDevice('Compass')
    compass.enable(timestep)
    gps: GPS = robot.getDevice('GPS')
    gps.enable(timestep)
    max_episodes = 100
    max_steps_per_episode = 1000
    final_position: (float, float) = (1.8, 1.8)

    # Define state and action space
    num_sectors = 8
    state_size = num_sectors * 2 + 2  # distances + num_obstacles + min_distance + num_obstacles_total
    action_size = 4  # Define the size based on your action representation
    linear_vel = 0.1  # Define linear velocity
    angular_vel = 1  # Define angular velocity

    # Initialize Q-learning agent
    agent = QLearningAgent(state_size, action_size)

    for episode in range(max_episodes):
        scan_count: int = 0
        while robot.step(timestep) != -1:
            # Read the LiDAR and update the map
            gps_readings: [float] = gps.getValues()
            robot_position: (float, float) = (gps_readings[0], gps_readings[1])
            compass_readings: [float] = compass.getValues()
            robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])
            robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)
            map.update_map(robot_tf, lidar.getPointCloud())

            # Extract features from LiDAR readings
            lidar_points = lidar.getPointCloud()
            distances, num_obstacles, min_distance, num_obstacles_total = extract_features(lidar_points)
            sector_distances = calculate_sector_distances(lidar_points)

            # Construct state
            state = np.concatenate((distances, num_obstacles, [min_distance, num_obstacles_total]))

            # Choose action based on Q-values
            action = agent.choose_action(state)

            # Execute action
            if action == 0:  # Move forward
                cmd_vel(robot, linear_vel, 0)
            elif action == 1:  # Move backward
                cmd_vel(robot, -linear_vel, 0)
            elif action == 2:  # Turn left
                cmd_vel(robot, 0, angular_vel)
            elif action == 3:  # Turn right
                cmd_vel(robot, 0, -angular_vel)
            else:  # No action
                cmd_vel(robot, 0, 0)

            # Recompensa por alcançar o objetivo
            goal_reached_reward = 100 if goal_reached else 0
            # Recompensa por distância ao objetivo
            distance_to_goal_reward = previous_distance_to_goal - current_distance_to_goal
            # Recompensa por evitar obstáculos
            avoid_obstacle_reward = -10 if collision_detected else 0

            # Definir a recompensa total como a soma das três recompensas
            reward = goal_reached_reward + distance_to_goal_reward + avoid_obstacle_reward

            # Update Q-table
            next_state = sector_distances  # Define next state based on LiDAR readings after executing action
            agent.update_q_table(state, action, reward, next_state)

            # Save map
            scan_count += 1
            plt.imshow(np.flip(map.occupancy_grid, 0))
            plt.savefig('ex1-episode' + str(episode) + '-scan' + str(scan_count) + '.png')

            if scan_count >= max_steps_per_episode:
                break


if __name__ == '__main__':
    main()
