"""
IRI - TP1 - Ex 1
By: Gonçalo Leão
"""

from controller import Robot, Keyboard,Lidar, LidarPoint
from controllers.utils import cmd_vel
import math

############################# Inicialize robot and sensors #########################
robot : Robot= Robot()
timestep: int = 100

lidar: Lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()


################################ important functions #########################################
def detect_obstacle(lidar_data):
    for measured_dist in lidar.getRangeImage():
        if measured_dist < 0.13:   # distance to object
            return True
    return False


def destination_direction(lidar_data):
    sum_x=0
    sum_y=0
    num_readings=len(lidar_data)

    for

############################### destination coordenates #######################

destination =(5,5)   # Choose better coordenates with other maps
r=0.5                # radius of the target



############################### main loop #########################################



while robot.step(timestep) != -1:
    if detect_obstacle():

        cmd_vel(robot, 0.1, 1)
    else:

        cmd_vel(robot, 1, 0)


"""
kb: Keyboard = Keyboard()
kb.disable()
kb.enable(timestep)

keyboard_linear_vel: float = 0.3
keyboard_angular_vel: float = 1.5

#codigo tp4 ex 2

while robot.step(timestep) != -1:
        key: int = kb.getKey()
        if key == ord('W'):
            cmd_vel(robot, keyboard_linear_vel, 0)
        elif key == ord('S'):
            cmd_vel(robot, -keyboard_linear_vel, 0)
        elif key == ord('A'):
            cmd_vel(robot, 0, keyboard_angular_vel)
        elif key == ord('D'):
            cmd_vel(robot, 0, -keyboard_angular_vel)
"""


