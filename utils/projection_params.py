import numpy as np


LIDAR2CAM_TRANSLATION = [-0.180, -0.009, -0.157]
LIDAR2CAM_ROTATION = [0.611, -0.503, 0.466, 0.395]
LIDAR2CAM_ROTATION_RPY = [3.084, -1.314, -1.334]
CAMERA_METRIX = [[1953.615879, 0.000000, 2007.041296],
                  [0.000000, 1952.694009, 1489.736671],
                  [0.000000, 0.000000, 1.000000]]
DISTORTION_COEFF = (-0.032929, 0.005035, -0.001666, -0.000796, 0.000000)
WIDTH = 3440
HEIGHT = 2880
DOWNSAMPLE_FACTOR = 0.4

lidar_to_sensor = np.array([[-1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, 1, 0.036180],
                             [0, 0, 0, 1]])
sensor_to_lidar = np.linalg.inv(lidar_to_sensor)
