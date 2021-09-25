import numpy as np
import utils.projection_params as P
from utils.util import RPY_to_rotMat, plot_xyz


def lidar2cam(XYZ):
    xyz = XYZ.reshape(-1, 3)

    # translation
    xyz = xyz - np.array(P.LIDAR2CAM_TRANSLATION)
    xyz = xyz.T

    # rotation
    rotation_matrix = RPY_to_rotMat(P.LIDAR2CAM_ROTATION_RPY[2], P.LIDAR2CAM_ROTATION_RPY[1],
                                    P.LIDAR2CAM_ROTATION_RPY[0]).T
    xyz = np.dot(rotation_matrix, xyz)

    # only project points with z > 0
    front_idx = xyz[2] > 0
    xyz = xyz[:, front_idx]
    xyz = np.dot(np.array(P.CAMERA_METRIX), xyz)
    for i in range(xyz.shape[1]):
        z = xyz[2, i]
        xyz[0, i] = (xyz[0, i]) / z
        xyz[1, i] = (xyz[1, i]) / z

    # get the camera FOV
    id_x = np.logical_and(xyz[0] >= 0, xyz[0] < P.WIDTH)
    id_y = np.logical_and(xyz[1] >= 0, xyz[1] < P.HEIGHT)
    fov_idx = np.logical_and(id_x, id_y)
    xyz = xyz[:, fov_idx]
    return xyz[:2, :], front_idx, fov_idx
