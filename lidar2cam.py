import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


_LIDAR2CAM_TRANSLATION = [-0.180, -0.009, -0.157]
_LIDAR2CAM_ROTATION = [0.611, -0.503, 0.466, 0.395]
_LIDAR2CAM_ROTATION_RPY = [3.084, -1.314, -1.334]
# _LIDAR2CAM_ROTATION_RPY = [3.084, -1.364, -1.32]
_CAMERA_METRIX = [[1953.615879, 0.000000, 2007.041296],
                  [0.000000, 1952.694009, 1489.736671],
                  [0.000000, 0.000000, 1.000000]]
_DISTORTION_COEFF = (-0.032929, 0.005035, -0.001666, -0.000796, 0.000000)
_WIDTH = 3440
_HEIGHT = 2880

# _lidar_to_sensor = np.array([[-1, 0, 0, 0], 
#                              [0, -1, 0, 0], 
#                              [0, 0, 1, 0.036180], 
#                              [0, 0, 0, 1]])
# _sensor_to_lidar = np.linalg.inv(_lidar_to_sensor)


def RPY_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat


def plot(x, y, z):
    ax = plt.axes(projection='3d')
    r = 5
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r/2, r/2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=60)
    z_col = np.minimum(np.absolute(z), 5)
    ax.scatter(x, y, z, c=z_col, s=0.1)
    # plt.axis('off')
    plt.show()


def lidar2cam(XYZ):
    xyz = XYZ.reshape(-1, 3).T
    
    # sensor to lidar
    # xyz = np.dot(_sensor_to_lidar, xyz)
    # xyz[2, :] -= 0.036180
    # plot(xyz[0], xyz[1], xyz[2])
    
    # translation
    xyz = xyz.T - np.array(_LIDAR2CAM_TRANSLATION)
    xyz = xyz.T
    
    # rotation
    rotation_matrix = RPY_to_rotMat(_LIDAR2CAM_ROTATION_RPY[2], _LIDAR2CAM_ROTATION_RPY[1], _LIDAR2CAM_ROTATION_RPY[0]).T
    xyz = np.dot(rotation_matrix, xyz)  
    # plot(xyz[0], xyz[1], xyz[2])
    
    xyz = xyz[:,xyz[2] > 0]
    xyz = np.matmul(np.array(_CAMERA_METRIX), xyz)

    for i in range(xyz.shape[1]):
        z = xyz[2, i]
        xyz[0,i] = (xyz[0,i]) / z
        xyz[1,i] = (xyz[1,i]) / z    
    
    id_x = np.logical_and(xyz[0]>=0, xyz[0]<_WIDTH)
    id_y = np.logical_and(xyz[1]>=0, xyz[1]<_HEIGHT)
    idx = np.logical_and(id_x, id_y)
    xyz = xyz[:, idx]
    plt.figure()
    plt.scatter(xyz[0], -xyz[1]+_HEIGHT, s=0.1, c='r')
    # plt.show()
    plt.axis('off')
    return xyz
    

if __name__ == '__main__':
    for i in range(22, 1754):
        XYZ = np.load('frames27/{}.npy'.format(i))
        result = lidar2cam(XYZ)
        plt.savefig('pt_im/{}.png'.format(i))
        plt.close()

    # image = np.array(Image.open('cam27/image0.png').convert('RGB'))
    # image = np.rot90(image, k=3)
    # plt.imshow(image)
    # plt.scatter(result[0]*0.4, result[1]*0.4, s=0.08, c='r')
    # plt.show()
    
