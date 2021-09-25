import numpy as np
import matplotlib.pyplot as plt


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


def plot_xyz(x, y, z, c=None):
    ax = plt.axes(projection='3d')
    r = 6
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r/2, r/3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(azim=100, elev=35)

    if c is None:
        color_map = np.minimum(np.absolute(z), 6)
    else:
        color_map = c
    ax.scatter(x, y, z, c=color_map, s=0.1)
    plt.axis('off')
    plt.show()


def assign_label(point_x, point_y, mask):
    assert np.max(point_x) <= mask.shape[1] and np.min(point_x) >= 0
    assert np.max(point_y) <= mask.shape[0] and np.min(point_y) >= 0

    point_x = point_x.reshape(-1,)
    point_y = point_y.reshape(-1,)
    assert point_x.shape[0] == point_y.shape[0]

    labels = np.zeros(point_x.shape)
    for i in range(point_x.shape[0]):
        pixel_x, pixel_y = int(np.floor(point_x[i])), int(np.floor(point_y[i]))
        labels[i] = mask[pixel_y, pixel_x]

    return labels
