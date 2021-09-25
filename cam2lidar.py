import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import utils.projection_params as P
from utils.util import assign_label, plot_xyz
from lidar2cam import lidar2cam
from utils.coco_constants import get_coco_colormap


def get_pt_labels(point_xy, mask):
    f = P.DOWNSAMPLE_FACTOR
    assert mask.shape == (P.HEIGHT, P.WIDTH) or mask.shape == (f*P.HEIGHT, f*P.WIDTH)

    labels = assign_label(point_xy[0,:], point_xy[1,:], mask)
    return labels


def get_labels_colormap(labels_fov):
    coco_colormap = get_coco_colormap()
    labels_fov_colormap = coco_colormap[labels_fov]
    return labels_fov_colormap


