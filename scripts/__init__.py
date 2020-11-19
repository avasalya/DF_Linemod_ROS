""" DenseFusion modules """
import os
import sys
import cv2
import copy
import time
import getpass
import itertools
import math as m
import numpy as np
import open3d as o3d
import random as rand
import numpy.ma as ma
from colorama import Fore, Style

import PIL.Image as pImage

import rospy
import std_msgs
import message_filters

import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image, CompressedImage

import geometry_msgs.msg as gm
from geometry_msgs.msg import Pose, PoseArray

import chainer
import chainer.utils as utils
import chainer_mask_rcnn as cmr

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix, rotation_matrix, concatenate_matrices, is_same_transform, is_same_quaternion, rotation_from_matrix

from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver


""" yolact modules """
# import roslib
# roslib.load_manifest('yolact_ros')

import pickle
import argparse
import cProfile
import threading
import pycocotools
import matplotlib.pyplot as plt

# from PIL import Image
# from queue import Queue
from colorama import Fore, Style
from collections import defaultdict, OrderedDict

import rospkg
from rospy.numpy_msg import numpy_msg

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage

import torch.backends.cudnn as cudnn

sys.path.append(os.path.join(os.path.dirname(__file__), "yolact"))
from yolact import Yolact

from layers.box_utils import jaccard, center_size, mask_iou
from layers.output_utils import postprocess, undo_image_transformation

from utils import timer
from utils.functions import SavePath
from utils.functions import MovingAverage, ProgressBar
from utils.augmentations import BaseTransform, FastBaseTransform, Resize

from data import COCODetection, get_label_map, MEANS, COLORS
from data import cfg, set_cfg, set_dataset