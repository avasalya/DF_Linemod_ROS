import os
import math as m
import cv2 as cv2
import numpy as np
import random as rand

import rospy
import std_msgs

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import Image, CompressedImage

import geometry_msgs.msg as gm
from geometry_msgs.msg import Pose, PoseArray

from lib.transformations import quaternion_matrix, rotation_matrix, concatenate_matrices

from PIL import Image
from PIL import ImageDraw
from colorama import Fore, Style

def draw_axis(img, R, t, K):
    # How+to+draw+3D+Coordinate+Axes+with+OpenCV+for+face+pose+estimation%3f
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, .1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_pointCloud(img, imgpts, color):
    for point in imgpts:
        img=cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color, -1)
    return img

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

def cv2pil(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = Image.fromarray(new_image)
    return new_image

""" adapted from DOPE """
def DrawLine(g_draw, point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    if not point1 is None and point2 is not None:
        g_draw.line([point1, point2], fill=lineColor, width=lineWidth)

def DrawDot(g_draw, point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        g_draw.ellipse(xy,
                    fill=pointColor,
                    outline=pointColor
                    )

def draw_cube(tar, img, g_draw, color, cam_fx, cam_fy, cam_cx, cam_cy):
    # pinhole camera model/ project the cubeoid on the image plane using camera intrinsics
    # u = fx * x/z + cx
    # v = fy * y/z + cy
    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
    p0 = (int((tar[0][0]/ tar[0][2])*cam_fx + cam_cx),  int((tar[0][1]/ tar[0][2])*cam_fy + cam_cy))
    p1 = (int((tar[1][0]/ tar[1][2])*cam_fx + cam_cx),  int((tar[1][1]/ tar[1][2])*cam_fy + cam_cy))
    p2 = (int((tar[2][0]/ tar[2][2])*cam_fx + cam_cx),  int((tar[2][1]/ tar[2][2])*cam_fy + cam_cy))
    p3 = (int((tar[3][0]/ tar[3][2])*cam_fx + cam_cx),  int((tar[3][1]/ tar[3][2])*cam_fy + cam_cy))
    p4 = (int((tar[4][0]/ tar[4][2])*cam_fx + cam_cx),  int((tar[4][1]/ tar[4][2])*cam_fy + cam_cy))
    p5 = (int((tar[5][0]/ tar[5][2])*cam_fx + cam_cx),  int((tar[5][1]/ tar[5][2])*cam_fy + cam_cy))
    p6 = (int((tar[6][0]/ tar[6][2])*cam_fx + cam_cx),  int((tar[6][1]/ tar[6][2])*cam_fy + cam_cy))
    p7 = (int((tar[7][0]/ tar[7][2])*cam_fx + cam_cx),  int((tar[7][1]/ tar[7][2])*cam_fy + cam_cy))

    points = []
    points.append(tuple(p0))
    points.append(tuple(p1))
    points.append(tuple(p2))
    points.append(tuple(p3))
    points.append(tuple(p4))
    points.append(tuple(p5))
    points.append(tuple(p6))
    points.append(tuple(p7))

    '''
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    '''
    lineWidthForDrawing = 2

    # draw front
    DrawLine(g_draw, points[0], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(g_draw, points[4], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[7], color, lineWidthForDrawing)
    DrawLine(g_draw, points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(g_draw, points[0], points[4], color, lineWidthForDrawing)
    DrawLine(g_draw, points[7], points[3], color, lineWidthForDrawing)
    DrawLine(g_draw, points[5], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(g_draw, points[0], pointColor=(0,101,255), pointRadius=4)
    DrawDot(g_draw, points[1], pointColor=(232,12,217), pointRadius=4)

    # draw x on the top
    DrawLine(g_draw, points[0], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[4], color, lineWidthForDrawing)


    r = 255 # int(np.random.choice(range(255)))
    g = 255 # int(np.random.choice(range(255)))
    b = 255 # int(np.random.choice(range(255)))

    # cv2.line(img, p0, p1, (0,0,b), 2)
    # cv2.line(img, p0, p3, (r,0,0), 2)
    # cv2.line(img, p0, p4, (0,g,0), 2)

    # cv2.line(img, p0, p1, color, 2)
    # cv2.line(img, p0, p3, color, 2)
    # cv2.line(img, p0, p4, color, 2)
    # cv2.line(img, p1, p2, color, 2)
    # cv2.line(img, p1, p5, color, 2)
    # cv2.line(img, p2, p3, color, 2)
    # cv2.line(img, p2, p6, color, 2)
    # cv2.line(img, p3, p7, color, 2)
    # cv2.line(img, p4, p5, color, 2)
    # cv2.line(img, p4, p7, color, 2)
    # cv2.line(img, p5, p6, color, 2)
    # cv2.line(img, p6, p7, color, 2)

    # print(p0, p1, p2, p3, p4, p5, p6, p7)
    # cv2.rectangle(img, p0, p7, (0,0,255))

    return p0, p7

def Publisher(model_pub, pose_pub, pose_sub, pa, cam_mat, dist, viz, objs_pose, modelPts, cloudPts):

        """ publish model cloud pints """
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_depth_optical_frame"

        """ publish pose to ros-msg """
        poses = objs_pose
        pose2msg = Pose()
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "camera_color_optical_frame"

        """ subscribe to pose in camera_color_optical_frame  """
        getPose = pa.poses
        header = pa.header
        # getRosPos = []
        # getRosQuat = []
        pos = np.zeros(3)
        quat = np.zeros(4)

        if poses is not None:

            print("total onigiri(s) found", len(poses))
            for p in range(len(poses)):
                # print(str(p), poses[p])
                pose2msg.position.x = poses[p]['tx']
                pose2msg.position.y = poses[p]['ty']
                pose2msg.position.z = poses[p]['tz']
                pose2msg.orientation.w = poses[p]['qw']
                pose2msg.orientation.x = poses[p]['qx']
                pose2msg.orientation.y = poses[p]['qy']
                pose2msg.orientation.z = poses[p]['qz']
                pose_array.poses.append(pose2msg)


                # TODO:
                # print(header.seq)
                # pos[0] = getPose[p].position.x
                # pos[1] = getPose[p].position.y
                # pos[2] = getPose[p].position.z
                # quat[0] = getPose[p].orientation.x
                # quat[1] = getPose[p].orientation.y
                # quat[2] = getPose[p].orientation.z
                # quat[3] = getPose[p].orientation.w


                #offset to align with obj-center
                objC = np.array([-0.01, 0.05, 0.])
                initRot = rotation_matrix(-m.pi, [0, 1, 0])

                pos = np.array([poses[p]['tx'], poses[p]['ty'], poses[p]['tz']])
                pos =  pos + objC
                q2rot = quaternion_matrix([poses[p]['qw'], poses[p]['qx'], poses[p]['qy'], poses[p]['qz']])
                # q2rot = concatenate_matrices(initRot, q2rot.T)

                """ transform modelPoints w.r.t estimated pose """
                modelPts = np.dot(modelPts, q2rot[0:3, 0:3])
                modelPts = np.add(modelPts, pos)

                """ send to ros """
                scaled_cloud = pcl2.create_cloud_xyz32(header, modelPts)
                model_pub.publish(scaled_cloud)

                """ publish/visualize pose """
                if viz is not None:
                    imgpts_cloud,_ = cv2.projectPoints(modelPts, np.identity(3), np.array([0.,0.,0.]), cam_mat, dist)
                    vizPnP = draw_pointCloud(viz, imgpts_cloud, [0,255,0]) # modelPts
                    draw_axis(viz, q2rot[0:3, 0:3], pos, cam_mat)
                    modelPts = np.zeros(shape=modelPts.shape)
                cv2.imshow("posePnP", cv2.cvtColor(vizPnP, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1), #cv2.moveWindow('posePnP', 0, 0)

                # getRosPos.append(pos)
                # getRosQuat.append(quat)

            pose_pub.publish(pose_array)
        else:
            print(f"{Fore.RED}no onigiri detected{Style.RESET_ALL}")


