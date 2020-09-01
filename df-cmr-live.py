#%%
import os
import sys
import cv2
import glob
import copy
import time
import getpass
import argparse
import math as m
import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt

from colorama import Fore, Style

from PIL import Image
from PIL import ImageDraw

import pyrealsense2 as rs

import chainer
import chainer.utils as utils
import chainer_mask_rcnn as cmr

import torch 
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.knn.__init__ import KNearestNeighbor
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix, rotation_matrix, concatenate_matrices, is_same_transform, is_same_quaternion, rotation_from_matrix

# clean terminal in the beginning
username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')

# specify which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # "0,1,2,3"

num_objects = 1
num_points = 500

path = os.path.dirname(__file__)

pretrained_model = os.path.join('txonigiri/', 'snapshot_model.npz')
# pretrained_model = os.path.join(path + '/txonigiri/', 'snapshot_model.npz')
pooling_func = cmr.functions.roi_align_2d
mask_rcnn = cmr.models.MaskRCNNResNet(
            n_layers=50,
            n_fg_class=1,
            pretrained_model= pretrained_model,
            pooling_func=pooling_func,
            anchor_scales=[4, 8, 16, 32],
            mean=(123.152, 115.903, 103.063),
            roi_size= 7, #7,14
            min_size=600,
            max_size=1000,
        )

chainer.cuda.get_device_from_id(0).use()
mask_rcnn.to_gpu()
print('maskrcnn model loaded %s' % pretrained_model)

pose = PoseNet(num_points, num_objects)
pose.cuda()
pose.load_state_dict(torch.load('txonigiri/pose_model.pth'))   
# pose.load_state_dict(torch.load(path + '/txonigiri/pose_model.pth'))   
pose.eval()
print("pose_model loaded...")

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load('txonigiri/pose_refine_model.pth'))
# refiner.load_state_dict(torch.load(path + '/txonigiri/pose_refine_model.pth'))
refiner.eval()
print("pose_refine_model loaded...")

#%%

class pose_estimation:
    
    def __init__(self, mask_rcnn, pose, refiner, object_index_, rgb, depth):
         
        self.rgb = rgb        
        self.rgb = np.transpose(self.rgb, (2, 0, 1))

        self.rgb_s = []
        self.rgb_s.append(self.rgb)

        self.depth = depth

        self.mask_rcnn = mask_rcnn
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) 
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

    def batch_predict(self):

        for batch in cmr.utils.batch(self.rgb_s, n=2):
            bboxes, masks, labels, scores = self.mask_rcnn.predict(batch)
            for bbox, mask, label, score in zip(bboxes, masks, labels, scores):
                yield bbox, mask, label, score

    def draw_seg(self, result):

        for img_chw, (bbox, mask, label, score) in zip(self.rgb_s, result):
            rgb = img_chw.transpose(1, 2, 0)
            del img_chw

            k = score >= 0.9
            bbox, mask, label, score = bbox[k], mask[k], label[k], score[k]
            i = np.argsort(score)
            bbox, mask, label, score = bbox[i], mask[i], label[i], score[i]

            captions = [
                '{}: {:.1%}'.format(class_names[l], s)
                for l, s in zip(label, score)
            ]
            # for caption in captions:
                # print(caption)
            
            viz = cmr.utils.draw_instance_bboxes(img=rgb, bboxes=bbox, labels=label + 1, n_class=len(class_names) + 1, captions=captions, masks=mask)

            # plt.imsave('seg_result/out/frame.png', viz)
            # plt.imshow(viz.copy()), plt.show()

        return (mask, bbox, viz)

    def pose_refiner(self, my_t, my_r, points, emb, idx):
        
        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = self.refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2
            # refine pose means two matrix multiplication
            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final
        
        return my_t, my_r

    def pose(self):
        
        my_result = []
        
        mask, bbox, viz = self.draw_seg(self.batch_predict())
        # cv2.imshow("mask", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)), cv2.moveWindow('mask', 0, 500) 
        
        pred = mask
        pred = pred *255
        pred = np.transpose(pred, (1, 2, 0)) # (CxHxW)->(HxWxC)

        # convert img into tensor
        rgb_original = self.rgb
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rgb = Variable(norm(torch.from_numpy(self.rgb.astype(np.float32)))).cuda()        

        all_masks = []
        mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))

        # iterate through detected masks
        for b in range(len(bbox)):

            mask = mask_depth * mask_label[:,:,b]        
            rmin = int(bbox[b,0])
            rmax = int(bbox[b,1])
            cmin = int(bbox[b,2])
            cmax = int(bbox[b,3])

            # visualize each masks
            # plt.imshow(mask), plt.show()

            img = np.transpose(rgb_original, (0, 1, 2)) #CxHxW
            choose = mask[rmin : rmax, cmin : cmax].flatten().nonzero()[0]
            if len(choose) == 0:
                cc = torch.LongTensor([0])
                return(cc, cc, cc, cc, cc, cc)
            
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1 
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
            
            depth_masked = self.depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])
            choose = torch.LongTensor(choose.astype(np.int32))

            cam_scale = mm2m
            pt2 = depth_masked/cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud = cloud /1000
            points = torch.from_numpy(cloud.astype(np.float32))

            img_masked = img[:, rmin : rmax, cmin : cmax ]
            img_ = norm(torch.from_numpy(img_masked.astype(np.float32)))
            idx = torch.LongTensor([self.object_index])
            img_ = Variable(img_).cuda().unsqueeze(0)
            points = Variable(points).cuda().unsqueeze(0)
            choose = Variable(choose).cuda()#.unsqueeze(0)
            idx = Variable(idx).cuda()#.unsqueeze(0)
    
            pred_r, pred_t, pred_c, emb = self.estimator(img_, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1) #1
            pred_t = pred_t.view(bs * num_points, 1, 3)

            # print("max confidence", how_max)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)

            # DF refiner ---results are better without refiner
            # my_t, my_r = self.pose_refiner(my_t, my_r, points, emb, idx)

#TODO  use cv.solvePnP or ICP

            """ get mean depth within a box as depth offset """
            depth = self.depth[rmin : rmax, cmin : cmax].astype(float)
            depth = depth * mm2m
            dep,_,_,_ = cv2.mean(depth)
            
            """ position mm2m """
            my_t = np.array(my_t*mm2m)
            # my_t[2] = dep # use this to get depth of obj centroid
            # my_t[1] = my_t[1] + 0.05
            # my_t[0] = my_t[0] + 0.05
            
            print("Pos xyz:{0}".format(my_t))

            """ rotation """
            mat_r = quaternion_matrix(my_r)[:3, :3]
            # print('estimated rotation is\n:{0}'.format(mat_r))

            """ project point cloud """
            imgpts_cloud,_ = cv2.projectPoints(np.dot(points.cpu().numpy(), mat_r), mat_r, my_t, cam_mat, dist)
            viz = draw_cloudPts(viz, imgpts_cloud, 1)

            """ draw cmr 2D box """
            cv2.rectangle(viz, (cmax, cmin), (rmax, rmin), (255,0,0))

            """ add estimated position and Draw 3D box, axis """
            # target_cam = np.add(edges, my_t)
            # new_image = cv2pil(viz)
            # g_draw = ImageDraw.Draw(new_image)
            # p0, p7 = draw_cube(target_cam, viz, g_draw, (255, 165, 0))
            # viz = pil2cv(new_image)
            # draw_axis(viz, np.eye(3), my_t, cam_mat)
            
            """ align 2d bbox with 3D box face """
            # cv2.rectangle(viz, p0, p7, (0,0,255))

            """ introduce offset in Rot """
            Rx = rotation_matrix(2*m.pi/3, [1, 0, 0], my_t)
            R = concatenate_matrices(Rx)[:3,:3]
            mat_r = np.dot(mat_r.T, R[:3, :3])
            
            """ transform 3D box and axis with estimated pose and Draw """
            target_df = np.dot(edges, mat_r)
            target_df = np.add(target_df, my_t)            
            new_image = cv2pil(viz)
            g_draw = ImageDraw.Draw(new_image)
            _,_ = draw_cube(target_df, viz, g_draw, (255, 255, 255))
            viz = pil2cv(new_image)
            draw_axis(viz, mat_r, my_t, cam_mat)

            """ viz pred pose  """
            cv2.imshow("pose", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            cv2.moveWindow('pose', 0, 0)
        
        else:
            if len(bbox) <= 1:
                print(f"{Fore.RED}unable to detect pose..{Style.RESET_ALL}")
        return viz


def draw_axis(img, R, t, K):
    # How+to+draw+3D+Coordinate+Axes+with+OpenCV+for+face+pose+estimation%3f
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, .1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_cloudPts(img, imgpts, label):
    color = [[254,0,0],[254,244,0],[171,242,0],[0,216,254],[1,0,254],[95,0,254],[254,0,221],[0,0,0],[153,56,0],[138,36,124],[107,153,0],[5,0,153],[76,76,76],[32,153,67],[41,20,240],[230,111,240],[211,222,6],[40,233,70],[130,24,70],[244,200,210],[70,80,90],[30,40,30]]
    for point in imgpts:

        img=cv2.circle(img,(int(point[0][0]),int(point[0][1])), 1, color[int(label)], -1)
        # cv2.imshow('color', img)
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

def draw_cube(tar, img, g_draw, color):
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
   
if __name__ == '__main__':
    
    autostop = 100
    
    bs = 1
    objId = 0
    objlist =[1]
    mm2m = 0.001
    iteration = 3
    class_names = ['txonigiri']

    # cam @ aist
    cam_fx = 605.286
    cam_cx = 320.075
    cam_fy = 605.699
    cam_cy = 247.877

    # cam depth @ aist
    # cam_cx = 160.037
    # cam_cy = 123.938
    # cam_fx = 302.643
    # cam_fy = 302.85

    # cam @ tx
    # cam_fx = 611.586
    # cam_cx = 324.002
    # cam_fy = 611.837
    # cam_cy = 235.856

    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cam_mat = np.matrix([ [cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1] ])

    edge = 70.
    edge = edge * mm2m

    edges  = np.array([
                    [edge, -edge*.5,  edge],
                    [edge, -edge*.5, -edge],
                    [edge,  edge*.5, -edge],
                    [edge,  edge*.5,  edge],
                    [-edge,-edge*.5,  edge],
                    [-edge,-edge*.5, -edge],
                    [-edge, edge*.5, -edge],
                    [-edge, edge*.5,  edge]])
    
    # edges = np.array([ [-edge,-edge, edge], # 1
    #                    [-edge,-edge,-edge], # 2
    #                    [ edge,-edge,-edge], # 3
    #                    [ edge,-edge, edge], # 4
    #                    [-edge, edge, edge], # 5
    #                    [-edge, edge,-edge], # 6
    #                    [ edge, edge,-edge], # 7
    #                    [ edge, edge, edge], # 8
    #                 ])
    

    # Stream (Color/Depth) settings
    config = rs.config()
    config.enable_stream(rs.stream.color, 640 , 480 , rs.format.bgr8, 60)
    config.enable_stream(rs.stream.depth, 640 , 480 , rs.format.z16, 60)

    # Start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    t0 = time.time()

    try:
        while True:
        
            t1 = time.time()

            # Wait for frame (Color & Depth)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = rs.align(rs.stream.color).process(frames).get_depth_frame()

            if  not depth_frame or  not color_frame:
                raise ValueError('No image found, camera not streaming?')

            # color image
            rgb = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Depth image
            depth = np.asanyarray(depth_frame.get_data())

            # merge depth with rgb
            gray_color = 153
            clipping_distance = 1 / mm2m # in mm
            depth_image_3d = np.dstack((depth, depth, depth))
            bg_removed = np.where((depth_image_3d >clipping_distance) | (depth_image_3d <= 0 ), gray_color, rgb)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha= 0.03 ), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            # cv2.imshow("merged rgbd", cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

            # DF pose estimation
            pe = pose_estimation(mask_rcnn, pose, refiner, objId, rgb, depth) 
            pe.pose()

            t2 = time.time()
            print('inference time is :{0}'.format(t2 - t1))
            
            key = cv2.waitKey(1) & 0xFF
            if  key == 27:
                print("stopping streaming...")
                break
            
            if t2-t0 > autostop:
                print("auto stop streaming after {} seconds".format(int(autostop)))
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
# %%
