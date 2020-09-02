#!/usr/bin/env python
'''
This ros node subscribes to two camera topics: '/camera/color/image_raw' and 
'/camera/aligned_depth_to_color/image_raw' in a synchronized way. It then runs 
semantic segmentation and pose estimation with trained models using DenseFusion
(https://github.com/j96w/DenseFusion). The whole code structure is adapted from: 
(http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)
'''

import os
import sys
import cv2
import copy
import time 
import rospy
import getpass
import argparse                                                             
import numpy as np
import numpy.ma as ma 

from colorama import Fore, Style
from helperFunc import *

import message_filters
from sensor_msgs.msg import Image

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
os.environ["CUDA_VISIBLE_DEVICES"]="0" # "0,1,2,3"

num_objects = 1
num_points = 500

knn = KNearestNeighbor(1)

path = os.path.dirname(__file__)

pretrained_model = os.path.join('txonigiri/', 'snapshot_model.npz')
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
pose.eval()
print("pose_model loaded...")

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load('txonigiri/pose_refine_model.pth'))
refiner.eval()
print("pose_refine_model loaded...")


class pose_estimation:
    
    def __init__(self, mask_rcnn, pose, refiner, object_index_):
        
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        
        # choose the same value as publish rate in rs_camera.launch
        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 30)
        ts.registerCallback(self.callback)

        self.mask_rcnn = mask_rcnn
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) 
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
    
        print('subscribed to rgb and depth topic in a sychronized way')

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

    def callback(self, rgb, depth):
        
        t1 = time.time()

        # print ('received depth image of type: ' +depth.encoding)
        # print ('received rgb image of type: ' + rgb.encoding)
            
        #https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
        depth = np.frombuffer(depth.data, dtype=np.uint16).reshape(depth.height, depth.width, -1)
        rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)

        # depth= np.asanyarray(depth.data)
        # rgb = np.asanyarray(rgb.data)

        self.rgb_s = []
        rgb_s = np.transpose(rgb, (2, 0, 1))
        self.rgb_s.append(rgb_s)
        mask, bbox, viz = self.draw_seg(self.batch_predict())
        # cv2.imshow("rgb", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)), cv2.waitKey(1)

        pred = mask
        pred = pred *255
        pred = np.transpose(pred, (1, 2, 0)) # (CxHxW)->(HxWxC)

        # convert img into tensor
        # rgb = rgb
        rgb_original = np.transpose(rgb, (2, 0, 1))
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb = Variable(norm(torch.from_numpy(rgb.astype(np.float32)))).cuda()        

        all_masks = []
        self.depth = depth.reshape(480, 640)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))

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
            # p0, p7 = draw_cube(target_cam, viz, g_draw, (255, 165, 0), cam_fx, cam_fy, cam_cx, cam_cy)
            # viz = pil2cv(new_image)
            # draw_axis(viz, np.eye(3), my_t, cam_mat)
            
            """ align 2d bbox with 3D box face """
            # cv2.rectangle(viz, p0, p7, (0,0,255))

            """ introduce offset in Rot """
            Rx = rotation_matrix(2*m.pi/3, [1, 0, 0], my_t)
            Ry = rotation_matrix(10*m.pi/180, [0, 1, 0], my_t)
            Rz = rotation_matrix(5*m.pi/180, [0, 0, 1], my_t)
            R = concatenate_matrices(Rx, Ry, Rz)[:3,:3]
            mat_r = np.dot(mat_r.T, R[:3, :3])
            
            
            # h, status = cv2.findHomography(cloud, np.dot(points.cpu().numpy(), mat_r))

            """ transform 3D box and axis with estimated pose and Draw """
            target_df = np.dot(edges, mat_r)
            target_df = np.add(target_df, my_t)
            new_image = cv2pil(viz)
            g_draw = ImageDraw.Draw(new_image)
            _,_ = draw_cube(target_df, viz, g_draw, (255, 255, 255), cam_fx, cam_fy, cam_cx, cam_cy)
            viz = pil2cv(new_image)
            draw_axis(viz, mat_r, my_t, cam_mat)

            """ viz pred pose  """
            cv2.imshow("pose", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            cv2.moveWindow('pose', 0, 0)
        
            t2 = time.time()
            print('inference time is :{0}'.format(t2 - t1))

        else:
            if len(bbox) <= 1:
                print(f"{Fore.RED}unable to detect pose..{Style.RESET_ALL}")
            
        return viz


if __name__ == '__main__':
    
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


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='txonigiri', help='object name --> txonigiri')
    opt = parser.parse_args()

    pe = pose_estimation(mask_rcnn, pose, refiner, objId) 
    rospy.init_node('txonigiri_pose', anonymous= False)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down ROS pose estimation module')
    cv2.destroyAllWindows()