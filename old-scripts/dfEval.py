#%%

import os
import sys
import cv2
import glob
import copy
import time
import json
import getpass
import argparse
import numpy as np
import numpy.ma as ma 
import random as rand
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from segnet_original import SegNet as segnet

from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.network import PoseNet, PoseRefineNet

from scipy.spatial.transform import Rotation as Rot

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
num_points = 1000

knn = KNearestNeighbor(1)

path = '/home/ash/catkin_ws/src/DF_ros_linemod/'

#%%

seg = segnet()
seg.cuda()
seg.load_state_dict(torch.load(path + 'txonigiri/seg_model_02_203.pth'))
seg.eval()
print("segnet_model loaded ...")

pose = PoseNet(num_points, num_objects)
pose.cuda()
pose.load_state_dict(torch.load(path + 'txonigiri/pose_model.pth'))   
pose.eval()
print("pose_model loaded...")

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load(path + 'txonigiri/pose_refine_model.pth'))
refiner.eval()
print("pose_refine_model loaded...")

DEBUG = False
IMGSAVE = True
bs = 1
objId = 0
objlist =[1]
iteration = 2
edge = 40.
scaled = np.array([ [-edge,-edge, edge],
                    [-edge,-edge,-edge],
                    [ edge,-edge,-edge],
                    [ edge,-edge, edge],
                    [-edge, edge, edge],
                    [-edge, edge,-edge],
                    [ edge, edge,-edge],
                    [ edge, edge, edge]])/ 1000

srcpath = [
        '/media/ash/SSD/Odaiba/dataset/synthetic/Home_living_640/',
        '/media/ash/SSD/Odaiba/dataset/synthetic/Home_kitchen_640/',
        '/media/ash/SSD/Odaiba/dataset/synthetic/Home_bedroom_640/',
        '/media/ash/SSD/Odaiba/dataset/synthetic/Home_balcony_640/'
        ]

n = 5000
randomList = rand.sample(range(0, n), n)

#%%

class pose_estimation:
    
    def __init__(self, seg, pose, refiner, object_index_, scaled_, fileName):
        
        self.rgb = np.array(Image.open(srcpath[folderNo[0]] + fileName + ".png").convert("RGB"))
        self.depth = np.array(Image.open(srcpath[folderNo[0]] + fileName  + ".depth.cm.8.png"))

        if DEBUG:
            print ('received rgb image of type: ', self.rgb.shape)
            print ('received depth image of type: ', self.depth.shape)
                
        self.model = seg
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_
        self.scaled = scaled_

        self.cam_fx = 605.2861938476562
        self.cam_cx = 320.0749206542969
        self.cam_fy = 605.69921875
        self.cam_cy = 247.87693786621094

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) 
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        # get ground truth from ndds json files
        with open(str(srcpath[folderNo[0]] + fileName + ".json")) as attributes:
            self.jdata = json.load(attributes)

        """ ROTATION """ #ndds has mirrored y-axis
        quaternion_obj2cam = Rot.from_quat(np.array(self.jdata['objects'][0]['quaternion_xyzw']))
        quaternion_cam2world = Rot.from_quat(np.array(self.jdata['camera_data']['quaternion_xyzw_worldframe']))
        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
        self.mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

    def get_bbox(self, bbox):

        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
        if bbx[0] < 0:
            bbx[0] = 0
        if bbx[1] >= 480:
            bbx[1] = 479
        if bbx[2] < 0:
            bbx[2] = 0
        if bbx[3] >= 640:
            bbx[3] = 639                
        rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > 480:
            delt = rmax - 480
            rmax = 480
            rmin -= delt
        if cmax > 640:
            delt = cmax - 640
            cmax = 640
            cmin -= delt
        return rmin, rmax, cmin, cmax

    def drawCube(self, tar, img, r, g, b):
        
        p0 = (int((tar[0][0]/ tar[0][2])*self.cam_fx + self.cam_cx),  int((tar[0][1]/ tar[0][2])*self.cam_fy + self.cam_cy))
        p1 = (int((tar[1][0]/ tar[1][2])*self.cam_fx + self.cam_cx),  int((tar[1][1]/ tar[1][2])*self.cam_fy + self.cam_cy))
        p2 = (int((tar[2][0]/ tar[2][2])*self.cam_fx + self.cam_cx),  int((tar[2][1]/ tar[2][2])*self.cam_fy + self.cam_cy))
        p3 = (int((tar[3][0]/ tar[3][2])*self.cam_fx + self.cam_cx),  int((tar[3][1]/ tar[3][2])*self.cam_fy + self.cam_cy))
        p4 = (int((tar[4][0]/ tar[4][2])*self.cam_fx + self.cam_cx),  int((tar[4][1]/ tar[4][2])*self.cam_fy + self.cam_cy))
        p5 = (int((tar[5][0]/ tar[5][2])*self.cam_fx + self.cam_cx),  int((tar[5][1]/ tar[5][2])*self.cam_fy + self.cam_cy))
        p6 = (int((tar[6][0]/ tar[6][2])*self.cam_fx + self.cam_cx),  int((tar[6][1]/ tar[6][2])*self.cam_fy + self.cam_cy))
        p7 = (int((tar[7][0]/ tar[7][2])*self.cam_fx + self.cam_cx),  int((tar[7][1]/ tar[7][2])*self.cam_fy + self.cam_cy))
        
        cv2.line(img, p0,p1,(r,g,b), 2)
        cv2.line(img, p0,p3,(r,g,b), 2)
        cv2.line(img, p0,p4,(r,g,b), 2)
        cv2.line(img, p1,p2,(r,g,b), 2)
        cv2.line(img, p1,p5,(r,g,b), 2)
        cv2.line(img, p2,p3,(r,g,b), 2)
        cv2.line(img, p2,p6,(r,g,b), 2)
        cv2.line(img, p3,p7,(r,g,b), 2)
        cv2.line(img, p4,p5,(r,g,b), 2)
        cv2.line(img, p4,p7,(r,g,b), 2)
        cv2.line(img, p5,p6,(r,g,b), 2)
        cv2.line(img, p6,p7,(r,g,b), 2)

    def callback(self):

        time1 = time.time()

        rgb_original = self.rgb
        self.rgb = np.transpose(self.rgb, (2, 0, 1))
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.rgb = norm(torch.from_numpy(self.rgb.astype(np.float32)))
        
        self.rgb = Variable(self.rgb).cuda()
        semantic = self.model(self.rgb.unsqueeze(0))
        _, pred = torch.max(semantic, dim=1)
        pred = pred *255
        if IMGSAVE:
            torchvision.utils.save_image(pred, path + '/seg_result/out/' + 'torchpred.png')

        pred = np.transpose(pred.cpu().numpy(), (1, 2, 0)) # (CxHxW)->(HxWxC)
        if IMGSAVE:
            cv2.imwrite(path + '/seg_result/out/' + 'numpypred.png', pred)
        
        contours, _ = cv2.findContours(np.uint8(pred),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        rmin, rmax, cmin, cmax = self.get_bbox([x, y, w, h])

        if IMGSAVE:
            print(self.get_bbox([x, y, w, h]))
            img_bbox = np.array(rgb_original.copy())
            cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
            cv2.imwrite(path + '/seg_result/out/' + 'bbox.png', img_bbox)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth,0))
        mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))
        # print(mask_depth.shape, mask_label.shape)
        mask = mask_depth * mask_label.reshape(480, 640)

        img = np.transpose(rgb_original, (2, 0, 1))
        
        img_masked = img[:, rmin:rmax, cmin:cmax]
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        #print("length of choose is :{0}".format(len(choose))) 
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1  # if number of object pixels are bigger than 500, we select just 500
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]  # now len(choose) = 500
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        depth_masked = self.depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        choose = np.array([choose])

        pt2 = depth_masked
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud /1000

        points = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        img = norm(torch.from_numpy(img_masked.astype(np.float32)))
        idx = torch.LongTensor([self.object_index])

        img = Variable(img).cuda().unsqueeze(0)
        points = Variable(points).cuda().unsqueeze(0)
        choose = Variable(choose).cuda().unsqueeze(0)
        idx = Variable(idx).cuda().unsqueeze(0)
 
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        print("how_max", how_max)
        print("which_max", which_max)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

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

            my_mat_final = np.dot(my_mat, my_mat_2) # refine pose means two matrix multiplication
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        my_r = quaternion_matrix(my_r)[:3, :3]
        my_t = np.array(my_t)

        ## custom scaling for 3Dbox
        col = [2,0,1]
        new_col = np.zeros((len(col), len(col)))
        for idx, i in enumerate(col):
            new_col[idx, i] = 1
        self.scaled = np.dot(self.scaled, new_col)

        # draw estimated pose 3Dbox
        target = np.dot(self.scaled, my_r.T)
        target = np.add(target, my_t)
        self.drawCube(target, rgb_original, 0, 0, 255)

        # ROTATION
        print('gt mirrored obj2world rotation is\n:{0}'.format(self.mirrored_y_axis))
        print('estimated rotation is\n:{0}'.format(my_r))

        # POSITION # ndds has cm units
        gt_t = self.jdata["objects"][0]["location"]
        gt_t = [ gt_t[i]*0.01 for i in range(len(gt_t))]
        print("ground truth translation is:{0}".format(gt_t))
        print('estimated translation is:{0}'.format(my_t))
        errorPos = np.linalg.norm((gt_t - my_t), ord=1)     
        print("errorPos:{0} cm".format(errorPos*100))

        # draw ground truth 3D box
        target_gt = np.dot(self.scaled, self.mirrored_y_axis.T)
        target_gt = np.add(target_gt, gt_t)        
        self.drawCube(target_gt, rgb_original, 255, 0, 0)

        plt.figure(figsize = (10,10))
        plt.imshow(rgb_original, cmap = 'gray', interpolation = 'nearest', aspect='auto')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        
        """ need python 3.x """
        ## https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented
        # cv2.imshow('rgb', cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB))  # OpenCV uses BGR model
        # cv2.waitKey(1) # pass any integr except 0, as 0 will freeze the display windows 

        time2 = time.time()
        print('inference time is :{0}'.format(time2-time1))


for i in randomList[1:3]:
    
    folderNo = rand.sample(range(0, 4), 1)
    # read ndds files
    if i<1:
        infileName = "00000"
    elif i>0 and i<10:
        infileName = "00000"
    elif i>9 and i<100:
        infileName = "0000"
    elif i>99 and i<1000:
        infileName = "000"
    else:
        infileName = "00"
    
    fileName = infileName + str(i)
    # print(folderNo[0], i, fileName)

    pe = pose_estimation(seg, pose, refiner, objId, scaled, fileName) 
    pe.callback()

print("DONE")
# %%
