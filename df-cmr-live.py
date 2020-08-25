#%%
import os
import sys
import cv2
import glob
import copy
import time
import getpass
import argparse
import numpy as np
import numpy.ma as ma 
import pyrealsense2 as rs
import matplotlib.pyplot as plt

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
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

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

path = os.path.dirname(__file__)

pretrained_model = os.path.join(path + '/txonigiri/', 'snapshot_model.npz')
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
pose.load_state_dict(torch.load(path + '/txonigiri/pose_model.pth'))   
pose.eval()
print("pose_model loaded...")

refiner = PoseRefineNet(num_points, num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load(path + '/txonigiri/pose_refine_model.pth'))
refiner.eval()
print("pose_refine_model loaded...")

#%%

class pose_estimation:
    
    def __init__(self, mask_rcnn, pose, refiner, object_index_, scaled_, rgb, depth):
         
        self.rgb = rgb        
        self.rgb = np.transpose(self.rgb, (2, 0, 1))

        self.rgb_s = []
        self.rgb_s.append(self.rgb)

        self.depth = depth
        
        if DEBUG:
            print ('received rgb image of type: ', self.rgb.shape)
            print ('received depth image of type: ', self.depth.shape)

        self.mask_rcnn = mask_rcnn
        self.estimator = pose
        self.refiner = refiner
        self.object_index = object_index_
        self.scaled = scaled_

        # cam @ aist
        self.cam_fx = 605.2861938476562
        self.cam_cx = 320.0749206542969
        self.cam_fy = 605.69921875
        self.cam_cy = 247.87693786621094

        # cam @ tx
        # self.cam_fx = 611.5845947265625
        # self.cam_cx = 324.00238037109375
        # self.cam_fy = 611.8369750976562
        # self.cam_cy = 235.85594177246094

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

    def draw_cube(self, tar, img):

        p0 = (int((tar[0][0]/ tar[0][2])*self.cam_fx + self.cam_cx),  int((tar[0][1]/ tar[0][2])*self.cam_fy + self.cam_cy))
        p1 = (int((tar[1][0]/ tar[1][2])*self.cam_fx + self.cam_cx),  int((tar[1][1]/ tar[1][2])*self.cam_fy + self.cam_cy))
        p2 = (int((tar[2][0]/ tar[2][2])*self.cam_fx + self.cam_cx),  int((tar[2][1]/ tar[2][2])*self.cam_fy + self.cam_cy))
        p3 = (int((tar[3][0]/ tar[3][2])*self.cam_fx + self.cam_cx),  int((tar[3][1]/ tar[3][2])*self.cam_fy + self.cam_cy))
        p4 = (int((tar[4][0]/ tar[4][2])*self.cam_fx + self.cam_cx),  int((tar[4][1]/ tar[4][2])*self.cam_fy + self.cam_cy))
        p5 = (int((tar[5][0]/ tar[5][2])*self.cam_fx + self.cam_cx),  int((tar[5][1]/ tar[5][2])*self.cam_fy + self.cam_cy))
        p6 = (int((tar[6][0]/ tar[6][2])*self.cam_fx + self.cam_cx),  int((tar[6][1]/ tar[6][2])*self.cam_fy + self.cam_cy))
        p7 = (int((tar[7][0]/ tar[7][2])*self.cam_fx + self.cam_cx),  int((tar[7][1]/ tar[7][2])*self.cam_fy + self.cam_cy))
        
        r = 255 # int(np.random.choice(range(255)))
        g = int(np.random.choice(range(255)))
        b = int(np.random.choice(range(255)))

        cv2.line(img, p0, p1, (r,g,b), 2)
        cv2.line(img, p0, p3, (r,g,b), 2)
        cv2.line(img, p0, p4, (r,g,b), 2)
        cv2.line(img, p1, p2, (r,g,b), 2)
        cv2.line(img, p1, p5, (r,g,b), 2)
        cv2.line(img, p2, p3, (r,g,b), 2)
        cv2.line(img, p2, p6, (r,g,b), 2)
        cv2.line(img, p3, p7, (r,g,b), 2)
        cv2.line(img, p4, p5, (r,g,b), 2)
        cv2.line(img, p4, p7, (r,g,b), 2)
        cv2.line(img, p5, p6, (r,g,b), 2)
        cv2.line(img, p6, p7, (r,g,b), 2)

    def pose(self):
        
        mask, bbox, viz = self.draw_seg(self.batch_predict())
        # cv2.imshow("rgb", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)), cv2.waitKey(1)
    
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

        for b in range(len(bbox)):

            mask = mask_depth * mask_label[:,:,b]        
            rmin = int(bbox[b,0])
            rmax = int(bbox[b,1])
            cmin = int(bbox[b,2])
            cmax = int(bbox[b,3])
            
            img = np.transpose(rgb_original, (0, 1, 2)) #CxHxW
            img_masked = img[:, rmin : rmax, cmin : cmax ]
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
        
            # visualize each masks
            # plt.imshow(mask), plt.show()
            
            depth_masked = self.depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = 1
            pt2 = depth_masked/cam_scale
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud = cloud /1000

            points = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            
            img_ = norm(torch.from_numpy(img_masked.astype(np.float32)))
            idx = torch.LongTensor([self.object_index])
            img_ = Variable(img_).cuda().unsqueeze(0)
            points = Variable(points).cuda().unsqueeze(0)
            choose = Variable(choose).cuda().unsqueeze(0)
            idx = Variable(idx).cuda().unsqueeze(0)
    
            pred_r, pred_t, pred_c, emb = self.estimator(img_, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 0) #1
            pred_t = pred_t.view(bs * num_points, 1, 3)

            # print("max confidence", how_max)

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

            # POSITION # ndds has cm units
            my_t = np.array(my_t)
            # my_t = np.array([my_t[0], my_t[1], 1-my_t[2]])
            # print('estimated translation is:{0}'.format(my_t))
            
            # ROTATION
            my_r = quaternion_matrix(my_r)[:3, :3]
            # my_r = np.dot(my_r, np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]))
            # print('estimated rotation is\n:{0}'.format(my_r))
            
            # Draw estimated pose 3Dbox
            target = np.dot(self.scaled, my_r.T) #my_r.T
            target = np.add(target, my_t)
            self.draw_cube(target, viz)

            # Norm pose
            NormPos = np.linalg.norm((my_t), ord=1)     
            # print("NormPos:{0}".format(NormPos))
            print("Pos xyz:{0}".format(my_t))

            # plt.figure(figsize = (10,10)), plt.imshow(viz), plt.show()
            cv2.imshow("pose", cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            
        return viz


if __name__ == '__main__':
        
    DEBUG = False
    IMGSAVE = False

    bs = 1
    objId = 0
    objlist =[1]
    iteration = 4
    class_names = ['txonigiri']

    edge = 60.
    scaled = np.array([ [-edge,-edge, edge],
                        [-edge,-edge,-edge],
                        [ edge,-edge,-edge],
                        [ edge,-edge, edge],
                        [-edge, edge, edge],
                        [-edge, edge,-edge],
                        [ edge, edge,-edge],
                        [ edge, edge, edge]])/ 1000

    # Stream (Color/Depth) settings
    config = rs.config()
    config.enable_stream(rs.stream.color, 640 , 480 , rs.format.bgr8, 0)
    config.enable_stream(rs.stream.depth, 640 , 480 , rs.format.z16, 0)

    # Start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

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
            # depth_frame = rs.colorizer().colorize(depth_frame)
            depth = np.asanyarray(depth_frame.get_data())

            # DF pose estimation
            pe = pose_estimation(mask_rcnn, pose, refiner, objId, scaled, rgb, depth) 
            pe.pose()

            t2 = time.time()
            print('inference time is :{0}'.format(t2 - t1))
            
            key = cv2.waitKey(1) & 0xFF
            if  key == 27:
                print("stopping streaming...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()