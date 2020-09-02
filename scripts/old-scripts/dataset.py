import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1] #txonigiri
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1] # get rid the new line feed '\n' 
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line)) # an array of paths
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))

                # len(list_rgb) = len(list_depth) = len(list_label) = len(list_obj) = len(list_rank) = 186, for object 1 in train mode

                if self.mode == 'eval':
                    self.list_label.append('{0}/data/{1}/seg_result/{2}.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            #print(self.list_rank)
            #print(self.list_obj)
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file)

            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item)) #pt[1].shape ->(5841,3)
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)
        #print('item_cout {0}'.format(item_count))
        #print('rgb_length {0}'.format(self.length))

        self.cam_fx = 605.2861938476562
        self.cam_cx = 320.0749206542969
        self.cam_fy = 605.69921875
        self.cam_cy = 247.87693786621094

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) 

        #print('self.xmap is of shape : {0}'.format(self.xmap.shape))        #(480, 640)
#xmap 
#array([[  0,   0,   0, ...,   0,   0,   0],
#       [  1,   1,   1, ...,   1,   1,   1],
#       [  2,   2,   2, ...,   2,   2,   2],
#       ...,
#       [477, 477, 477, ..., 477, 477, 477],
#       [478, 478, 478, ..., 478, 478, 478],
#       [479, 479, 479, ..., 479, 479, 479]])

        self.ymap = np.array([[i for i in range(640)] for j in range(480)])    

        #print('self.ymap is of shape : {0}'.format(self.ymap.shape))         #(480, 640)
#ymap
#array([[  0,   1,   2, ..., 637, 638, 639],
#       [  0,   1,   2, ..., 637, 638, 639],
#       [  0,   1,   2, ..., 637, 638, 639],
#       ...,
#       [  0,   1,   2, ..., 637, 638, 639],
#       [  0,   1,   2, ..., 637, 638, 639],
#       [  0,   1,   2, ..., 637, 638, 639]])



        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 1000
        self.num_pt_mesh_small = 1000
        self.symmetry_obj_idx = [1]


    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)   #480x640x3
        depth = np.array(Image.open(self.list_depth[index]))  #480x640
        label = np.array(Image.open(self.list_label[index]))  #480x640x3
        obj = self.list_obj[index]  # is always 1 as soon as index is between 0 and (186-1) 
        rank = self.list_rank[index]   # 4, 9, 21 ... 

        meta = self.meta[obj][rank][0]  # acquire different ground truth poses of one object from gt.yaml

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))  
        #print('mask_depth is of shape : {0}'.format(mask_depth.shape))  #(480, 640)
        # True if it's not 0 and False if it's is , i.e., where the background is.

        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            

        #print('mask_label is of shape : {0}'.format(mask_label.shape))    #(480, 640)
        
        mask = mask_label * mask_depth # mask_depth only segments the background, mask_label segments the object 

        #print('mask is of shape : {0}'.format(mask.shape))                #(480, 640)
#depth        
#array([[   0,    0,    0, ...,    0,    0,    0],
#       [   0,    0,    0, ...,    0,    0,    0],
#       [   0,    0,    0, ...,    0,    0,    0],
#       ...,
#       [   0,    0,    0, ..., 1579, 1579, 1586],
#       [   0,    0,    0, ..., 1579, 1579, 1579],
#       [   0,    0,    0, ..., 1557, 1557, 1557]], dtype=int32)
#mask_depth
#array([[False, False, False, ..., False, False, False],
#       [False, False, False, ..., False, False, False],
#       [False, False, False, ..., False, False, False],
#       ...,
#       [False, False, False, ...,  True,  True,  True],
#       [False, False, False, ...,  True,  True,  True],
#       [False, False, False, ...,  True,  True,  True]])


        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]       #(480, 640, 3)
        img = np.transpose(img, (2, 0, 1))  #(3, 480, 640)
        img_masked = img

        rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]  #(3, 80, 80)
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        #print('target_r is of shape : {0}'.format(target_r.shape))        #(3, 3)

        target_t = np.array(meta['cam_t_m2c'])  # it's in mm,  e.g, array([ -71.54985682,  -36.76336895, 1015.5891147 ])

        #print('target_t is of shape : {0}'.format(target_t.shape))         #(3,)


        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)]) #从(-0.03, 0.03)中选取
        #print('add_t is of shape : {0}'.format(add_t.shape))               #(3,)

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # nonzero() will return the indices(locations), 
                                                                   # it's an array of array  
        

        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1  # if number of object pixels are bigger than 500, we select just 500
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]  # now len(choose) = 500
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        #print('choose is of shape :{0}'.format(choose.shape))  (500,)
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        #print('depth_masked is of shape :{0}'.format(depth_masked.shape))     #(500, 1)
        # (500,)->(500,1)

        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        #print('xmap_masked is of shape :{0}'.format(xmap_masked.shape))       #(500, 1)

        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        #print('ymap_masked is of shape :{0}'.format(ymap_masked.shape))       #(500, 1)


        choose = np.array([choose])
        #print('choose as array is of shape :{0}'.format(choose.shape))        #(1, 500)

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale   # it's still in mm 
        #print('pt2 is of shape :{0}'.format(pt2.shape))                       #(500, 1)

        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        #print('pt0 is of shape :{0}'.format(pt0.shape))                       #(500, 1)

        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        #print('pt1 is of shape :{0}'.format(pt1.shape))                        #(500, 1)
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)                         #(500, 3)
        
        #print('target_t is of shape :{0}'.format(target_t.shape))     #(3,)
        #print('cloud is of shape : {0}'.format(cloud.shape))          #(500, 3)

        cloud = np.add(cloud, -1.0 * target_t) / 1000.0
        cloud = np.add(cloud, target_t / 1000.0)

        #divide every point by factor 1000

        #print('final cloud is of shape : {0}'.format(cloud.shape))          #(500, 3)

        

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        model_points = self.pt[obj] / 1000.0                #len(model_points)=5841, model_points.shape = (5841,3)
        dellist = [j for j in range(0, len(model_points))]  #(0,1,2,3...5840)
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)   #len(dellist)=5341
        model_points = np.delete(model_points, dellist, axis=0)                        #len(model_points) = 500

        #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        #for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        target = np.dot(model_points, target_r.T)      #对model_points进行旋转  target.shape = (500, 3)


        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)  #对model_points进行平移和噪声
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)  # target.shape = (500,3), in m
            out_t = target_t / 1000.0                  # this is even not used 

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        
        #cloud ->在crop出来的像素区域随机选取500个点,利用相机内参结合深度值算出来的点云cloud
        #choose ->随机选取的500个包含物体的像素点
        #img_masked ->彩色图像crop出来物体后进行一次augumentation操作后的图像
        #target ->真实模型上随机选取的mesh点进行ground truth pose变换后得到的点
        #model_points ->真实模型上随机选取的mesh点在进行pose变换前的点
        #obj ->当前是哪个物体的index




        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])



    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
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


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
