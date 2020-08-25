#%%

import os
import numpy as np
import random as rand
from PIL import Image

import chainer
import chainer_mask_rcnn as cmr
import chainer.utils as utils

import matplotlib.pyplot as plt
import skimage.io
from skimage.viewer import ImageViewer

# specify which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # "0,1,2,3"

path = '/home/ash/catkin_ws/src/DF_ros_linemod/'

pretrained_model = os.path.join(path + 'txonigiri/', 'snapshot_model.npz')

# pooling_func = 'align', 'pooling', 'resize'
pooling_func = cmr.functions.roi_align_2d
# pooling_func = cmr.functions.roi_pooling_2d
# pooling_func = cmr.functions.crop_and_resize

mask_rcnn = cmr.models.MaskRCNNResNet(
            n_layers=50,
            n_fg_class=1,
            pretrained_model= pretrained_model,
            pooling_func=pooling_func,
            anchor_scales=[4, 8, 16, 32],
            mean=(123.152, 115.903, 103.063),
            roi_size=7,# 14,
            min_size=600,
            max_size=1000,
        )

chainer.cuda.get_device_from_id(0).use()
mask_rcnn.to_gpu()
print('maskrcnn model loaded %s' % pretrained_model)

#%%

class Chainer_maskrcnn:
    
    def __init__(self, files):
        
        self.imgs_chw = []
        for fileName in files:
            print(fileName)
            img = skimage.io.imread(fileName)
            # plt.imshow(img), plt.show()
            img_chw = img.transpose(2, 0, 1)
            self.imgs_chw.append(img_chw)
            del img, img_chw

    def batch_predict(self, mask_rcnn):

        for batch in cmr.utils.batch(self.imgs_chw, n=2):
            bboxes, masks, labels, scores = mask_rcnn.predict(batch)
            for bbox, mask, label, score in zip(bboxes, masks, labels, scores):
                yield bbox, mask, label, score

    def draw_seg(self, result):

        for fileName, img_chw, (bbox, mask, label, score) in zip(files, self.imgs_chw, result):
            img = img_chw.transpose(1, 2, 0)
            del img_chw

            k = score >= 0.7
            bbox, mask, label, score = bbox[k], mask[k], label[k], score[k]
            i = np.argsort(score)
            bbox, mask, label, score = bbox[i], mask[i], label[i], score[i]

            # print("mask type", type(mask))
            # print("mask shape", mask.shape)
    
            print("bbox", bbox)

            # print("label type", type(label))
            # print("label shape", label.shape)
            
            captions = [
                '{}: {:.1%}'.format(class_names[l], s)
                for l, s in zip(label, score)
            ]
            for caption in captions:
                print(caption)
            viz = cmr.utils.draw_instance_bboxes(
                img=img,
                bboxes=bbox,
                labels=label + 1,
                n_class=len(class_names) + 1,
                captions=captions,
                masks=mask,
            )
            plt.imshow(viz)
            plt.show()
        
#%%

if __name__ == '__main__':

    n = 100
    files = []
    class_names = ['txonigiri']
    randomList = rand.sample(range(0, n), 1)
    scrpath = ['/media/ash/SSD/Odaiba/dataset/txonigiri-yolo/']

    for i in randomList:
        files.append(scrpath[0] + 'img' + str(i) + ".png")

    cmr_ = Chainer_maskrcnn(files)
    result = cmr_.batch_predict(mask_rcnn)

    for i in randomList:
        bbox = cmr_.draw_seg(result)
    
    print("done")
