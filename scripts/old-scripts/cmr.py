#%%

import os
import numpy as np
import random as rand
from PIL import Image

import chainer
import chainer_mask_rcnn as cmr

from chainercv import utils
from utils.vis_bbox import vis_bbox
from utils.bn_utils import bn_to_affine
import matplotlib.pyplot as plt


# specify which gpu to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # "0,1,2,3"

path = '/home/ash/catkin_ws/src/DF_ros_linemod/'

pretrained_model = os.path.join(path + 'txonigiri/', 'snapshot_model.npz')
print('Using pretrained_model: %s' % pretrained_model)

# pooling_func = 'align'
pooling_func = cmr.functions.roi_align_2d
# # pooling_func = 'pooling'
# pooling_func = cmr.functions.roi_pooling_2d
# # pooling_func = 'resize'
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
print("maskrcnn model loaded ...")

# chainer.serializers.load_npz(path + 'txonigiri/snapshot_model.npz', model)
# with np.load(path + 'txonigiri/snapshot_model.npz') as maskrcnn:



#%%

class maskrcnn:
    
    def __init__(self, fileName):
        
        # self.rgb = np.array(Image.open(fileName).convert("RGB"))
        self.rgb = utils.read_image(fileName, color=True)
        print(self.rgb.shape)

        bboxes, labels, scores, masks = model.predict([self.rgb])        
        bbox, label, score, mask = bboxes[0], np.asarray(labels[0],dtype=np.int32), scores[0], masks[0]
        print(bbox)

        vis_bbox(self.rgb, bbox, label=label, score=score, mask=mask, label_names=('onigiri'), contour=False, labeldisplay=True)
        plt.show()
    
        # plt.figure(figsize = (10,10))
        # plt.imshow(fileName, cmap = 'gray', interpolation = 'nearest', aspect='auto')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        # # plt.savefig(filename)


n = 100 #5000
randomList = rand.sample(range(0, n), n)
scrpath = ['/media/ash/SSD/Odaiba/dataset/txonigiri-yolo/']
fileName = scrpath[0] + 'img' + str(randomList[0]) + ".png"
print(fileName)

maskrcnn(fileName)
print("done")
