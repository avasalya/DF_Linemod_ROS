#%%
from segnet_original import SegNet as segnet
import torch 
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import glob
import sys
import cv2

path = '/home/ash/catkin_ws/src/DF_ros_linemod/'

model = segnet()
model.cuda()
model.load_state_dict(torch.load(path + 'txonigiri/seg_model_96.pth'))
model.eval()

#%%

name = '5.png'

# """ for single images """
rgb = np.array(Image.open(path + 'seg_result/' + name).convert("RGB"))
# rgb = cv2.imread(path + 'seg_result/' + name)
# print(rgb.shape)

# ## resize the image
# dim = (640, 480)
# rgb = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
# print('Resized Dimensions : ',resized.shape)


rgb = np.transpose(rgb, (2, 0, 1))
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
rgb = norm(torch.from_numpy(rgb.astype(np.float32)))

rgb = Variable(rgb).cuda()
semantic = model(rgb.unsqueeze(0))
_, pred = torch.max(semantic, dim=1)
pred = pred*255
torchvision.utils.save_image(pred, path + 'seg_result/' + '_' + name)
# print(semantic.shape)
# print(pred.shape)

# img = np.transpose(pred.cpu().numpy(), (1, 2, 0))
# cv2.imwrite(path + 'seg_result/out/' + 'rgb.png', img)




#%%

""" for batch of images """

# colors = [np.array(Image.open(file).convert("RGB")) for file in sorted(glob.glob(path + 'seg_result/*.png')) ]
# colors_trans = [np.transpose(rgb, (2, 0, 1)) for rgb in colors]
# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# colors_norm = [ norm(torch.from_numpy(rgb.astype(np.float32))) for rgb in colors_trans]


# for idx, rgb in enumerate(colors_norm):
#     # rgb = np.transpose(rgb, (2, 0, 1))
#     # rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
#     rgb = Variable(rgb).cuda()
#     semantic = model(rgb.unsqueeze(0))
#     _, pred = torch.max(semantic, dim=1)
#     pred = pred*255
#     # pred = np.transpose(pred, (1, 2, 0))  # (CxHxW)->(HxWxC)
#     # ret, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
#     torchvision.utils.save_image(pred, path + 'seg_result/out/' + str(idx) + '.png')
# print("done")
