import numpy as np
import cv2
import albumentations as A
from PIL import Image

import ultralytics.yolo.data.dataloaders.npyloader as npyloader

hyp={'flipud': 0,
     'fliplr': 0.0,
     'mosaic': 0.0,
     'mixup': 0.0,
     'degrees': 0,
     'translate': 0.0,
     'scale': 0.0,
     'shear': 0.01,
     'perspective': 0, #0.0001,
     'copy_paste': 0.0,
     'border': (10, 10)}
nloader, ndata = npyloader.create_dataloader('/home/dkrst/GIT/STRIBOR-dev/LOCAL/work-DATA/fixed_size/samples_last/YOLO/valid/npy',
                                             640, 16,32, nchannels=6, augment=True, hyp=hyp)

im, l, f, s = ndata.testGetItem(39)
nim = im.numpy().transpose((1,2,0))

cv2.imshow('T', cv2.cvtColor(nim[:,:,:3], cv2.COLOR_RGB2BGR))
cv2.imshow('S', cv2.cvtColor(nim[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()

# Dobre transformacije
#T = A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.02)
#T = A.Blur(p=0.02)
#T = MedianBlur(p=0.02)
#T = A.RandomBrightnessContrast(p=0.01)
#T = A.RandomGamma(p=0.01)
#T = A.ImageCompression(quality_lower=75, p=0.0)

aug = T(image=nim[:,:,3:])['image']
cv2.imshow('ORIG', cv2.cvtColor(nim[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.imshow('T', cv2.cvtColor(aug[:,:,:3], cv2.COLOR_RGB2BGR))
#cv2.imshow('S', cv2.cvtColor(aug[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()

'''
alpha=10
sigma=150
alpha_affine=50
interpolation=1
border_mode=4
value=17
mask_value=105
elastic = A.ElasticTransform(alpha, sigma, alpha_affine, interpolation,
                             border_mode, value, mask_value, p=1.0) 
aug = elastic(image=nim)['image']
'''
aug = elastic(image=nim)['image']
cv2.imshow('T', cv2.cvtColor(aug[:,:,:3], cv2.COLOR_RGB2BGR))
cv2.imshow('S', cv2.cvtColor(aug[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()




#########################
# TEST v5loader
import ultralytics.yolo.data.dataloaders.v5loader as v5loader
from ultralytics.yolo.data.dataloaders.v5augmentations import Albumentations
from ultralytics.yolo.utils.ops import clean_str, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn
import albumentations as A


#
index = 39
albumentations = Albumentations(640)
im, h, w = imdata.load_image(index)
labels = imdata.labels[index].copy()
ratio = [1, 1]
pad = [0, 0]
labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
im, labels = self.albumentations(img, labels)
###
# STARO iz Yolov5

from os.path import join
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from matplotlib import pyplot as plt

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def compare(img, aug):
    f = plt.figure(figsize=(10, 15))
    f.add_subplot(2, 1, 1)
    plt.imshow(img)
    f.add_subplot(2, 1, 2)
    plt.imshow(aug)
    plt.axis('off')
    plt.show()

img = cv2.imread('image_00603-frame-09.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#visualize(img)


#
# ElasticTransform example
#
# DOBRA!
'''
class albumentations.augmentations.geometric.transforms.ElasticTransform (alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5)
'''

alpha=10
sigma=150
alpha_affine=50
interpolation=1
border_mode=4
value=17
mask_value=105

elastic = A.ElasticTransform(alpha, sigma, alpha_affine, interpolation,
                             border_mode, value, mask_value, p=1.0) 

aug = elastic(image=img)['image']
#visualize(aug)
compare(img, aug)

#
# GridDistortion example
#
# - ne vidim razliku, jos pogledati
'''
class albumentations.augmentations.geometric.transforms.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, normalized=False, always_apply=False, p=0.5) 
'''

num_steps=50
distort_limit=0.5
interpolation=1
border_mode=4
value=None
mask_value=None

grid_distortion = A.GridDistortion(num_steps, distort_limit, interpolation, border_mode, value, mask_value, p=1.0)

