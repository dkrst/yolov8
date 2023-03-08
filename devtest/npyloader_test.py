#  Store multichannel image in NPY format
import numpy as np
import cv2

s=cv2.imread('still/000333_s_000283.jpg')
t=cv2.imread('temporal/000333_s_000283.jpg')
m = np.concatenate((s,t), 2)
np.save('m.npy', m , allow_pickle=False)

nm = np.load('m.npy', allow_pickle=False)
nm.shape

cv2.imshow('S', nm[:,:,:3])
cv2.imshow('T', nm[:,:,3:])
cv2.waitKey()
cv2.destroyAllWindows()

###################

# Test dataloader
import numpy as np
import cv2

# TEST v5loader
import ultralytics.yolo.data.dataloaders.v5loader as v5loader

imloader, imdata = v5loader.create_dataloader('/home/dkrst/GIT/STRIBOR-dev/LOCAL/work-DATA/fixed_size/samples_last/YOLO/valid/images', 640, 16,32)

im, l, f, s = imdata.testGetItem(39)
nim = im.numpy().transpose((1,2,0))
nim = cv2.cvtColor(nim, cv2.COLOR_RGB2BGR)
cv2.imshow('W', nim)
cv2.waitKey()
cv2.destroyAllWindows()

# TEST npyloader
import numpy as np
import cv2
import ultralytics.yolo.data.dataloaders.npyloader as npyloader
nloader, ndata = npyloader.create_dataloader('/home/dkrst/GIT/STRIBOR-dev/LOCAL/work-DATA/fixed_size/samples_last/YOLO/valid/npy',
                                             640, 16,32, nchannels=6, augment=True)
nim, h, w = ndata.load_image(39)
nim.shape
cv2.imshow('S', nim[:,:,:3])
cv2.imshow('T', nim[:,:,3:])
cv2.waitKey()
cv2.destroyAllWindows()

mim, l = ndata.load_mosaic(39)
#mim, l = ndata.load_mosaic9(39)
mim.shape
cv2.imshow('S', mim[:,:,:3])
cv2.imshow('T', mim[:,:,3:])
cv2.waitKey()
cv2.destroyAllWindows()


#
# TEST npyloader
import numpy as np
import cv2
import ultralytics.yolo.data.dataloaders.npyloader as npyloader

hyp={'flipud': 0,
     'fliplr': 0.5,
     'mosaic': 1.0,
     'mixup': 0.01,
     'cout': 0.01,
     'replicate': 0.9,
     'degrees': 9,
     'translate': 0.1,
     'scale': 0.1,
     'shear': 10,
     'perspective': 0, #0.0001,
     'copy_paste': 0.5,
     'border': (10, 10)}
nloader, ndata = npyloader.create_dataloader('/home/dkrst/GIT/STRIBOR-dev/LOCAL/work-DATA/fixed_size/samples_last/YOLO/valid/npy',
                                             640, 16,32, nchannels=6, augment=True, hyp=hyp)

im, l, f, s = ndata.testGetItem(39)
nim = im.numpy().transpose((1,2,0))

cv2.imshow('T', cv2.cvtColor(nim[:,:,:3], cv2.COLOR_RGB2BGR))
cv2.imshow('S', cv2.cvtColor(nim[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()

# VODI RACUNA: prva slika je zadnja!

sim = nim[:,:,3:]
tim = nim[:,:,:3]

sim = cv2.cvtColor(sim, cv2.COLOR_RGB2BGR)
tim = cv2.cvtColor(tim, cv2.COLOR_RGB2BGR)
cv2.imshow('S', sim)
cv2.imshow('T', tim)
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.resize radi na proizvoljnom broju kanala:
rim = cv2.resize(nim, (1280,1280),  interpolation=cv2.INTER_LINEAR)
cv2.imshow('S', cv2.cvtColor(rim[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.imshow('T', cv2.cvtColor(rim[:,:,:3], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.rotate radi na proizvoljnom broju kanala:
rim = cv2.rotate(nim, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('S', cv2.cvtColor(rim[:,:,3:], cv2.COLOR_RGB2BGR))
cv2.imshow('T', cv2.cvtColor(rim[:,:,:3], cv2.COLOR_RGB2BGR))
cv2.waitKey()
cv2.destroyAllWindows()
