#import utils.temporalloaders as temp
#import utils.dataloaders as still
import cv2
import yaml
import importlib

import ultralytics.yolo.data.dataloaders.v5loader as v5loader
import ultralytics.yolo.data.dataloaders.npyloader as npyloader


# importlib.reload(still)
# importlib.reload(temp)
fhyp = '/home/dkrst/GIT/YOLOv5/data/hyp.yaml'
with open(fhyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)

still_loader, stilldata = still.create_dataloader('/home/dkrst/GIT/YOLOv5/data/YOLO-ST-640/data_still/images/train', 640, 16, 32, hyp=hyp, augment=True)
tmp_loader, tmpdata = temp.create_dataloader('/home/dkrst/GIT/YOLOv5/data/YOLO-ST-640/train', 640, 16, 32, hyp=hyp, augment=True)

# STILL IMAGE
im, l, f, s = stilldata.testGetItem(9)
nim = im.numpy().transpose((1,2,0))

nim = cv2.cvtColor(nim, cv2.COLOR_RGB2BGR)

cv2.imshow('W', nim)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# USPOREDBA:
import subprocess
subprocess.call(['gwenview', f])

# 6 CHANNEL TEMPORAL
tim, tl, tf,  ts = tmpdata.testGetItem(9)
ntim = tim.numpy().transpose((1,2,0))

def getStillTemporal(tim, rgb2bgr=False):
    tm_img = tim[:,:,:3]
    st_img = tim[:,:,3:]
    if rgb2bgr:
        tm_img = cv2.cvtColor(tm_img, cv2.COLOR_RGB2BGR)
        st_img = cv2.cvtColor(st_img, cv2.COLOR_RGB2BGR)
    return st_img, tm_img

st_img, tm_img = getStillTemporal(ntim, True)
cv2.imshow('S', st_img)
cv2.waitKey(3000)
cv2.destroyAllWindows()

        
    
