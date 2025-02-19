import numpy as np

import ultralytics.yolo.data.dataloaders.npyloader as npyloader


#nloader, ndata = npyloader.create_dataloader('/home/dkrst/GIT/YOLO/data/NPY-new/test/still/', 640, 16,32)

# Na laptopu:
nloader, ndata = npyloader.create_dataloader('LOCAL/VALID/still/', 640, 16,32)

nim, h, w = ndata.load_image(3)
nim.shape

from ultralytics import YOLO
#model = YOLO('/home/dkrst/GIT/YOLO/yolov8/YOLOv8-Restart/stillRGB_3C/weights/last.pt')
# Na laptopu:
model = YOLO('LOCAL/MODELS/stillRGB_3C-Small/best.pt')

results = model([nim])

#
# ili
#
#im = np.load('/home/dkrst/GIT/YOLO/data/NPY-new/test/still/000453_s_000878.npz')['im']

# Na laptopu
im = np.load('LOCAL/VALID/still/000332_s_000266.npz')['im']
im.shape

results = model([im])

#
# Test za 5C
#
import numpy as np
import ultralytics.yolo.data.dataloaders.npyloader as npyloader
from ultralytics import YOLO

model = YOLO('LOCAL/MODELS/joinST_5C-Small/best.pt')

im = np.load('LOCAL/VALID/join_st/000339_s_000074.npz')['im']
im.shape

results = model([im])

nloader, ndata = npyloader.create_dataloader('LOCAL/VALID/join_st/', 640, 16,32)
nim, h, w = ndata.load_image(3)
nim.shape
results = model([nim])

