import numpy as np

import ultralytics.yolo.data.dataloaders.npyloader as npyloader


nloader, ndata = npyloader.create_dataloader('/home/dkrst/GIT/YOLO/data/NPY-new/test/still/', 640, 16,32)

nim, h, w = ndata.load_image(39)
nim.shape

from ultralytics import YOLO
model = YOLO('/home/dkrst/GIT/YOLO/yolov8/YOLOv8-Restart/stillRGB_3C/weights/last.pt')

results = model([nim])

#
# ili
#
im = np.load('/home/dkrst/GIT/YOLO/data/NPY-new/test/still/000453_s_000878.npz')['im']
im.shape

results = model([im])


