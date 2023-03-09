# STILL

from ultralytics import YOLO
model = YOLO('/home/dkrst/GIT/YOLO/data/yolov8l.yaml')
results = model.train(
    data='/home/dkrst/GIT/YOLO/data/NPY-ST-640/data_still.yaml',
    mgsz=640,
    epochs=9,
    batch=8,
    name='STILL0-cli',
    v5loader=True
)

# NPY

from ultralytics import YOLO
model = YOLO('/home/dkrst/GIT/YOLO/data/yolov8l_6c.yaml')
results = model.train(
    data='/home/dkrst/GIT/YOLO/data/NPY-ST-640/data_npy.yaml',
    imgsz=640,
    epochs=9,
    batch=8,
    name='NPY0-cli',
    v5loader=True
)
