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
#model = YOLO('/home/dkrst/GIT/YOLO/yolov8/runs/detect/VAL-dev118/weights/last.pt')
results = model.train(
    data='/home/dkrst/GIT/YOLO/data/NPY-ST-640/data_npy.yaml',
    imgsz=640,
    epochs=333,
    batch=16,
    augment=True,
    v5loader=True,
    #name='NPY0-cli'
    cache=True,
    device='cuda:0',
    name='NPY0'
)
from clearml import Task
task = Task.current_task()
task.get_status()
task.mark_completed()
#task.mark_failed()
