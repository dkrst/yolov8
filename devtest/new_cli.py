from ultralytics import YOLO
model = YOLO('/home/dkrst/GIT/YOLO/data/Paper-Yolov8/yolov8s_3C.yaml')
results = model.train(
    #cfg='/home/dkrst/GIT/YOLO/data/NPY-new/temporal_3C-cfg.yaml',
    cfg='/home/dkrst/GIT/YOLO/data/Paper-Yolov8/stillRGB_3C-cfg.yaml',
    #cfg='/home/dkrst/GIT/YOLO/data/NPY-new/dist_4C-cfg.yaml',
    #cfg='/home/dkrst/GIT/YOLO/data/NPY-new/fgr_5C-cfg.yaml',
    #
    # data='/home/dkrst/GIT/YOLO/data/NPY-new/stillRGB_3C-data.yaml',
    # epochs=3,
    # v5loader=True,
    # device='cuda:1',
    # name='TEMPORAL_3C'
)
from clearml import Task
task = Task.current_task()
task.get_status()
task.mark_completed()
