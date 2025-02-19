from ultralytics import YOLO

model = YOLO('LOCAL/MODELS/yolov8s_5C.yaml')
model = YOLO('LOCAL/MODELS/joinST_5C-Small/best.pt')

metrics = model.val(data='LOCAL/MODELS/joinST_5C-data.yaml')
