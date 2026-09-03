import os
import ultralytics
from ultralytics import YOLO

# 强制设置数据集目录
os.environ['DATASETS_DIR'] = 'D:/datasets'
ultralytics.utils.downloads.DATASETS_DIR = 'D:/datasets'

print("当前工作目录:", os.getcwd())
print("数据集目录:", os.environ.get('DATASETS_DIR'))

# 加载模型
model = YOLO('ultralytics/cfg/models/cfg2025/YOLOv8-Backbone/SwinV2-Transformer/YOLOv8-CSwinTRv2.yaml')

# 训练模型
results = model.train(
    data='ultralytics/cfg/datasets/coco128.yaml',
    epochs=20,
    batch=2,
    imgsz=640,
    name='train42',
    workers=0  # 在Windows上建议设置为0
)