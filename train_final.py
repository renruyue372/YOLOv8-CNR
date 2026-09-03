from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO('ultralyticsPro0401-YOLOv8/ultralytics/cfg/models/cfg2025/YOLOv8-Backbone/SwinV2-Transformer/YOLOv8-CSwinTRv2.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
    model.train(data='coco.yaml', epochs=400, imgsz=640, device='0')
