from ultralytics import YOLO

model = YOLO('D:\PythonProject\\ultralyticsPro0401-YOLOv8\\ultralytics\\cfg\\models\\cfg2025\\YOLOv8-Conv改进\\YOLOv8-ShiftConv.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights


# Train the model
if __name__ == '__main__':
    model.train(data='D:\\PythonProject\\ultralyticsPro0401-YOLOv8\\ultralytics\\cfg\\datasets\\VOC-test.yaml',
                epochs=300,
                imgsz=640,
                workers=0,
                batch=16,
                device=0,
                )