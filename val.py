from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("D:\\PythonProject\\ultralyticsPro0401-YOLOv8\\WSJ\\train111\\weights\\best.pt")
    model.val(batch=16,workers=0,device=0)