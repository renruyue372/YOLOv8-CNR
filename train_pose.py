import sys
import argparse
import os

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-mango') # Path

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    # model = YOLO('yolov8n-pose.pt')
    model = YOLO(yaml)

    model.info()
    
    # 姿态估计训练
    results = model.train(
        data='pose01.yaml', 
        epochs=300, 
        imgsz=640, 
        workers=2, 
        batch=4
        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\cfg2024\YOLOv8-Pose\yolov8-pose-05.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)