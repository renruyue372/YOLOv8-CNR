import sys
import argparse
import os

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-') # Path

from ultralytics import YOLO

'''
python train_v8.py --cfg 

'''

def main(opt):
    yaml = opt.cfg
    #weights = opt.weights
    # model = YOLO(weights)
    model = YOLO(yaml)


    # print(model)

    model.info()

    #results = model.train(data='D:\\PythonProject\\ultralyticsPro0401-YOLOv8\\M4SFWD Dataset\\data.yaml',
     #                   epochs=300,
      #                  imgsz=640,
       #                 workers=0,
        #                batch=32,
         #               pretrained=False,
          #              project='M4SFWD',  # 自定义项目路径
           #             name='train100',# 自定义实验名称
            #            )

    results = model.train(data='D:\\PythonProject\\ultralyticsPro0401-YOLOv8\\M4SFWD Dataset\\data.yaml',
                        epochs=300,
                        imgsz=640,
                        workers=0,
                        batch=16,
                        pretrained=False,
                        device=0,
                        project='WSJ',  # 自定义项目路径
                        name='train1',# 自定义实验名称
                        # weight_decay=0.001,
                        # dropout=0.2,
                        # cos_lr=True,
                        # lrf=0.001,
                        # warmup_epochs=5.0,
                        # hsv_h=0.015,
                        # hsv_s=0.5,
                        # hsv_v=0.3,
                        # box=5.0,
                        # cls=0.8

     )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\v8\yolov8.yaml', help='initial weights path')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)