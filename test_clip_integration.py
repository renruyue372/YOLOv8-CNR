from ultralytics import YOLO
import cv2

# 测试CLIP集成
def test_clip_integration():
    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')
    
    # 测试图像路径
    test_image = 'ultralytics/assets/bus.jpg'
    
    # 执行推理
    results = model(test_image)
    
    # 显示结果
    for result in results:
        # 打印原始检测结果
        print("\n原始检测结果:")
        print(result.verbose())
        
        # 显示带有检测框的图像
        im_array = result.plot()
        cv2.imshow('CLIP Enhanced Detection', im_array)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_clip_integration()