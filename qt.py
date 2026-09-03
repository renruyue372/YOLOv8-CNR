import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
                             QHBoxLayout, QMessageBox, QFileDialog, QLineEdit, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
from ultralytics import YOLO


class Worker:
    def __init__(self):
        self.model = None
        self.model_path = None

    def load_model(self, model_path):
        if model_path:
            self.model_path = model_path
            self.model = YOLO(model_path)
            return self.model is not None
        return False

    def detect_image(self, image, conf_threshold=0.25, iou_threshold=0.7):
        results = self.model.predict(image, conf=conf_threshold, iou=iou_threshold)
        return results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("@author：笑脸惹桃花")
        # self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(300, 150, 800, 450)

        # 创建参数设置区域
        params_widget = QWidget()
        params_layout = QHBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)  # 减少边距

        # Conf 输入框
        conf_label = QLabel("Conf:")
        self.conf_input = QLineEdit()
        self.conf_input.setPlaceholderText("0.25")
        self.conf_input.setText("0.25")

        # IOU 输入框
        iou_label = QLabel("IOU:")
        self.iou_input = QLineEdit()
        self.iou_input.setPlaceholderText("0.7")
        self.iou_input.setText("0.7")

        # 添加标签和输入框到布局，并设置拉伸因子
        params_layout.addWidget(conf_label)
        params_layout.addWidget(self.conf_input, 1)  # 1表示拉伸因子，使其占据更多空间
        params_layout.addWidget(iou_label)
        params_layout.addWidget(self.iou_input, 1)  # 1表示拉伸因子，使其占据更多空间

        # 创建路径显示区域
        path_widget = QWidget()
        path_layout = QVBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 5, 0, 5)

        # 模型路径显示
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("模型路径:")
        self.model_path_display = QLabel("未选择模型")
        self.model_path_display.setStyleSheet("color: gray; border: 1px solid #ccc; padding: 2px;")
        self.model_path_display.setWordWrap(True)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_display, 1)

        # 图片路径显示
        image_path_layout = QHBoxLayout()
        image_path_label = QLabel("图片路径:")
        self.image_path_display = QLabel("未选择图片")
        self.image_path_display.setStyleSheet("color: gray; border: 1px solid #ccc; padding: 2px;")
        self.image_path_display.setWordWrap(True)
        image_path_layout.addWidget(image_path_label)
        image_path_layout.addWidget(self.image_path_display, 1)

        path_layout.addLayout(model_path_layout)
        path_layout.addLayout(image_path_layout)

        # 创建两个 QLabel 分别显示左右图像
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumSize(580, 450)  # 设置大小
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumSize(580, 450)  # 设置大小
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        # 水平布局，用于放置左右两个 QLabel
        layout = QVBoxLayout()
        layout.addWidget(params_widget)  # 添加参数设置区域到主布局
        layout.addWidget(path_widget)  # 添加路径显示区域

        hbox_video = QHBoxLayout()
        hbox_video.addWidget(self.label1)  # 左侧显示原始图像
        hbox_video.addWidget(self.label2)  # 右侧显示检测后的图像
        layout.addLayout(hbox_video)

        self.worker = Worker()

        # 创建按钮布局 - 布满一整行
        buttons_widget = QWidget()
        hbox_buttons = QHBoxLayout(buttons_widget)
        hbox_buttons.setContentsMargins(0, 0, 0, 0)

        # 添加模型选择按钮
        self.load_model_button = QPushButton("📁 模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setMinimumHeight(40)

        # 添加图片检测按钮
        self.image_detect_button = QPushButton("💾 图片检测")
        self.image_detect_button.clicked.connect(self.detect_image)
        self.image_detect_button.setEnabled(False)
        self.image_detect_button.setMinimumHeight(40)

        # 设置按钮拉伸因子，使其均匀分布
        hbox_buttons.addWidget(self.load_model_button, 1)
        hbox_buttons.addWidget(self.image_detect_button, 1)

        layout.addWidget(buttons_widget)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.current_results = None
        self.current_image_path = None

    def get_thresholds(self):
        """获取conf和iou阈值"""
        try:
            conf = float(self.conf_input.text())
        except ValueError:
            conf = 0.25  # 默认值

        try:
            iou = float(self.iou_input.text())
        except ValueError:
            iou = 0.7  # 默认值

        return conf, iou

    def detect_image(self):
        conf, iou = self.get_thresholds()
        image_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)")
        if image_path:
            self.current_image_path = image_path
            self.image_path_display.setText(image_path)
            self.image_path_display.setStyleSheet("color: black; border: 1px solid #ccc; padding: 2px;")

            image = cv2.imread(image_path)
            if image is not None:
                self.current_results = self.worker.detect_image(image, conf, iou)
                if self.current_results:
                    annotated_image = self.current_results[0].plot()
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height1, width1, channel1 = image_rgb.shape
                    bytesPerLine1 = 3 * width1
                    qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
                    pixmap1 = QPixmap.fromImage(qimage1)
                    self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height2, width2, channel2 = annotated_image.shape
                    bytesPerLine2 = 3 * width2
                    qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
                    pixmap2 = QPixmap.fromImage(qimage2)
                    self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt)")
        if model_path and self.worker.load_model(model_path):
            self.model_path_display.setText(model_path)
            self.model_path_display.setStyleSheet("color: black; border: 1px solid #ccc; padding: 2px;")
            self.image_detect_button.setEnabled(True)

    def exit_application(self):
        # 终止程序运行
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())