# YOLOv8-CNR
This repository contains the code and configuration files for the paper: YOLOv8-CNR: A Lightweight Forest Fire Detection Model with Collaborative Optimization of Backbone, Neck, and Loss Function.

Model Description

YOLOv8-CNR is improved from YOLOv8n with collaborative optimization at three levels. In the backbone, the C3_ConvNeXtV2 module is introduced to enhance feature representation of flame textures and smoke diffuse edges. In the neck, the RepNCSPELAN4 module is incorporated to strengthen multi-scale information flow and alleviate feature attenuation of small-target smoke. For the loss function, SIoU is adopted to improve localization accuracy for irregularly shaped targets.

Dataset

This study uses the M4SFWD synthetic forest fire dataset, which contains 3,974 images. The dataset is divided into a training set of 2,820 images containing 12,666 instances (6,929 fire, 5,737 smoke), a validation set of 763 images containing 3,381 instances (1,807 fire, 1,574 smoke), and a test set of 391 images containing 1,716 instances (891 fire, 825 smoke). Two real-world datasets, SMOKE_dataset (737 images, smoke only) and FLAME_dataset (2,000 images, flame only), are used for generalization evaluation. Due to copyright restrictions, the M4SFWD dataset is not included in this repository. Please obtain it from the original source.

Hyperparameters

All models were trained with tuned hyperparameters determined via grid search on the YOLOv8n validation set. The training uses 300 epochs with a batch size of 32 and input size of 640x640. The optimizer is SGD with an initial learning rate of 0.01, final learning rate of 0.001, momentum of 0.937, and weight decay of 0.0003. The loss weights are box loss 8.0, classification loss 0.8, and DFL loss 5.0. The confidence threshold is 0.25 and NMS threshold is 0.7. All models use COCO pretrained weights. For complete training details, please refer to Table 1 in the paper.

Checkpoint Selection

The final model weights are selected based on the highest mAP50-95 on the validation set. All experiments are repeated with three random seeds (0, 42, and 123) for statistical reliability, and all results are reported as mean ± standard deviation.

Dependencies

The code requires Python 3.8 or higher, PyTorch 2.0.1 or higher, CUDA 11.7, and Ultralytics 8.2.103.

Repository Notes

This repository is for review purposes. The dataset must be prepared separately and organized according to the split described above. For questions, please contact the corresponding author.

License

For academic review purposes only.

许可

仅供学术审稿使用。
