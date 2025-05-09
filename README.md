# HeadPoseEstimation

[English](./README.md) | [中文](./README_zh.md)

### Project Introduction
**Head Pose Estimation**

This is a lightweight version created for deployment on edge-side chips (based on MobileNetV2) that not only estimates head pose but also includes a lightened version of face detection based on MobileNetV2, with ONNX files provided.
For a better-performing version, you could modify the network and use more data (the current version uses only the BIWI dataset).

![Alt text](./assets/head_pose_definition.png "head pose metrics")

<img src="./assets/result3.jpg" width="400">

**Dataset**    
BIWI

**Model**    
HopeNet

**Weight Files**    
Head Pose Estimation    
>./weights/retinaface_mbv2_s840_fixfpn_relu_no_postproc_20210630.onnx

Face Detection    
>./weights/head_pose_estimation_hopenet_biwi_mbv2_20211223.onnx
    
### Inference Method    
```python
from deploy.head_pose_est_api import HeadPoseEstAPI

hpe_api = HeadPoseEstAPI(model_path='./weights/head_pose_estimation_hopenet_biwi_mbv2_20211223.onnx')
result = hpe_api(face_img)
```
Refer to the implementation in `deploy/head_pose_est_api.py`.
If you want to process video streams using a local camera, you can also use the code in `run/realtime_time.py`.


### Training Code    
1. Configure the file path for biwi_dataset in the code
2. cd run & python3 train_mbv2_v3.py

**Adjustments for Data Augmentation can be made in the following code:**         
Augmentation techniques include: random flipping, cropping or expansion, rotation, distortion, color adjustments, brightness, blurring, Mixup, sharpening, etc.
```python
class Pipeline(object):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_crop = RandomCrop(crop_ratio=[0.6, 0.9])
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()
        self.random_effect2 = RandomEffect2()
        self.rotate_func = Rotate()

        # self.mixup_func is MixUp()
        expand_params = dict(expand_ratio=[1.01, 1.2])
        mosaic_params = dict(mosaic_weight=0.1, base=224)
        self.expand_func = Expand(**expand_params)
        self.mosaic_func = Mosaic(**mosaic_params)
        self.sharpen_func = Sharpness()
```

### Points to Optimize
1. More data for training
2. Gradient accumulation
3. Exponential Moving Average (EMA)
4. Optuna for Data Augmentation policy
5. Loss function    


### References
1. https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
2. https://github.com/natanielruiz/deep-head-pose
