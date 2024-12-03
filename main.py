from Webcam import save_frame_camera_key, dir_path
import torch

from ResNet50 import test_model
from VisionTransformerPy import test_model
from AlexNet import test_model

ViT = torch.load('ViT_model.pth')
ResNet = torch.load('resnet50_model.pth')
AlexNet = torch.load('alexnet_model.pth')

save_frame_camera_key('camera_capture')

test_model(ViT, dir_path)

test_model(ResNet, dir_path)

test_model(AlexNet, dir_path)


