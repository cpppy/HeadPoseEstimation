import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from model_design.mobilenetv2 import MobileNetV2



class HopenetMBV2(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, num_bins=66):
        self.inplanes = 64
        super(HopenetMBV2, self).__init__()
        self.backbone = MobileNetV2()
        self.avgpool = nn.AvgPool2d(4)
        self.fc_yaw = nn.Linear(320, num_bins)
        self.fc_pitch = nn.Linear(320, num_bins)
        self.fc_roll = nn.Linear(320, num_bins)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        if not self.training:
            output = torch.cat((pre_yaw, pre_pitch, pre_roll), dim=0)
            output = F.softmax(output, dim=1)
            return output
        return pre_yaw, pre_pitch, pre_roll



if __name__=='__main__':

    model = HopenetMBV2(66)
    model.eval()

    x = torch.randn(2, 3, 112, 112, dtype=torch.float32)
    # yaw, pitch, roll = model(x)
    # print(yaw.shape, pitch.shape, roll.shape)
    y = model(x)
    print(y.shape)


