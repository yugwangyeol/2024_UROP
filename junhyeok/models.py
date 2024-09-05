import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x  # 입력을 그대로 유지
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # 입력을 출력에 더함 (skip connection)
        out = self.relu(out)  # 마지막 ReLU 적용
        return out

class NoiseEncoder(nn.Module):
    def __init__(self):
        super(NoiseEncoder, self).__init__()
        # 간단한 CNN 구조에 Residual Block 추가
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(64, 64)
        self.resblock2 = ResidualBlock(64, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()  # 출력을 -1에서 1 사이로 제한

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.final_conv(x)
        return self.tanh(x)

# Example of using the NoiseEncoder
# encoder = NoiseEncoder()
# output = encoder(torch.randn(1, 3, 32, 32))  # 임의의 입력 이미지에 대한 출력

def get_vgg16():
    # 사전 학습된 VGG16 모델 로드
    vgg16 = models.vgg16(pretrained=True)
    # 모델의 가중치를 고정
    for param in vgg16.parameters():
        param.requires_grad = False
    return vgg16