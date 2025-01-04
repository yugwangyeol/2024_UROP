import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),  # stride=2로 시퀀스 길이 절반
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # stride=2로 시퀀스 길이 절반
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # stride=2로 시퀀스 길이 절반
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(256)  # 고정된 크기로 줄이기
        self.flatten = nn.Flatten()

        # Conv 레이어 출력 크기 자동 계산
        dummy_input = torch.randn(1, 1, input_length)
        with torch.no_grad():
            conv_output = self.conv(dummy_input)
            pooled_output = self.pool(conv_output)
            flattened_size = pooled_output.numel()

        # Linear 레이어 생성
        self.linear_layer = nn.Linear(flattened_size, 1)

        # Freeze 설정 (가중치와 바이어스 고정)
        self.linear_layer.weight.requires_grad = False  # 가중치 고정
        self.linear_layer.bias.requires_grad = False  # 바이어스 고정

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return torch.sigmoid(x)
