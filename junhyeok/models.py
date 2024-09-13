import torch
import torch.nn as nn

class NoiseEncoder(nn.Module):
    def __init__(self):
        super(NoiseEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)  # 입력: [batch_size, 1, time]
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # 출력: [batch_size, 64, time]
        self.conv3 = nn.Conv1d(64, 1, kernel_size=3, padding=1)  # 출력: [batch_size, 1, time]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # [batch_size, 64, time]
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))
        return x
