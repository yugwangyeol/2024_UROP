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
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 64->32
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Dropout 추가
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 128->64
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 256->128
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1024)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 1024, 1)  # 256->128

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.sigmoid(x)
