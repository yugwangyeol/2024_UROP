import torch
import torch.nn as nn

class NoiseEncoder(nn.Module):
    def __init__(self):
        super(NoiseEncoder, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

