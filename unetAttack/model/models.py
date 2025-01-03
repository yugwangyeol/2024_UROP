import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, initial_features=32):
        super(Generator, self).__init__()
        
        # 입력을 바로 처리
        self.first_conv = ConvBlock(in_channels, initial_features)
        
        # Encoder
        self.enc1 = ConvBlock(initial_features, initial_features)      # 32 -> 32
        self.enc2 = ConvBlock(initial_features, initial_features*2)    # 32 -> 64
        self.enc3 = ConvBlock(initial_features*2, initial_features*4)  # 64 -> 128
        self.enc4 = ConvBlock(initial_features*4, initial_features*8)  # 128 -> 256
        
        # Decoder
        self.dec4 = ConvBlock(initial_features*8, initial_features*4)  # 256 -> 128
        self.dec3 = ConvBlock(initial_features*6, initial_features*2)  # 192(128+64) -> 64
        self.dec2 = ConvBlock(initial_features*3, initial_features)    # 96(64+32) -> 32
        self.dec1 = ConvBlock(initial_features*2, initial_features)    # 64(32+32) -> 32
        
        self.final = nn.Conv1d(initial_features, in_channels, 1)
        
        self.pool = nn.MaxPool1d(2, 2)
        
    def forward(self, x):
        # Initial conv
        x = self.first_conv(x)  # [B, 32, L]
        
        # Encoder path
        enc1 = self.enc1(x)          # [B, 32, L]
        enc2 = self.enc2(self.pool(enc1))   # [B, 64, L/2]
        enc3 = self.enc3(self.pool(enc2))   # [B, 128, L/4]
        enc4 = self.enc4(self.pool(enc3))   # [B, 256, L/8]
        
        # Decoder path with skip connections
        dec4 = self.dec4(F.interpolate(enc4, size=enc3.shape[-1], mode='linear'))  # [B, 128, L/4]
        
        dec3_input = torch.cat([F.interpolate(dec4, size=enc2.shape[-1], mode='linear'), enc2], dim=1)
        dec3 = self.dec3(dec3_input)  # [B, 64, L/2]
        
        dec2_input = torch.cat([F.interpolate(dec3, size=enc1.shape[-1], mode='linear'), enc1], dim=1)
        dec2 = self.dec2(dec2_input)  # [B, 32, L]
        
        dec1_input = torch.cat([dec2, x], dim=1)
        dec1 = self.dec1(dec1_input)  # [B, 32, L]
        
        # Final output - 원본 차원으로 복원
        return self.final(dec1)  # [B, 1, L]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        
        

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        linear_layer = nn.Linear(x.size(1), 1).to(x.device)
        x = linear_layer(x)
        return torch.sigmoid(x)
