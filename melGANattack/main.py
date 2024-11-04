import torch
import torchaudio
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from train import train_noise_encoder
import wandb
import os

# wandb 초기화
wandb.init(project="mel_GAN_noise_attack")

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
batch_size = 4
num_epochs = 26
learning_rate = 0.001

# 모델 저장 경로 설정
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모델 초기화
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 옵티마이저 설정
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 데이터 로드
train_dataset = torchaudio.datasets.LIBRISPEECH(root='/home/work/rvc/wav_attack/data', url='train-clean-100', download=True)

# 데이터 로더 설정
def collate_fn(batch):
    waveforms = []
    max_length = max(waveform.shape[1] for waveform, *_ in batch)
    for waveform, *_ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 학습 실행
train_noise_encoder(generator, discriminator, None, train_loader, optimizer_G, optimizer_D, num_epochs=num_epochs, batch_size=batch_size, device=device)

# 모델 저장
torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))

print(f"모델이 {save_dir}에 저장되었습니다.")
