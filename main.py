import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import WavLMModel
from models import Discriminator, Generator
from train import train_noise_encoder
import os

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 저장 경로 설정
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 하이퍼파라미터
batch_size = 4
num_epochs = 10
learning_rate = 0.0002
lambda_emb = 0.3
lambda_wav = 0.9

# 데이터 로드
train_dataset = torchaudio.datasets.LIBRISPEECH(root='/home/work/Conference/VCAttack_wavGANAttack/data', url='train-clean-100', download=True)

def collate_fn(batch):
    waveforms = []
    max_length = max(waveform.shape[1] for waveform, _, _, _, _, _ in batch)
    for waveform, _, _, _, _, _ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 모델 초기화
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 학습 실행
img_losses, emb_losses, gan_losses = train_noise_encoder(generator, discriminator, wavlm, train_loader, num_epochs, batch_size, device, lambda_wav, lambda_emb)

# 학습 완료 메시지 출력
print("학습 완료!")

# 모델 저장
torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))

print(f"모델이 {save_dir}에 저장되었습니다.")