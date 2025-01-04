import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import WavLMModel
from model import Discriminator, Generator
from train import train_noise_encoder
import os

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 저장 경로 설정
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 하이퍼파라미터
batch_size = 2  # 배치 크기
num_epochs = 10
learning_rate = 0.001
lambda_emb = 0.2
lambda_wav = 0.8

# 데이터 로드
train_dataset = torchaudio.datasets.VCTK_092(root="./Data", download=True)

def get_dataset_max_length(dataset):
    """데이터셋의 최대 길이를 계산"""
    print("데이터셋 최대 길이 계산 중...")
    max_length = 0
    for waveform, *_ in dataset:
        max_length = max(max_length, waveform.shape[1])
    print(f"데이터셋 최대 길이: {max_length}")
    return max_length

def collate_fn(batch, max_length):
    """전체 데이터셋의 최대 길이로 패딩"""
    waveforms = []
    for waveform, *_ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

# 데이터셋의 최대 길이 계산 및 시퀀스 길이 설정
dataset_max_length = get_dataset_max_length(train_dataset)
sequence_length = dataset_max_length  # Discriminator에 필요한 sequence_length 정의

# collate_fn에 최대 길이 전달
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=lambda b: collate_fn(b, sequence_length)
)

# 모델 초기화
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
generator = Generator().to(device)
discriminator = Discriminator(input_length=sequence_length).to(device)  # 수정: stride 및 AdaptiveAvgPool1d 사용

# 더미 forward pass 및 출력 크기 확인
dummy_input = torch.randn(1, 1, sequence_length).to(device)
conv_output = discriminator.conv(dummy_input)
print(f"Conv output shape: {conv_output.shape}")

# 옵티마이저 초기화
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# 학습 실행
img_losses, emb_losses, gan_losses = train_noise_encoder(
    generator, discriminator, wavlm, train_loader, 
    optimizer_G, optimizer_D, num_epochs, batch_size, 
    device, lambda_wav, lambda_emb
)

# 학습 완료 메시지 출력
print("학습 완료!")

# 모델 저장
torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
print(f"모델이 {save_dir}에 저장되었습니다.")
