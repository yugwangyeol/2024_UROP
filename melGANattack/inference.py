import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from models import Generator
import torchaudio.transforms as T

# 경로 설정
wav_name = "contents_4"
model_path = "/home/work/rvc/melGANattack/checkpoints/generator.pth"
input_wav_path = f"/home/work/rvc/FreeVC/voice/content/{wav_name}.wav"
output_mel_path = f"/home/work/rvc/FreeVC/voice/content/noisy_{wav_name}.pt"
output_wav_path = f"/home/work/rvc/FreeVC/voice/content/noisy_{wav_name}.wav"
save_path = f"/home/work/rvc/melGANattack"

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Noise Generator 모델 불러오기
noise_generator = Generator().to(device)
noise_generator.load_state_dict(torch.load(model_path, map_location=device))
noise_generator.eval()

# 2. 입력 wav 파일 로드 및 mel spectrogram 변환
waveform, sample_rate = torchaudio.load(input_wav_path)
waveform = waveform.unsqueeze(0).to(device)  # [1, num_channels, num_samples]

# MelSpectrogram 변환 설정
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=80
).to(device)

# STFT 변환 설정
stft_transform = T.Spectrogram(n_fft=1024, hop_length=256).to(device)
griffin_lim = T.GriffinLim(n_fft=1024, hop_length=256).to(device)

# 3. Mel spectrogram으로 변환
mel_spectrogram = mel_transform(waveform)  # [1, n_mels, time]

# 4. 노이즈 추가 (Mel spectrogram 상에서)
with torch.no_grad():
    noise = noise_generator(mel_spectrogram)  # [1, 1, n_mels, time]
    noisy_mel_spectrogram = mel_spectrogram + noise # 노이즈 추가

# 5. Mel을 STFT로 변환 후 Griffin-Lim 적용
# Mel을 원래 STFT 크기로 변환합니다.
stft_spectrogram = T.InverseMelScale(n_stft=513, n_mels=80, sample_rate=sample_rate).to(device)(noisy_mel_spectrogram)
noisy_waveform = griffin_lim(stft_spectrogram)

# 6. Mel 시각화 함수
def plot_mel_spectrograms(original_mel, noisy_mel, save_path="mel_spectrogram_comparison.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_mel.squeeze().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Original Mel Spectrogram")
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_mel.squeeze().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Noisy Mel Spectrogram")
    
    plt.tight_layout()
    plt.savefig(save_path)  # 이미지 파일로 저장
    print(f"Mel spectrogram 비교 이미지가 {save_path}에 저장되었습니다.")
    plt.close()

# Mel 비교 시각화
plot_mel_spectrograms(mel_spectrogram, noisy_mel_spectrogram)
print(noisy_waveform.shape)
noisy_waveform = noisy_waveform.squeeze(0)
# 7. 결과 noisy waveform 저장
if not os.path.exists(os.path.dirname(output_wav_path)):
    os.makedirs(os.path.dirname(output_wav_path))

# noisy_waveform을 CPU로 옮기기
torchaudio.save(output_wav_path, noisy_waveform.cpu(), sample_rate)

print(f"노이즈가 적용된 파일이 {output_wav_path}에 저장되었습니다.")
