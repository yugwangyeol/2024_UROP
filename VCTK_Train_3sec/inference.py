import torch
import torchaudio
import os
from model import Generator as NoiseEncoder  # NoiseEncoder를 정의한 파일에서 import

# 경로 설정
wav_name = "org_content_1"
model_path = "./checkpoints/generator.pth"
input_wav_path = f"/home/work/2024_UROP/Test Data/{wav_name}.wav"
output_wav_dir = "/home/work/2024_UROP/Result"
output_wav_path = os.path.join(output_wav_dir, f"noisy_{wav_name}.wav")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. NoiseEncoder 모델 불러오기
noise_encoder = NoiseEncoder().to(device)
noise_encoder.load_state_dict(torch.load(model_path, map_location=device))
noise_encoder.eval()

# 2. 입력 wav 파일 로드
waveform, sample_rate = torchaudio.load(input_wav_path)

# 배치 차원 추가 (모델에 넣기 위해)
waveform = waveform.unsqueeze(0).to(device)  # [1, num_channels, num_samples]

# 3. 노이즈 적용
with torch.no_grad():
    noise = noise_encoder(waveform).to(device)  # [1, num_channels, num_samples]
    noisy_waveform = torch.clamp(waveform + noise, -1, 1)  # [-1, 1] 범위로 클램핑
(waveform + noise, -1, 1)  # [-1, 1] 범위로 클램핑
# 4. 노이즈가 적용된 wav 파일 저장
if not os.path.exists(output_wav_dir):
    os.makedirs(output_wav_dir)

torchaudio.save(output_wav_path, noisy_waveform.squeeze(0).cpu(), sample_rate)

print(f"노이즈가 적용된 파일이 {output_wav_path}에 저장되었습니다.")
