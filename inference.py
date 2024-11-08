import torch
import torchaudio
import os
from models import Generator  # Generator를 정의한 파일에서 import

# 경로 설정
wav_name = "contents_2"
model_path = "/home/work/Conference/VCAttack_wavGANAttack/checkpoints/generator.pth"
input_wav_path = f"/home/work/Conference/VCAttack_wavGANAttack/{wav_name}.wav"
output_wav_dir = "/home/work/Conference/VCAttack_wavGANAttack"
output_wav_path = os.path.join(output_wav_dir, f"noisy_{wav_name}.wav")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Generator 모델 불러오기
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# 2. 입력 wav 파일 로드
waveform, sample_rate = torchaudio.load(input_wav_path)

# 배치 차원 추가 (모델에 넣기 위해)
waveform = waveform.unsqueeze(0).to(device)  # [1, num_channels, num_samples]

# 3. Generator로 노이즈 생성 및 적용
with torch.no_grad():
    delta = generator(waveform).to(device)
    # delta: [1, 1, sequence_length]
    noisy_waveform = torch.clamp(waveform + delta, -1, 1)

# 4. 노이즈가 적용된 wav 파일 저장
if not os.path.exists(output_wav_dir):
    os.makedirs(output_wav_dir)
    
torchaudio.save(output_wav_path, noisy_waveform.squeeze(0).cpu(), sample_rate)

print(f"노이즈가 적용된 파일이 {output_wav_path}에 저장되었습니다.")