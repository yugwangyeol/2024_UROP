import torch
from models import NoiseEncoder

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_encoder = NoiseEncoder().to(device)

# 학습된 모델 파라미터를 로드하세요
#noise_encoder.load_state_dict(torch.load('trained_noise_encoder.pth'))

# 모델 파라미터 저장
torch.save(noise_encoder.state_dict(), "saved_noise_encoder.pth")
print("NoiseEncoder 파라미터가 별도의 파일로 저장되었습니다.")
