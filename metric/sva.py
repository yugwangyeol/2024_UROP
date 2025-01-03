import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from models import Generator
import subprocess
import sys
'''
# resemblyzer 패키지 설치
subprocess.check_call([sys.executable, "-m", "pip", "install", "resemblyzer"])
'''
from resemblyzer import VoiceEncoder, preprocess_wav

# 경로 설정
noise_generator_path = "/home/work/rvc/wavGANattack/checkpoints/generator_E1W1.pth"
test_data_path = "/home/work/rvc/wavGANattack/test_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# d-vector 모델 초기화
d_vector = VoiceEncoder()

# Noise Generator 모델 불러오기
noise_generator = Generator().to(device)
noise_generator.load_state_dict(torch.load(noise_generator_path, map_location=device))
noise_generator.eval()

# DataLoader 설정
test_dataset = torchaudio.datasets.LIBRISPEECH(root="/home/work/rvc/wavGANattack/test_data", url='test-other', download=True)

def collate_fn(batch):
    waveforms = []
    max_length = max(waveform.shape[1] for waveform, *_ in batch)
    for waveform, *_ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

test_loader = DataLoader(test_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)

# SVA 평가 함수
def evaluate_sva():
    correct_defense = 0
    correct_quality = 0
    total = 0
    threshold = 0.683  # EER 기반 threshold 설정

    with torch.no_grad():
        for waveforms in tqdm(test_loader, desc="Evaluating SVA"):
            waveforms = waveforms.to(device)
            
            # 원본 d-vector 추출
            original_waveform = waveforms[0].cpu().numpy().squeeze()
            original_d_vector = d_vector.embed_utterance(preprocess_wav(original_waveform))
            
            # 노이즈 추가하여 변형된 음성 생성
            noise = noise_generator(waveforms)
            noisy_waveforms = torch.clamp(waveforms + noise, -1, 1)
            noisy_waveform = noisy_waveforms[0].cpu().numpy().squeeze()
            
            # 노이즈가 추가된 d-vector 추출
            noisy_d_vector = d_vector.embed_utterance(preprocess_wav(noisy_waveform))
            
            # SVAquality와 SVAdefense 유사도 계산
            similarity_quality = np.dot(original_d_vector, noisy_d_vector) / (np.linalg.norm(original_d_vector) * np.linalg.norm(noisy_d_vector))
            similarity_defense = 1 - similarity_quality  # 방해 성공률
            
            # threshold와 비교하여 SVAquality 및 SVAdefense 평가
            if similarity_quality > threshold:
                correct_quality += 1
            if similarity_defense < threshold:
                correct_defense += 1  # 성공적으로 방해된 경우
            
            total += 1
    
    sva_quality = correct_quality / total
    sva_defense = correct_defense / total
    print(f"SVA Quality (유사성 유지 평가): {sva_quality:.4f}")
    print(f"SVA Defense (방해 성공률 평가): {sva_defense:.4f}")

if __name__ == "__main__":
    evaluate_sva()
