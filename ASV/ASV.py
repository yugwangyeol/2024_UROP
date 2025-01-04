import os
import torchaudio
import torch
import numpy as np
from speechbrain.pretrained import SpeakerRecognition

# 1. 사전 학습된 ASV 모델 로드 (SpeechBrain의 ECAPA-TDNN 사용)
asv_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/asv")

# 2. 유틸리티 함수: 음성 파일에서 화자 임베딩 추출
def extract_embedding(file_path):
    signal, sr = torchaudio.load(file_path)  # 음성 로드
    embedding = asv_model.encode_batch(signal)  # 화자 임베딩 추출
    return embedding.squeeze(0).detach().cpu().numpy()

# 3. 유사도 계산 함수 (코사인 유사도 사용)
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# 4. ASV 평가 함수
def evaluate_asv(original_dir, generated_dir):
    original_files = sorted([os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith(".wav")])
    generated_files = sorted([os.path.join(generated_dir, f) for f in os.listdir(generated_dir) if f.endswith(".wav")])

    assert len(original_files) == len(generated_files), "원본과 생성 데이터의 파일 수가 같아야 합니다."

    similarities = []
    for orig_file, gen_file in zip(original_files, generated_files):
        orig_embedding = extract_embedding(orig_file).flatten()  # 원본 화자 임베딩
        gen_embedding = extract_embedding(gen_file).flatten()   # 생성 화자 임베딩

        similarity = cosine_similarity(orig_embedding, gen_embedding)
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    print(f"평균 코사인 유사도: {avg_similarity:.4f}")

    return similarities

if __name__ == "__main__":
    # 5. 데이터 경로 설정
    original_dir = "/home/work/2024_Conference/UROP/ASV_Data/Style"  # VCTK 원본 데이터 경로
    generated_dir = "/home/work/2024_Conference/UROP/ASV_Data/Noise_FreeVC"    # 생성된 데이터 경로

    # 6. ASV 평가 실행
    similarities = evaluate_asv(original_dir, generated_dir)