import os
import torchaudio
import numpy as np
from gammatone.filters import make_erb_filters, erb_filterbank
import scipy.fftpack

# 1. GFCC 추출 함수
import numpy as np
from gammatone.filters import make_erb_filters, erb_filterbank

def extract_gfcc(file_path, num_coefficients=20):
    signal_data, sr = torchaudio.load(file_path)
    signal_data = signal_data.squeeze(0).numpy()

    # 필터 중심 주파수 배열 생성 (64개 필터)
    low_freq = 50  # 필터의 최저 주파수
    high_freq = sr / 2  # Nyquist 주파수
    num_channels = 64
    centre_freqs = np.geomspace(low_freq, high_freq, num_channels)  # 로그 스케일로 필터 생성

    # 감마톤 필터 생성
    erb_filters = make_erb_filters(sr, centre_freqs)

    # 필터 적용
    filtered_signal = erb_filterbank(signal_data, erb_filters)

    # GFCC 계산
    power_spectrum = np.abs(filtered_signal) ** 2
    log_power_spectrum = np.log(np.maximum(power_spectrum, 1e-10))
    gfcc = scipy.fftpack.dct(log_power_spectrum, type=2, axis=0, norm='ortho')[:num_coefficients]

    return gfcc.mean(axis=1)


# 2. 유사도 계산 함수 (변경 없음)
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# 3. ASV 평가 함수
def evaluate_asv(original_dir, generated_dir):
    original_files = sorted([os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith(".wav")])
    generated_files = sorted([os.path.join(generated_dir, f) for f in os.listdir(generated_dir) if f.endswith(".wav")])

    assert len(original_files) == len(generated_files), 

    similarities = []
    for orig_file, gen_file in zip(original_files, generated_files):
        orig_embedding = extract_gfcc(orig_file)
        gen_embedding = extract_gfcc(gen_file)

        similarity = cosine_similarity(orig_embedding, gen_embedding)
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)*100
    print(f"평균 코사인 유사도: {avg_similarity:.4f}%")

    return similarities

# 4. 메인 실행 코드
if __name__ == "__main__":
    original_dir = "/home/work/2024_Conference/UROP/ASV_Data/Style"
    generated_dir = "/home/work/2024_Conference/UROP/ASV_Data/Noise_FreeVC"

    similarities = evaluate_asv(original_dir, generated_dir)
