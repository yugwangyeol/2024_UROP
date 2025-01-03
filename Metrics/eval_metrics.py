import os
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

def signal_power(signal):
    """
    신호의 RMS 전력을 계산하는 함수
    """
    return np.mean(signal ** 2) ** 0.5

def evaluate_pesq(ref, deg, rate):
    """
    PESQ 점수 계산 함수
    """
    score_wb = pesq(rate, ref, deg, mode='wb')  # 와이드밴드 PESQ 계산
    return score_wb

def evaluate_stoi(ref, deg, rate):
    """
    STOI 점수 계산 함수
    """
    score_stoi = stoi(ref, deg, rate, extended=False)  # 기본 STOI 계산
    return score_stoi

def evaluate_snr(ref, deg):
    """
    SNR 점수 계산 함수
    """
    min_len = min(len(ref), len(deg))  # 신호 길이 맞추기
    ref = ref[:min_len]
    deg = deg[:min_len]

    # 정규화
    ref = ref / np.max(np.abs(ref))
    deg = deg / np.max(np.abs(deg))

    noise = ref - deg
    signal_power_value = signal_power(ref)
    noise_power_value = signal_power(noise)

    if noise_power_value == 0:
        return float('inf')

    return 10 * np.log10((signal_power_value / noise_power_value) ** 2)

if __name__ == "__main__":
    path = "./baseline"  # 평가할 폴더 경로
    dirs = os.listdir(path)  # 폴더 내 모든 디렉토리 가져오기

    snrs = []  # SNR 점수 리스트
    pesqs = []  # PESQ 점수 리스트
    stois = []  # STOI 점수 리스트

    for dir_name in tqdm(dirs):
        p = os.path.join(path, dir_name)
        
        # 파일 경로 설정
        ref_path = os.path.join(p, f"style{dir_name[-1]}.wav")  # style1.wav, style2.wav 등
        deg_path = os.path.join(p, f"noisy_style{dir_name[-1]}.wav")  # noisy_style1.wav, noisy_style2.wav 등

        if os.path.exists(ref_path) and os.path.exists(deg_path):  # 파일 존재 확인
            # librosa를 사용하여 오디오 로드
            ref, ref_rate = librosa.load(ref_path, sr=16000)
            deg, deg_rate = librosa.load(deg_path, sr=16000)
            assert ref_rate == deg_rate, "Sampling rates must be the same"

            # PESQ
            pesq_score = evaluate_pesq(ref, deg, ref_rate)
            pesqs.append(pesq_score)

            # STOI
            stoi_score = evaluate_stoi(ref, deg, ref_rate)
            stois.append(stoi_score)

            # SNR
            snr_score = evaluate_snr(ref, deg)
            snrs.append(snr_score)

            # 개별 결과 출력
            print(f"{dir_name}: SNR: {snr_score:.2f} dB, PESQ (WB): {pesq_score:.2f}, STOI: {stoi_score:.2f}")

    # 평균 결과 출력
    print("\nOverall Average Metrics:")
    print(f"Average SNR: {np.mean(snrs):.2f} dB")
    print(f"Average PESQ (WB): {np.mean(pesqs):.2f}")
    print(f"Average STOI: {np.mean(stois):.2f}")
