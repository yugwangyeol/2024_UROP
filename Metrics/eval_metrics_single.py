import numpy as np
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi

def evaluate_pesq(ref_path, deg_path):
    """
    PESQ(Perceptual Evaluation of Speech Quality) 점수를 계산하는 함수
    
    Args:
        ref_path (str): 원본 오디오 파일 경로
        deg_path (str): 변형된 오디오 파일 경로
        
    Returns:
        dict: 와이드밴드와 내로우밴드 PESQ 점수
    """
    # 파일 읽기
    ref_rate, ref = wavfile.read(ref_path)
    deg_rate, deg = wavfile.read(deg_path)
    
    # 샘플링 레이트 확인
    assert ref_rate == deg_rate, "Sampling rates of ref and deg must be the same."
    
    # PESQ 점수 계산
    score_wb = pesq(ref_rate, ref, deg, mode='wb')
    score_nb = pesq(ref_rate, ref, deg, mode='nb')
    
    return {
        'wb_pesq': score_wb,
        'nb_pesq': score_nb
    }

def evaluate_stoi(ref_path, deg_path):
    """
    STOI(Short-Time Objective Intelligibility) 점수를 계산하는 함수
    
    Args:
        ref_path (str): 원본 오디오 파일 경로
        deg_path (str): 변형된 오디오 파일 경로
        
    Returns:
        dict: STOI와 ESTOI 점수
    """
    # 파일 읽기
    ref_rate, ref = wavfile.read(ref_path)
    deg_rate, deg = wavfile.read(deg_path)
    
    # 샘플링 레이트 확인
    assert ref_rate == deg_rate, "Sampling rates of ref and deg must be the same."
    
    # 신호 길이 맞추기
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # STOI 계산
    score_stoi = stoi(ref, deg, ref_rate, extended=False)
    score_estoi = stoi(ref, deg, ref_rate, extended=True)
    
    return {
        'stoi': score_stoi,
        'estoi': score_estoi
    }

def evaluate_snr(ref_path, deg_path):
    """
    SNR(Signal-to-Noise Ratio)을 계산하는 함수
    
    Args:
        ref_path (str): 원본 오디오 파일 경로
        deg_path (str): 변형된 오디오 파일 경로
        
    Returns:
        float: SNR 값 (dB)
    """
    # 파일 읽기
    _, ref = wavfile.read(ref_path)
    _, deg = wavfile.read(deg_path)
    
    # 신호 길이 맞추기
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]
    
    # 정규화
    ref = ref / np.max(np.abs(ref))
    deg = deg / np.max(np.abs(deg))
    
    # SNR 계산
    noise = ref - deg
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    ref_path = "./audio/style0.wav"
    deg_path = "./audio/noisy_style0.wav"
    
    # 1. PESQ 평가
    pesq_scores = evaluate_pesq(ref_path, deg_path)
    print(f"Wideband PESQ Score: {pesq_scores['wb_pesq']:.2f}")
    # print(f"Narrowband PESQ Score: {pesq_scores['nb_pesq']:.2f}")
    
    # 2. STOI 평가
    stoi_scores = evaluate_stoi(ref_path, deg_path)
    print(f"STOI Score: {stoi_scores['stoi']:.2f}")
    # print(f"ESTOI Score: {stoi_scores['estoi']:.2f}")
    
    # 3. SNR 평가
    snr = evaluate_snr(ref_path, deg_path)
    print(f"SNR: {snr:.2f} dB")
    
    