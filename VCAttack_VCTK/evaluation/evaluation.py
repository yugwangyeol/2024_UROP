import os
import torch
import torchaudio
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
from speechbrain.pretrained import SpeakerRecognition
from tqdm import tqdm
from typing import Dict
import argparse

class UnifiedEvaluator:
    def __init__(self, device: str = 'cuda'):
        """
        통합 평가를 위한 클래스 초기화
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # SpeechBrain의 ECAPA-TDNN 모델 로드
        self.speaker_encoder = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="evaluation/pretrained_models/asv"
        )

    def evaluate_pesq(self, ref, deg, rate):
        return pesq(rate, ref, deg, mode='wb')

    def evaluate_stoi(self, ref, deg, rate):
        return stoi(ref, deg, rate, extended=False)

    def evaluate_snr(self, ref, deg):
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        
        signal_power = np.sum(ref ** 2)
        noise_power = np.sum((deg - ref) ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        if signal_power < 1e-10:
            return float('-inf')
            
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def extract_embedding(self, file_path):
        signal, sr = torchaudio.load(file_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        signal = signal.to(self.device)
        embedding = self.speaker_encoder.encode_batch(signal)
        
        return embedding.squeeze(0).detach().cpu().numpy()

    def cosine_similarity(self, embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def verify_speaker(self, wav1_path: str, wav2_path: str, threshold: float) -> bool:
        emb1 = self.extract_embedding(wav1_path).flatten()
        emb2 = self.extract_embedding(wav2_path).flatten()
        similarity = self.cosine_similarity(emb1, emb2)
        return similarity >= threshold

    def evaluate_all_metrics(self, 
                        test_pairs_path: str,
                        test_noisy_pairs_path: str,
                        threshold: float) -> Dict:
        results = {
            'snrs': [], 'pesqs': [], 'stois': [],
            'asr_count': 0, 'psr_count': 0, 'total': 0
        }

        with open(test_pairs_path, 'r') as f:
            original_pairs = [line.strip().split() for line in f.readlines()]
        with open(test_noisy_pairs_path, 'r') as f:
            noisy_pairs = [line.strip().split() for line in f.readlines()]
        
        if not all(len(pair) == 3 for pair in noisy_pairs):
            raise ValueError("Each line in noisy_pairs file must contain three paths")

        if len(original_pairs) != len(noisy_pairs):
            raise ValueError("Number of pairs in both files must match")

        for i in tqdm(range(len(original_pairs)), desc="Evaluating", ncols=70):
            try:
                x_path = original_pairs[i][0]
                x_prime_path = noisy_pairs[i][0]
                f_x_prime_t_path = noisy_pairs[i][2]

                ref, ref_rate = librosa.load(x_path, sr=16000)
                deg, deg_rate = librosa.load(x_prime_path, sr=16000)

                # 음질 평가
                snr_score = self.evaluate_snr(ref, deg)
                pesq_score = self.evaluate_pesq(ref, deg, ref_rate)
                stoi_score = self.evaluate_stoi(ref, deg, ref_rate)

                results['snrs'].append(snr_score)
                results['pesqs'].append(pesq_score)
                results['stois'].append(stoi_score)

                # ASV 기반 평가
                if not self.verify_speaker(f_x_prime_t_path, x_path, threshold):
                    results['asr_count'] += 1

                if self.verify_speaker(x_prime_path, x_path, threshold):
                    results['psr_count'] += 1

                results['total'] += 1

            except Exception as e:
                print(f"\nError processing pair {i+1}: {str(e)}")
                continue

        final_results = {
            'SNR': np.mean(results['snrs']) if results['snrs'] else 0,
            'PESQ': np.mean(results['pesqs']) if results['pesqs'] else 0,
            'STOI': np.mean(results['stois']) if results['stois'] else 0,
            'ASR': results['asr_count'] / results['total'] if results['total'] > 0 else 0,
            'PSR': results['psr_count'] / results['total'] if results['total'] > 0 else 0,
            'total_evaluated': results['total']
        }

        return final_results

def main():
    parser = argparse.ArgumentParser(description='VCAttack evaluation')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'PH'], default='FreeVC',
                      help='Type of voice conversion model (FreeVC or PH)')
    args = parser.parse_args()

    # 파일 경로
    test_pairs_path = f"data/{args.model}_test_pairs.txt"
    test_noisy_pairs_path = f"data/{args.model}_test_noisy_pairs.txt"
    
    # 통합 평가기 초기화
    evaluator = UnifiedEvaluator(device='cuda')

    # 임계값 고정
    threshold = 0.328  # RW-Voiceshield의 임계값
    # threshold = 0.359  # 계산한 임계값

    # 모든 메트릭 평가
    print("\nEvaluating all metrics...")
    print(f"Model: {args.model}")
    results = evaluator.evaluate_all_metrics(
        test_pairs_path=test_pairs_path,
        test_noisy_pairs_path=test_noisy_pairs_path,
        threshold=threshold
    )
    
    # 결과 출력
    print("\nFinal Evaluation Results:")
    print(f"Model: {args.model}")
    print(f"Total pairs evaluated: {results['total_evaluated']}")
    print(f"Average SNR: {results['SNR']:.2f} dB")
    print(f"Average PESQ: {results['PESQ']:.2f}")
    print(f"Average STOI: {results['STOI']:.2f}")
    print(f"ASR: {results['ASR']:.3f}")
    print(f"PSR: {results['PSR']:.3f}")

if __name__ == "__main__":
    main()