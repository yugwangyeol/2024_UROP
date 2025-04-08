import os
import torch
import torchaudio
import numpy as np
from scipy import stats
import librosa
from pesq import pesq
from pystoi import stoi
from speechbrain.pretrained import SpeakerRecognition
from tqdm import tqdm
from typing import Dict, Tuple
import argparse

class UnifiedEvaluator:
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.speaker_encoder = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="evaluation/pretrained_models/asv"
        )

    def evaluate_pesq(self, ref, deg, rate):
        return pesq(rate, ref, deg, mode='wb')

    def evaluate_stoi(self, ref, deg, rate):
        return stoi(ref, deg, rate, extended=False)

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

    def calculate_statistics(self, data: list) -> Tuple[float, Tuple[float, float]]:
        if not data:
            return 0, (0, 0)
            
        mean = np.mean(data)
        ci = stats.t.interval(confidence=0.95, 
                            df=len(data)-1,
                            loc=mean,
                            scale=stats.sem(data))
                            
        return mean, ci

    def wilson_score_interval(self, success: int, total: int, alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
        if total == 0:
            return 0, (0, 0)
            
        proportion = success / total
        z = stats.norm.ppf(1 - alpha/2)
        
        denominator = 1 + z**2/total
        center = (proportion + z**2/(2*total))/denominator
        
        err = z * np.sqrt((proportion*(1-proportion) + z**2/(4*total))/total)/denominator
        
        ci = (max(0.0, center - err), min(1.0, center + err))
        return proportion, ci

    def evaluate_all_metrics(self, 
                        test_pairs_path: str,
                        test_noisy_pairs_path: str,
                        threshold: float) -> Dict:
        results = {
            'pesqs': [], 'stois': [],
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

                pesq_score = self.evaluate_pesq(ref, deg, ref_rate)
                stoi_score = self.evaluate_stoi(ref, deg, ref_rate)

                results['pesqs'].append(pesq_score)
                results['stois'].append(stoi_score)

                if not self.verify_speaker(f_x_prime_t_path, x_path, threshold):
                    results['asr_count'] += 1

                if self.verify_speaker(x_prime_path, x_path, threshold):
                    results['psr_count'] += 1

                results['total'] += 1

            except Exception as e:
                print(f"\nError processing pair {i+1}: {str(e)}")
                continue

        # PESQ, STOI에 대한 평균, 신뢰구간 계산
        pesq_mean, pesq_ci = self.calculate_statistics(results['pesqs'])
        stoi_mean, stoi_ci = self.calculate_statistics(results['stois'])
        
        # ASR, PSR에 대한 Wilson score interval 계산
        asr_mean, asr_ci = self.wilson_score_interval(results['asr_count'], results['total'])
        psr_mean, psr_ci = self.wilson_score_interval(results['psr_count'], results['total'])

        final_results = {
            'PESQ': {
                'mean': pesq_mean,
                'ci': pesq_ci,
            },
            'STOI': {
                'mean': stoi_mean,
                'ci': stoi_ci,
            },
            'ASR': {
                'mean': asr_mean,
                'ci': asr_ci,
            },
            'PSR': {
                'mean': psr_mean,
                'ci': psr_ci,
            },
            'total_evaluated': results['total']
        }

        return final_results

def main():
    parser = argparse.ArgumentParser(description='VCAttack evaluation')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'TriAAN-VC'], default='FreeVC',
                    help='Type of voice conversion model (FreeVC or TriAAN-VC)')
    parser.add_argument('--attack_type', type=str, choices=['white', 'black'], default='white',
                    help='Type of attack (white-box or black-box)')
    args = parser.parse_args()

    # attack_type을 w/b로 축약
    attack_abbr = 'w' if args.attack_type == 'white' else 'b'

    test_pairs_path = f"data/{args.model}_test_pairs.txt"
    test_noisy_pairs_path = f"data/{args.model}_test_noisy_pairs_{attack_abbr.upper()}.txt"
    
    evaluator = UnifiedEvaluator(device='cuda')
    threshold = 0.328  # RW-Voiceshield의 임계값
    
    print("\nEvaluating all metrics...")
    print(f"Model: {args.model}")
    results = evaluator.evaluate_all_metrics(
        test_pairs_path=test_pairs_path,
        test_noisy_pairs_path=test_noisy_pairs_path,
        threshold=threshold
    )
    
    # 결과 출력
    print("\nEvaluation Results:")
    print(f"Model: {args.model}")
    print(f"Total pairs evaluated: {results['total_evaluated']}")
    
    metrics = ['PESQ', 'STOI', 'ASR', 'PSR']
    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  {results[metric]['mean']:.3f} [{results[metric]['ci'][0]:.3f}, {results[metric]['ci'][1]:.3f}]")

if __name__ == "__main__":
    main()