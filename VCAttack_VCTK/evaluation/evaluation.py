import os
import torch
import torchaudio
import numpy as np
import librosa
import json
from pesq import pesq
from pystoi import stoi
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from typing import Dict, Tuple, List
import argparse

class UnifiedEvaluator:
    def __init__(self, device: str = 'cuda', threshold_path: str = 'asv_threshold.json'):
        """
        통합 평가를 위한 클래스 초기화
        Args:
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
            threshold_path: ASV 임계값을 저장할 파일 경로
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold_path = threshold_path
        print(f"Using device: {self.device}")
        
        # ECAPA-TDNN 모델 로드
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="evaluation/pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.speaker_encoder.eval()
        self.speaker_encoder = self.speaker_encoder.to(self.device)

    def load_threshold(self) -> Tuple[float, float]:
        """
        저장된 임계값과 EER 로드
        Returns:
            (threshold, eer) 또는 (None, None)
        """
        try:
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, 'r') as f:
                    data = json.load(f)
                return data['threshold'], data['eer']
            return None, None
        except Exception as e:
            print(f"Error loading threshold: {str(e)}")
            return None, None

    def save_threshold(self, threshold: float, eer: float):
        """
        임계값과 EER 저장
        """
        try:
            with open(self.threshold_path, 'w') as f:
                json.dump({
                    'threshold': threshold,
                    'eer': eer,
                    'timestamp': str(np.datetime64('now'))
                }, f, indent=4)
        except Exception as e:
            print(f"Error saving threshold: {str(e)}")

    def get_or_calculate_threshold(self, vctk_path: str, utterances_per_speaker: int = 256) -> Tuple[float, float]:
        """
        저장된 임계값을 로드하거나 새로 계산
        Args:
            vctk_path: VCTK 데이터셋 경로
            utterances_per_speaker: 각 화자당 사용할 최대 발화 수
        Returns:
            (threshold, eer)
        """
        threshold, eer = self.load_threshold()
        if threshold is not None and eer is not None:
            print(f"Loaded existing threshold: {threshold:.3f} (EER: {eer:.3f})")
            return threshold, eer
        
        print("Calculating new threshold from VCTK dataset...")
        threshold, eer = self.find_threshold_from_vctk(vctk_path, utterances_per_speaker)
        self.save_threshold(threshold, eer)
        print(f"Calculated and saved new threshold: {threshold:.3f} (EER: {eer:.3f})")
        return threshold, eer

    def evaluate_pesq(self, ref, deg, rate):
        return pesq(rate, ref, deg, mode='wb')

    def evaluate_stoi(self, ref, deg, rate):
        return stoi(ref, deg, rate, extended=False)

    def evaluate_snr(self, ref, deg):
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        noise = deg - ref
        signal_power = np.mean(ref ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        if signal_power == 0:
            return float('-inf')
            
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        오디오 데이터를 전처리하여 16kHz로 샘플링
        """
        waveform = waveform.to('cpu')
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        return waveform.to(self.device)

    def extract_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        오디오 데이터에서 임베딩 추출
        """
        try:
            with torch.no_grad():
                feats = self.speaker_encoder.mods.compute_features(waveform)
                embeddings = self.speaker_encoder.mods.embedding_model(feats)
                return embeddings.to(self.device)
        except Exception as e:
            print(f"Error in extract_embeddings: {str(e)}")
            raise

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        두 임베딩 간의 코사인 유사도를 계산
        """
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).mean().item()

    def find_threshold_from_vctk(self, vctk_path: str, utterances_per_speaker: int = 256) -> Tuple[float, float]:
        """
        VCTK 데이터셋을 사용해 임계값과 EER 계산
        """
        dataset = torchaudio.datasets.VCTK_092(root=vctk_path, download=False)
        
        speaker_utterances: Dict[str, List[torch.Tensor]] = {}
        for waveform, sample_rate, _, speaker_id, _ in dataset:
            if speaker_id not in speaker_utterances:
                speaker_utterances[speaker_id] = []
            if len(speaker_utterances[speaker_id]) < utterances_per_speaker:
                waveform = self.preprocess_audio(waveform, sample_rate)
                speaker_utterances[speaker_id].append(waveform)
        
        genuine_scores = []
        impostor_scores = []
        
        for speaker_id, utterances in speaker_utterances.items():
            half = len(utterances) // 2
            for i in range(half):
                emb1 = self.extract_embeddings(utterances[i])
                emb2 = self.extract_embeddings(utterances[i + half])
                genuine_scores.append(self.compute_similarity(emb1, emb2))
            
            other_speakers = list(set(speaker_utterances.keys()) - {speaker_id})
            for _ in range(half):
                other_id = np.random.choice(other_speakers)
                other_utterance = np.random.choice(speaker_utterances[other_id])
                emb1 = self.extract_embeddings(utterances[i])
                emb2 = self.extract_embeddings(other_utterance)
                impostor_scores.append(self.compute_similarity(emb1, emb2))
        
        eer, threshold = self.compute_eer(np.array(genuine_scores), np.array(impostor_scores))
        return threshold, eer

    def compute_eer(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
        """
        EER 및 최적 임계값 계산
        """
        thresholds = np.linspace(0, 1, 1000)
        fars = []
        frrs = []
        
        for threshold in thresholds:
            far = np.mean(impostor_scores >= threshold)
            frr = np.mean(genuine_scores < threshold)
            fars.append(far)
            frrs.append(frr)
        
        fars = np.array(fars)
        frrs = np.array(frrs)
        
        min_diff_idx = np.argmin(np.abs(fars - frrs))
        eer = (fars[min_diff_idx] + frrs[min_diff_idx]) / 2
        optimal_threshold = thresholds[min_diff_idx]
        
        return eer, optimal_threshold

    def verify_speaker(self, wav1: torch.Tensor, wav2: torch.Tensor, threshold: float) -> bool:
        """
        두 오디오 간 화자 유사성 검증
        """
        emb1 = self.extract_embeddings(wav1)
        emb2 = self.extract_embeddings(wav2)
        similarity = self.compute_similarity(emb1, emb2)
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

                snr_score = self.evaluate_snr(ref, deg)
                pesq_score = self.evaluate_pesq(ref, deg, ref_rate)
                stoi_score = self.evaluate_stoi(ref, deg, ref_rate)

                results['snrs'].append(snr_score)
                results['pesqs'].append(pesq_score)
                results['stois'].append(stoi_score)

                x_wav, sr = torchaudio.load(x_path)
                x_prime_wav, sr = torchaudio.load(x_prime_path)
                f_x_prime_t_wav, sr = torchaudio.load(f_x_prime_t_path)

                x_wav = self.preprocess_audio(x_wav, sr)
                x_prime_wav = self.preprocess_audio(x_prime_wav, sr)
                f_x_prime_t_wav = self.preprocess_audio(f_x_prime_t_wav, sr)

                if not self.verify_speaker(f_x_prime_t_wav, x_wav, threshold):
                    results['asr_count'] += 1

                if self.verify_speaker(x_prime_wav, x_wav, threshold):
                    results['psr_count'] += 1

                results['total'] += 1

                # print(f"\nPair {i+1}:")
                # print(f"Reference: {os.path.basename(x_path)}")
                # print(f"Degraded: {os.path.basename(x_prime_path)}")
                # print(f"Converted: {os.path.basename(f_x_prime_t_path)}")
                # print(f"Metrics - SNR: {snr_score:.2f} dB, PESQ: {pesq_score:.2f}, STOI: {stoi_score:.2f}")

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
    vctk_path = "data/VCTK-Corpus-0.92"  
    
    # 통합 평가기 초기화
    evaluator = UnifiedEvaluator(device='cuda')

    # 임계값 고정
    threshold = 0.328  # RW-Voiceshield의 임계값
    eer = None  
    
    # # 임계값 로드 또는 계산
    # threshold, eer = evaluator.get_or_calculate_threshold(vctk_path)

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