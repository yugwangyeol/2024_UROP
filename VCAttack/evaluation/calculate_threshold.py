import os
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from tqdm import tqdm
import argparse

class ThresholdCalculator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # SpeechBrain의 ECAPA-TDNN 모델 로드
        self.speaker_encoder = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/asv"
        )

    def extract_embedding(self, file_path):
        """음성 파일로부터 화자 임베딩 추출"""
        signal, sr = torchaudio.load(file_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        signal = signal.to(self.device)
        embedding = self.speaker_encoder.encode_batch(signal)
        
        return embedding.squeeze(0).detach().cpu().numpy()

    def cosine_similarity(self, embedding1, embedding2):
        """두 임베딩 간의 코사인 유사도 계산"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def compute_eer(self, genuine_scores, impostor_scores):
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
        
        return eer, optimal_threshold, fars[min_diff_idx], frrs[min_diff_idx]

    def calculate_threshold(self, vctk_path, utterances_per_speaker=256):
        """VCTK 데이터셋으로부터 EER 기준 임계값 계산"""
        print("Loading VCTK dataset...")
        dataset = torchaudio.datasets.VCTK_092(root=vctk_path, download=False)

        # 1. 화자별로 utterances_per_speaker개의 발화 수집
        speaker_files = {}
        for waveform, _, _, speaker_id, utterance_id in dataset:
            if speaker_id not in speaker_files:
                speaker_files[speaker_id] = []
            if len(speaker_files[speaker_id]) < utterances_per_speaker:
                # mic1, mic2 버전 모두 추가
                for mic in ['mic1', 'mic2']:
                    file_path = os.path.join(vctk_path, "VCTK-Corpus-0.92/wav48_silence_trimmed", speaker_id, f"{speaker_id}_{utterance_id}_{mic}.flac")
                    speaker_files[speaker_id].append(file_path)

        genuine_scores = []  # 긍정 샘플 (같은 화자)
        impostor_scores = [] # 부정 샘플 (다른 화자)
        
        print("Computing similarity scores...")
        # 2. 각 화자에 대해 긍정/부정 샘플 생성
        for speaker_id in tqdm(speaker_files.keys(), desc="Processing speakers"):
            files = speaker_files[speaker_id]
            if len(files) < utterances_per_speaker:
                continue
                
            # 발화를 두 그룹으로 나눔 (각 utterances_per_speaker/2개)
            half = utterances_per_speaker // 2
            first_group = files[:half]
            second_group = files[half:]
            
            # Positive samples: 첫 그룹과 두 번째 그룹에서 랜덤 선택
            for _ in range(half):
                wav1 = np.random.choice(first_group)
                wav2 = np.random.choice(second_group)
                emb1 = self.extract_embedding(wav1).flatten()
                emb2 = self.extract_embedding(wav2).flatten()
                genuine_scores.append(self.cosine_similarity(emb1, emb2))
            
            # Negative samples: 첫 그룹과 다른 화자의 발화를 랜덤 선택
            other_speakers = list(set(speaker_files.keys()) - {speaker_id})
            for _ in range(half):
                wav1 = np.random.choice(first_group)
                other_spk = np.random.choice(other_speakers)
                wav2 = np.random.choice(speaker_files[other_spk])
                emb1 = self.extract_embedding(wav1).flatten()
                emb2 = self.extract_embedding(wav2).flatten()
                impostor_scores.append(self.cosine_similarity(emb1, emb2))
        
        # 3. EER 지점의 임계값 계산
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)

        print("\nComputing EER and threshold...")
        eer, optimal_threshold, far, frr = self.compute_eer(genuine_scores, impostor_scores)

        # 결과 출력
        print(f"\nThreshold Calculation Results:")
        print(f"Equal Error Rate (EER): {eer*100:.2f}%")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"FAR at EER: {far*100:.2f}%")
        print(f"FRR at EER: {frr*100:.2f}%")
        print(f"Total genuine pairs evaluated: {len(genuine_scores)}")
        print(f"Total impostor pairs evaluated: {len(impostor_scores)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return optimal_threshold

def main():
    parser = argparse.ArgumentParser(description='Calculate threshold from VCTK dataset')
    parser.add_argument('--vctk_path', type=str, default='data',
                      help='Path to VCTK dataset for threshold calculation (optional)')
    parser.add_argument('--utterances_per_speaker', type=int, default=256,
                      help='Number of utterances to use per speaker')
    args = parser.parse_args()

    calculator = ThresholdCalculator(device='cuda')
    threshold = calculator.calculate_threshold(
        vctk_path=args.vctk_path,
        utterances_per_speaker=args.utterances_per_speaker
    )
    
    print(f"\nFinal calculated threshold: {threshold:.3f}")

if __name__ == "__main__":
    main()