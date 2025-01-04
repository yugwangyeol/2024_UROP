import torch
import torchaudio
from sklearn.model_selection import train_test_split
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import os

class ASVEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # SpeechBrain의 ECAPA-TDNN 모델 로드
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        self.speaker_encoder.eval()
        
        # 모델의 모든 부분이 올바른 디바이스에 있는지 확인
        self.speaker_encoder = self.speaker_encoder.to(self.device)
        for param in self.speaker_encoder.parameters():
            param.data = param.data.to(self.device)
    
    def preprocess_audio(self, waveform, sample_rate):
        """오디오 전처리 함수"""
        # CPU에서 처리
        waveform = waveform.cpu()
        
        # 필요한 경우 리샘플링 (CPU에서)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        # 디바이스로 이동
        return waveform.to(self.device)
    
    def extract_embeddings(self, waveform):
        """음성에서 화자 임베딩 추출"""
        try:
            if isinstance(waveform, list):
                # 패딩을 위해 최대 길이 찾기
                max_length = max(wav.shape[1] for wav in waveform)
                padded_wavs = []
                
                for wav in waveform:
                    # 먼저 디바이스로 이동
                    wav = wav.to(self.device)
                    if wav.shape[1] < max_length:
                        padding = torch.zeros(1, max_length - wav.shape[1], device=self.device)
                        padded_wav = torch.cat([wav, padding], dim=1)
                    else:
                        padded_wav = wav
                    padded_wavs.append(padded_wav)
                
                waveform = torch.stack(padded_wavs).squeeze(1)
            
            if waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            
            # 명시적으로 디바이스 확인
            waveform = waveform.to(self.device)
            
            with torch.no_grad():
                # SpeechBrain의 encode_batch 대신 직접 처리
                feats = self.speaker_encoder.mods.compute_features(waveform)
                feats = feats.to(self.device)
                embeddings = self.speaker_encoder.mods.embedding_model(feats)
                embeddings = embeddings.to(self.device)
            
            return embeddings
            
        except Exception as e:
            print(f"Error in extract_embeddings: {str(e)}")
            raise
    
    def compute_scores(self, original_wavs, noisy_wavs, speakers):
        """원본과 노이즈가 추가된 음성의 코사인 유사도 계산"""
        original_embs = self.extract_embeddings(original_wavs)
        noisy_embs = self.extract_embeddings(noisy_wavs)
        
        # 임베딩 차원 확인 및 조정
        if len(original_embs.shape) == 2:
            original_embs = original_embs.unsqueeze(1)
        if len(noisy_embs.shape) == 2:
            noisy_embs = noisy_embs.unsqueeze(0)
            
        # 차원 출력하여 디버깅
        #print(f"Original embeddings shape: {original_embs.shape}")
        #print(f"Noisy embeddings shape: {noisy_embs.shape}")
        
        # 코사인 유사도 계산
        similarity_matrix = torch.nn.functional.cosine_similarity(
            original_embs,
            noisy_embs,
        )
        
        genuine_scores = []
        impostor_scores = []

        print(similarity_matrix.shape)
        # 각 화자 쌍에 대한 스코어 수집
        for i in range(len(speakers)):
            for j in range(len(speakers)):
                score = similarity_matrix[i, j].item()
                if speakers[i] == speakers[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
        
        return np.array(genuine_scores), np.array(impostor_scores)
        
    def compute_eer(self, genuine_scores, impostor_scores):
        """EER 계산"""
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
        
        # EER 지점 찾기
        min_diff_idx = np.argmin(np.abs(fars - frrs))
        eer = (fars[min_diff_idx] + frrs[min_diff_idx]) / 2
        
        return eer, fars, frrs, thresholds

def evaluate_asv_performance(original_dir, noisy_dir, num_samples=100):
    """ASV 성능 평가 메인 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ASVEvaluator(device=device)
    
    # 데이터 로드 및 전처리
    dataset = torchaudio.datasets.VCTK_092(root=original_dir, download=False)
    noisy_list = sorted(os.listdir(noisy_dir))  # 정렬하여 일관성 유지
    
    # 테스트용 샘플 선택
    num_samples = min(num_samples, len(dataset), len(noisy_list))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    test_wavs = []
    test_speakers = []
    
    for idx in indices:
        waveform, sample_rate, _, speaker_id, _ = dataset[idx]
        # 전처리 및 디바이스 이동
        waveform = evaluator.preprocess_audio(waveform, sample_rate)
        test_wavs.append(waveform)
        test_speakers.append(speaker_id)
    
    # 노이즈가 추가된 음성 로드
    noisy_wavs = []
    for idx in noisy_list[:num_samples]:
        waveform_path = os.path.join(noisy_dir, idx)
        waveform, sample_rate = torchaudio.load(waveform_path)
        # 전처리 및 디바이스 이동
        waveform = evaluator.preprocess_audio(waveform, sample_rate)
        noisy_wavs.append(waveform)
    
    # 스코어 계산
    with torch.no_grad():
        genuine_scores, impostor_scores = evaluator.compute_scores(
            test_wavs, noisy_wavs, test_speakers
        )
    
    # EER 계산
    eer, fars, frrs, thresholds = evaluator.compute_eer(
        genuine_scores, impostor_scores
    )
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'eer': eer,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'fars': fars,
        'frrs': frrs,
        'thresholds': thresholds
    }

# 사용 예시
if __name__ == "__main__":
    results = evaluate_asv_performance(
        original_dir="../Data",
        noisy_dir="../Result",
        num_samples=100
    )
    
    print(f"EER: {results['eer']*100:.2f}%")