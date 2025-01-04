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
        
        self.speaker_encoder = self.speaker_encoder.to(self.device)
        for param in self.speaker_encoder.parameters():
            param.data = param.data.to(self.device)
    
    def preprocess_audio(self, waveform, sample_rate):
        waveform = waveform.cpu()
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        return waveform.to(self.device)
    
    def extract_embeddings(self, waveform):
        try:
            if isinstance(waveform, list):
                max_length = max(wav.shape[1] for wav in waveform)
                padded_wavs = []
                
                for wav in waveform:
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
            
            waveform = waveform.to(self.device)
            
            with torch.no_grad():
                feats = self.speaker_encoder.mods.compute_features(waveform)
                feats = feats.to(self.device)
                embeddings = self.speaker_encoder.mods.embedding_model(feats)
                embeddings = embeddings.to(self.device)
            
            return embeddings
            
        except Exception as e:
            print(f"Error in extract_embeddings: {str(e)}")
            raise
    
    def compute_scores(self, original_wavs, noisy_wavs, speakers):
        original_embs = self.extract_embeddings(original_wavs)
        noisy_embs = self.extract_embeddings(noisy_wavs)
        
        if len(original_embs.shape) == 2:
            original_embs = original_embs.unsqueeze(1)
        if len(noisy_embs.shape) == 2:
            noisy_embs = noisy_embs.unsqueeze(0)
            
        #print(f"Original embeddings shape: {original_embs.shape}")
        #print(f"Noisy embeddings shape: {noisy_embs.shape}")
        
        similarity_matrix = torch.nn.functional.cosine_similarity(
            original_embs,
            noisy_embs,
        )
        
        genuine_scores = []
        impostor_scores = []

        print(similarity_matrix.shape)
        for i in range(len(speakers)):
            for j in range(len(speakers)):
                score = similarity_matrix[i, j].item()
                if speakers[i] == speakers[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
        
        return np.array(genuine_scores), np.array(impostor_scores)
        
    def compute_eer(self, genuine_scores, impostor_scores):
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
        
        return eer, fars, frrs, thresholds

def evaluate_asv_performance(original_dir, noisy_dir, num_samples=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ASVEvaluator(device=device)
    
    dataset = torchaudio.datasets.VCTK_092(root=original_dir, download=False)
    noisy_list = sorted(os.listdir(noisy_dir))  
    
    num_samples = min(num_samples, len(dataset), len(noisy_list))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    test_wavs = []
    test_speakers = []
    
    for idx in indices:
        waveform, sample_rate, _, speaker_id, _ = dataset[idx]
        waveform = evaluator.preprocess_audio(waveform, sample_rate)
        test_wavs.append(waveform)
        test_speakers.append(speaker_id)
    
    noisy_wavs = []
    for idx in noisy_list[:num_samples]:
        waveform_path = os.path.join(noisy_dir, idx)
        waveform, sample_rate = torchaudio.load(waveform_path)
        waveform = evaluator.preprocess_audio(waveform, sample_rate)
        noisy_wavs.append(waveform)
    
    with torch.no_grad():
        genuine_scores, impostor_scores = evaluator.compute_scores(
            test_wavs, noisy_wavs, test_speakers
        )
    
    eer, fars, frrs, thresholds = evaluator.compute_eer(
        genuine_scores, impostor_scores
    )
    
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

if __name__ == "__main__":
    results = evaluate_asv_performance(
        original_dir="../Data",
        noisy_dir="../Result",
        num_samples=100
    )
    
    print(f"EER: {results['eer']*100:.2f}%")
