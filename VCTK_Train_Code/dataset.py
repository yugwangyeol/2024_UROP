import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class VCTKDataset(Dataset):
    def __init__(self, root, sample_rate=16000, duration=3):
        try:
            self.dataset = torchaudio.datasets.VCTK_092(root=root, download=True)
        except Exception as e:
            try:
                self.dataset = torchaudio.datasets.VCTK(root=root, version="0.92", download=True)
                
        self.sample_rate = sample_rate
        self.target_length = duration * sample_rate
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            waveform, sample_rate, utterance, speaker_id, utterance_id = self.dataset[idx]
        except ValueError:
            waveform, sample_rate, _, _, _, _ = self.dataset[idx]
        
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform

def get_dataloader(root, batch_size=4, sample_rate=16000, duration=3, num_workers=4):
    def collate_fn(batch):
        waveforms = []
        target_length = duration * sample_rate
        
        for waveform in batch:
            current_length = waveform.size(1)
            
            if current_length > target_length:
                # 랜덤한 위치에서 지정된 길이만큼 자르기
                start = torch.randint(0, current_length - target_length, (1,))
                waveform = waveform[:, start:start + target_length]
            elif current_length < target_length:
                # 패딩 추가
                padding = torch.zeros(1, target_length - current_length)
                waveform = torch.cat([waveform, padding], dim=1)
                
            waveforms.append(waveform)
        
        return torch.stack(waveforms)

    dataset = VCTKDataset(root=root, sample_rate=sample_rate, duration=duration)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )