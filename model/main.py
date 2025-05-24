import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import WavLMModel, HubertModel
from model import Discriminator, Generator
from train import train_noise_encoder
import os
import argparse

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature extractor 로드
def get_feature_extractor(extractor_type, device):
    if extractor_type == "wavlm":
        print("Loading WavLM model...")
        return WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    elif extractor_type == "hubert":
        print("Loading HuBERT model...")
        return HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    else:
        raise ValueError(f"지원하지 않는 특성 추출기: {extractor_type}")

def main():
    parser = argparse.ArgumentParser(description='Train noise encoder with different feature extractors')
    parser.add_argument('--feature_extractor', type=str, choices=['wavlm', 'hubert'], default='wavlm',
                      help='Type of feature extractor (wavlm or hubert)')
    args = parser.parse_args()

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using feature extractor: {args.feature_extractor}")

    # 모델 저장 경로 설정
    save_dir = 'model/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 하이퍼파라미터
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    lambda_emb = 0.2
    lambda_wav = 0.8

    # Dataset 클래스 정의
    class VCTKDataset(Dataset):
        def __init__(self, txt_file, transform=None):
            # txt_file에서 오디오 경로 읽기
            with open(txt_file, "r") as f:
                self.audio_paths = [line.strip() for line in f.readlines()]
            self.transform = transform

        def __len__(self):
            return len(self.audio_paths)

        def __getitem__(self, idx):
            audio_path = self.audio_paths[idx]
            waveform, sample_rate = torchaudio.load(audio_path)  # 오디오 파일 로드
            if self.transform:
                waveform = self.transform(waveform)
            return waveform

    # 데이터 경로
    torchaudio.datasets.VCTK_092(root="data", download=True)
    train_txt_path = "data/train.txt"  # train.txt 파일 경로

    # Dataset 및 DataLoader 생성
    train_dataset = VCTKDataset(train_txt_path)

    # collate_fn 함수 정의
    def collate_fn(batch):
        waveforms = []
        max_length = max(waveform.shape[1] for waveform in batch)
        for waveform in batch:
            if waveform.shape[1] < max_length:
                padding = torch.zeros(1, max_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            waveforms.append(waveform)
        return torch.stack(waveforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 모델 초기화
    feature_extractor = get_feature_extractor(args.feature_extractor, device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 옵티마이저 설정
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # 학습 실행
    print("Train VCTK dataset")
    img_losses, emb_losses, gan_losses = train_noise_encoder(
        generator, discriminator, feature_extractor, train_loader, 
        optimizer_G, optimizer_D, num_epochs, batch_size, device, 
        lambda_wav, lambda_emb
    )

    # 학습 완료 메시지 출력
    print("학습 완료!")

    # 모델 저장
    generator_path = os.path.join(save_dir, f'generator_{args.feature_extractor}.pth')
    discriminator_path = os.path.join(save_dir, f'discriminator_{args.feature_extractor}.pth')
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"모델이 {save_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
