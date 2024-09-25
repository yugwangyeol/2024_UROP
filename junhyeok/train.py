import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Model
import wandb  # wandb 추가

# wandb 초기화
wandb.init(project="noise_DAE_project")

def cosine_similarity_loss(x, y):
    cos_sim = F.cosine_similarity(x, y)
    return cos_sim.mean()

def transform_data(waveform, device):
    # waveform의 차원 [batch_size, num_channels, sequence_length]으로 변환
    return waveform.squeeze(1).to(device)  # [batch_size, sequence_length]

def train_noise_encoder(noise_encoder, wav2vec2, dataloader, optimizer, num_epochs, device, lambda_img=0.95, lambda_emb=0.05):
    noise_encoder.train()  # 학습 모드 전환

    for epoch in range(num_epochs):
        epoch_img_loss = 0.0
        epoch_emb_loss = 0.0
        epoch_total_loss = 0.0

        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)

            waveforms = transform_data(waveforms, device)

            # NoiseEncoder 통과
            noise = noise_encoder(waveforms.unsqueeze(1)).to(device)
            # [batch_size, time]->[batch_size, 1, time]
            # noise_encoder의 입력 size에 맞춤.
            noisy_waveforms = torch.clamp(waveforms.unsqueeze(1) + noise, 0, 1).to(device)  # [batch_size, 1, time]
            # 원본 waveforms에 noise_encoder에서 생성된 노이즈를 더함.
            # waveform의 범위가 보통 [-1,1] or [0,1]사이 값이기 때문임.
            # Wav2Vec 2.0 모델을 사용하여 임베딩 계산
            with torch.no_grad():
                wav2vec2_original_output = wav2vec2(waveforms).last_hidden_state.to(device)
                wav2vec2_noisy_output = wav2vec2(noisy_waveforms.squeeze(1)).last_hidden_state.to(device)
                # squeeze : 다시 원본 상태로 돌려놓기

            # Loss 계산
            img_loss = torch.nn.functional.mse_loss(noisy_waveforms, waveforms.unsqueeze(1)).to(device)
            emb_loss = cosine_similarity_loss(wav2vec2_noisy_output, wav2vec2_original_output).to(device)
            total_loss = lambda_img * img_loss + lambda_emb * emb_loss

            # 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 손실 값 저장
            epoch_img_loss += img_loss.item()
            epoch_emb_loss += emb_loss.item()
            epoch_total_loss += total_loss.item()

        # wandb 로그 기록
        wandb.log({
            "epoch": epoch + 1,
            "Spectrogram Loss": epoch_img_loss / len(dataloader),
            "Embedding Loss": epoch_emb_loss / len(dataloader),
            "Total Loss": epoch_total_loss / len(dataloader),
        })

    return epoch_img_loss / len(dataloader), epoch_emb_loss / len(dataloader), epoch_total_loss / len(dataloader)
