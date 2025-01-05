import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.transforms as T
import wandb

def train_noise_encoder(generator, discriminator, extractor, dataloader, optimizer_G, optimizer_D, num_epochs, batch_size, device, lambda_wav, lambda_emb):
    # wandb 초기화
    wandb.init(project="Discriminator Conv2d")
    
    generator.train()
    discriminator.train()
    extractor.eval()  # WavLM 모델 고정
    
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)
    
    for epoch in range(num_epochs):
        epoch_wav_loss = 0.0
        epoch_emb_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)
            
            # -----------------
            # Generator 학습
            # -----------------
            optimizer_G.zero_grad()
            
            # Generator에서 노이즈 생성 후 Wav에 추가
            noise = generator(waveforms)  # [batch_size, 1, time]
            noisy_waveforms = torch.clamp(waveforms + noise, -1, 1)

            # Mel 변환
            mel_spectrograms = mel_transform(waveforms)  # 원본 mel
            noisy_mel_spectrograms = mel_transform(noisy_waveforms)  # 노이즈가 추가된 mel

            # Discriminator가 mel 스펙트로그램 상에서 판단
            validity_fake = discriminator(noisy_mel_spectrograms)
            valid = torch.ones_like(validity_fake) # 노이지 mel

            # GAN 손실 계산 (Generator는 Discriminator가 noise를 1이라고 판단하도록 학습)
            g_loss = F.binary_cross_entropy(validity_fake, valid)

            # 이미지 손실
            wav_loss = F.mse_loss(noisy_waveforms, waveforms)
            
            # 임베딩 손실 (WavLM을 통한 L2 distance)
            with torch.no_grad():
                extractor_original_output = extractor(waveforms.squeeze(1)).last_hidden_state
                extractor_noisy_output = extractor(noisy_waveforms.squeeze(1)).last_hidden_state
            emb_loss = -torch.norm(extractor_noisy_output - extractor_original_output, p=2, dim=-1).mean()

            # 최종 손실 계산
            total_loss = lambda_wav * wav_loss + lambda_emb * emb_loss + g_loss
            total_loss.backward()
            optimizer_G.step()

            # -----------------
            # Discriminator 학습
            # -----------------
            optimizer_D.zero_grad()

            # Discriminator가 real mel과 fake mel을 구분하도록 학습
            validity_fake = discriminator(noisy_mel_spectrograms.detach())
            validity_real = discriminator(mel_spectrograms.detach())
            valid = torch.ones_like(validity_real)
            fake = torch.zeros_like(validity_fake)
            d_real_loss = F.binary_cross_entropy(validity_real, valid)
            d_fake_loss = F.binary_cross_entropy(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # 각 손실 값 기록
            epoch_wav_loss += wav_loss.item()
            epoch_emb_loss += emb_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # wandb에 평균 손실 값 기록
        wandb.log({
            "epoch": epoch + 1,
            "g_loss": epoch_g_loss / len(dataloader),
            "d_loss": epoch_d_loss / len(dataloader),
            "wav_loss": epoch_wav_loss / len(dataloader),
            "emb_loss": epoch_emb_loss / len(dataloader),
            "total_loss": (epoch_g_loss + epoch_d_loss) / len(dataloader)
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {epoch_g_loss / len(dataloader):.4f}, "
              f"D Loss: {epoch_d_loss / len(dataloader):.4f}, "
              f"Wav Loss: {epoch_wav_loss / len(dataloader):.4f}, "
              f"Emb Loss: {epoch_emb_loss / len(dataloader):.4f}")
    
    return epoch_wav_loss / len(dataloader), epoch_emb_loss / len(dataloader), epoch_g_loss / len(dataloader)