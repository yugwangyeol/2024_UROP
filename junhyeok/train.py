import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.transforms as T
import wandb

# wandb 초기화
wandb.init(project="mel_GAN_noise_attack")

def train_noise_encoder(generator, discriminator, extractor, dataloader, optimizer_G, optimizer_D, num_epochs, batch_size, device, lambda_wav, lambda_centroid, lambda_gan):
    # 사전 학습된 centroid 로드
    centroids = torch.load('wavlm_centroids.pt').to(device)
    
    generator.train()
    discriminator.train()
    extractor.eval()


    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)

    for epoch in range(num_epochs):
        epoch_wav_loss = 0.0
        epoch_centroid_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)

            # -----------------
            # Generator 학습 - g_loss와 centroid_loss
            # -----------------
            optimizer_G.zero_grad()

            noise = generator(waveforms)
            noisy_waveforms = torch.clamp(waveforms + noise, -1, 1)

            mel_spectrograms = mel_transform(waveforms)
            noisy_mel_spectrograms = mel_transform(noisy_waveforms)

            validity_fake = discriminator(noisy_mel_spectrograms.reshape(batch_size, 1, -1))
            valid = torch.ones_like(validity_fake)

            g_loss = F.binary_cross_entropy(validity_fake, valid)
            wav_loss = F.mse_loss(noisy_waveforms, waveforms)

            # Centroid loss
            with torch.no_grad():

                original_emb = extractor(waveforms.squeeze(1)).last_hidden_state.mean(dim=1)
                noisy_emb = extractor(noisy_waveforms.squeeze(1)).last_hidden_state.mean(dim=1)
                
                distances_to_centroids = torch.cdist(noisy_emb, centroids)
                farthest_centroid_idx = torch.argmax(distances_to_centroids, dim=1)
                target_centroids = centroids[farthest_centroid_idx]
                
            centroid_loss = F.mse_loss(noisy_emb, target_centroids)
            total_loss_G = lambda_wav * wav_loss + lambda_centroid * centroid_loss + lambda_gan * g_loss

            total_loss_G.backward()
            optimizer_G.step()

            # -----------------
            # Discriminator 학습 - d_loss
            # -----------------
            optimizer_D.zero_grad()

            validity_real = discriminator(mel_spectrograms.detach().reshape(batch_size, 1, -1))
            validity_fake = discriminator(noisy_mel_spectrograms.detach().reshape(batch_size, 1, -1))

            valid = torch.ones_like(validity_real)
            fake = torch.zeros_like(validity_fake)

            d_real_loss = F.binary_cross_entropy(validity_real, valid)
            d_fake_loss = F.binary_cross_entropy(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 손실 기록
            epoch_wav_loss += wav_loss.item()
            epoch_centroid_loss += centroid_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # wandb에 평균 손실 값 기록

        wandb.log({
            "epoch": epoch + 1,
            "wav_loss": epoch_wav_loss / len(dataloader),
            "centroid_loss": epoch_centroid_loss / len(dataloader),
            "g_loss": epoch_g_loss / len(dataloader),
            "d_loss": epoch_d_loss / len(dataloader),
            "total_loss_G": (epoch_wav_loss + epoch_centroid_loss + epoch_g_loss) / len(dataloader),
            "total_loss_D": epoch_d_loss / len(dataloader)
        })

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Wav Loss: {epoch_wav_loss / len(dataloader):.4f}, "
              f"Centroid Loss: {epoch_centroid_loss / len(dataloader):.4f}, "
              f"G Loss: {epoch_g_loss / len(dataloader):.4f}, "
              f"D Loss: {epoch_d_loss / len(dataloader):.4f}")

    return wav_loss.item(), centroid_loss.item(), g_loss.item(), d_loss.item()
