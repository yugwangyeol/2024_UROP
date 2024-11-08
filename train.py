import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.transforms as T
import wandb

# wandb 초기화
wandb.init(project="wavGANattack")

REAL_LABEL_SMOOTH = 0.9
FAKE_LABEL_SMOOTH = 0.1

def train_noise_encoder(generator, discriminator, extractor, dataloader, num_epochs, batch_size, device, lambda_wav, lambda_emb):
    generator.train()
    discriminator.train()
    extractor.eval()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

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

        for batch_idx, waveforms in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            waveforms = waveforms.to(device)

            # Generator 학습
            optimizer_G.zero_grad()

            noise = generator(waveforms)
            noisy_waveforms = torch.clamp(waveforms + noise, -1, 1)

            mel_spectrograms = mel_transform(waveforms)
            noisy_mel_spectrograms = mel_transform(noisy_waveforms)

            validity_fake = discriminator(noisy_mel_spectrograms.reshape(batch_size, 1, -1))
            
            valid_g = torch.ones_like(validity_fake)
            g_loss = F.binary_cross_entropy(validity_fake, valid_g)
            wav_loss = F.mse_loss(noisy_waveforms, waveforms)

            with torch.no_grad():
                extractor_original_output = extractor(waveforms.squeeze(1)).last_hidden_state
                extractor_noisy_output = extractor(noisy_waveforms.squeeze(1)).last_hidden_state
            emb_loss = -torch.norm(extractor_noisy_output - extractor_original_output, p=2, dim=-1).mean()

            total_loss = lambda_wav * wav_loss + lambda_emb * emb_loss + g_loss
            total_loss.backward()
            optimizer_G.step()

            # Discriminator 학습 (2번에 1번)
            if batch_idx % 2 == 0:
                optimizer_D.zero_grad()

                validity_real = discriminator(mel_spectrograms.detach().reshape(batch_size, 1, -1))
                validity_fake = discriminator(noisy_mel_spectrograms.detach().reshape(batch_size, 1, -1))

                # Discriminator에만 Label Smoothing 적용
                valid = torch.ones_like(validity_fake) * REAL_LABEL_SMOOTH
                fake = torch.zeros_like(validity_fake) + FAKE_LABEL_SMOOTH

                d_real_loss = F.binary_cross_entropy(validity_real, valid)
                d_fake_loss = F.binary_cross_entropy(validity_fake, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                epoch_d_loss += d_loss.item()

            epoch_wav_loss += wav_loss.item()
            epoch_emb_loss += emb_loss.item()
            epoch_g_loss += g_loss.item()

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

    return wav_loss.item(), emb_loss.item(), g_loss.item()