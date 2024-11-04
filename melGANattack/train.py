import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.transforms as T
import wandb


def train_noise_encoder(generator, discriminator, extractor, dataloader, optimizer_G, optimizer_D, num_epochs, batch_size, device):
    generator.train()
    discriminator.train()

    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)

            # -----------------
            # Generator 학습
            # -----------------
            optimizer_G.zero_grad()

            # Mel 변환
            mel_spectrograms = mel_transform(waveforms)  # 원본 mel

            # Generator에서 노이즈 생성 후 Mel에 추가
            noise = generator(mel_spectrograms)
            noisy_mel_spectrograms = mel_spectrograms + noise


            # Discriminator가 mel 스펙트로그램 상에서 판단
            validity_fake = discriminator(noisy_mel_spectrograms)  # Generator 결과만 사용
            valid = torch.ones_like(validity_fake) # 노이지 mel

            # GAN 손실 계산 (Generator는 Discriminator가 noise를 1이라고 판단하도록 학습한다.) 
            g_loss = F.binary_cross_entropy(validity_fake, valid) # valid : 1 , validiry_fake : D(noisy mel)

            g_loss.backward()
            optimizer_G.step()

            # -----------------
            # Discriminator 학습
            # -----------------
            optimizer_D.zero_grad()

            # Discriminator가 real mel과 fake mel을 구분하도록 학습
            validity_real = discriminator(mel_spectrograms.detach())  # detach()로 그래프 끊기
            validity_fake = discriminator(noisy_mel_spectrograms.detach())  # detach()로 그래프 끊기

            valid = torch.ones_like(validity_real)
            fake = torch.zeros_like(validity_fake)

            d_real_loss = F.binary_cross_entropy(validity_real, valid)
            d_fake_loss = F.binary_cross_entropy(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()  # retain_graph=True 제거
            optimizer_D.step()

            # 각 손실 값 기록
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # wandb에 평균 손실 값 기록
        wandb.log({
            "epoch": epoch + 1,
            "g_loss": epoch_g_loss / len(dataloader),
            "d_loss": epoch_d_loss / len(dataloader),

        })

        print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {epoch_g_loss / len(dataloader):.4f}, "
              f"D Loss: {epoch_d_loss / len(dataloader):.4f}")

    return d_loss.item(), g_loss.item()
