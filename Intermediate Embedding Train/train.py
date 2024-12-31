import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.transforms as T
import wandb
wandb.init(project="mel_GAN_noise_attack")

def train_noise_encoder(generator, discriminator, extractor, dataloader, optimizer_G, optimizer_D, num_epochs, batch_size, device, lambda_wav, lambda_emb, target_layers=[12,23]):  # target_layers 파라미터 추가
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
        epoch_emb_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)
            
            # Generator 학습
            optimizer_G.zero_grad()
            noise = generator(waveforms)
            noisy_waveforms = torch.clamp(waveforms + noise, -1, 1)
            
            # Mel 변환 및 Discriminator 
            mel_spectrograms = mel_transform(waveforms)
            noisy_mel_spectrograms = mel_transform(noisy_waveforms)
            validity_fake = discriminator(noisy_mel_spectrograms.reshape(batch_size, 1, -1))
            valid = torch.ones_like(validity_fake)
            g_loss = F.binary_cross_entropy(validity_fake, valid)
            
            # Wav 손실
            wav_loss = F.mse_loss(noisy_waveforms, waveforms)
            
            # 임베딩 손실 계산 - 중간 layer에서
            with torch.no_grad():
                # output_hidden_states=True로 설정하여 모든 layer의 hidden states를 얻음
                original_output = extractor(waveforms.squeeze(1), output_hidden_states=True)
                noisy_output = extractor(noisy_waveforms.squeeze(1), output_hidden_states=True)
                
                # hidden_states는 tuple 형태로, (input_embedding, layer1, layer2, ..., last_layer)
                original_hidden_states = original_output.hidden_states
                noisy_hidden_states = noisy_output.hidden_states
                
                # 선택한 layer들의 embedding에 대해 거리 계산
                emb_loss = 0
                for layer_idx in target_layers:
                    original_layer_emb = original_hidden_states[layer_idx]
                    noisy_layer_emb = noisy_hidden_states[layer_idx]
                    # L2 거리를 계산하고 멀어지도록 음수 부호
                    layer_distance = -torch.norm(noisy_layer_emb - original_layer_emb, p=2, dim=-1).mean()
                    emb_loss += layer_distance
                
                # layer 개수로 정규화
                emb_loss = emb_loss / len(target_layers)
            
            # 최종 손실 계산
            total_loss = lambda_wav * wav_loss + lambda_emb * emb_loss + g_loss
            total_loss.backward()
            optimizer_G.step()
            
            # Discriminator 학습 (이하 동일)
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
            
            # 손실 값 기록
            epoch_wav_loss += wav_loss.item()
            epoch_emb_loss += emb_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
        # wandb logging
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