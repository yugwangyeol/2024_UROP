import torch
import torch.nn.functional as F
from tqdm import tqdm



def cosine_similarity_loss(x, y):
    cos_sim = F.cosine_similarity(x, y)
    return cos_sim.mean()
def train_noise_encoder(noise_encoder, vgg16, dataloader, optimizer, num_epochs, device, lambda_img=0.95, lambda_emb=0.05):
    noise_encoder.train()
    img_losses = []
    emb_losses = []
    total_losses = []
    # VGG16 특징 추출기 정의 (한 번만 수행)
    vgg16_features = torch.nn.Sequential(
        vgg16.features,
        vgg16.avgpool,
        torch.nn.Flatten(),
        vgg16.classifier[0]
    )

    for epoch in range(num_epochs):
        epoch_img_loss = 0
        epoch_emb_loss = 0
        epoch_total_loss = 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)

            # 노이즈 생성 및 적용
            noise = noise_encoder(images)
            noisy_images = torch.clamp(images + noise, 0, 1)

            # VGG16으로 예측 (마지막 fc 층 제외)
            original_embedding = vgg16_features(images)
            noisy_embedding = vgg16_features(noisy_images)

            # mse loss
            image_loss = F.mse_loss(images, noisy_images)

            # cosine sim
            embedding_loss = cosine_similarity_loss(original_embedding, noisy_embedding)
            # 전체 손실 계산
            total_loss = lambda_img * image_loss + lambda_emb * embedding_loss

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_img_loss += image_loss.item()
            epoch_emb_loss += embedding_loss.item()
            epoch_total_loss += total_loss.item()

        img_losses.append(epoch_img_loss / len(dataloader))
        emb_losses.append(epoch_emb_loss / len(dataloader))
        total_losses.append(epoch_total_loss / len(dataloader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Image Loss: {img_losses[-1]:.4f}, Embedding Loss: {emb_losses[-1]:.4f}, Total Loss: {total_losses[-1]:.4f}")

    return img_losses, emb_losses, total_losses

