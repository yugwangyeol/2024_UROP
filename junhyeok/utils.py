import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

def plot_learning_curve(img_losses, emb_losses, total_losses):
    plt.figure()

    plt.plot(img_losses, label='Image Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Image Loss over Epochs')
    plt.legend()
    plt.savefig('img_loss_curve.png')
    plt.clf()

    plt.plot(emb_losses, label='Embedding Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Embedding Loss over Epochs')
    plt.legend()
    plt.savefig('emb_loss_curve.png')
    plt.clf()

    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss over Epochs')
    plt.legend()
    plt.savefig('total_loss_curve.png')
    plt.clf()

def save_noisy_images(noise_encoder, dataloader, device, num_images=5):
    noise_encoder.eval()
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        noise = noise_encoder(images)
        noisy_images = torch.clamp(images + noise, 0, 1)

    # 원본 이미지, 노이즈 추가 이미지, 노이즈 자체를 격자로 저장
    img_grid = vutils.make_grid(
        torch.cat([images, noisy_images, noise], dim=0), 
        nrow=num_images
    )
    
    vutils.save_image(img_grid, "noisy_images_comparison.png", normalize=True)

    noise_encoder.train()

