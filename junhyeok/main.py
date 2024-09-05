import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import NoiseEncoder, get_vgg16
from train import train_noise_encoder
from utils import plot_learning_curve, save_noisy_images

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터
batch_size = 64
num_epochs = 50
learning_rate = 0.001
lambda_emb = 0.05
lambda_img = 0.95

# 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
vgg16 = get_vgg16().to(device)
noise_encoder = NoiseEncoder().to(device)
torch.save(noise_encoder.state_dict(), "saved_noise_encoder.pth")
# 옵티마이저 설정
optimizer = optim.Adam(noise_encoder.parameters(), lr=learning_rate)

# 학습 실행
img_losses, emb_losses, total_losses = train_noise_encoder(noise_encoder, vgg16, train_loader, optimizer, num_epochs, device, lambda_img=lambda_img, lambda_emb=lambda_emb)
# 학습 곡선 그리기
plot_learning_curve(img_losses, emb_losses, total_losses)

# 노이즈가 추가된 이미지 저장
save_noisy_images(noise_encoder, train_loader, device)

print("학습 완료!")