import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
from models import NoiseEncoder

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# input_data 폴더에 있는 이미지 로드
image_path = '/home/work/rvc/experiment/input/bird_012.jpg'  # 이미지 경로를 실제 파일명으로 교체
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # 배치 차원 추가
image = image.to(device)

# VGG16 모델 불러오기
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

# 학습된 NoiseEncoder 불러오기
noise_encoder = NoiseEncoder().to(device)
# 학습된 모델 파라미터를 로드합니다. 실제 경로로 교체하세요
noise_encoder.load_state_dict(torch.load('/home/work/rvc/experiment/saved_noise_encoder.pth'))
noise_encoder.eval()

# 정확도 평가 함수
def evaluate_single_image_accuracy(model, image, label):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == label).sum().item()
    return 100 * correct

# CIFAR-10 클래스와 레이블 매핑
cifar10_classes = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
}

# 레이블 설정 (이미지의 실제 클래스에 맞게 설정)
# 예시: "cat" 클래스의 이미지를 사용할 경우, label = torch.tensor([3])
label = torch.tensor([3]).to(device)  # 클래스에 맞게 변경 필요

# 1. 원본 이미지의 정확도 평가
original_accuracy = evaluate_single_image_accuracy(vgg16, image, label)

# 2. 노이즈가 추가된 이미지 생성 및 평가
with torch.no_grad():
    noise = noise_encoder(image)
    noisy_image = torch.clamp(image + noise, 0, 1)

# 3. 노이즈가 추가된 이미지의 정확도 평가
noisy_accuracy = evaluate_single_image_accuracy(vgg16, noisy_image, label)

# 결과 출력
print(f"원본 이미지 정확도: {original_accuracy:.2f}%")
print(f"노이즈가 추가된 이미지 정확도: {noisy_accuracy:.2f}%")

# 4. 원본 이미지, 노이즈가 추가된 이미지, 정확도를 포함한 결과 이미지 저장
def save_comparison_image(original_image, noisy_image, original_accuracy, noisy_accuracy):
    plt.figure(figsize=(12, 6))
    
    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.cpu().squeeze().permute(1, 2, 0))
    plt.title(f"Original Image\nAccuracy: {original_accuracy:.2f}%")
    plt.axis('off')
    
    # 노이즈가 추가된 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image.cpu().squeeze().permute(1, 2, 0))
    plt.title(f"Noisy Image\nAccuracy: {noisy_accuracy:.2f}%")
    plt.axis('off')
    
    # 저장
    plt.savefig("comparison_image.png")
    plt.close()

# 정확도와 이미지를 비교하는 이미지 저장
save_comparison_image(image, noisy_image, original_accuracy, noisy_accuracy)

print("결과 이미지가 저장되었습니다.")
