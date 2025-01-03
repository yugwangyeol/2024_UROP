import torch
from transformers import WavLMModel
from model import Discriminator, Generator
from train import train_noise_encoder
from dataset import get_dataloader
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='음성 적대적 공격 학습')
    parser.add_argument('--data_dir', type=str, default='../../Data',
                      help='데이터셋 저장 경로')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                      help='모델 저장 경로')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='학습률')
    parser.add_argument('--lambda_emb', type=float, default=0.2,
                      help='임베딩 손실 가중치')
    parser.add_argument('--lambda_wav', type=float, default=0.8,
                      help='파형 손실 가중치')
    parser.add_argument('--sample_rate', type=int, default=16000,
                      help='목표 샘플링 레이트')
    parser.add_argument('--duration', type=int, default=3,
                      help='오디오 클립 길이(초)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 저장 경로 설정
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 데이터로더 생성
    train_loader = get_dataloader(
        root=args.data_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration=args.duration
    )

    # 모델 초기화
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 옵티마이저 초기화
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # 학습 실행
    img_losses, emb_losses, gan_losses = train_noise_encoder(
        generator=generator,
        discriminator=discriminator,
        extractor=wavlm,
        dataloader=train_loader,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=device,
        lambda_wav=args.lambda_wav,
        lambda_emb=args.lambda_emb
    )

    print("학습 완료!")

    # 모델 저장
    torch.save(generator.state_dict(), os.path.join(args.save_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.save_dir, 'discriminator.pth'))
    print(f"모델이 {args.save_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()