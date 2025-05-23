import torch
import torchaudio
import os
import argparse
import numpy as np  
import random      
from model import Generator as NoiseEncoder

def process_audio_file(noise_encoder, input_path, output_path, device):
    # 입력 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)
    # 배치 차원 추가
    waveform = waveform.unsqueeze(0).to(device)
    # 노이즈 적용
    with torch.no_grad():
        noise = noise_encoder(waveform)
        noisy_waveform = torch.clamp(waveform + noise * 0.7, -1, 1)
    # 저장
    torchaudio.save(output_path, noisy_waveform.squeeze(0).cpu(), sample_rate)
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Add noise to test dataset pairs')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'TriAAN-VC'], default='FreeVC',
                      help='Type of voice conversion model (FreeVC or TriAAN-VC)')
    parser.add_argument('--attack_type', type=str, choices=['white', 'black'], default='white',
                      help='Type of attack (white-box or black-box)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    args = parser.parse_args()

    # 시드 설정 (인자로 제공된 경우)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # 모델 경로 설정
    feature_extractor = "wavlm" if args.attack_type == 'white' else "hubert"
    model_path = f"model/checkpoints/generator_{feature_extractor}.pth"

    # attack_type을 w/b로 축약
    attack_abbr = 'w' if args.attack_type == 'white' else 'b'

    # 입출력 경로 설정
    input_pairs_file = f"data/{args.model}_test_pairs.txt"
    output_pairs_file = f"data/{args.model}_test_noisy_pairs_{attack_abbr.upper()}.txt"
    output_wav_dir = f"data/{args.model}_noisy_style_{attack_abbr.upper()}"
    
    # 출력 디렉토리 생성
    if not os.path.exists(output_wav_dir):
        os.makedirs(output_wav_dir)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NoiseEncoder 모델 불러오기
    noise_encoder = NoiseEncoder().to(device)
    noise_encoder.load_state_dict(torch.load(model_path, map_location=device))
    noise_encoder.eval()
    
    # pairs 파일 읽기
    with open(input_pairs_file, "r") as f:
        pairs = [line.strip().split() for line in f.readlines()]
    
    # 새로운 쌍을 저장할 리스트
    new_pairs = []
    
    # 각 콘텐츠 파일에 대해 처리
    for i, (style_path, contents_path) in enumerate(pairs):
        style_filename = os.path.splitext(os.path.basename(style_path))[0]
        wav_name = f"noisy_{style_filename}.wav"
        output_wav_path = os.path.join(output_wav_dir, wav_name)  
        
        # 노이즈 적용 및 저장
        try:
            processed_path = process_audio_file(noise_encoder, style_path, output_wav_path, device)
            # 새로운 쌍 추가
            new_pairs.append(f"{processed_path} {contents_path}\n")
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({style_path}): {str(e)}")
    
    # 새로운 쌍을 파일로 저장
    with open(output_pairs_file, "w") as f:
        f.writelines(new_pairs)
    
    print(f"\n처리된 파일 쌍이 {output_pairs_file}에 저장되었습니다.")
    print(f"총 {len(new_pairs)}개의 쌍이 생성되었습니다.")
    if args.seed is not None:
        print(f"시드 {args.seed}로 실행되었습니다.")

if __name__ == "__main__":
    main()
