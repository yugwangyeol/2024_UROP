import torch
import torchaudio
import os
from model import Generator as NoiseEncoder

def process_audio_file(noise_encoder, input_path, output_path, device):
    # 입력 오디오 파일 로드
    waveform, sample_rate = torchaudio.load(input_path)
    
    # 배치 차원 추가
    waveform = waveform.unsqueeze(0).to(device)
    
    # 노이즈 적용
    with torch.no_grad():
        noise = noise_encoder(waveform)
        noisy_waveform = torch.clamp(waveform + noise, -1, 1)
    
    # 저장
    torchaudio.save(output_path, noisy_waveform.squeeze(0).cpu(), sample_rate)
    return output_path

def main():
    # 입출력 경로 설정
    model_path = "model/checkpoints/generator.pth"
    input_pairs_file = "data/FreeVC_test_pairs.txt"
    output_pairs_file = "data/FreeVC_test_noisy_pairs.txt"
    output_wav_dir = "data/FreeVC_noisy_style"
    
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
    
    # 각 소스 파일에 대해 처리
    for i, (src_path, tar_path) in enumerate(pairs):
        src_filename = os.path.splitext(os.path.basename(src_path))[0]
        wav_name = f"noisy_{src_filename}.wav"
        output_wav_path = os.path.join(output_wav_dir, wav_name)  
        
        # 노이즈 적용 및 저장
        try:
            processed_path = process_audio_file(noise_encoder, src_path, output_wav_path, device)
            # 새로운 쌍 추가
            new_pairs.append(f"{os.path.abspath(processed_path)} {os.path.abspath(tar_path)}\n")
            print(f"처리 완료: {src_path} -> {processed_path}")
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({src_path}): {str(e)}")
    
    # 새로운 쌍을 파일로 저장
    with open(output_pairs_file, "w") as f:
        f.writelines(new_pairs)
    
    print(f"\n처리된 파일 쌍이 {output_pairs_file}에 저장되었습니다.")
    print(f"총 {len(new_pairs)}개의 쌍이 생성되었습니다.")


if __name__ == "__main__":
    main()
