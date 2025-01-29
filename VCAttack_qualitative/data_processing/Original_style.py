import os
import shutil
from pathlib import Path
import soundfile as sf

def convert_flac_to_wav(flac_path, wav_path):
    """FLAC 파일을 WAV 파일로 변환"""
    try:
        # FLAC 파일 읽기
        data, samplerate = sf.read(flac_path)
        # WAV 파일로 저장
        sf.write(wav_path, data, samplerate)
        return True
    except Exception as e:
        print(f"변환 실패: {flac_path}")
        print(f"에러: {str(e)}")
        return False

def copy_style_files(input_file, output_dir):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 읽기
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 각 줄에서 첫 번째 파일 경로만 추출
    style_files = []
    for line in lines:
        paths = line.strip().split()
        if paths:  # 빈 줄이 아닌 경우
            style_files.append(paths[0])
    
    # 파일 복사 및 변환
    copied_count = 0
    converted_count = 0
    for file_path in style_files:
        if os.path.exists(file_path):
            # 원본 파일명만 추출
            filename = os.path.basename(file_path)
            # 파일 확장자 확인
            is_flac = filename.lower().endswith('.flac')
            
            if is_flac:
                # FLAC 파일인 경우 WAV로 변환
                wav_filename = filename[:-5] + '.wav'  # .flac를 .wav로 변경
                dest_path = os.path.join(output_dir, wav_filename)
                if convert_flac_to_wav(file_path, dest_path):
                    print(f"변환됨: {file_path} -> {dest_path}")
                    converted_count += 1
                    copied_count += 1
            else:
                # 일반 파일 복사
                dest_path = os.path.join(output_dir, filename)
                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"복사됨: {file_path} -> {dest_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"복사 실패: {file_path}")
                    print(f"에러: {str(e)}")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print(f"\n총 {len(style_files)}개 파일 중 {copied_count}개 처리 완료")
    print(f"- 변환된 FLAC 파일: {converted_count}개")
    print(f"- 복사된 파일: {copied_count - converted_count}개")

if __name__ == "__main__":
    input_file = "data/Original_style.txt"
    output_dir = "data/Original_style"
    
    copy_style_files(input_file, output_dir)