import random
from collections import defaultdict
import os
import shutil

def read_speaker_info(speaker_info_path):
    # 화자 정보 파일을 읽어 성별 정보를 추출함
    speaker_gender = {}
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:  # 헤더 라인 스킵
                continue
            # 공백으로 분리 (탭이나 스페이스 모두 처리)
            parts = line.strip().split()
            if len(parts) >= 3:
                speaker_id = parts[0].strip()  # ID
                gender = parts[2].strip()      # GENDER
                if gender in ['M', 'F']:
                    speaker_gender[speaker_id] = gender
    
    return speaker_gender

def read_test_pairs(test_pairs_path):
    # FreeVC_test_pairs.txt를 읽어 스타일 화자별 페어 정보를 추출함
    style_pairs = defaultdict(list)
    with open(test_pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 스타일과 콘텐츠 경로를 공백으로 분리
            paths = line.strip().split()
            if len(paths) == 2:
                # 스타일 경로에서 화자 ID 추출 (첫 번째 경로 사용)
                style_path = paths[0]
                style_speaker = style_path.split('/')[-2]  
                style_pairs[style_speaker].append(line.strip())
    
    return style_pairs

def analyze_gender_distribution(style_pairs, speaker_gender):
    # 스타일 화자들의 성별 분포를 분석함
    gender_dist = {'M': [], 'F': []}
    
    print("\n화자별 페어 수:")
    for speaker, pairs in style_pairs.items():
        if speaker in speaker_gender:
            gender = speaker_gender[speaker]
            gender_dist[gender].append(speaker)
            print(f"화자 {speaker} ({gender}): {len(pairs)}개 페어")
    
    print(f"\n성별 분포:")
    print(f"- 남성 화자: {len(gender_dist['M'])}명")
    print(f"- 여성 화자: {len(gender_dist['F'])}명")
    
    return gender_dist

def balanced_sampling(style_pairs, gender_dist):
    # 성별당 25개씩 샘플링
    sampled_pairs = []
    
    # 남성 화자 (9명) - 7명에서 3개씩, 2명에서 2개씩
    male_speakers = gender_dist['M']
    random.shuffle(male_speakers)
    
    print("\n남성 화자 샘플링 계획:")
    print(f"- 7명에서 3개씩: {male_speakers[:7]}")
    print(f"- 2명에서 2개씩: {male_speakers[7:]}")
    
    # 7명에서 3개씩 선택
    for speaker in male_speakers[:7]:
        selected = random.sample(style_pairs[speaker], 3)
        sampled_pairs.extend(selected)
    
    # 2명에서 2개씩 선택
    for speaker in male_speakers[7:]:
        selected = random.sample(style_pairs[speaker], 2)
        sampled_pairs.extend(selected)
    
    # 여성 화자 (12명) - 1명에서 3개, 11명에서 2개씩
    female_speakers = gender_dist['F']
    random.shuffle(female_speakers)
    
    print("\n여성 화자 샘플링 계획:")
    print(f"- 1명에서 3개: {female_speakers[0]}")
    print(f"- 11명에서 2개씩: {female_speakers[1:]}")
    
    # 1명에서 3개 선택
    selected = random.sample(style_pairs[female_speakers[0]], 3)
    sampled_pairs.extend(selected)
    
    # 11명에서 2개씩 선택
    for speaker in female_speakers[1:]:
        selected = random.sample(style_pairs[speaker], 2)
        sampled_pairs.extend(selected)
    
    return sampled_pairs

def copy_paired_files(sampled_pairs, freevc_original_dir, output_dir):
    # Original_style.txt의 페어에 해당하는 wav 파일을 찾아 복사함
    os.makedirs(output_dir, exist_ok=True)
    
    copied_files = []
    missing_files = []
    
    for pair in sampled_pairs:
        style_path, contents_path = pair.split()
        # 스타일과 콘텐츠 화자 ID 추출
        style_speaker = style_path.split('/')[-2]
        contents_speaker = contents_path.split('/')[-2]
        # 파일 이름 추출 및 확장자 변경 (.flac -> .wav)
        style_file = style_path.split('/')[-1].replace('.flac', '.wav')
        contents_file = contents_path.split('/')[-1].replace('.flac', '.wav')
        
        # 새로운 파일 이름 생성 (style_contents.wav 형식)
        new_filename = f"{style_file.split('.')[0]}_{contents_file}"
        
        # FreeVC_original 디렉토리에서 파일 찾기
        original_file_path = os.path.join(freevc_original_dir, new_filename)
        
        if os.path.exists(original_file_path):
            # 파일 복사
            shutil.copy2(original_file_path, os.path.join(output_dir, new_filename))
            copied_files.append(new_filename)
        else:
            missing_files.append(new_filename)
    
    # 결과 출력
    print(f"\n파일 복사 결과:")
    print(f"복사된 파일 수: {len(copied_files)}")
    print(f"찾지 못한 파일 수: {len(missing_files)}")
    
    if missing_files:
        print("\n찾지 못한 파일들:")
        for file in missing_files:
            print(f"- {file}")

def main():
    # 파일 경로
    speaker_info_path = "data/VCTK-Corpus-0.92/speaker-info.txt"
    test_pairs_path = "data/FreeVC_test_pairs.txt"
    freevc_original_dir = "data/FreeVC_original"  
    output_dir = "data/Original_FreeVC"  
    
    # 데이터 로드
    speaker_gender = read_speaker_info(speaker_info_path)
    style_pairs = read_test_pairs(test_pairs_path)
    
    # 성별 분포 분석
    gender_dist = analyze_gender_distribution(style_pairs, speaker_gender)
    
    # 샘플링
    sampled_pairs = balanced_sampling(style_pairs, gender_dist)
    
    # 결과 저장
    output_path = "data/Original_style.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in sampled_pairs:
            f.write(f"{pair}\n")
    
    # 파일 복사 수행
    copy_paired_files(sampled_pairs, freevc_original_dir, output_dir)
    
    # 최종 샘플링 결과 분석
    final_speakers = defaultdict(int)
    final_gender_count = {'M': 0, 'F': 0}
    
    for pair in sampled_pairs:
        style_path = pair.split()[0]  # 첫 번째 경로를 스타일로 사용
        style_speaker = style_path.split('/')[-2]  # 경로에서 화자 ID 추출
        if style_speaker in speaker_gender:
            gender = speaker_gender[style_speaker]
            final_speakers[style_speaker] += 1
            final_gender_count[gender] += 1
    
    print("\n최종 샘플링 결과:")
    print(f"총 샘플링된 페어 수: {len(sampled_pairs)}개")
    print("성별 카운트:", dict(final_gender_count))
    print("\n화자별 페어 수:")
    for speaker, count in sorted(final_speakers.items()):
        gender = speaker_gender[speaker]
        print(f"화자 {speaker} ({gender}): {count}개 페어")

if __name__ == "__main__":
    main()