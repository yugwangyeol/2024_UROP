import os
import math
import random 
import glob
import numpy as np
import torch

# 시드 고정
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# VCTK 데이터셋의 스피커 정보 파일 경로
with open("data/VCTK-Corpus-0.92/speaker-info.txt", "r") as f:
    lines = f.readlines()

# 경로 설정
root = "data/VCTK-Corpus-0.92/wav48_silence_trimmed"  # VCTK 데이터 경로
target_pth = "data"  # train.txt, valid.txt, test.txt 저장 경로
lines = [line.rstrip() for line in lines]
lines = lines[1:-1]  # 헤더 제거

# 남성(M)과 여성(F) 스피커를 구분
m_spks, f_spks = [], []

for line in lines:
    L = line.split()
    if L[2] == "M":  # 성별이 M인 경우
        m_spks.append(L[0])
    else:  # 성별이 F인 경우
        f_spks.append(L[0])

# Train/Valid/Test 스피커 리스트 초기화
train_spks, val_spks, test_spks = [], [], []
random.shuffle(m_spks)  # 남성 스피커 랜덤 셔플
random.shuffle(f_spks)  # 여성 스피커 랜덤 셔플

# 남성 스피커 나누기
train_spks += m_spks[:math.ceil(len(m_spks)*0.6)]
val_spks += m_spks[math.ceil(len(m_spks)*0.6):math.ceil(len(m_spks)*0.8)]
test_spks += m_spks[math.ceil(len(m_spks)*0.8):]

# 여성 스피커 나누기
train_spks += f_spks[:math.ceil(len(f_spks)*0.6)]
val_spks += f_spks[math.ceil(len(f_spks)*0.6):math.ceil(len(f_spks)*0.8)]
test_spks += f_spks[math.ceil(len(f_spks)*0.8):]

# 나뉜 스피커 수 출력
print(f"split: Train={len(train_spks)}, Valid={len(val_spks)}, Test={len(test_spks)}")

# Train 데이터의 오디오 경로 저장
lines = []
for spk in train_spks:
    spk_path = os.path.join(root, spk)
    flac_files = glob.glob(spk_path + '/*.flac') 
    lines += flac_files
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "train.txt"), "w") as f:
    f.writelines(lines)

# Valid 데이터의 오디오 경로 저장
lines = []
for spk in val_spks:
    spk_path = os.path.join(root, spk)
    flac_files = glob.glob(spk_path + '/*.flac') 
    lines += flac_files
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "val.txt"), "w") as f:
    f.writelines(lines)

# Test 데이터의 오디오 경로 저장
lines = []
for spk in test_spks:
    spk_path = os.path.join(root, spk)
    flac_files = glob.glob(spk_path + '/*.flac') 
    lines += flac_files
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "test.txt"), "w") as f:
    f.writelines(lines)
