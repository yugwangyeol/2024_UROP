import os
import random
from tqdm import tqdm
import sys
import soundfile as sf
import torch
import torchaudio
import glob

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("FreeVC"))
sys.path.append(os.path.abspath("VC_inference"))
sys.path.append(os.path.abspath("evaluation"))

from FreeVC import utils as vc_utils
from FreeVC_inference import load_freevc_models, vc_infer, get_output_name
from evaluation import UnifiedEvaluator

os.environ["NUMBA_DISABLE_ERROR_MESSAGE"] = "1" 
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"      

def main(audio_output_dir, vctk_path):
    """
    test.txt에서 pair 파일을 생성하고, VC를 수행한 뒤 ASV 검증하는 함수
    """
    os.makedirs(audio_output_dir, exist_ok=True)

    root = "data/VCTK-Corpus-0.92/wav48_silence_trimmed"

    # 화자 리스트 생성
    with open("data/VCTK-Corpus-0.92/speaker-info.txt", "r") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    lines = lines[1:-1]
    spk_info = {} 
    for line in lines:
        L = line.split()
        spk_info[L[0]] = L[2]

    # 남성, 여성 화자 분리
    m_spks, f_spks = [], []
    with open("data/test.txt", "r") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        spk = line.split("/")[-2]
        if spk_info[spk] == "M" and (spk not in m_spks):
            m_spks.append(spk)
        elif spk_info[spk] == "F" and (spk not in f_spks):
            f_spks.append(spk)
    print(len(m_spks), len(f_spks))

    # FreeVC 모델 로드 
    smodel, net_g, cmodel, hps = load_freevc_models()

    # ASV evaluator 초기화
    print("Initializing ASV evaluator...")
    evaluator = UnifiedEvaluator(device='cuda')

    # 고정된 threshold 사용
    threshold = 0.328  # RW-Voiceshield의 임계값
    eer = None  
    print(f"Using threshold: {threshold:.3f}")

    # 남성 화자 처리
    for spk in tqdm(m_spks):
        print(f"processing spk{spk}")
        wavs = glob.glob(os.path.join(root, spk) + '/*.flac')
        random.shuffle(wavs)
        i = 0
        for style_path in wavs:
            for j in range(10):
                while True:
                    contents_path = random.choice(lines)
                    if contents_path.split("/")[-2] != spk:
                        break
                        
                # FreeVC 변환 수행
                out = vc_infer(contents_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
                out_path = 'out.wav'
                sf.write(out_path, out, 16000)

                # 화자 유사성 검증
                if evaluator.verify_speaker(style_path, out_path, threshold):
                    while True:
                        adv_path = random.choice(lines)
                        adv_spk = adv_path.split("/")[-2]
                        if adv_spk in f_spks:  
                            break

                    # 검증 성공 시 오디오 저장 및 pair 추가
                    output_name = get_output_name(style_path, contents_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    os.rename(out_path, output_path_full)

                    # 원본 pairs 파일에 기록
                    with open("data/FreeVC_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {contents_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open("data/FreeVC_test_pairs_adv.txt", "a") as f:
                        f.writelines([f"{contents_path} {style_path} {adv_path}\n"])

                    i = i + 1
                    break
            if i >= 50:
                break               

    # 여성 화자 처리                    
    for spk in tqdm(f_spks):
        print(f"processing spk{spk}")
        wavs = glob.glob(os.path.join(root, spk) + '/*.flac')
        random.shuffle(wavs)
        i = 0
        for style_path in wavs:
            for j in range(10):
                while True:
                    contents_path = random.choice(lines)
                    if contents_path.split("/")[-2] != spk:
                        break
                        
                # FreeVC 변환 수행
                out = vc_infer(contents_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
                out_path = 'out.wav'
                sf.write(out_path, out, 16000)

                # 화자 유사성 검증
                if evaluator.verify_speaker(style_path, out_path, threshold):
                    while True:
                        adv_path = random.choice(lines)
                        adv_spk = adv_path.split("/")[-2]
                        if adv_spk in m_spks: 
                            break

                    # 검증 성공 시 오디오 저장 및 pair 추가
                    output_name = get_output_name(style_path, contents_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    os.rename(out_path, output_path_full)

                    # 원본 pairs 파일에 기록
                    with open("data/FreeVC_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {contents_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open("data/FreeVC_test_pairs_adv.txt", "a") as f:
                        f.writelines([f"{contents_path} {style_path} {adv_path}\n"])

                    i = i + 1
                    break
                    
            if i >= 50:
                break
                                     
if __name__ == '__main__':
    audio_output_dir = "data/FreeVC_original"
    vctk_path = "data"
    
    main(
        audio_output_dir=audio_output_dir,
        vctk_path=vctk_path
    )