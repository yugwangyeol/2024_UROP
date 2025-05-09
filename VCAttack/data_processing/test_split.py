import os
import random
from tqdm import tqdm
import sys
import soundfile as sf
import torch
import torchaudio
import argparse
import glob

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("FreeVC"))
sys.path.append(os.path.abspath("VC_inference"))
sys.path.append(os.path.abspath("evaluation"))

from FreeVC import utils as vc_utils
from FreeVC_inference import load_freevc_models, freevc_infer, get_output_name
from TriAANVC_inference import load_triaanvc_models, triaanvc_infer, get_output_name
from evaluation import UnifiedEvaluator

os.environ["NUMBA_DISABLE_ERROR_MESSAGE"] = "1" 
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"      

def process_conversion(contents_path, style_path, model_type, models, model_infer):
    """
    모델 타입에 따라 voice conversion을 수행하는 함수
    
    Args:
        contents_path: 콘텐츠 오디오 경로
        style_path: 스타일 오디오 경로
        model_type: 모델 타입 ('FreeVC' or 'TriAAN-VC')
        models: 로드된 모델들 (FreeVC: (smodel, net_g, cmodel, hps), TriAAN-VC : ())
        model_infer: 변환 함수
    """
    if model_type == 'FreeVC':
        smodel, net_g, cmodel, hps = models
        return model_infer(contents_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
    else:  # TriAAN-VC 
        model, cfg, mean, std = models
        return model_infer(contents_path, style_path, model, cfg, mean, std)

def main(audio_output_dir, vctk_path, model_type, model_loader, model_infer):
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

    # VC 모델 로드 
    models = model_loader() 

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
                        
                # 모델 타입에 따라 적절한 변환 수행
                out = process_conversion(contents_path, style_path, model_type, models, model_infer)
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
                    with open(f"data/{model_type}_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {contents_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open(f"data/{model_type}_test_pairs_adv.txt", "a") as f:
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
                        
                # 모델 타입에 따라 적절한 변환 수행
                out = process_conversion(contents_path, style_path, model_type, models, model_infer)
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
                    with open(f"data/{model_type}_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {contents_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open(f"data/{model_type}_test_pairs_adv.txt", "a") as f:
                        f.writelines([f"{contents_path} {style_path} {adv_path}\n"])

                    i = i + 1
                    break
                    
            if i >= 50:
                break
                                     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice conversion pair creation')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'TriAAN-VC'], default='FreeVC',
                      help='Type of voice conversion model (FreeVC or TriAAN-VC)')
    args = parser.parse_args()

    audio_output_dir = f"data/{args.model}_original"
    vctk_path = "data"

    if args.model == "FreeVC":
        model_loader = load_freevc_models
        model_infer = freevc_infer
    else:  # TriAAN-VC
        model_loader = load_triaanvc_models
        model_infer = triaanvc_infer
    
    main(
        audio_output_dir=audio_output_dir,
        vctk_path=vctk_path,
        model_type=args.model,
        model_loader=model_loader,
        model_infer=model_infer
    )
