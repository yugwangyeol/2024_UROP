import os
import random
from tqdm import tqdm
import sys
import soundfile as sf
import torchaudio
import argparse
import glob
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("FreeVC"))
sys.path.append(os.path.abspath("VC_inference"))
sys.path.append(os.path.abspath("evaluation"))

from FreeVC_inference import vc_infer, get_output_name
from speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from FreeVC import utils as vc_utils
# from PH_inference import ph_infer, get_output_name

from evaluation import UnifiedEvaluator

os.environ["NUMBA_DISABLE_ERROR_MESSAGE"] = "1" 
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"      

def load_freevc_models():
    print("Loading FreeVC models...")
    smodel = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
    hps = vc_utils.get_hparams_from_file('FreeVC/logs/freevc.json')
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = vc_utils.load_checkpoint('FreeVC/checkpoints/freevc.pth', net_g, None, True)
    cmodel = vc_utils.get_cmodel(0)
      
    return smodel, net_g, cmodel, hps

def main(audio_output_dir, vctk_path, model_loader, model_infer):
    """
    test.txt에서 pair 파일을 생성하고, VC를 수행한 뒤 ASV 검증하는 함수
    Args:
        audio_output_dir: F(x,t) wav 파일 저장 경로 ({VC_model}_original)
        vctk_path : VCTK 데이터 경로
        model_loader: VC 모델 로드 함수
        model_infer: VC 모델 인퍼런스 함수
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
    print("Loading VC models...")
    smodel, net_g, cmodel, hps = model_loader() 

    # ASV evaluator 초기화
    print("Initializing ASV evaluator...")
    evaluator = UnifiedEvaluator(device='cuda')

    # 고정된 threshold 사용
    threshold = 0.328  # RW-Voiceshield의 임계값
    eer = None  
    print(f"Using threshold: {threshold:.3f}")

    # # threshold 계산 or 로드
    # print("Getting ASV threshold...")
    # threshold, eer = evaluator.get_or_calculate_threshold(vctk_path)
    # print(f"Using threshold: {threshold:.3f} (EER: {eer:.3f})")

    # 남성 화자 처리
    for spk in tqdm(m_spks):
        print(f"processing spk{spk}")
        wavs = glob.glob(os.path.join(root, spk) + '/*.flac')
        random.shuffle(wavs)
        i = 0
        for style_path in wavs:
            for j in range(10):
                while True:
                    source_path = random.choice(lines)
                    if source_path.split("/")[-2] != spk:
                        break
                out = model_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
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
                    output_name = get_output_name(style_path, source_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    os.rename(out_path, output_path_full)

                    # 원본 pairs 파일에 기록
                    with open(f"data/{args.model}_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {source_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open(f"data/{args.model}_test_pairs_adv.txt", "a") as f:
                        f.writelines([f"{source_path} {style_path} {adv_path}\n"])

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
                    source_path = random.choice(lines)
                    if source_path.split("/")[-2] != spk:
                        break
                out = model_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
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
                    output_name = get_output_name(style_path, source_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    os.rename(out_path, output_path_full)

                    # 원본 pairs 파일에 기록
                    with open(f"data/{args.model}_test_pairs.txt", "a") as f:
                        f.writelines([f"{style_path} {source_path}\n"])
                    
                    # adversarial pairs 파일에 기록
                    with open(f"data/{args.model}_test_pairs_adv.txt", "a") as f:
                        f.writelines([f"{source_path} {style_path} {adv_path}\n"])

                    i = i + 1
                    break
                    
            if i >= 50:
                break
                                     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice conversion pair creation')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'PH'], default='FreeVC',
                      help='Type of voice conversion model (FreeVC or PH)')
    args = parser.parse_args()

    audio_output_dir = f"data/{args.model}_original"
    vctk_path = "data"

    if args.model == "FreeVC":
        model_loader = load_freevc_models
        model_infer = vc_infer
    else:  # PH
        model_loader = load_ph_models
        model_infer = ph_infer
    
    main(
        audio_output_dir=audio_output_dir,
        vctk_path=vctk_path,
        model_loader=model_loader,
        model_infer=model_infer
    )
