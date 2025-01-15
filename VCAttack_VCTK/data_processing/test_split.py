import os
import random
from tqdm import tqdm
import sys
import soundfile as sf
import torchaudio
import argparse
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

os.environ["NUMBA_DISABLE_ERROR_MESSAGE"] = "1"  # 에러 로그 비활성화
os.environ["NUMBA_LOG_LEVEL"] = "WARNING"       # 로그 레벨을 WARNING으로 설정

def load_vc_models():
    """
    FreeVC 모델을 로드하고, 인퍼런스에 필요한 component를 반환
    Returns:
        smodel: Speaker encoder model
        net_g: Generator model
        cmodel: Content model
        hps: Hyperparameters
    """
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

def create_pairs_from_test_file(test_file, output_path, audio_output_dir, vctk_path, model_loader, model_infer):
    """
    test.txt에서 pair 파일을 생성하고, VC를 수행한 뒤 ASV 검증하는 함수
    Args:
        test_file: test.txt 경로
        output_path: (x,t) txt 파일 저장 경로 ({VC_model}_test_pairs.txt)
        audio_output_dir: F(x,t) wav 파일 저장 경로 ({VC_model}_original)
        model_loader: VC 모델 로드 함수
        model_infer: VC 모델 인퍼런스 함수
    """
    os.makedirs(audio_output_dir, exist_ok=True)

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
    
    # VC 모델 로드 
    print("Loading VC models...")
    smodel, net_g, cmodel, hps = model_loader() 
    
    # test.txt의 오디오 파일 불러오기
    with open(test_file, "r") as f:
        all_files = [line.strip() for line in f.readlines()]
    
    print(f"Total files in test.txt: {len(all_files)}")
    
    # 매칭된 페어를 저장할 리스트
    verified_pairs = []

    # 화자 리스트 생성
    unique_speakers = list(set(file.split("/")[-2] for file in all_files))

    # 각 화자에 대해 
    for spk in tqdm(unique_speakers):
        print(f"processing speakers : {spk}")
        # 해당 화자의 모든 wav 파일 수집
        wavs = [file for file in all_files if file.split("/")[-2] == spk]
        random.shuffle(wavs)
        i = 0  # 현재 화자의 매칭된 페어 수
        
        for style_path in wavs:  # 현재 화자(spk)의 wav 파일을 순회 (스타일 파일 순회)
            for j in range(10):  # 스타일 파일당 최대 10개의 소스 파일 매칭 시도
                while True:
                    source_path = random.choice(all_files)  # 소스 파일 랜덤 선택
                    if source_path.split("/")[-2] != spk:  # 소스 화자와 스타일 화자가 다를 때만 선택
                        break
                
                converted_audio = model_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
                    
                # 변환된 음성을 임시 파일로 저장
                temp_output = "temp_output.wav"
                sf.write(temp_output, converted_audio, hps.data.sampling_rate)
                    
                # 검증용 wav 파일 로드
                style_wav, sr = torchaudio.load(style_path)
                conv_wav, sr = torchaudio.load(temp_output)
                        
                # 오디오 전처리
                style_wav = evaluator.preprocess_audio(style_wav, sr)
                conv_wav = evaluator.preprocess_audio(conv_wav, sr)
                        
                # 화자 유사성 검증
                if evaluator.verify_speaker(conv_wav, style_wav, threshold):
                     # 검증 성공 시 오디오 저장 및 pair 추가
                    output_name = get_output_name(style_path, source_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    os.rename(temp_output, output_path_full)
                    verified_pairs.append(f"{style_path} {source_path}\n")
                    i += 1  # 매칭된 페어 수 증가
                    break
                else:
                    # 검증 실패 시 임시 파일 삭제
                    os.remove(temp_output)
                            
            if i >= 50:  # 현재 화자에 대해 50개의 매칭된 페어를 찾으면, 해당 화자에 대한 처리를 중단
                break              

    # 검증된 페어를 출력 파일에 저장
    with open(output_path, "w") as f:
        f.writelines(verified_pairs)
    
    print(f"Created {len(verified_pairs)} verified pairs and saved to {output_path}")
    print(f"Converted audio files saved to {audio_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice conversion pair creation')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'PH'], default='FreeVC',
                      help='Type of voice conversion model (FreeVC or PH)')
    args = parser.parse_args()

    test_file = "data/test.txt"  
    output_path = f"data/{args.model}_test_pairs.txt"
    audio_output_dir = f"data/{args.model}_original"
    vctk_path = "data"          
    
    if args.model == "FreeVC":
        model_loader = load_vc_models
        model_infer = vc_infer
    else:  # PH
        model_loader = load_ph_models
        model_infer = ph_infer
    
    create_pairs_from_test_file(
        test_file=test_file,
        output_path=output_path,
        audio_output_dir=audio_output_dir,
        vctk_path=vctk_path,
        model_loader=model_loader,
        model_infer=model_infer
    )