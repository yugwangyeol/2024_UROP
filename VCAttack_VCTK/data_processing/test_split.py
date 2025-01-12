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
        audio_output_dir: F(x,t) wav 파일 저장 경로 ({data/Fre}_original)
        vctk_path: ASV 임계값 계산 시 필요한 VCTK 데이터셋 경로
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
    print(f"Using fixed threshold: {threshold:.3f}")
    
    # # threshold 계산 or 로드
    # print("Getting ASV threshold...")
    # threshold, eer = evaluator.get_or_calculate_threshold(vctk_path)
    # print(f"Using threshold: {threshold:.3f} (EER: {eer:.3f})")
    
    # VC 모델 로드 
    print("Loading models...")
    smodel, net_g, cmodel, hps = model_loader()
    
    # test.txt의 오디오 파일 불러오기
    with open(test_file, "r") as f:
        all_files = [line.strip() for line in f.readlines()]
    
    print(f"Total files in test.txt: {len(all_files)}")
    
    # 검증된 pair 저장 리스트
    verified_pairs = []
    
    for source_path in tqdm(all_files, desc="Processing source"):
        source_spk = source_path.split("/")[-2] # 소스 화자 ID 추출
        style_candidates = []
        
        # num_pairs만큼 스타일 후보 선택
        while len(style_candidates) < args.num_pairs:
            style_path = random.choice(all_files) # 랜덤으로 스타일 파일 선택
            style_spk = style_path.split("/")[-2]
            
            # 소스 화자와 스타일 화자가 다르고, 이미 선택되지 않은 경우
            if style_spk != source_spk and style_path not in style_candidates: # 스타일 화자 ID 추출
                try:
                    # 출력 파일 이름 생성
                    output_name = get_output_name(style_path, source_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    
                    # 출력 파일이 이미 존재하면 건너뜀
                    if os.path.exists(output_path_full):
                        print(f"Skipping {output_name} - already exists")
                        continue
                    
                    # voice conversion 수행
                    converted_audio = model_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
                    
                    # 임시 파일에 변환된 음성 저장
                    temp_output = "temp_output.wav"
                    sf.write(temp_output, converted_audio, hps.data.sampling_rate)
                    
                    # ASV로 검증
                    try:
                        # 검증용 wav 파일 로드
                        style_wav, sr = torchaudio.load(style_path)
                        conv_wav, sr = torchaudio.load(temp_output)
                        
                        # 오디오 전처리
                        style_wav = evaluator.preprocess_audio(style_wav, sr)
                        conv_wav = evaluator.preprocess_audio(conv_wav, sr)
                        
                        # 화자 유사성 검증
                        if evaluator.verify_speaker(conv_wav, style_wav, threshold):
                            # 검증 성공 시 오디오 저장 및 pair 추가
                            os.rename(temp_output, output_path_full)
                            verified_pairs.append(f"{style_path} {source_path}\n")
                            style_candidates.append(style_path)
                        else:
                            os.remove(temp_output)
                            
                    except Exception as e:
                        # print(f"\nError in verification: {str(e)}")
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                        continue
                        
                except Exception as e:
                    print(f"\nError in conversion: {str(e)}")
                    continue
    
    # 검증된 pair를 출력 파일에 저장
    with open(output_path, "w") as f:
        f.writelines(verified_pairs)
    
    print(f"Created {len(verified_pairs)} verified pairs and saved to {output_path}")
    print(f"Converted audio files saved to {audio_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice conversion pair creation with model selection')
    parser.add_argument('--model', type=str, choices=['FreeVC', 'PH'], required=True,
                    help='Type of voice conversion model (FreeVC or PH)')
    parser.add_argument('--test_file', type=str, default="data/test.txt",
                    help='Path to test file')
    parser.add_argument('--vctk_path', type=str, default="data",
                    help='Path to VCTK dataset')
    parser.add_argument('--num_pairs', type=int, default=5,
                    help='Number of pairs per source')
    
    args = parser.parse_args()
    
    output_path = f"data/{args.model}_test_pairs.txt"
    audio_output_dir = f"data/{args.model}_original"
    
    if args.model == "FreeVC":
        model_loader = load_vc_models
        model_infer = vc_infer
    else:  # PH
        model_loader = load_ph_models
        model_infer = ph_infer
    
    create_pairs_from_test_file(
        test_file=args.test_file,
        output_path=output_path,
        audio_output_dir=audio_output_dir,
        vctk_path=args.vctk_path,
        model_loader=model_loader,
        model_infer=model_infer
    )