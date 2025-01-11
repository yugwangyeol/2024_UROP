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
    Load FreeVC models and return required components for inference
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
    Create pairs file from test.txt and perform voice conversion with ASV verification
    Args:
        test_file: Path to test.txt containing pre-selected audio files
        output_path: Where to save the pairs file
        audio_output_dir: Directory to save converted audio files
        vctk_path: Path to VCTK dataset for threshold calculation
        model_loader: Function to load models
        model_infer: Function to perform inference
    """
    os.makedirs(audio_output_dir, exist_ok=True)
    
    # Initialize ASV evaluator
    print("Initializing ASV evaluator...")
    evaluator = UnifiedEvaluator(device='cuda')

    # 고정된 임계값 사용
    threshold = 0.328  # EER threshold manually set
    eer = None  # EER is not calculated in this case
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
    
    # pair 생성
    verified_pairs = []
    
    for source_path in tqdm(all_files, desc="Processing source"):
        source_spk = source_path.split("/")[-2]
        style_candidates = []
        
        while len(style_candidates) < args.num_pairs:
            style_path = random.choice(all_files)
            style_spk = style_path.split("/")[-2]
            
            if style_spk != source_spk and style_path not in style_candidates:
                try:
                    # Generate output filename
                    output_name = get_output_name(source_path, style_path)
                    output_path_full = os.path.join(audio_output_dir, output_name)
                    
                    # Skip if output already exists
                    if os.path.exists(output_path_full):
                        print(f"Skipping {output_name} - already exists")
                        continue
                    
                    # Perform voice conversion
                    converted_audio = model_infer(style_path, source_path, smodel, cmodel, net_g, hps, vc_utils)
                    
                    # Save converted audio temporarily
                    temp_output = "temp_output.wav"
                    sf.write(temp_output, converted_audio, hps.data.sampling_rate)
                    
                    # ASV로 검증
                    try:
                        # Load wavs for verification
                        source_wav, sr = torchaudio.load(source_path)
                        conv_wav, sr = torchaudio.load(temp_output)
                        
                        # Preprocess audio
                        source_wav = evaluator.preprocess_audio(source_wav, sr)
                        conv_wav = evaluator.preprocess_audio(conv_wav, sr)
                        
                        # Verify speaker similarity
                        if evaluator.verify_speaker(conv_wav, source_wav, threshold):
                            # If verification successful, save the audio and add to pairs
                            os.rename(temp_output, output_path_full)
                            verified_pairs.append(f"{style_path} {source_path}\n")
                            style_candidates.append(style_path)
                            print(f"\nSuccessfully converted and verified: {output_name}")
                        else:
                            print(f"\nConversion failed verification for: {output_name}")
                            os.remove(temp_output)
                            
                    except Exception as e:
                        # print(f"\nError in verification: {str(e)}")
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                        continue
                        
                except Exception as e:
                    print(f"\nError in conversion: {str(e)}")
                    continue
    
    # 검증된 pair를 output file에 저장하기
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
    parser.add_argument('--vctk_path', type=str, default="data/VCTK-Corpus-0.92",
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