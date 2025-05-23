import os
import sys
import argparse
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("FreeVC"))
import torch
import numpy as np
import librosa
from pathlib import Path
from speaker_encoder.voice_encoder import SpeakerEncoder
from FreeVC.models import SynthesizerTrn
from FreeVC import utils as vc_utils
import soundfile as sf
from tqdm import tqdm

def load_freevc_models():
    print("Loading FreeVC models...")
    
    # Speaker Encoder 모델 로드
    smodel = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
    
    # 하이퍼파라미터 로드
    hps = vc_utils.get_hparams_from_file('FreeVC/logs/freevc.json')
    
    # Synthesizer 모델 로드
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = vc_utils.load_checkpoint('FreeVC/checkpoints/freevc.pth', net_g, None, True)
    
    # Content 추출 모델 로드
    cmodel = vc_utils.get_cmodel(0)
    
    return smodel, net_g, cmodel, hps

def freevc_infer(contents_audio_path, style_audio_path, smodel, cmodel, net_g, hps, utils):
    # 스타일 음성을 로드하여 화자 임베딩(g_tgt)을 생성
    wav_tgt, _ = librosa.load(style_audio_path, sr=hps.data.sampling_rate)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()

    # 콘텐츠 음성을 로드하여 음성 내용(c)을 추출
    wav_cnts, _ = librosa.load(contents_audio_path, sr=hps.data.sampling_rate)
    wav_cnts = torch.from_numpy(wav_cnts).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_cnts)
    
    # 변환된 음성 생성
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()
    return audio

def get_output_name(style_pth, contents_pth):
    # 스타일과 콘텐츠 파일 이름을 추출하여 결합된 출력 이름 생성
    style_name = os.path.basename(style_pth)
    style_name = os.path.splitext(style_name)[0]
    
    contents_name = os.path.basename(contents_pth)
    contents_name = os.path.splitext(contents_name)[0]
    
    output_name = f"{style_name}_{contents_name}.wav"
    return output_name

def main():
    parser = argparse.ArgumentParser(description='Voice conversion for noisy pairs')
    parser.add_argument('--attack_type', type=str, choices=['white', 'black'], default='white',
                        help='Type of attack (white-box or black-box)')
    args = parser.parse_args()

    # attack_type을 w/b로 축약
    attack_abbr = 'w' if args.attack_type == 'white' else 'b'

    # 경로 설정
    output_dir = f'data/FreeVC_noise_{attack_abbr.upper()}'
    pairs_file = f'data/FreeVC_test_noisy_pairs_{attack_abbr.upper()}.txt'
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 모델 로드
    smodel, net_g, cmodel, hps = load_freevc_models()

    with open(pairs_file, "r") as f:
        pairs = f.readlines()
    
    print(f"Found {len(pairs)} pairs to process")
    
    # 업데이트된 경로를 저장할 리스트
    updated_pairs = []
    
    # 각 페어를 처리
    for pair in tqdm(pairs, desc="Processing voice conversion"):
        style_path, contents_path = pair.strip().split() 
        
        # 출력 파일 이름 생성
        output_name = get_output_name(style_path, contents_path)
        output_path = os.path.join(output_dir, output_name)
        
        # voice conversion 수행
        converted_audio = freevc_infer(contents_path, style_path, smodel, cmodel, net_g, hps, vc_utils)

        # 변환된 음성 저장
        sf.write(output_path, converted_audio, hps.data.sampling_rate)

        # 경로 정보 저장
        updated_pairs.append(f"{style_path} {contents_path} {output_path}\n")
    
    # 업데이트된 경로 정보를 파일에 저장
    with open(pairs_file, "w") as f:
        f.writelines(updated_pairs)
    
    print(f"Updated pairs file saved to {pairs_file}")

if __name__ == "__main__":
    main()