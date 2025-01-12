import sys
import os
import torch
import numpy as np
import librosa
from pathlib import Path
from speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from FreeVC import utils as vc_utils
import soundfile as sf
from tqdm import tqdm
sys.path.append(os.path.abspath("FreeVC"))

def vc_infer(source_audio_path, style_audio_path, smodel, cmodel, net_g, hps, utils):
    # 스타일 음성을 로드하여 화자 임베딩(g_tgt)을 생성
    wav_tgt, _ = librosa.load(style_audio_path, sr=hps.data.sampling_rate)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    # 소스 음성을 로드하여 음성 내용(c)을 추출
    wav_src, _ = librosa.load(source_audio_path, sr=hps.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    # 변환된 음성 생성
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()
    return audio

def get_output_name(style_pth, source_pth):
    # 스타일과 소스 파일 이름을 추출하여 결합된 출력 이름 생성
    style_name = os.path.basename(style_pth)
    style_name = os.path.splitext(style_name)[0]
    
    source_name = os.path.basename(source_pth)
    source_name = os.path.splitext(source_name)[0]
    
    output_name = f"{style_name}_{source_name}.wav"
    return output_name

def main():
    print("Loading models...")
    smodel = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
    hps = vc_utils.get_hparams_from_file('FreeVC/logs/freevc.json')
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = vc_utils.load_checkpoint('FreeVC/checkpoints/freevc.pth', net_g, None, True)
    cmodel = vc_utils.get_cmodel(0)

    # 출력 디렉토리 생성
    output_dir = "data/FreeVC_output"
    os.makedirs(output_dir, exist_ok=True)

    # 변환할 페어 파일 로드
    with open("data/FreeVC_test_noisy_pairs.txt", "r") as f:
        pairs = f.readlines()
    
    print(f"Found {len(pairs)} pairs to process")
    
    # 각 페어를 처리
    for pair in tqdm(pairs, desc="Processing voice conversion"):
        style_path, source_path = pair.strip().split() 
        
        # 출력 파일 이름 생성
        output_name = get_output_name(style_path, source_path)
        output_path = os.path.join(output_dir, output_name)
        
        # 출력 파일이 이미 존재하면 건너뜀
        if os.path.exists(output_path):
            print(f"Skipping {output_name} - already exists")
            continue
        
        # voice conversion 수행
        try:
            converted_audio = vc_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
            # 변환된 음성 저장
            sf.write(output_path, converted_audio, hps.data.sampling_rate)
            
        except Exception as e:
            print(f"\nError processing {style_path} -> {source_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()