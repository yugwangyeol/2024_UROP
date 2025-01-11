import sys
import os
sys.path.append(os.path.abspath("FreeVC"))

import torch
import numpy as np
import librosa
from pathlib import Path
from speaker_encoder.voice_encoder import SpeakerEncoder
from models import SynthesizerTrn
from FreeVC import utils as vc_utils
import soundfile as sf
from tqdm import tqdm

def vc_infer(src_pth, tar_pth, smodel, cmodel, net_g, hps, utils):
    wav_tgt, _ = librosa.load(tar_pth, sr=hps.data.sampling_rate)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    wav_src, _ = librosa.load(src_pth, sr=hps.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()
    return audio

def get_output_name(style_path, source_path):
    style_name = os.path.basename(style_path)
    style_name = os.path.splitext(style_name)[0]
    
    source_name = os.path.basename(source_path)
    source_name = os.path.splitext(source_name)[0]
    
    # Combine names
    output_name = f"{style_name}_{source_name}.wav"
    return output_name

def main():
    # Load models
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

    # Create output directory if it doesn't exist
    output_dir = "data/FreeVC_output"
    os.makedirs(output_dir, exist_ok=True)

    # Read pairs from file
    with open("data/FreeVC_test_noisy_pairs.txt", "r") as f:
        pairs = f.readlines()
    
    print(f"Found {len(pairs)} pairs to process")
    
    # Process each pair
    for pair in tqdm(pairs, desc="Processing voice conversion"):
        style_path, source_path = pair.strip().split() 
        
        # Generate output filename
        output_name = get_output_name(style_path, source_path)
        output_path = os.path.join(output_dir, output_name)
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_name} - already exists")
            continue
        
        # Perform voice conversion
        try:
            converted_audio = vc_infer(source_path, style_path, smodel, cmodel, net_g, hps, vc_utils)
            
            # Save the converted audio
            sf.write(output_path, converted_audio, hps.data.sampling_rate)
            print(f"\nSuccessfully converted and saved: {output_name}")
            
        except Exception as e:
            print(f"\nError processing {style_path} -> {source_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()