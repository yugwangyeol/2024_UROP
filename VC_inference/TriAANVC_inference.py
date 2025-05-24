import os
import sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("TriAAN-VC"))

import soundfile as sf
import torch
import numpy as np
import kaldiio
import argparse
from tqdm import tqdm
from pathlib import Path

from src.train import *
from src.dataset import *
from src.utils import *
from src.vocoder import decode
from src.cpc import *
from config import *

def normalize_lf0(lf0):      
    zero_idxs    = np.where(lf0 == 0)[0]
    nonzero_idxs = np.where(lf0 != 0)[0]
    if len(nonzero_idxs) > 0:
        mean = np.mean(lf0[nonzero_idxs])
        std  = np.std(lf0[nonzero_idxs])
        if std == 0:
            lf0 -= mean
            lf0[zero_idxs] = 0.0
        else:
            lf0 = (lf0 - mean) / (std + 1e-8)
            lf0[zero_idxs] = 0.0
    return lf0    

def GetTestData(path, cfg):
    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db)

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
        
    mel = logmelspectrogram(
            x=wav,
            fs=cfg.sampling_rate,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            n_shift=cfg.n_shift,
            win_length=cfg.win_length,
            window=cfg.window,
            fmin=cfg.fmin,
            fmax=cfg.fmax
            )
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])
    
    return wav, mel, lf0

def load_triaanvc_models(config_path='TriAAN-VC/config/base.yaml', 
                         checkpoint_path='TriAAN-VC/checkpoints', 
                         model_name='model-cpc-split.pth', 
                         device='cuda:0'):

    print("Loading TriAAN-VC models...")
    
    cfg = Config(config_path)
    cfg.device = device
    cfg.checkpoint = checkpoint_path
    cfg.model_name = model_name
    
    # 모델 초기화 및 로드
    model = TriAANVC(cfg.model.encoder, cfg.model.decoder).to(cfg.device)
    checkpoint = torch.load(f'{cfg.checkpoint}/{cfg.model_name}', map_location=cfg.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # mel statistics 로드
    mel_stats = np.load(os.path.join(cfg.data_path, 'mel_stats.npy'))
    mean = np.expand_dims(mel_stats[0], -1)
    std = np.expand_dims(mel_stats[1], -1)
    
    return model, cfg, mean, std

def triaanvc_infer(contents_path, style_path, model, cfg, mean, std):
    with torch.no_grad():
        contents_wav, contents_mel, contents_lf0 = GetTestData(contents_path, cfg.setting)
        style_wav, style_mel, _ = GetTestData(style_path, cfg.setting)
        
        if cfg.train.cpc:
            cpc_model = load_cpc(os.path.join(cfg.cpc_path, 'cpc.pt')).to(cfg.device)
            cpc_model.eval()
            with torch.no_grad():
                contents_wav = torch.from_numpy(contents_wav).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
                style_wav = torch.from_numpy(style_wav).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
                contents_feat = cpc_model(contents_wav, None)[0].transpose(1,2)
                style_feat = cpc_model(style_wav, None)[0].transpose(1,2)
        else:
            contents_feat = (contents_mel.T - mean) / (std + 1e-8) 
            style_feat = (style_mel.T - mean) / (std + 1e-8)
            contents_feat = torch.from_numpy(contents_feat).unsqueeze(0).to(cfg.device)
            style_feat = torch.from_numpy(style_feat).unsqueeze(0).to(cfg.device)
            
        contents_lf0 = torch.from_numpy(normalize_lf0(contents_lf0)).unsqueeze(0).to(cfg.device)
        
        output = model(contents_feat, contents_lf0, style_feat)
        output = output.squeeze(0).cpu().numpy().T * (std.squeeze(1) + 1e-8) + mean.squeeze(1)
        
        # Mel-spectrogram을 wav로 변환
        temp_dir = Path("temp_conversion")
        temp_dir.mkdir(exist_ok=True)
        
        feat_writer = kaldiio.WriteHelper(f"ark,scp:{temp_dir}/feats.ark,{temp_dir}/feats.scp")
        feat_writer['converted'] = output
        feat_writer.close()
        
        # vocoder를 통한 wav 생성
        decode(f'{temp_dir}/feats.scp', str(temp_dir), cfg.device)

        # 생성된 파일 읽기
        converted_wav, _ = sf.read(os.path.join(temp_dir, 'converted_gen.wav'))
        converted_wav = converted_wav.reshape(-1, 1)
        
        # 임시 파일 정리
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()
        
        return converted_wav

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
    parser.add_argument('--feature_extractor', type=str, choices=['wavlm', 'hubert'], required=True,
                        help='Type of feature extractor (wavlm or hubert)')
    args = parser.parse_args()

    # 경로 설정
    config_path = 'TriAAN-VC/config/base.yaml'
    checkpoint_path = 'TriAAN-VC/checkpoints'
    model_name = 'model-cpc-split.pth'
    device = 'cuda:0'
    output_dir = f'data/TriAAN-VC_converted_{args.feature_extractor}'
    pairs_file = f'data/TriAAN-VC_test_noisy_pairs_{args.feature_extractor}.txt'
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델과 설정 로드
    model, cfg, mean, std = load_triaanvc_models(
        config_path, 
        checkpoint_path,
        model_name,
        device
    )
    
    # 변환할 페어 파일 로드
    with open(pairs_file, "r") as f:
        pairs = f.readlines()
    
    print(f"Found {len(pairs)} pairs to process")
    
    # 업데이트된 경로를 저장할 리스트
    updated_pairs = []
    
    # 각 페어 처리
    for pair in tqdm(pairs, desc="Processing voice conversion"):
        style_path, contents_path = pair.strip().split()
        
        # 출력 파일 이름 생성
        output_name = get_output_name(style_path, contents_path)
        output_path = os.path.join(output_dir, output_name)
        
        # voice conversion 수행
        converted_audio = triaanvc_infer(contents_path, style_path, model, cfg, mean, std)
        
        # 변환된 음성 저장
        sf.write(output_path, converted_audio, cfg.setting.sampling_rate)
        
        # 경로 정보 저장
        updated_pairs.append(f"{style_path} {contents_path} {output_path}\n")
    
    # 업데이트된 경로 정보를 파일에 저장
    with open(pairs_file, "w") as f:
        f.writelines(updated_pairs)
    
    print(f"Updated pairs file saved to {pairs_file}")

if __name__ == "__main__":
    main()