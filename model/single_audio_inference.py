import torch
import torchaudio
import os
import argparse
from model import Generator as NoiseEncoder

def process_audio_file(noise_encoder, input_path, output_path, device):
    # Load audio file
    waveform, sample_rate = torchaudio.load(input_path)
    waveform = waveform.unsqueeze(0).to(device)

    # Generate noise and add to input
    with torch.no_grad():
        noise = noise_encoder(waveform)
        noisy_waveform = torch.clamp(waveform + noise * 0.7, -1, 1)

    # Save noisy audio
    torchaudio.save(output_path, noisy_waveform.squeeze(0).cpu(), sample_rate)
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Add noise to a single audio file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the noisy output audio file')
    parser.add_argument('--model_path', type=str, default='model/checkpoints/generator_wavlm.pth',)
    args = parser.parse_args()

    # Load model
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_encoder = NoiseEncoder().to(device)
    noise_encoder.load_state_dict(torch.load(model_path, map_location=device))
    noise_encoder.eval()

    # Process single audio file
    try:
        processed_path = process_audio_file(noise_encoder, args.input_path, args.output_path, device)
        print(f"Noisy audio saved to: {processed_path}")
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    main()
