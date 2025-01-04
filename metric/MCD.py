import os
import math
import librosa
import pyworld
import pysptk
import numpy as np


def load_wav(wav_file, sr):
    """
    Load a wav file with librosa.
    :param wav_file: path to wav file
    :param sr: sampling rate
    :return: audio time series numpy array
    """
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav


def log_spec_dB_dist(x, y):
    """
    Log-spectral dB distance metric.
    """
    log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    diff = x - y
    return log_spec_dB_const * np.sqrt(np.sum(diff ** 2))


def wav2mcep(wavfile, alpha=0.65, fft_size=512, mcep_size=34, sampling_rate=22050, frame_period=5.0):
    """
    Convert WAV to Mel-Cepstral Coefficients (MCEP).
    """
    # Load WAV file
    loaded_wav = load_wav(wavfile, sr=sampling_rate)

    # WORLD vocoder analysis
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=sampling_rate, 
                                  frame_period=frame_period, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mgc


def calculate_mcd(ref_mcep, synth_mcep):
    """
    Calculate MCD between reference and synthesized MCEP features.
    """
    # Remove the 0th dimension (log-energy) and transpose
    ref_mcep = ref_mcep[:, 1:].T
    synth_mcep = synth_mcep[:, 1:].T

    # Validate dimensions
    if ref_mcep.shape[0] != synth_mcep.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: ref={ref_mcep.shape[0]}, synth={synth_mcep.shape[0]}"
        )

    # Dynamic Time Warping (DTW) with optimal path
    cost_matrix, path = librosa.sequence.dtw(ref_mcep, synth_mcep, metric="euclidean")

    # Compute MCD along the optimal path
    log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    total_cost = 0.0

    for i, j in zip(path[0], path[1]):
        diff = ref_mcep[:, i] - synth_mcep[:, j]
        total_cost += log_spec_dB_const * np.sqrt(np.sum(diff ** 2))

    # Normalize by path length
    mean_mcd = total_cost / len(path[0])

    return mean_mcd


if __name__ == "__main__":
    # File paths
    ref_wav = "/home/work/rvc/FreeVC/voice/content/contents_4.wav"
    synth_wav = "/home/work/rvc/unetAttack/output/noisy_contents_4.wav"


    # Convert WAV to MCEP
    ref_mcep = wav2mcep(ref_wav)
    synth_mcep = wav2mcep(synth_wav)

    # Calculate MCD
    try:
        mcd = calculate_mcd(ref_mcep, synth_mcep)
        print(f"MCD between {os.path.basename(ref_wav)} and {os.path.basename(synth_wav)}: {mcd:.4f} dB")
    except ValueError as e:
        print(f"Error calculating MCD: {e}")
