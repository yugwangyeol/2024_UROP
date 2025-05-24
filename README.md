<<<<<<< HEAD
# Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors


Official Implementation of the Interspeech 2025 paper Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors

![Image](https://github.com/user-attachments/assets/3529efb5-0c14-4447-9dc4-a696156eee48)

Voice conversion (VC) enables natural speech synthesis with minimal data; however, it poses security risks, e.g., identity theft and privacy breaches. To address this, we propose Mimic Blocker, an active defense mechanism that prevents VC models from extracting speaker characteristics while preserving audio quality. Our method employs adversarial training, an audio quality preservation strategy, and an attack strategy. It relies on only publicly available pretrained feature extractors, which ensures model-agnostic protection. Furthermore, it enables self-supervised learning using only the original speaker's speech. Experimental results demonstrate that our method achieves robust defense performance in both white-box and black-box scenarios. Notably, the proposed approach maintains audio quality by generating noise imperceptible to human listeners, thereby enabling protection while retaining natural voice characteristics in practical applications.

## Demo

You can find our Demo here

## How to Use
### 1. Clone the repository
```bash
git clone https://github.com/yugwangyeol/Mimic-Blocker.git
cd Mimic-Blocker
```
```bash
git clone https://github.com/OlaWod/FreeVC.git
```
Download WavLM-Large and put it under directory 'wavlm/'
```bash
git clone https://github.com/winddori2002/TriAAN-VC.git
```

### 2. Install requirements
```bash
pip install -r requirements.txt --no-deps
pip install "typing-extensions<4.6.0"
```

### 3. Download VCTK Dataset
``` python
import torch
import torchaudio

torchaudio.datasets.VCTK_092(root="data", download=True)
```

### 
