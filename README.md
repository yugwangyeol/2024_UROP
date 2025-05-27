# Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-1234.1234-b31b1b.svg?style=flat-square)]()
[![LLaMA](https://img.shields.io/badge/Project_Page-MimicBlocekr-FFB000.svg?style=flat-square)](https://2junhyeok.github.io/MimicBlocker/)

Official Implementation of the Interspeech 2025 paper Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors

![Image](https://github.com/user-attachments/assets/3529efb5-0c14-4447-9dc4-a696156eee48)

Voice conversion (VC) enables natural speech synthesis with minimal data; however, it poses security risks, e.g., identity theft and privacy breaches. To address this, we propose Mimic Blocker, an active defense mechanism that prevents VC models from extracting speaker characteristics while preserving audio quality. Our method employs adversarial training, an audio quality preservation strategy, and an attack strategy. It relies on only publicly available pretrained feature extractors, which ensures model-agnostic protection. Furthermore, it enables self-supervised learning using only the original speaker's speech. Experimental results demonstrate that our method achieves robust defense performance in both white-box and black-box scenarios. Notably, the proposed approach maintains audio quality by generating noise imperceptible to human listeners, thereby enabling protection while retaining natural voice characteristics in practical applications.

## Demo

You can find our [Demo](https://2junhyeok.github.io/MimicBlocker/) here

## Project Structure

```
Mimic-Blocker/
│
├── requirements.txt
│
├── FreeVC/ (utils.py updated)                
│
├── TriAAN-VC/ (src/vocoder.py, config/base.yaml updated)
│
├── data/
│   ├── VCTK-Corpus-0.92/                     
│   ├── {FreeVC/TriAAN-VC}_original/          
│   ├── {FreeVC/TriAAN-VC}_noise_{wavlm/hubert}/ 
│   ├── {FreeVC/TriAAN-VC}_noisy_style_{wavlm/hubert}/ 
│   ├── {FreeVC/TriAAN-VC}_test_pairs_{wavlm/hubert}.txt 
│   ├── {FreeVC/TriAAN-VC}_test_noisy_pairs_{wavlm/hubert}.txt 
│   ├── train.txt                             
│   ├── val.txt                                
│   └── test.txt                               
│
├── model/
│   ├── checkpoints/                          
│   │   └── generator_{wavlm/hubert}.pth
│   ├── inference.py                          
│   ├── main.py                               
│   ├── model.py                              
│   ├── train.py                              
│   └── single_audio_inference.py             
│
├── VC_inference/
│   ├── Freevc_inference.py                   
│   └── TriAANVC_inference.py                 
│
├── evaluation/  
│   ├── pretrained models/                    
│   └── evaluation.py                         
│
└── data_processing/ 
    ├── test_split.py                         
    └── train_test_split.py                   
```

## Pre-requisites
### 1. Clone the repository
```bash
git clone https://github.com/yugwangyeol/Mimic-Blocker.git
cd Mimic-Blocker
```
```bash
git clone https://github.com/OlaWod/FreeVC.git
```
- Download [freevc.pth](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbnZ1a1ZubFEzWlR4MXJqck9aMmFiQ3d1QkFoP2U9VWxoUlI1&id=537643E55991EE7B%219178&cid=537643E55991EE7B) and put it under directory 'checkpoints/'

- Download [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) and put it under directory 'wavlm/'

- Rename the folder 'configs' to 'logs'
```bash
git clone https://github.com/winddori2002/TriAAN-VC.git
```
- Download [model-cpc-split.pth](https://github.com/winddori2002/TriAAN-VC/releases/tag/v1.0) and put it under directory 'checkpoints/'

- Download [cpc.pt](https://github.com/winddori2002/TriAAN-VC/releases/tag/v1.0) and put it under directory 'cpc/'

- Download [vocoder.pkl](https://github.com/winddori2002/TriAAN-VC/releases/tag/v1.0) and put it under directory 'vocoder/'

<details>
  <summary>Modify the code in 'FreeVC' and 'TriAAN-VC'</summary>
  <div markdown="1">
    <ul>
      <li><code>FreeVC/utils.py</code></li>
    </ul>

<pre><code class="language-python">
# Original code (line 24)
checkpoint = torch.load('wavlm/WavLM-Large.pt')

# Modified
checkpoint = torch.load('FreeVC/wavlm/WavLM-Large.pt')
</code></pre>

<ul>
      <li><code>TriAAN-VC/src/vocoder.py</code></li>
    </ul>

<pre><code class="language-python">
# Original code (line 20)
checkpoint = './vocoder/vocoder.pkl'

# Modified
checkpoint = 'TriAAN-VC/vocoder/vocoder.pkl'
</code></pre>

<ul>
      <li><code>TriAAN-VC/config/base.yaml</code></li>
    </ul>

<pre><code class="language-python">
# Original code (line 1~8)
data_path:       ./base_data
wav_path:        ./vctk/wav48_silence_trimmed
txt_path:        ./vctk/txt
spk_info_path:   ./vctk/speaker-info.txt
converted_path: 
vocoder_path:    ./vocoder
cpc_path:        ./cpc
n_uttr:

# Modified
data_path:       TriAAN-VC/base_data
wav_path:        TriAAN-VC/vctk/wav48_silence_trimmed
txt_path:        TriAAN-VC/vctk/txt
spk_info_path:   TriAAN-VC/vctk/speaker-info.txt
converted_path: 
vocoder_path:    TriAAN-VC/vocoder
cpc_path:        TriAAN-VC/cpc
n_uttr:
</code></pre>

  </div>
</details>


### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Download VCTK Dataset
``` python
import torch
import torchaudio

torchaudio.datasets.VCTK_092(root="data", download=True)
```

### 4. Split Dataset (Train/Val/Test)
```python
python data_processing/train_test_split.py
```
- Generate `train.txt`, `val.txt`, and `text.txt` based on the VCTK speakers

## Inference
```python
python model/single_audio_inference.py --input_path </path/to/wavs> --output_path </path/to/outputdir> --model_path </path/to/pretrained_model>
```

- `input_path` :  Path to a single `.wav` file to which adversarial noise will be added.
- Download [pretrained models](https://drive.google.com/drive/folders/1diE18-47wxZdk1B48KS4zJSF1sP8ogYG?usp=sharing) and put it under directory 'model/checkpoints/'

## Training

Use `train.txt` to train the adversarial noise generator and generate checkpoints.
```python
python model/main.py --feature_extractor wavlm
```
- You can choose the `feature_extractor`:
  - If `wavlm` is selected, the model will generate `generator_wavlm.pth`.
  - If `hubert` is selected, the model will generate `generator_hubert.pth`.
  
## Evaluation
### 1. Generate Test Pairs for VC

Extract valid `(x, t)` pairs based on ASV results for VC input generation.
```python
# FreeVC model (default)
python data_processing/test_split.py --model FreeVC

# TriAAN-VC model
python data_processing/test_split.py --model TriAAN-VC
```
- Saves `(x, t)` pairs to `data/{model}_test_pairs.txt`
- Saves `F(x, t)` to `data/{model}_original`

### 2. Generate Noisy Styles `(x → x')`

Add adversarial noise to `x` to generate `x'` for VC input.

```bash
# FreeVC model (default)
python model/inference.py --model FreeVC --feature_extractor wavlm

# TriAAN-VC model
python model/inference.py --model TriAAN-VC --feature_extractor wavlm
```

- Takes `x` from `data/{model}_test_pairs.txt`
- Saves `x'` to `data/{model}_noisy_style_{feature_extractor}`
- Saves `(x', t)` pairs to `data/{model}_test_noisy_pairs_{feature_extractor}.txt`

<details>
  <summary><strong>Note on feature extractor (default: <code>wavlm</code>)</strong></summary>

- **FreeVC:**
  - `wavlm`: White-box scenario  
  - `hubert`: Black-box scenario  
- **TriAAN-VC:** Always Black-box (both `wavlm` and `hubert`)

</details>


### 3. Convert Noisy Pairs with VC Model

Generate `F(x', t)` by feeding `(x', t)` into the VC model.

```bash
# FreeVC
python VC_inference/FreeVC_inference.py --feature_extractor wavlm

# TriAAN-VC
python VC_inference/TriAANVC_inference.py --feature_extractor wavlm
```

- Takes `(x', t)` from `data/{model}_test_noisy_pairs_{feature_extractor}.txt`
- Saves `F(x', t)` to `data/{model}_noise_{feature_extractor}`
- Appends `F(x', t)` path to the 3rd column of `data/{model}_test_noisy_pairs_{feature_extractor}.txt`

### 4. Evaluate Defense Performance

Evaluate performance using PESQ/STOI/ASR/PSR metrics.

```bash
# FreeVC model (default)
python evaluation/evaluation.py --model FreeVC --feature_extractor wavlm

# TriAAN-VC model
python evaluation/evaluation.py --model TriAAN-VC --feature_extractor wavlm
```

- Retrieves:
  - `x` from `data/{model}_test_pairs.txt`
  - `x'`, `F(x', t)` from `data/{model}_test_noisy_pairs_{feature_extractor}.txt`

# Ciations

```
@article{2025mimicblocekr,
  title={Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors},
  author={Yu, Gwang Yeol and Lee, Jun Hyeok and Kim, Seo Ryeong and Lee, Ji Min},
  journal={},
  year={2025}
}
```
