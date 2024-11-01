import torch
import torchaudio
from transformers import WavLMModel
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

def extract_embeddings():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WavLM 모델 로드
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    wavlm.eval()

    # 데이터 로드
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=f'{root}/data',
        url='train-clean-100',
        download=True
    )

    # DataLoader 설정 (배치 사이즈를 더 크게 설정 가능)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 임베딩 저장용 리스트
    all_embeddings = []

    # 임베딩 추출
    with torch.no_grad():
        for waveforms in tqdm(loader, desc="Extracting embeddings"):
            waveforms = waveforms.to(device)

            # WavLM 임베딩 추출
            outputs = wavlm(waveforms.squeeze(1)).last_hidden_state

            # 각 음성의 임베딩을 평균하여 하나의 벡터로 만듦
            embeddings = outputs.mean(dim=1)  # [batch_size, hidden_dim]

            all_embeddings.append(embeddings.cpu().numpy())

    # 모든 임베딩을 하나의 배열로 합침
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    return all_embeddings

def perform_clustering(embeddings, n_clusters=10):
    # KMeans 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    # 센트로이드 추출
    centroids = kmeans.cluster_centers_

    # 텐서로 변환하여 저장
    centroids_tensor = torch.from_numpy(centroids).float()
    torch.save(centroids_tensor, 'wavlm_centroids.pt')

    return centroids_tensor

def collate_fn(batch):
    waveforms = []
    max_length = max(waveform.shape[1] for waveform, *_ in batch)
    for waveform, *_ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

if __name__ == "__main__":
    print("Starting embedding extraction...")
    embeddings = extract_embeddings()

    print("Starting clustering...")
    centroids = perform_clustering(embeddings)

    print(f"Clustering complete. Centroids shape: {centroids.shape}")
    print("Centroids saved to 'wavlm_centroids.pt'")
