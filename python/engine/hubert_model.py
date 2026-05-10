import os
import torch
import requests
import torchaudio
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
CACHE_DIR = Path(os.path.expanduser("~")) / ".zer0voices" / "models"

def get_hubert_path() -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(CACHE_DIR / "hubert_base.pt")

def is_hubert_downloaded() -> bool:
    return os.path.exists(get_hubert_path())

def download_hubert(progress_callback=None) -> str:
    path = get_hubert_path()
    print("[HuBERT] Downloading hubert_base.pt...")
    response = requests.get(HUBERT_URL, stream=True)
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total:
                progress_callback(downloaded / total)
    print("[HuBERT] Download complete.")
    return path

def load_hubert():
    path = get_hubert_path()
    if not is_hubert_downloaded():
        download_hubert()

    print("[HuBERT] Loading with torchaudio bundle...")
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().to(DEVICE)
    model.eval()
    print("[HuBERT] Ready.")
    return model

def extract_features(model, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Extract HuBERT content features from audio"""
    with torch.no_grad():
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.to(DEVICE)
        features, _ = model.extract_features(audio)
        return features[-1]