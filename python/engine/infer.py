import torch
import numpy as np
import librosa
from .pitch import extract_pitch, pitch_to_coarse
from .hubert_model import load_hubert, extract_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVCInfer:
    def __init__(self):
        self.hubert = None
        self.model = None
        self.model_info = None
        self.is_ready = False

    def load(self, model, model_info: dict):
        """Load RVC model and HuBERT for inference"""
        print("[RVCInfer] Loading HuBERT...")
        self.hubert = load_hubert()
        self.model = model
        self.model_info = model_info
        self.is_ready = True
        print("[RVCInfer] Ready for inference")

    def infer_chunk(
        self,
        audio_chunk: np.ndarray,
        input_sr: int = 44100,
        pitch_shift: int = 0,
    ) -> np.ndarray:
        if not self.is_ready or self.model is None:
            return audio_chunk

        sr_raw = self.model_info.get("sample_rate", 48000)
        if isinstance(sr_raw, str):
            target_sr = int(sr_raw.replace("k", "000"))
        else:
            target_sr = int(sr_raw)

        try:
            if input_sr != target_sr:
                audio_16k = librosa.resample(audio_chunk, orig_sr=input_sr, target_sr=16000)
                audio_target = librosa.resample(audio_chunk, orig_sr=input_sr, target_sr=target_sr)
            else:
                audio_16k = librosa.resample(audio_chunk, orig_sr=input_sr, target_sr=16000)
                audio_target = audio_chunk

            f0 = extract_pitch(audio_16k, 16000)
            if pitch_shift != 0:
                f0[f0 > 0] *= 2 ** (pitch_shift / 12)
            f0_coarse = pitch_to_coarse(f0)

            audio_tensor = torch.FloatTensor(audio_16k).unsqueeze(0).to(DEVICE)
            feats = extract_features(self.hubert, audio_tensor)

            feats = torch.nn.functional.interpolate(
                feats.transpose(1, 2),
                size=len(f0_coarse),
                mode="nearest"
            ).transpose(1, 2)

            f0_tensor = torch.LongTensor(f0_coarse).unsqueeze(0).to(DEVICE)
            f0_float = torch.FloatTensor(f0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = self.model.infer(
                    feats,
                    torch.LongTensor([feats.shape[1]]).to(DEVICE),
                    f0_tensor,
                    f0_float,
                    torch.LongTensor([0]).to(DEVICE),
                    nosplit=True
                )[0][0, 0].data.cpu().float().numpy()

            if target_sr != input_sr:
                output = librosa.resample(output, orig_sr=target_sr, target_sr=input_sr)

            return output

        except Exception as e:
            print(f"[RVCInfer] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return audio_chunk
