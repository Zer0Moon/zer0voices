import torch
import numpy as np
import torchcrepe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_pitch(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Extract f0 pitch using torchcrepe on GPU"""
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

    f0, periodicity = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=50,
        fmax=1100,
        model="tiny",
        batch_size=512,
        device=DEVICE,
        return_periodicity=True
    )

    # Zero out unvoiced frames
    f0 = torchcrepe.threshold.Silence(-60)(f0, audio_tensor, sr, hop_length)
    f0[periodicity < 0.1] = 0

    return f0.squeeze().cpu().numpy()

def pitch_to_coarse(f0: np.ndarray) -> np.ndarray:
    """Convert f0 to coarse pitch bins for RVC"""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - 550) / (6500 - 550) * 254 + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int32)