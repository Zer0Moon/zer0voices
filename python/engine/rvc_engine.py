import os
import torch
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RVCEngine:
    def __init__(self):
        self.model = None
        self.net_g = None
        self.model_path = None
        self.model_info = {}
        self.is_loaded = False
        self.infer = None
        print(f"[RVCEngine] Using device: {DEVICE}")

    def detect_version(self, cpt: dict) -> str:
        if "version" in cpt:
            return cpt["version"]
        if "weight" in cpt:
            keys = list(cpt["weight"].keys())
            if any("enc_q" in k for k in keys):
                return "v2"
        return "v1"

    def load_model(self, pth_path: str) -> dict:
        try:
            print(f"[RVCEngine] Loading: {pth_path}")
            cpt = torch.load(pth_path, map_location="cpu", weights_only=False)

            version = self.detect_version(cpt)
            sr = cpt.get("sr", 40000)
            f0 = cpt.get("f0", 1)
            cfg = cpt.get("config", [])

            # Build model from checkpoint config
            from .synthesizer import Synthesizer
            net_g = Synthesizer(*cfg, is_half=False, version=version)
            net_g.eval()

            # Load weights
            weights = {k.replace("module.", ""): v for k, v in cpt["weight"].items()}
            net_g.load_state_dict(weights, strict=False)
            net_g = net_g.float().to(DEVICE)

            self.net_g = net_g
            self.model_path = pth_path
            self.model_info = {
                "path": pth_path,
                "name": Path(pth_path).stem,
                "version": version,
                "sample_rate": sr,
                "f0": f0,
                "size_mb": round(os.path.getsize(pth_path) / (1024**2), 1)
            }
            self.is_loaded = True

            # Init inferencer
            from .infer import RVCInfer
            self.infer = RVCInfer()
            self.infer.load(net_g, self.model_info)

            print(f"[RVCEngine] Loaded: {self.model_info}")
            return {"success": True, "info": self.model_info}

        except Exception as e:
            print(f"[RVCEngine] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def convert(self, audio: np.ndarray, sr: int, pitch_shift: int = 0) -> np.ndarray:
        if not self.is_loaded or self.infer is None:
            return audio
        return self.infer.infer_chunk(audio, input_sr=sr, pitch_shift=pitch_shift)

    def unload(self):
        self.net_g = None
        self.infer = None
        self.is_loaded = False
        self.model_info = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[RVCEngine] Unloaded")

    def get_onnx_path(self, pth_path: str) -> str:
        p = Path(pth_path)
        cache_dir = p.parent / ".zer0voices_cache"
        cache_dir.mkdir(exist_ok=True)
        return str(cache_dir / f"{p.stem}.onnx")

    def is_onnx_cached(self, pth_path: str) -> bool:
        return os.path.exists(self.get_onnx_path(pth_path))