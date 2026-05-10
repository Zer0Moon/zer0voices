import os
import torch
import numpy as np
from pathlib import Path

class RVCEngine:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.onnx_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.model_info = {}
        print(f"[RVCEngine] Using device: {self.device}")

    def detect_version(self, cpt: dict) -> str:
        """Auto-detect RVC v1 or v2 from checkpoint"""
        if "version" in cpt:
            return cpt["version"]
        # Fallback: detect by model architecture
        if "weight" in cpt:
            keys = list(cpt["weight"].keys())
            if any("enc_q" in k for k in keys):
                return "v2"
        return "v1"

    def load_model(self, pth_path: str) -> dict:
        """Load a .pth model file and return info about it"""
        try:
            print(f"[RVCEngine] Loading model: {pth_path}")
            cpt = torch.load(pth_path, map_location="cpu", weights_only=False)

            version = self.detect_version(cpt)
            sr = cpt.get("sr", 40000)
            f0 = cpt.get("f0", 1)

            self.model_path = pth_path
            self.model_info = {
                "path": pth_path,
                "name": Path(pth_path).stem,
                "version": version,
                "sample_rate": sr,
                "f0": f0,
                "size_mb": round(os.path.getsize(pth_path) / (1024 * 1024), 1)
            }

            print(f"[RVCEngine] Model loaded: {self.model_info}")
            return {"success": True, "info": self.model_info}

        except Exception as e:
            print(f"[RVCEngine] Error loading model: {e}")
            return {"success": False, "error": str(e)}

    def get_onnx_path(self, pth_path: str) -> str:
        """Get the expected ONNX cache path for a model"""
        p = Path(pth_path)
        cache_dir = p.parent / ".zer0voices_cache"
        cache_dir.mkdir(exist_ok=True)
        return str(cache_dir / f"{p.stem}.onnx")

    def is_onnx_cached(self, pth_path: str) -> bool:
        """Check if ONNX version already exists"""
        return os.path.exists(self.get_onnx_path(pth_path))

    def unload(self):
        """Unload current model and free GPU memory"""
        self.model = None
        self.is_loaded = False
        self.model_info = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[RVCEngine] Model unloaded")