# ===== foa_dataset.py =====
"""FOA → Dataset
  (メタデータに基づくラベル y の返却を省略し、I_act と I_rea のみを返します)
"""
import ast
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from foa_iv_features import foa_to_iv
import numpy as np
import re
import joblib
from functools import lru_cache

@lru_cache(maxsize=None)
def _load_scalers():
    drt_sc   = joblib.load("scale/drt_scaler.joblib")     # DRR+T30 用 (40 dim)
    voldist_sc = joblib.load("scale/scalar_scaler.joblib")# volume+distance 用 (2 dim)
    return drt_sc, voldist_sc

class FOADataset(Dataset):
    """
    audio_folder にある FOA WAV/FLAC ファイルから
    ・I_act (active intensity)
    ・I_rea (reactive intensity)
    のペアのみを返す Dataset。
    メタデータに基づく y ベクトルは含みません。
    """
    def __init__(self, audio_folder: str, metadata_csv: str, sr: int = 16_000):
        self.audio_dir = Path(audio_folder)
        self.meta = pd.read_csv(metadata_csv, index_col=0, dtype={"Unnamed: 0": str})
        self.meta.index = self.meta.index.map(lambda x: str(x).zfill(6))
        self.sr = sr
        self.T = sr * 10
        # フォルダ内の *.flac とメタデータ行が一致するものだけをリスト化
        self.files = [p for p in sorted(self.audio_dir.glob("*.flac")) if p.stem in self.meta.index]
        if not self.files:
            raise FileNotFoundError("No matching FLACs and metadata rows")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        # ① WAV/FLAC 読み込み → 音声長を T 秒以内に切り詰め
        path = self.files[idx]
        wav, sr = torchaudio.load(path)  # (channels, num_samples)
        wav = wav[:, : self.T]

        # ② FOA → I_act, I_rea の計算
        #    foa_to_iv は (1, 4, T) の形を受け取って
        #    I_act, I_rea を (1, 3, H, W) で返す関数と想定
        I_act, I_rea = foa_to_iv(wav.unsqueeze(0))
        # バッチ次元を外して (3, H, W) の形に
        I_act, I_rea = I_act[0], I_rea[0]

        # ③ y は不要なので返さない
        return I_act, I_rea
