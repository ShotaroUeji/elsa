# ===== foa_dataset.py =====
"""FOA + metadata → Dataset
ラベル構成
    - scalar_cols  : 4 要素 (azimuth, elevation, distance, room/volume)
    - list_cols    : 2 要素 (drr_db, t30_ms) それぞれ 33 長さのリスト → idx 10‑29 (20要素)
    => y.length = 4 + 20 + 20 = 44
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

class FOALabeledDataset(Dataset):
    SCALAR_COLS = [
        "speech/azimuth",
        "speech/elevation",
        "speech/distance",
        "room/volume",
    ]
    LIST_COLS = [
        "acoustics/drr_db",
        "acoustics/t30_ms",
    ]

    def __init__(self, audio_folder: str, metadata_csv: str, sr: int = 16_000):
        self.audio_dir = Path(audio_folder)
        self.meta = pd.read_csv(metadata_csv, index_col=0, dtype={"Unnamed: 0": str})
        self.meta.index = self.meta.index.map(lambda x: str(x).zfill(6))
        self.sr = sr
        self.T = sr * 10
        # flac と meta 行が一致するものだけ
        self.files = [p for p in sorted(self.audio_dir.glob("*.flac")) if p.stem in self.meta.index]
        if not self.files:
            raise FileNotFoundError("No matching FLACs and metadata rows")

    def __len__(self):
        return len(self.files)

    def _parse_list_field(self, field_str: str) -> list[float]:
        """
        '[a b …]' または '[a, b, …]' を float list に変換し、
        33 要素中 10〜29 番目 (index 9–28) の 20 要素だけ返す。
        """
        cleaned = re.sub(r"[\[\],]", " ", field_str)           # "," → 空白
        nums = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        if len(nums) != 33:
            raise ValueError(f"Expected 33 elems, got {len(nums)} : {field_str[:60]}")
        return nums[9:29].tolist()     # 20 要素

    def __getitem__(self, idx: int):
        # ① 既存処理 ----------------------------
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        wav = wav[:, : self.T]
        I_act, I_rea = foa_to_iv(wav.unsqueeze(0))
        I_act, I_rea = I_act[0], I_rea[0]
        row = self.meta.loc[path.stem]

        # ② スカラー抽出（順序は変えない）
        az, el, dist, vol = [row[c] for c in self.SCALAR_COLS]

        # ③ DRR / T30 抽出 & 20 バンドにスライス
        drr_20 = self._parse_list_field(row["acoustics/drr_db"])   # 20 要素
        t30_20 = self._parse_list_field(row["acoustics/t30_ms"])   # 20 要素

        # ④ ----------- 正規化を適用 -----------
        drt_sc, distvol_sc = _load_scalers()

        drt_vec  = np.hstack([drr_20, t30_20]).reshape(1, -1)      # (1,40)
        drt_norm = drt_sc.transform(drt_vec)[0]                    # (40,)

        vd_vec   = np.array([[dist, vol]], dtype=np.float32)       # (1,2)
        vd_norm  = distvol_sc.transform(vd_vec)[0]                 # (2,) 0-1

        # ⑤ y ベクトルを元と同じ順で連結 (44,)
        y = torch.tensor(
            [az, el] + vd_norm.tolist() + drt_norm.tolist(),
            dtype=torch.float32
        )
        return I_act, I_rea, y
