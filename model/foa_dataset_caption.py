# ===== foa_dataset_caption.py =====
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
from pathlib import Path

class FOADatasetWithCaptionOnly(Dataset):
    """
    FOA (4 チャンネル) の生波形とキャプションだけを返す Dataset。
    metadata_csv は少なくとも以下の列を持つものとする:
      - 'filename': 音声ファイル名の stem (例: "000123")
      - 'caption' : その音声に対応するキャプション文字列

    1 つの音声ファイルに対して複数行のキャプションがある場合は、
    CSV の行数分だけ同じ音声を読み込み、別々のキャプションを返す。
    """
    def __init__(self, audio_folder: str, metadata_csv: str, sr: int = 48000):
        super().__init__()
        self.audio_dir = Path(audio_folder)
        # CSV を DataFrame として読み込み ('filename' と 'caption' の2列が必要)
        self.meta = pd.read_csv(metadata_csv, dtype={'filename': str, 'caption': str})

        # 存在しないファイル名はスキップしておく
        valid_indices = []
        for idx, row in self.meta.iterrows():
            fname = row['filename']
            p = self.audio_dir / f"{fname}.flac"
            if p.is_file():
                valid_indices.append(idx)
            else:
                # 警告を出してスキップ
                print(f"[Warning] Audio file not found, skipping: {p}")
        self.meta = self.meta.loc[valid_indices].reset_index(drop=True)

        self.sr = sr
        self.max_length = sr * 10  # 最大 10 秒分のサンプル数

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int):
        """
        Returns:
          - foa_waveform: Tensor of shape (4, L)  ･･･ 生波形, L ≤ max_length
          - caption     : str
        """
        row = self.meta.iloc[idx]
        fname = row['filename']
        caption_text = row['caption']

        # --- 1. FOA 音声ファイルを読み込む ---
        audio_path = self.audio_dir / f"{fname}.flac"
        # torchaudio.load は (channels, num_samples) の FloatTensor を返す (dtype=float32)
        wav, original_sr = torchaudio.load(audio_path)
        # wav.shape = (4, original_length)

        # --- 2. サンプリングレートが異なる場合はリサンプリング ---
        if original_sr != self.sr:
            wav = torchaudio.functional.resample(wav, orig_freq=original_sr, new_freq=self.sr)

        # --- 3. 最大 10 秒分 (max_length) に切り詰める ---
        if wav.size(1) > self.max_length:
            wav = wav[:, :self.max_length]  # (4, max_length)
        # 10 秒より短い場合はそのまま (4, L<L_max)

        # 戻り値は (FOA_waveform, キャプション文字列)
        return wav, caption_text
