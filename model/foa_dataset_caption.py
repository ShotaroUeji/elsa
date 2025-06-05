import math
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# W, Y, Z, XはFOAの形式によって変える必要あり
#############################################
# Constants
#############################################
# 10‑second fixed duration (cannot be changed)
MAX_DURATION_SEC: float = 10.0
# Fixed FOA sample‑rate
FOA_SR: int = 48_000  # Omni is kept at this rate
# Sample‑rate used for I_act / I_rea computation
IV_SR: int = 16_000


# -----------------------------------------------------------------------------
# Utility: FOA → (I_act, I_rea)
# -----------------------------------------------------------------------------

def foa_to_iv(
    foa_wave: torch.Tensor,
    sr: int = IV_SR,
    n_fft: int = 400,
    hop: int = 100,
    eps: float = 1e-6,
):
    """Convert FOA waveform to *active* / *reactive* intensity vectors.

    Parameters
    ----------
    foa_wave : Tensor
        (B, 4, T) — FOA (W, Y, Z, X) waveform **at ``sr``**.
    sr : int, optional
        Sample‑rate of *foa_wave* (default: 16 kHz).  
        *For API consistency only — the algorithm itself is SR‑agnostic.*
    n_fft, hop, eps : int / float, optional
        STFT parameters & small value to avoid division by zero.

    Returns
    -------
    I_act, I_rea : Tensor, Tensor
        Shape: (B, 3, F, N)
    """
    B, C, T = foa_wave.shape
    assert C == 4, "FOA waveform must have 4 channels (W, Y, Z, X)"

    win = torch.hann_window(n_fft, device=foa_wave.device)

    # STFT: (B, 4, F, N)
    spec = (
        torch.stft(
            foa_wave.view(-1, T),
            n_fft=n_fft,
            hop_length=hop,
            window=win,
            center=True,
            return_complex=True,
        )
        .view(B, 4, n_fft // 2 + 1, -1)
        .contiguous()
    )
 
    W, Y, Z, X = spec[:, 0], spec[:,1], spec[:, 2], spec[:, 3]
    conjW = W.conj()

    # Active / Reactive intensity
    I_act = torch.stack([(conjW * Y).real, (conjW * Z).real, (conjW * X).real], dim=1)
    I_rea = torch.stack([(conjW * Y).imag, (conjW * Z).imag, (conjW * X).imag], dim=1)

    # Unit‑norm (skip zero‑vectors)
    norm = torch.linalg.norm(I_act, dim=1, keepdim=True)  # (B,1,F,N)
    I_act = torch.where(norm > eps, I_act / norm, I_act)
    I_rea = torch.where(norm > eps, I_rea / norm, I_rea)

    return I_act.float(), I_rea.float()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class FOADatasetWithIV(Dataset):
    """Dataset that returns **(I_act, I_rea, Omni‑48k, caption)**.

    * **I_act / I_rea** are computed on 16‑kHz down‑sampled FOA.
    * **Omni** (W‑channel) remains at the original 48‑kHz.
    * All examples are *exactly 10 s* long.
      If the original clip is shorter, it is **cyclically repeated**
      until 10 s and then truncated.

    The ``metadata_csv`` must have at least two columns:
    ``filename`` (stem without extension) and ``caption``.
    """

    def __init__(
        self,
        audio_folder: str,
        metadata_csv: str,
        n_fft: int = 400,
        hop: int = 100,
    ) -> None:
        super().__init__()

        self.audio_dir = Path(audio_folder)
        self.meta = pd.read_csv(metadata_csv, dtype={"audio_filename": str, "caption": str})

        # Keep only rows whose audio exists.
        valid_idx = []
        for idx, row in self.meta.iterrows():
            if (self.audio_dir / f"{row['audio_filename']}.wav").is_file():
                valid_idx.append(idx)
            else:
                print(f"[Warning] Audio file not found, skipping: {row['audio_filename']}.wav")
        self.meta = self.meta.loc[valid_idx].reset_index(drop=True)

        # Pre‑compute lengths
        self.foa_len = int(FOA_SR * MAX_DURATION_SEC)  # 480 000 samples
        self.iv_len = int(IV_SR * MAX_DURATION_SEC)    # 160 000 samples

        self.n_fft = n_fft
        self.hop = hop

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.meta)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):  # noqa: D401
        row = self.meta.iloc[idx]
        fname = row["audio_filename"]
        caption_text = row["caption"]

        # --------------------------------------------------------------
        # 1) Load FOA audio (expecting 48‑kHz FLAC, 4 channels)
        # --------------------------------------------------------------
        path = self.audio_dir / f"{fname}.wav"
        wav, orig_sr = torchaudio.load(path)  # (4, L_orig)

        # Resample to 48 k if necessary
        if orig_sr != FOA_SR:
            wav = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=FOA_SR)

        # --------------------------------------------------------------
        # 2) Enforce EXACT 10‑second length (repeat‑&‑trim)
        # --------------------------------------------------------------
        if wav.size(1) < self.foa_len:
            # Repeat cyclically until ≥ 10 s, then cut
            repeat_factor = math.ceil(self.foa_len / wav.size(1))
            wav = wav.repeat(1, repeat_factor)[:, : self.foa_len]
        else:
            wav = wav[:, : self.foa_len]

        # --------------------------------------------------------------
        # 3) Omni at 48 kHz
        # --------------------------------------------------------------
        omni_48k = wav[0]  # (T=480 000)

        # --------------------------------------------------------------
        # 4) Down‑sample FOA to 16 kHz for IV computation
        # --------------------------------------------------------------
        wav_16k = torchaudio.functional.resample(wav, orig_freq=FOA_SR, new_freq=IV_SR)
        if wav_16k.size(1) < self.iv_len:  # In theory should match exactly, but be safe
            rep = math.ceil(self.iv_len / wav_16k.size(1))
            wav_16k = wav_16k.repeat(1, rep)[:, : self.iv_len]
        elif wav_16k.size(1) > self.iv_len:
            wav_16k = wav_16k[:, : self.iv_len]

        # --------------------------------------------------------------
        # 5) FOA (16 k) → (I_act, I_rea)
        # --------------------------------------------------------------
        I_act, I_rea = foa_to_iv(
            wav_16k.unsqueeze(0),  # add batch dim
            sr=IV_SR,
            n_fft=self.n_fft,
            hop=self.hop,
        )
        I_act, I_rea = I_act.squeeze(0), I_rea.squeeze(0)  # (3,F,N)
        print(f"[Info] Loaded {fname}: I_act shape {I_act.shape}, I_rea shape {I_rea.shape}")
        return I_act, I_rea, omni_48k, caption_text
