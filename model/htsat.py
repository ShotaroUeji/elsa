import torch, soundfile as sf, librosa, numpy as np
from transformers import ClapProcessor, ClapAudioModelWithProjection

# -------------------- 1. モデルとプロセッサ --------------------
CKPT = "laion/clap-htsat-fused"               # unfused 版でも可
processor = ClapProcessor.from_pretrained(CKPT)
audio_model = ClapAudioModelWithProjection.from_pretrained(CKPT)  # ★

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
audio_model.to(DEVICE)

# -------------------- 2. FOA → W チャンネルを 48 kHz へ ---------------
def foa_w_to_48k(path: str, target_sr: int = 48_000) -> np.ndarray:
    """FOA 4-ch wav を読み、W(= ch0) を float32 で 48 kHz に整形"""
    wav4, sr = sf.read(path)                      # shape: (N, 4)
    w = wav4[:, 0].astype(np.float32)             # W チャンネル (モノラル)
    if sr != target_sr:
        w = librosa.resample(w, sr, target_sr).astype(np.float32)
    # 完全無音なら微小ノイズで NaN 連鎖を防止
    if np.allclose(w, 0.0):
        w += 1e-8 * np.random.randn(*w.shape).astype(np.float32)
    # 1 s 未満はゼロ埋め（CLAP の安定動作範囲）
    min_len = target_sr
    if w.size < min_len:
        w = np.pad(w, (0, min_len - w.size))
    return w
foa_paths = [
    "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/train/0_A/foa.wav",
    "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/train/0_B/foa.wav"
]

batch_wavs = [foa_w_to_48k(p) for p in foa_paths]

# -------------------- 3. ClapProcessor で input_features ---------------
inputs = processor(
    audios=batch_wavs,
    sampling_rate=48_000,
    return_tensors="pt",
    padding="repeatpad",     # 10 s (=480 000) にそろえる
    truncation="fusion",     # True は未実装エラーになるので注意
    max_length=480_000,
).to(DEVICE)

assert not torch.isnan(inputs["input_features"]).any(), "NaN in features!"

# -------------------- 4. 推論 → audio_embeds (512 dim) ------------------
with torch.no_grad():
    outputs = audio_model(**inputs)        # ClapAudioModelWithProjection
    audio_embeds = outputs.audio_embeds    # shape: [B, 512] (L2 正規化済)

print(audio_embeds.shape)  # e.g. torch.Size([2, 512])
print(audio_embeds)        # NaN が無ければ成功
