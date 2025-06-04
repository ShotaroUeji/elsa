# import torch
# import soundfile as sf
# import numpy as np
# import librosa
# from transformers import ClapProcessor, ClapAudioModel

# # ----------------------------------------------------------------------------
# # 1. 事前学習済み CLAP AudioEncoder と Processor を読み込む
# # ----------------------------------------------------------------------------
# processor = ClapProcessor.from_pretrained("laion/clap-htsat-iused")
# audio_encoder: ClapAudioModel = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")

# # ----------------------------------------------------------------------------
# # 2. 処理したい FOA WAV ファイルのパスをリストで用意する
# # ----------------------------------------------------------------------------
# # 例: ここでは 3 つのファイルをバッチ処理するイメージ
# foa_paths = [
#     "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/train/0_A/foa.wav",
#     "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/train/0_B/foa.wav"
# ]

# # バッチ用に「Omni チャンネル (W) のモノラル波形」を格納するリストを用意
# batch_mono = []
# target_sr = 48000  # CLAP が期待するサンプリングレート

# for path in foa_paths:
#     # (a) FOA 4ch WAV を読み込む
#     audio_4ch, orig_sr = sf.read(path)  # shape: (num_samples, 4)
#     if audio_4ch.ndim != 2 or audio_4ch.shape[1] != 4:
#         raise RuntimeError(f"{path} が FOA (4ch) 形式ではありません。")

#     # (b) W チャンネル (= index 0) だけを抽出してモノラル化
#     omni = audio_4ch[:, 0].astype(np.float32)  # shape: (num_samples,)

#     # (c) 必要であれば 48kHz にリサンプリング
#     if orig_sr != target_sr:
#         omni = librosa.resample(omni, orig_sr, target_sr)
#     # ここまでで `omni` は 1 次元の NumPy 配列 (long,) になっている

#     # (d) バッチ用リストに追加
#     batch_mono.append(omni)

# # 例: float64 -0.75 0.83   → こういう場合は float32 にキャストし、値域を正規化する

# # ------------------------------------------------
# # ----------------------------
# # 3. ClapProcessor にまとめて渡し、モデル入力用テンソルを作る
# # ----------------------------------------------------------------------------
# #    - `processor(audios=[np1, np2, ...], sampling_rate=48000, return_tensors="pt")`
# #      のようにリストを渡すと、内部でパディング（長さ揃え）や attention_mask 生成をしてくれる
# # ----------------------------------------------------------------------------
# inputs = processor(
#     audios=batch_mono,       # Python のリスト; 要素はすべて 1 次元 NumPy 配列
#     sampling_rate=target_sr, # 48000
#     return_tensors="pt"   
# )

# # デバッグ: キーを確認すると普通は "input_features" だけが返ってくる
# print("processor が返すキー:", inputs.keys())
# # 例: dict_keys(['input_features'])
# feat= inputs["input_features"]
# nan_mask = torch.isnan(feat)
# print("[Debug] processor 出力の input_features に NaN があるか:", nan_mask.any().item())
# for i in range(feat.shape[0]):
#     cnt = torch.isnan(feat[i]).sum().item()
#     print(f"  サンプル {i} の NaN 個数 = {cnt}")
# # ----------------------------------------------------------------------------
# # 4. Audio Encoder (HTSAT 部分) をバッチで呼び出し、[batch_size, 768] を得る
# # ----------------------------------------------------------------------------
# with torch.no_grad():
#     # attention_mask がなければ渡さずとも動くため、input_features のみ渡せば OK
#     audio_outputs = audio_encoder(**inputs)
#     # ここで得られる pooler_output が [batch_size, 768] のテンソル
#     semantic_embeds = audio_outputs.pooler_output  # dddddddddfdf68] など
#     aud= audio_outputs.last_hidden_state
# # ----------------------------------------------------------------------------
# # 5. 結果の確認
# # ----------------------------------------------------------------------------
# print("バッチで得られた埋め込みの形状:", semantic_embeds.shape)
# #  → torch.Size([3, 768]) のように、foa_paths の要素数に応じた第一次元を持つ
# print(semantic_embeds)
# print(aud)

import torch, soundfile as sf, librosa, numpy as np
from transformers import ClapProcessor, ClapAudioModelWithProjection

# -------------------- 1. モデルとプロセッサ --------------------
CKPT = "laion/clap-htsat-fused"               # unfused 版でも可
processor = ClapProcessor.from_pretrained(CKPT)
audio_model = ClapAudioModelWithProjection.from_pretrained(CKPT).eval()  # ★

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
