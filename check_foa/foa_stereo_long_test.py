#!/usr/bin/env python3
"""
piyo_toggle_pyroom.py
---------------------
Pyroom-Acoustics を使って
  * ビープ信号を ±Y へ 2 秒ごとに配置
  * A→B 変換 → Blumlein (α=6) ステレオ化
  * WAV 書き出し & 波形プロット
"""

import math, numpy as np, soundfile as sf, matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.signal import fftconvolve

# ---------- マイク（正四面体＋カーディオイド） ----------
def build_tetra(center, fs, r=0.05):
    v = r / math.sqrt(3)
    tet = np.array([[ v,  v,  v],
                    [ v, -v, -v],
                    [-v,  v, -v],
                    [-v, -v,  v]], float).T
    dirs = []
    for x, y, z in tet.T:
        az  = math.degrees(math.atan2(y, x)) % 360
        col = math.degrees(math.acos(z / r))
        dirs.append(pra.directivities.CardioidFamily(
            orientation=pra.directivities.DirectionVector(
                azimuth=az, colatitude=col, degrees=True),
            p=0.5, gain=1.0))
    return pra.MicrophoneArray(center[:, None] + tet, fs, directivity=dirs)

# ---------- A → B ----------
def a2b(m0, m1, m2, m3):
    W = (m0 + m1 + m2 + m3) / 2
    X = (m0 + m1 - m2 - m3) / 2
    Y = (m0 - m1 + m2 - m3) / 2
    Z = (m0 - m1 - m2 + m3) / 2
    return W, X, Y, Z

# ---------- B → Stereo（Blumlein＋Y ブースト α=6） ----------
def b2stereo(W, X, Y, alpha=6.0):
    L = (X + alpha * Y) / math.sqrt(2)
    R = (X - alpha * Y) / math.sqrt(2)
    st = np.stack([L, R], axis=1)
    st /= np.max(np.abs(st) + 1e-12)
    return st.astype(np.float32)

# ---------- パラメータ ----------
fs         = 48_000
room_dim   = (12, 12, 6)
ctr        = np.array(room_dim) / 2
seg_sec    = 2
cycles     = 2                  # 4 秒 (= 左2 + 右2) ×2 = 8 s
y_offset   = 4.0                # ±4 m

# ---------- ビープ (“ぴよ”) 生成 (0.1 s ON / 0.1 s OFF) ----------
beep_len = int(0.1 * fs)
beep     = np.sin(2 * np.pi * 2000 * np.arange(beep_len) / fs).astype(np.float32) * 0.6
piyo_seg = np.tile(np.concatenate([beep, np.zeros_like(beep)]), int(seg_sec / 0.2))

# ---------- Pyroom-Acoustics シミュレーション ----------
room = pra.ShoeBox(room_dim, fs=fs, max_order=0)
room.add_microphone_array(build_tetra(ctr, fs))

stereo_buf = []
for _ in range(cycles):
    for sgn in (+1, -1):                   # +Y (左) → -Y (右)
        room.sources = []
        room.add_source((ctr + np.array([0, sgn * y_offset, 0])).tolist(),
                        signal=piyo_seg)
        room.compute_rir()

        outs = [fftconvolve(piyo_seg, room.rir[m][0])[:len(piyo_seg)]
                for m in range(4)]
        W, X, Y, Z = a2b(*outs)
        stereo_buf.append(b2stereo(W, X, Y))

stereo = np.concatenate(stereo_buf, axis=0)
sf.write("piyo_toggle.wav", stereo, fs)
print("✅  piyo_toggle.wav  を出力しました（ヘッドホンで 2 s ごとに左↔右）")

# ---------- 波形プロット (先頭 4 s) ----------
plot_len = int(4 * fs)
t = np.arange(plot_len) / fs
plt.figure(figsize=(10, 4))
plt.plot(t, stereo[:plot_len, 0], label="Left", alpha=0.8)
plt.plot(t, stereo[:plot_len, 1], label="Right", alpha=0.8)
plt.title("Stereo waveform (first 4 s): 'piyo' left-right toggle")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
