#!/usr/bin/env python3
"""
debug_bformat_waveform.py
- A-to-B変換によって生成されたB-formatの4チャンネルの波形を直接プロットし、
  問題の根本原因を視覚的に特定するためのデバッグ用スクリプト。
- 変換式は提示された論文(tetraproc.pdf)に準拠。
"""
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
import math
import matplotlib.pyplot as plt
import os

def setup_mic_array_from_paper(room, mic_center, mic_radius):
    """論文に準拠して指向性マイクを設置する"""
    v = mic_radius / math.sqrt(3)
    r = mic_radius
    tet = np.array([[v, v, v], [v, -v, -v], [-v, v, -v], [-v, -v, v]], dtype=float).T
    pos_mat = mic_center.reshape(3, 1) + tet
    dirs = []
    for x, y, z in tet.T:
        az = math.degrees(math.atan2(y, x)) % 360
        col = math.degrees(math.acos(z / r))
        dirs.append(CardioidFamily(orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True), p=0.5, gain=1.0))
    mic_array = pra.MicrophoneArray(pos_mat, fs=room.fs, directivity=dirs)
    room.add_microphone_array(mic_array)
    return room

def generate_and_plot_bformat_rir(source_azimuth_deg, source_elevation_deg):
    """B-format RIRを生成し、その4チャンネルの波形をプロットする"""
    print(f"\n音源方向: Az={source_azimuth_deg}°, El={source_elevation_deg}° でシミュレーションを実行します...")
    
    FS = 48000
    ROOM_DIMS = [10, 10, 10]
    mic_center = np.array(ROOM_DIMS) / 2.0
    
    # シミュレーション実行
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, max_order=0)
    room = setup_mic_array_from_paper(room, mic_center, mic_radius=0.05)
    az_rad, el_rad = math.radians(source_azimuth_deg), math.radians(source_elevation_deg)
    source_pos = mic_center + 2.0 * np.array([math.cos(el_rad)*math.cos(az_rad), math.cos(el_rad)*math.sin(az_rad), math.sin(el_rad)])
    room.add_source(source_pos)
    room.compute_rir()
    
    # A-format -> B-format 変換 (論文/標準規約に準拠)
    m = [rir[0] for rir in room.rir]
    Tmax = max(len(o) for o in m)
    m0, m1, m2, m3 = [np.pad(o, (0, Tmax - len(o))) for o in m]
    
    # LFU=m0, RFD=m1, LBD=m2, RBU=m3
    W = (m0 + m1 + m2 + m3) / 2
    X = (m0 + m1 - m2 - m3) / 2
    Y = (m0 - m1 + m2 - m3) / 2
    Z = (m0 - m1 - m2 + m3) / 2
    
    # --- グラフ描画 ---
    print("B-formatの各チャンネル波形をプロットしています...")
    time_axis = np.arange(Tmax) / FS
    channels = {'W (Omni)': W, 'X (Front-Back)': X, 'Y (Left-Right)': Y, 'Z (Up-Down)': Z}
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'B-format Channel Waveforms (Source at Az={source_azimuth_deg}°, El={source_elevation_deg}°)', fontsize=16)
    
    for i, (name, data) in enumerate(channels.items()):
        axes[i].plot(time_axis, data)
        axes[i].set_title(name)
        axes[i].grid(True, linestyle=':')
        axes[i].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Time (s)')
    
    output_dir = "debug_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'b_format_channels_az{source_azimuth_deg}_el{source_elevation_deg}.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename)
    
    print(f"グラフを '{output_filename}' に保存しました。")
    plt.close(fig)

if __name__ == "__main__":
    # 真横（方位角90°, 仰角0°）のケースをテスト
    generate_and_plot_bformat_rir(source_azimuth_deg=90.0, source_elevation_deg=0.0)
    # 比較のため、正面（方位角0°, 仰角0°）のケースもテスト
    generate_and_plot_bformat_rir(source_azimuth_deg=0.0, source_elevation_deg=0.0)