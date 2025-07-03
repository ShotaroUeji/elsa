#!/usr/bin/env python3
"""
【最終版】主要な3視点からの静止画を生成する、シンプルな検証スクリプト
"""
import numpy as np
import pyroomacoustics as pra
import math
from scipy.signal import fftconvolve
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_key_views(mic_positions, source_pos, dirs, axis):
    """
    3つの主要な視点から静止画を生成し、向きを最終確認する関数
    """
    # 描画する視点（ elevation, azimuth ）とファイル名のリスト
    views = {
        'diagonal': (30, 45),    # 全体像が分かりやすい、斜めからの視点
        'top_down': (90, 0),     # 真上からの視点
        'side_on':  (5, 270)     # ほぼ真横からの視点
    }

    for view_name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 1. 音源
        ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='*', s=200, label='Sound Source')

        # 2. マイクカプセルと番号
        ax.scatter(mic_positions[0, :], mic_positions[1, :], mic_positions[2, :], c='b', s=100, label='Mic Capsules')
        for i in range(mic_positions.shape[1]):
             ax.text(mic_positions[0, i], mic_positions[1, i], mic_positions[2, i], f' {i}', color='k')

        # 3. カプセルの向き
        colors = ['cyan', 'magenta', 'yellow', 'lime']
        for i in range(mic_positions.shape[1]):
            pos = mic_positions[:, i]
            dv = dirs[i]._orientation
            az_rad, col_rad = np.deg2rad(dv._azimuth), np.deg2rad(dv._colatitude)
            dx = 0.4 * np.sin(col_rad) * np.cos(az_rad)
            dy = 0.4 * np.sin(col_rad) * np.sin(az_rad)
            dz = 0.4 * np.cos(col_rad)
            ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, color=colors[i])

        # グラフの見た目を設定
        ax.set_xlabel('X-Axis'); ax.set_ylabel('Y-Axis'); ax.set_zlabel('Z-Axis')
        ax.set_title(f"View: {view_name.replace('_', ' ').title()} | Source on {axis}-Axis")
        ax.legend()
        ax.view_init(elev=elev, azim=azim)

        # ファイルに保存
        output_filename = f"foa_geometry_{axis}_{view_name}.png"
        plt.savefig(output_filename)
        print(f"✅ Image saved to '{output_filename}'")
        plt.close(fig)

def validate_foa_simulation(axis='X', visualize=False):
    print(f"\n--- 🧪 Verifying {axis}-Axis ---")
    fs = 48000; room_dims = [10, 8, 6]
    wav_impulse = np.zeros(fs, dtype=np.float32); wav_impulse[100] = 1.0
    room = pra.ShoeBox(room_dims, fs=fs, max_order=0)
    mic_center = np.array(room_dims) / 2
    distance = 2.0
    if axis == 'X': source_pos = mic_center + np.array([distance, 0, 0])
    elif axis == 'Y': source_pos = mic_center + np.array([0, distance, 0])
    elif axis == 'Z': source_pos = mic_center + np.array([0, 0, distance])
    room.add_source(source_pos.tolist(), signal=wav_impulse)
    r = 0.05; v = r / math.sqrt(3)
    tet = np.array([[v,v,v], [v,-v,-v], [-v,v,-v], [-v,-v,v]], dtype=float).T
    dirs = []
    for x, y, z in tet.T:
        az, col = np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arccos(z / r))
        dirs.append(CardioidFamily(orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True), p=0.5, gain=1.0))
    mic_positions = mic_center.reshape(3, 1) + tet
    
    if visualize:
        visualize_key_views(mic_positions, source_pos, dirs, axis)

    # (音響計算部分は省略しても可視化はできます)

if __name__ == "__main__":
    # Z軸の可視化のみを実行して確認
    validate_foa_simulation(axis='Z', visualize=True)