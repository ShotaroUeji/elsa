#!/usr/bin/env python3
"""
validate_corrected_with_viz.py - 頷き（チルト）と回転アニメーションを生成する最終検証スクリプト
"""
import numpy as np
import pyroomacoustics as pra
import math
from scipy.signal import fftconvolve
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import io

def create_frame(ax, mic_positions, source_pos, dirs, axis, angle_elev, angle_azim):
    """
    指定されたアングルで1フレームを描画するヘルパー関数
    """
    ax.cla() # 毎回プロットをクリア
    
    # 1. 音源
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='*', s=200, label=f'Sound Source ({axis}-Axis)')

    # 2. マイクカプセル
    ax.scatter(mic_positions[0, :], mic_positions[1, :], mic_positions[2, :], c='b', s=100, label='Mic Capsules')

    # 3. カプセルの向き
    colors = ['cyan', 'magenta', 'yellow', 'lime']
    for i in range(mic_positions.shape[1]):
        pos = mic_positions[:, i]
        dv = dirs[i]._orientation
        az_rad = np.deg2rad(dv._azimuth)
        col_rad = np.deg2rad(dv._colatitude)
        dx, dy, dz = 0.5*np.sin(col_rad)*np.cos(az_rad), 0.5*np.sin(col_rad)*np.sin(az_rad), 0.5*np.cos(col_rad)
        ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, color=colors[i], length=0.4, normalize=True, label=f'Capsule {i}' if angle_elev == 90 and angle_azim == 270 else "")

    # 4. 正四面体のエッジ
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot(*zip(mic_positions[:, i], mic_positions[:, j]), color='gray', linestyle='--')

    # グラフの見た目
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    ax.set_title(f"FOA Geometry - Source on {axis}-Axis")
    all_points = np.hstack([mic_positions, source_pos[:, np.newaxis]])
    means = all_points.mean(axis=1)
    max_range = (all_points.max(axis=1) - all_points.min(axis=1)).max() * 0.8
    ax.set_xlim(means[0] - max_range, means[0] + max_range); ax.set_ylim(means[1] - max_range, means[1] + max_range); ax.set_zlim(means[2] - max_range, means[2] + max_range)
    if angle_elev == 90 and angle_azim == 270: ax.legend()
    ax.grid(True)
    ax.view_init(elev=angle_elev, azim=angle_azim)

def visualize_tilting_animation(mic_center, mic_positions, source_pos, dirs, axis):
    """
    シミュレーションのジオメトリを「頷き」と「回転」のGIFアニメーションとして保存する
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    images = []

    # Part 1: 上下に頷く（チルト）アニメーション (真横からの視点を含む)
    # 90度（真上）-> 0度（真横）-> -90度（真下）-> 0度（真横）
    tilt_angles = list(range(90, -91, -5)) + list(range(-85, 1, 5))
    for elev_angle in tilt_angles:
        create_frame(ax, mic_positions, source_pos, dirs, axis, angle_elev=elev_angle, angle_azim=270)
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        images.append(imageio.imread(buf)); buf.close()
    
    # Part 2: 水平に回転するアニメーション
    for azim_angle in range(0, 360, 10):
        create_frame(ax, mic_positions, source_pos, dirs, axis, angle_elev=25, angle_azim=azim_angle)
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        images.append(imageio.imread(buf)); buf.close()

    output_filename = f"foa_geometry_{axis}_tilt_and_rotate.gif"
    imageio.mimsave(output_filename, images, duration=0.1)
    print(f"📹 Final animation saved to '{output_filename}'")
    plt.close(fig)

# --- これ以降の validate_foa_simulation と main の部分は変更なし ---
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
    print(f"Source placed on {axis}-axis at {np.round(source_pos, 2)}")
    r = 0.05; v = r / math.sqrt(3)
    tet = np.array([[v,v,v], [v,-v,-v], [-v,v,-v], [-v,-v,v]], dtype=float).T
    dirs = []
    for x, y, z in tet.T:
        az = math.degrees(math.atan2(y, x)) % 360; col = math.degrees(math.acos(z / r))
        dirs.append(CardioidFamily(orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True), p=0.5, gain=1.0))
    mic_positions = mic_center.reshape(3, 1) + tet
    mic_array = pra.MicrophoneArray(mic_positions, fs=fs, directivity=dirs)
    room.add_microphone_array(mic_array)
    if visualize:
        # 新しい可視化関数を呼び出す
        visualize_tilting_animation(mic_center, mic_positions, source_pos, dirs, axis)
    room.compute_rir()
    outs = [fftconvolve(wav_impulse, room.rir[m][0], mode="full") for m in range(4)]
    Tmax = max(len(o) for o in outs); m0, m1, m2, m3 = [np.pad(o, (0, Tmax - len(o))) for o in outs]
    W = (m0+m1+m2+m3)/2; X = (m0+m1-m2-m3)/2; Y = (m0-m1+m2-m3)/2; Z = (m0-m1-m2+m3)/2
    foa_channels = {'W':W,'X':X,'Y':Y,'Z':Z}
    print("--- RMS Energy ---")
    rms_values = {}
    for name, signal in foa_channels.items():
        rms = np.sqrt(np.mean(signal**2)); rms_values[name] = rms; print(f"[{name}-ch]: {rms:.8f}")
    dominant_ch_val = rms_values[axis]; other_chs_vals = [rms_values[ax] for ax in ['X', 'Y', 'Z'] if ax != axis]
    is_dominant = all(dominant_ch_val > val * 100 for val in other_chs_vals if val > 1e-12)
    print("--- Conclusion ---")
    if is_dominant and dominant_ch_val > 0:
        print(f"✅ PASSED: {axis}-channel energy is dominant as expected.")
    else:
        print(f"❌ FAILED: Channel energy distribution is unexpected for {axis}-axis.")

if __name__ == "__main__":
    print("🚀 Starting FOA validation with CORRECTED Y/Z formulas...")
    validate_foa_simulation(axis='X', visualize=True)
    validate_foa_simulation(axis='Y', visualize=True)
    validate_foa_simulation(axis='Z', visualize=True)