#!/usr/bin/env python3
"""
verify_doa_accuracy_final_v2.py
- Zチャンネルの勾配計算における致命的なバグを修正。
- これが、シミュレーションの空間的正確性を評価する最終確定版のスクリプトとなる。
"""
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from scipy.signal import stft
import math

def setup_mic_array(room, mic_center, mic_radius):
    """指向性マイクアレイを設置する"""
    v = mic_radius / math.sqrt(3)
    r = mic_radius
    tet = np.array([[v, v, v], [v, -v, -v], [-v, v, -v], [-v, -v, v]], dtype=float).T
    pos_mat = mic_center.reshape(3, 1) + tet
    dirs = []
    for x, y, z in tet.T:
        az, col = math.degrees(math.atan2(y, x)) % 360, math.degrees(math.acos(z / r))
        dirs.append(CardioidFamily(orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True), p=0.5, gain=1.0))
    mic_array = pra.MicrophoneArray(pos_mat, fs=room.fs, directivity=dirs)
    room.add_microphone_array(mic_array)
    return room

def calculate_doa_from_a_format(a_format_rirs, fs):
    """A-formatのRIR信号から、音響インテンシティ法でDoAを推定する"""
    m0, m1, m2, m3 = a_format_rirs
    nperseg = 256
    freqs, _, P0 = stft(m0, fs, nperseg=nperseg)
    _, _, P1 = stft(m1, fs, nperseg=nperseg)
    _, _, P2 = stft(m2, fs, nperseg=nperseg)
    _, _, P3 = stft(m3, fs, nperseg=nperseg)

    P_center = (P0 + P1 + P2 + P3) / 4.0
    
    # X軸方向の勾配 (Front - Back)
    grad_x = (P0 + P1) - (P2 + P3)
    # Y軸方向の勾配 (Left - Right)
    grad_y = (P0 + P2) - (P1 + P3)
    # Z軸方向の勾配 (Up - Down)
    # ★★★ ここが致命的なバグでした ★★★
    grad_z = (P0 + P3) - (P1 + P2)  # 正しくは P1, P2 を使用
    
    # インテンシティ計算
    I_x_tf = np.imag(P_center * np.conj(grad_x))
    I_y_tf = np.imag(P_center * np.conj(grad_y))
    I_z_tf = np.imag(P_center * np.conj(grad_z))
    
    # 時間と周波数でインテンシティを平均（総和）
    I_x_total, I_y_total, I_z_total = np.sum(I_x_tf), np.sum(I_y_tf), np.sum(I_z_tf)
    avg_I = np.array([I_x_total, I_y_total, I_z_total])
    
    # ベクトルから角度を計算
    if np.linalg.norm(avg_I) < 1e-9: return 0.0, 0.0
    avg_I /= np.linalg.norm(avg_I)
    est_az_rad = math.atan2(avg_I[1], avg_I[0])
    est_el_rad = math.asin(np.clip(avg_I[2], -1.0, 1.0))
    return math.degrees(est_az_rad), math.degrees(est_el_rad)

def angular_error(az1, el1, az2, el2):
    """角度誤差を計算する"""
    az1_rad, el1_rad, az2_rad, el2_rad = map(math.radians, [az1, el1, az2, el2])
    p1 = np.array([math.cos(el1_rad)*math.cos(az1_rad), math.cos(el1_rad)*math.sin(az1_rad), math.sin(el1_rad)])
    p2 = np.array([math.cos(el2_rad)*math.cos(az2_rad), math.cos(el2_rad)*math.sin(az2_rad), math.sin(el2_rad)])
    return math.degrees(math.acos(np.clip(np.dot(p1, p2), -1.0, 1.0)))

def main():
    """複数のシナリオでDoA推定精度をテストする"""
    print("--- FOA音源の空間的正確性（DoA推定精度）の連続検証【最終確定版】---")
    test_scenarios = [
        {"name": "正面", "az": 0, "el": 0},
        {"name": "右横", "az": 90, "el": 0},
        {"name": "左斜め上", "az": -45, "el": 30},
        {"name": "後ろ下", "az": 180, "el": -20},
        {"name": "もとのテストケース", "az": 45, "el": 30},
    ]
    max_overall_error = 0.0
    for scenario in test_scenarios:
        true_az, true_el = scenario['az'], scenario['el']
        
        # --- シミュレーション実行 ---
        FS = 48000
        mic_center = np.array([5, 5, 5])
        az_rad, el_rad = map(math.radians, [true_az, true_el])
        source_pos = mic_center + 2.0 * np.array([math.cos(el_rad)*math.cos(az_rad), math.cos(el_rad)*math.sin(az_rad), math.sin(el_rad)])
        
        room = pra.ShoeBox([10, 10, 10], fs=FS, max_order=0)
        room = setup_mic_array(room, mic_center, mic_radius=0.05)
        room.add_source(source_pos)
        room.compute_rir()
        
        # --- 解析 ---
        a_format_rirs = [rir[0] for rir in room.rir]
        est_az, est_el = calculate_doa_from_a_format(a_format_rirs, FS)
        error = angular_error(true_az, true_el, est_az, est_el)
        max_overall_error = max(max_overall_error, error)
        
        # --- 結果表示 ---
        print("\n" + "="*50)
        print(f"シナリオ: '{scenario['name']}'")
        print(f"  真の方向      : Az={true_az:>6.2f}°, El={true_el:>6.2f}°")
        print(f"  推定方向    : Az={est_az:>6.2f}°, El={est_el:>6.2f}°")
        print(f"  方向誤差      : {error:.3f}°")
        print("="*50)

    print("\n【総合評価】")
    print(f"全シナリオにおける最大の方向誤差: {max_overall_error:.3f}°")
    if max_overall_error < 1.0:
        print("結論: 全てのテストケースで誤差は1度未満。極めて高い空間的正確性が確認されました。")
    else:
        print("結論: 良好な精度ですが、一部のケースで誤差が1度を超えました。")

if __name__ == "__main__":
    main()