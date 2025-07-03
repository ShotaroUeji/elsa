#!/usr/bin/env python3
"""
verify_tdoa_robustness_fully_aligned.py
– SpatialAudio.pyのシミュレーション設定（指向性マイクを含む）に完全準拠し、
  様々な条件下でTDOAの妥当性を体系的に検証する最終決定版スクリプト。
"""
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
import math
import matplotlib.pyplot as plt
import os

def setup_mic_array_from_training_script(room, mic_center, mic_radius):
    """
    訓練データ作成用スクリプトのロジックに完全準拠して
    指向性を持つ正四面体マイクアレイを部屋に設置する関数
    """
    v = mic_radius / math.sqrt(3)
    r = mic_radius
    tet = np.array([
        [ v,  v,  v], [ v, -v, -v], [-v,  v, -v], [-v, -v,  v]
    ], dtype=float).T
    pos_mat = mic_center.reshape(3, 1) + tet

    dirs = []
    for x, y, z in tet.T:
        az  = math.degrees(math.atan2(y, x)) % 360
        col = math.degrees(math.acos(z / r))
        dirs.append(
            CardioidFamily(
                orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True),
                p=0.5,
                gain=1.0,
            )
        )
        
    mic_array = pra.MicrophoneArray(
        pos_mat,
        fs=room.fs,
        directivity=dirs,
    )
    room.add_microphone_array(mic_array)
    return room

def run_single_verification(scenario, save_plots=True):
    """単一のシナリオに対してTDOA検証を実行する関数"""
    
    scenario_name = scenario['name']
    room_dims = scenario['dims']
    source_pos = scenario['source_pos']
    
    print("-" * 80)
    print(f"シナリオ '{scenario_name}' の検証を開始します...")
    print(f"  部屋の寸法: {room_dims} m, 音源の座標: {source_pos} m")

    FS = 48000
    mic_center = np.array(room_dims) / 2.0
    C = 343.0
    MAX_ORDER = 0
    MIC_RADIUS = 0.05
    
    v = MIC_RADIUS / math.sqrt(3)
    tet = np.array([[ v,  v,  v], [ v, -v, -v], [-v,  v, -v], [-v, -v,  v]], dtype=float).T
    mic_positions = mic_center.reshape(3, 1) + tet
    
    distances = [np.linalg.norm(source_pos - mic_pos) for mic_pos in mic_positions.T]
    theoretical_arrival_times = [d / C for d in distances]
    ref_time_th = theoretical_arrival_times[0]
    theoretical_tdoa = [t - ref_time_th for t in theoretical_arrival_times]
    
    room = pra.ShoeBox(room_dims, fs=FS, max_order=MAX_ORDER)
    room.c = C
    room.add_source(source_pos)
    
    # ★★★ 指向性マイクアレイをセットアップ ★★★
    room = setup_mic_array_from_training_script(room, mic_center, mic_radius=MIC_RADIUS)
    
    room.compute_rir()
    
    measured_peak_samples = [np.argmax(room.rir[i][0]) for i in range(len(room.rir))]
    measured_arrival_times = [s / FS for s in measured_peak_samples]
    ref_time_measured = measured_arrival_times[0]
    measured_tdoa = [t - ref_time_measured for t in measured_arrival_times]
    
    th_tdoa_ms_list = [t * 1000 for t in theoretical_tdoa]
    ms_tdoa_ms_list = [t * 1000 for t in measured_tdoa]
    tdoa_errors_ms = [th_tdoa_ms_list[i] - ms_tdoa_ms_list[i] for i in range(len(th_tdoa_ms_list))]
    
    max_error_us = max(abs(e) for e in tdoa_errors_ms) * 1000
    sampling_period_us = (1 / FS) * 1e6
    
    is_success = max_error_us < sampling_period_us
    result_text = "成功" if is_success else "警告"
    print(f"  -> 最大誤差: {max_error_us:.3f} μs (サンプリング周期: {sampling_period_us:.3f} μs) ... 【{result_text}】")

    # (グラフ描画部分は省略)
    
    return is_success

def main():
    """メイン実行関数"""
    test_scenarios = [
        {"name": "標準的な部屋", "dims": [8, 6, 3.5], "source_pos": [2, 4.5, 2]},
        {"name": "広い部屋", "dims": [15, 12, 4], "source_pos": [3, 10, 2.5]},
        {"name": "細長い部屋", "dims": [3, 10, 3], "source_pos": [1, 2, 1.5]},
        {"name": "音源が隅にある場合", "dims": [8, 8, 3], "source_pos": [0.5, 0.5, 0.5]},
        {"name": "音源が中心にある場合", "dims": [8, 6, 4], "source_pos": [4, 3, 2]}
    ]
    
    print("="*80)
    print("【最終決定版】TDOA頑健性検証（指向性マイク使用）を開始します。")
    print(f"合計 {len(test_scenarios)} 個のシナリオをテストします。")
    print("="*80)

    results = []
    for scenario in test_scenarios:
        success = run_single_verification(scenario, save_plots=False)
        results.append(success)

    success_count = sum(results)
    total_count = len(results)
    print("\n" + "="*80)
    print("全シナリオの検証が完了しました。")
    print(f"最終結果: {total_count} 個のシナリオ中、{success_count} 個のテストが成功しました。")
    print("="*80)

    if success_count == total_count:
        print("結論: 指向性を考慮した、訓練データ作成時と同一の条件下においても、\n      シミュレーションの空間的な正確性（TDOA）は完全に保たれることが確認されました。")
    else:
        print("結論: いくつかの条件下で誤差が許容範囲を超えました。結果を確認してください。")

if __name__ == "__main__":
    main()