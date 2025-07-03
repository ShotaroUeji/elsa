#!/usr/bin/env python3
"""
verify_tdoa.py – 正四面体マイクアレイのシミュレーションにおける
                 音の到達時間差（TDOA）の妥当性を検証します。

このスクリプトは以下の処理を行います:
1.  シミュレーションのパラメータを定義します。
2.  理論的なTDOAを計算します。
3.  RIRをシミュレートし、実測のTDOAを測定します。
4.  理論値と実測値の誤差をテキストで評価します。
5.  RIR波形を可視化して保存します。
6.  【IMPROVED】TDOA比較と誤差分析を1枚の画像にまとめて保存します。
"""
import numpy as np
import pyroomacoustics as pra
import math
import matplotlib.pyplot as plt

def verify_tdoa_and_visualize_full():
    """TDOAの検証を実行し、2種類のサマリーグラフを画像ファイルとして保存する"""
    
    print("--- FOA音源シミュレーションのTDOA（到達時間差）検証を開始します ---")

    # --- 1. パラメータ設定からTDOA測定まで ---
    # (この部分は前回から変更ありません)
    FS = 48000
    ROOM_DIMS = [8, 6, 3.5]
    MIC_CENTER = np.array(ROOM_DIMS) / 2.0
    SOURCE_POS = np.array([1, 4.5, 2])
    C = 343.0
    MAX_ORDER = 0
    MIC_RADIUS = 0.05
    print("\n[シミュレーション設定]")
    print(f"  サンプリング周波数: {FS} Hz")
    print(f"  部屋の寸法: {ROOM_DIMS} m")
    print(f"  音源の座標: {SOURCE_POS} m")
    print(f"  マイクアレイ中心座標: {MIC_CENTER} m (部屋の中心)")
    print(f"  音速: {C} m/s")
    v = MIC_RADIUS / math.sqrt(3)
    tet = np.array([[ v,  v,  v], [ v, -v, -v], [-v,  v, -v], [-v, -v,  v]], dtype=float).T
    mic_positions = MIC_CENTER.reshape(3, 1) + tet
    print("\n[1. 理論的なTDOAを計算中...]")
    distances = [np.linalg.norm(SOURCE_POS - mic_pos) for mic_pos in mic_positions.T]
    theoretical_arrival_times = [d / C for d in distances]
    ref_time_th = theoretical_arrival_times[0]
    theoretical_tdoa = [t - ref_time_th for t in theoretical_arrival_times]
    print("[2. pyroomacousticsでRIR（部屋インパルス応答）をシミュレート中...]")
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, max_order=MAX_ORDER, materials=pra.Material(0.1))
    room.c = C
    room.add_source(SOURCE_POS)
    room.add_microphone_array(mic_positions)
    room.compute_rir()
    print("[3. シミュレートされたRIRからTDOAを測定中...]")
    measured_peak_samples = [np.argmax(room.rir[i][0]) for i in range(mic_positions.shape[1])]
    measured_arrival_times = [s / FS for s in measured_peak_samples]
    ref_time_measured = measured_arrival_times[0]
    measured_tdoa = [t - ref_time_measured for t in measured_arrival_times]
    print("\n[4. 検証結果の比較]")
    print("-" * 75)
    print(f"{'マイクロホン':<12} | {'理論 TDOA (ms)':<25} | {'実測 TDOA (ms)':<22} | {'誤差 (ms)':<15}")
    print("-" * 75)
    th_tdoa_ms_list = [t * 1000 for t in theoretical_tdoa]
    ms_tdoa_ms_list = [t * 1000 for t in measured_tdoa]
    for i in range(mic_positions.shape[1]):
        error_ms = th_tdoa_ms_list[i] - ms_tdoa_ms_list[i]
        print(f"  Mic {i:<8} | {th_tdoa_ms_list[i]:<25.6f} | {ms_tdoa_ms_list[i]:<22.6f} | {error_ms:<15.6f}")
    print("-" * 75)
    
    # --- RIR波形グラフの描画と保存 ---
    print("\n[5. RIR波形をグラフとしてファイルに保存します...]")
    # (この部分は変更ありません)
    desired_plot_samples = max(measured_peak_samples) + int(0.002 * FS)
    min_rir_length = min(len(r[0]) for r in room.rir)
    actual_plot_samples = min(desired_plot_samples, min_rir_length)
    time_axis = np.arange(actual_plot_samples) / FS
    fig1, axes1 = plt.subplots(mic_positions.shape[1], 1, figsize=(12, 8), sharex=True, sharey=True)
    fig1.suptitle('Simulated Room Impulse Response (RIR) for each Microphone', fontsize=16)
    for i in range(mic_positions.shape[1]):
        axes1[i].plot(time_axis, room.rir[i][0][:actual_plot_samples], label=f'Mic {i} RIR')
        peak_time = measured_arrival_times[i]
        axes1[i].axvline(x=peak_time, color='r', linestyle='--', linewidth=1.5, label=f'Detected Peak ({peak_time*1000:.3f} ms)')
        axes1[i].set_title(f'Microphone {i}')
        axes1[i].legend()
        axes1[i].grid(True, linestyle=':', alpha=0.6)
    axes1[-1].set_xlabel('Time (s)', fontsize=12)
    fig1.text(0.06, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    output_filename_1 = 'rir_plot.png'
    plt.savefig(output_filename_1)
    print(f"RIR波形グラフを '{output_filename_1}' という名前で保存しました。")
    plt.close(fig1)

    # ★★★ ここからが修正部分 ★★★
    # --- TDOA比較グラフと誤差分析グラフを横に並べて描画・保存 ---
    print("[6. TDOA比較・誤差分析グラフを生成し、ファイルに保存します...]")

    # 1行2列のサブプロットを作成。figsizeで全体のサイズを調整。
    fig2, (ax_comp, ax_error) = plt.subplots(1, 2, figsize=(20, 7))
    fig2.suptitle('TDOA Validation Summary', fontsize=16)
    
    labels = [f'Mic {i}' for i in range(mic_positions.shape[1])]
    x = np.arange(len(labels))

    # --- 左側のグラフ: TDOA比較棒グラフ ---
    width = 0.35
    rects1 = ax_comp.bar(x - width/2, th_tdoa_ms_list, width, label='Theoretical TDOA')
    rects2 = ax_comp.bar(x + width/2, ms_tdoa_ms_list, width, label='Measured TDOA')
    ax_comp.set_ylabel('TDOA relative to Mic 0 (ms)')
    ax_comp.set_title('TDOA Comparison: Theoretical vs. Measured')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(labels)
    ax_comp.legend()
    ax_comp.grid(axis='y', linestyle='--', alpha=0.7)
    ax_comp.bar_label(rects1, padding=3, fmt='%.3f')
    ax_comp.bar_label(rects2, padding=3, fmt='%.3f')
    
    # --- 右側のグラフ: TDOA誤差分析グラフ ---
    tdoa_errors_ms = [th_tdoa_ms_list[i] - ms_tdoa_ms_list[i] for i in range(len(th_tdoa_ms_list))]
    sampling_period_ms = (1 / FS) * 1000
    
    ax_error.bar(x, tdoa_errors_ms, color='gray', label='TDOA Error')
    ax_error.axhline(y=sampling_period_ms, color='r', linestyle='--', label=f'±1 sample period ({sampling_period_ms:.4f} ms)')
    ax_error.axhline(y=-sampling_period_ms, color='r', linestyle='--')
    ax_error.set_ylabel('Error (Theoretical - Measured) [ms]')
    ax_error.set_title('TDOA Error vs. Sampling Period')
    ax_error.set_xticks(x)
    ax_error.set_xticklabels(labels)
    ax_error.legend()
    ax_error.grid(axis='y', linestyle='--', alpha=0.7)
    max_abs_error_plot = max(abs(e) for e in tdoa_errors_ms)
    plot_limit = max(max_abs_error_plot, sampling_period_ms) * 1.5
    ax_error.set_ylim(-plot_limit, plot_limit)

    # 全体のレイアウトを調整して保存
    fig2.tight_layout(rect=[0, 0, 1, 0.96]) # suptitleとの重なりを調整
    output_filename_2 = 'tdoa_analysis_summary.png'
    plt.savefig(output_filename_2)
    print(f"TDOA比較・誤差分析グラフを '{output_filename_2}' という名前で保存しました。")
    plt.close(fig2)

    print("\n--- TDOA検証を終了します ---")


if __name__ == "__main__":
    verify_tdoa_and_visualize_full()