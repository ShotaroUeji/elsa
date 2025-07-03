#!/usr/bin/env python3
"""
test_a2b_rir_only.py
  – A-format → B-format 妥当性テスト（room.compute_rir の RIR を直接使用）
"""
import math, numpy as np, pyroomacoustics as pra

# ---------- 正四面体アレイ ----------
def build_tetra_array(center, fs, r=0.05):
    v = r / math.sqrt(3)
    tet = np.array([[ v,  v,  v],
                    [ v, -v, -v],
                    [-v,  v, -v],
                    [-v, -v,  v]], float).T
    dirs, pos = [], center[:, None] + tet
    for x, y, z in tet.T:
        az  = math.degrees(math.atan2(y, x)) % 360
        col = math.degrees(math.acos(z / r))
        dirs.append(pra.directivities.CardioidFamily(
            orientation=pra.directivities.DirectionVector(
                azimuth=az, colatitude=col, degrees=True),
            p=0.5, gain=1.0))
    return pra.MicrophoneArray(pos, fs, directivity=dirs)

# ---------- A → B 変換 ----------
def a2b(m0, m1, m2, m3):
    W = (m0 + m1 + m2 + m3) / 2
    X = (m0 + m1 - m2 - m3) / 2
    Y = (m0 - m1 + m2 - m3) / 2
    Z = (m0 - m1 - m2 + m3) / 2
    return dict(W=W, X=X, Y=Y, Z=Z)

# ---------- 1 軸テスト ----------
def run_axis(axis, dist=2.0, fs=48_000, room_dim=(10,8,6)):
    room = pra.ShoeBox(room_dim, fs=fs, max_order=0)
    ctr  = np.array(room_dim)/2
    shift = {'X':(dist,0,0), 'Y':(0,dist,0), 'Z':(0,0,dist)}[axis]
    src_pos = ctr + np.array(shift)
    room.add_source(src_pos)                     # ★ 信号を渡さない
    room.add_microphone_array(build_tetra_array(ctr, fs))
    room.compute_rir()                           # ★ RIR を生成

    # RIR(= impulse response) をそのまま A-format とみなす
    m0, m1, m2, m3 = [room.rir[i][0] for i in range(4)]
    # 長さを揃える
    Tmax = max(map(len, (m0,m1,m2,m3)))
    m0, m1, m2, m3 = [np.pad(h, (0, Tmax-len(h))) for h in (m0,m1,m2,m3)]

    B = a2b(m0,m1,m2,m3)
    rms = {k: np.sqrt(np.mean(v**2)) for k,v in B.items()}
    dom  = rms[axis]; others = [rms[k] for k in 'XYZ' if k!=axis]
    assert all(dom > 10*o for o in others), f"{axis} failed: {rms}"
    print(f"[{axis}] OK  dominant={dom:.3e}  others={others}")

# ---------- 全軸テスト ----------
def main():
    for ax in 'XYZ':
        run_axis(ax)
    print("\nAll axis tests passed ✔")

if __name__ == "__main__":
    main()
