#!/usr/bin/env python3
"""
python sanity_check.py metadata.csv scalers_dir/
戻り値 0 : 問題なし
戻り値 1 : NaN/Inf を含む行あり（行番号を表示）
"""
import sys, re, joblib, numpy as np, pandas as pd

csv_path, sc_dir = sys.argv[1], sys.argv[2]
drt_sc  = joblib.load(f"{sc_dir}/drt_scaler.joblib")      # 40 dim
vd_sc   = joblib.load(f"{sc_dir}/scalar_scaler.joblib")   # 2 dim

pat = re.compile(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?")
def arr20(txt):                # 33→20 スライス
    v = np.asarray([float(x) for x in pat.findall(txt.replace("\n"," "))], dtype=np.float32)
    return v[9:29] if v.size==33 else None

df = pd.read_csv(csv_path, dtype=str)
bad_rows = []

for idx, row in df.iterrows():
    drr = arr20(row["acoustics/drr_db"])
    t30 = arr20(row["acoustics/t30_ms"])
    if drr is None or t30 is None:
        bad_rows.append(idx); continue

    # --- スケール変換 ---
    drt_norm = drt_sc.transform(np.hstack([drr, t30]).reshape(1,-1))[0]
    vol  = float(row["room/volume"]); dist = float(row["speech/distance"])
    vd_norm = vd_sc.transform([[vol, dist]])[0]

    # --- NaN / Inf チェック ---
    az = float(row["speech/azimuth"])
    el = float(row["speech/elevation"])
    y = np.asarray([az, el, *vd_norm, *drt_norm], dtype=np.float32)  # ★
    if not np.isfinite(y).all():
        bad_rows.append(idx)

if bad_rows:
    print("❌ 問題のある行: ", bad_rows[:20], " ...")   # 多い場合は先頭20だけ表示
    sys.exit(1)
print("✅ 全行 OK — NaN/Inf なし")
