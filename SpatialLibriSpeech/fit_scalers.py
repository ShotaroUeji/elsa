#!/usr/bin/env python3
import re, sys, os, ast, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

csv_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

_num_pat = re.compile(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?")

def parse_arr(txt: str):
    if not isinstance(txt, str):
        return None
    txt = txt.replace("\n", " ").replace("　", " ")
    nums = _num_pat.findall(txt)
    return np.asarray([float(n) for n in nums], dtype=np.float32) if nums else None

df = pd.read_csv(csv_path, dtype=str)

# ---- バンド長 D を決定 ----
for s in df["acoustics/drr_db"]:
    arr = parse_arr(s)
    if arr is not None:
        D = len(arr); break
else:
    raise ValueError("DRR が読み取れません")

drr_lst, t30_lst = [], []
dist_vol_lst     = []        # distanceとvolume だけ

for _, row in df.iterrows():
    drr_full = parse_arr(row["acoustics/drr_db"])
    t30_full = parse_arr(row["acoustics/t30_ms"])
    if drr_full is None or t30_full is None: 
        continue
    drr = drr_full[9:29]      # ★ 20 要素にスライス
    t30 = t30_full[9:29]
    D = 20                    # 決め打ちでも可
    if drr is None or t30 is None or len(drr)!=D or len(t30)!=D:
        continue

    try:
        dist = float(row["speech/distance"])
        vol  = float(row["room/volume"])
        
    except (TypeError, ValueError):
        continue

    drr_lst.append(drr)
    t30_lst.append(t30)
    dist_vol_lst.append([dist, vol])

N = len(drr_lst)
if N == 0:
    raise ValueError("有効行が 0 行です")

print(f"fit rows = {N}, band length D = {D}")

# ----------- fit -----------
X_drt = np.hstack([np.vstack(drr_lst), np.vstack(t30_lst)])  # (N, 2D)
X_scl = np.vstack(dist_vol_lst)                              # (N, 2)

drt_scaler    = StandardScaler().fit(X_drt)
scalar_scaler = MinMaxScaler((0,1)).fit(X_scl)

joblib.dump(drt_scaler,    f"{out_dir}/drt_scaler.joblib")
joblib.dump(scalar_scaler, f"{out_dir}/scalar_scaler.joblib")
print("✅ scalers saved to", out_dir)
