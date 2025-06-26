#!/usr/bin/env python3
"""SpatialAudio.py  –  mono/stereo 音声 → 4ch 正四面体 + FOA + (任意で stereo)
   ・RIR は gen_room_pool.py が作った JSON から 1 件取得
   ・split 指定で trainval / test どちらのプールを使うか自動切替
"""
import random, json, yaml, math, sys, hashlib, shutil
from pathlib import Path
import numpy as np, librosa, soundfile as sf
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from typing import Literal
import math, numpy as np, pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
# ───────────────── 設定ロード
cfg = yaml.safe_load(Path("Spatial_AudioCaps/spatial_ranges.yml").read_text())

# ──────────────────── 量子化グリッド & ヘルパ ──────────────────────
# 距離: 1 cm (=0.01 m) 単位  /  角度: 1° 単位
GRID_CM  = 1       # [cm]
GRID_DEG = 1        # [deg]



POOL = {
    "train": json.loads(Path("Spatial_AudioCaps/room_pool_trainval.json").read_text()),
    "val"  : json.loads(Path("Spatial_AudioCaps/room_pool_trainval.json").read_text()),
    "test" : json.loads(Path("Spatial_AudioCaps/room_pool_test.json").read_text()),
}


def _snap(val: float, grid: float) -> float:
    """val を grid 単位に丸めて返す"""
    return round(val / grid) * grid


def _load_audio(infile: Path):
    """
    ① soundfile で読めれば最速（libsndfile が mp3 対応なら mp3 も OK）
    ② 失敗したら librosa+audioread にフォールバック
       mono=True で単 ch、 sr=None で原本サンプリング周波数
    """
    try:
        wav, fs = sf.read(str(infile))
        if wav.ndim > 1:           # ステレオ→mono 平均
            wav = wav.mean(axis=1)
    except Exception:              # mp3 コーデック無しなど
        wav, fs = librosa.load(str(infile), sr=None, mono=True)
    return wav.astype(np.float32), fs



def _rand_room(split):
    return random.choice(POOL[split])

# ──────────────────────── 位置サンプリング ─────────────────────────
def _rand_position(
    split    : str,
    ctr      : np.ndarray,
    dist_rng : tuple[float,float] | None = None,
    el_rng   : tuple[float,float] | None = None,
    dims     : tuple[float,float,float] | None = None,   # ← 追加
    margin   : float = 0.1                               # ← 追加
) -> tuple[np.ndarray,float,float,float]:
    """
    壁マージンに抵触した場合は while ループを continue して
    “距離・方位・仰角の量子化セル” を変えずに **再サンプリング** する版。
    """
    if dist_rng is None or el_rng is None:
        rng_cfg = cfg["TEST"] if split == "test" else cfg["TRAINVAL"]
        dist_min, dist_max = rng_cfg["DIST_MIN"], rng_cfg["DIST_MAX"]
        el_min,   el_max   = rng_cfg["EL_MIN"],   rng_cfg["EL_MAX"]
    else:
        dist_min, dist_max = dist_rng
        el_min,   el_max   = el_rng

    if dims is None:
        raise ValueError("room dimensions `dims` must be given")

    w, h, H = dims
    while True:
        # ─── ① 連続乱数を引く
        dist   = random.uniform(dist_min, dist_max)
        az_deg = random.uniform(-180.0, 180.0)
        el_deg = random.uniform(el_min, el_max)

        # ─── ② 量子化（セル偶奇チェックは従来どおり）
        dist_q = _snap(dist, GRID_CM / 100)
        az_q   = _snap(az_deg, GRID_DEG)
        el_q   = _snap(el_deg, GRID_DEG)

        cell_d = int(round(dist_q / (GRID_CM / 100)))
        cell_a = int(round((az_q + 180.0) / GRID_DEG))
        cell_e = int(round((el_q - el_min) / GRID_DEG))
        if ((cell_d + cell_a + cell_e) & 1) != (1 if split == "test" else 0):
            continue

        # ─── ③ 極座標 → 直交座標
        az_rad = math.radians(az_q)
        el_rad = math.radians(el_q)
        src = ctr + dist_q * np.array([
            math.cos(el_rad)*math.cos(az_rad),
            math.cos(el_rad)*math.sin(az_rad),
            math.sin(el_rad)
        ])

        # ─── ④ マージン判定。壁を跨ぐなら「やり直し」
        if ((src[0] < margin) or (src[0] > w-margin) or
            (src[1] < margin) or (src[1] > h-margin) or
            (src[2] < margin) or (src[2] > H-margin)):
            continue   # ← ここでループ先頭に戻る

        # 条件クリア
        return src, dist_q, az_q, el_q

    

def trim_pad(x, fs, min_sec=4.0):
    y, _ = librosa.effects.trim(x, top_db=30)
    need = int(min_sec*fs) - len(y)
    if need>0: y = np.tile(y, math.ceil(need/len(y)))[:need+len(y)]
    return y

def spatial_foa(in_wav: Path | str, out_dir: Path, split: Literal["train","val","test"],
                room_conf=None, stereo_out=False):
    in_wav = Path(in_wav)
    
    wav, fs = _load_audio(in_wav)

    wav = trim_pad(wav, fs)
    print(f"Loaded {in_wav.name} : {len(wav)/fs:.2f}s @ {fs} Hz")  # 修正
    # ---------- インパルス信号の生成 ----------
    impulse = np.zeros_like(wav)
    impulse[0] = 1.0
    wav = impulse

    room_cfg = room_conf if room_conf else _rand_room(split)
    w,h,H = room_cfg["dims"]; alpha = room_cfg["alpha"]
    room = pra.ShoeBox([w,h,H], fs=fs,
                       materials=pra.Material(alpha), max_order=10)
    ctr = np.array([w/2,h/2,H/2])

    rng = cfg["TEST"] if split=="test" else cfg["TRAINVAL"]
    src, dist, az_deg, el_deg = _rand_position(
        split, ctr, (rng["DIST_MIN"], rng["DIST_MAX"]),
        (rng["EL_MIN"],   rng["EL_MAX"]),dims=(w,h,H)
    )

    # ─── ソース位置を必ず部屋の内側に収める ──────────

    room.add_source(src.tolist(), signal=wav)
    # 正四面体マイク
    r=0.05; v=r/math.sqrt(3)
    tet = np.array([[ v,  v,  v],           # LFU
                [ v, -v, -v],           # RFD
                [-v,  v, -v],           # RBU
                [-v, -v,  v]], dtype=float).T    # LBD   (shape = 3×4)

    dirs = []
    for x, y, z in tet.T:                   # 列ごとに取り出す
        az  = math.degrees(math.atan2(y, x)) % 360            # 0–360°
        col = math.degrees(math.acos(z / r))                  # 0°=真上, 180°=真下
        dirs.append(
            CardioidFamily(
                orientation=DirectionVector(azimuth=az, colatitude=col, degrees=True),
                p=0.5,
                gain=1.0,
            )
        )
        # ---------- マイクロホンアレイを作成（指向性付き） ----------
    pos_mat = ctr.reshape(3,1) + tet  # 形状は (3, 4)
    mic_array = pra.MicrophoneArray(
        pos_mat,   # 3×4
        fs=fs,
        directivity=dirs,                    # 4 個の Directivity オブジェクト
    )
    room.add_microphone_array(mic_array)
    room.compute_rir()

    # ---------- RIR 畳み込み（A-format 4ch） ----------
    outs = [fftconvolve(wav, room.rir[m][0], mode="full")
            for m in range(4)]
    Tmax = max(len(o) for o in outs)
    m0, m1, m2, m3 = [np.pad(o, (0, Tmax - len(o))) for o in outs]

    # ---------- A → B (First-order Ambisonics, FOA) ----------
    W =  (m0 +  m1 +  m2 +  m3)/2
    X =  (m0 +  m1 -  m2 -  m3)/2
    Y =  (m0 -  m1 -  m2 +  m3)/2
    Z =  (m0 -  m1 +  m2 -  m3)/2
    foa = np.stack([W, Y, Z, X])

    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Save temporary FOA (will be moved by make_pairs) ───
    sf.write(out_dir / 'foa.wav', foa.T, fs)
    room_id = hashlib.md5(json.dumps(room_cfg, sort_keys=True).encode()).hexdigest()[:8]
    meta = dict(
        dims           = room_cfg["dims"],             # [w,h,H]
        area_m2        = room_cfg["area_m2"],
        alpha          = float(alpha),
        fullband_T30_ms= room_cfg["T30_ms"],
        source_distance_m = round(dist, 3),
        azimuth_deg    = round(az_deg, 2),
        elevation_deg  = round(el_deg, 2),
        source_pos_xyz = [round(float(v), 3) for v in src],
        fs             = fs,
        room_id        = room_id,
        split          = split
    )
    Path(out_dir / 'meta.yml').write_text(yaml.dump(meta, sort_keys=False))

# CLI ≒ quick test
if __name__=="__main__":
    if len(sys.argv)!=4:
        print("usage: SpatialAudio.py in.wav out_dir train|val|test")
        sys.exit(1)
    spatial_foa(Path(sys.argv[1]), Path(sys.argv[2]), split=sys.argv[3])
