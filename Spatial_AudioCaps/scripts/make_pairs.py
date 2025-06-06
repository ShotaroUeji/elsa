#!/usr/bin/env python3
"""
make_pairs.py  –  AudioCaps を FOA 拡張 + GPT 空間キャプション化

例)
~/elsa$ python3 Spatial_AudioCaps/scripts/make_pairs.py --split train --csv Spatial_AudioCaps/AudioCaps_csv/train.csv --audio-dir Spatial_AudioCaps/takamichi09/AudioCaps_mp3/train --out-dir Spatial_AudioCaps/takamichi09/audiopre --workers 1
例)
~/elsa$ python3 Spatial_AudioCaps/scripts/make_pairs.py 
--split train 
--csv Spatial_AudioCaps/AudioCaps_csv/train.csv 
--audio-dir Spatial_AudioCaps/takamichi09/AudioCaps_mp3/train 
--out-dir Spatial_AudioCaps/takamichi09/audiopre 
--workers 1
"""
from pathlib import Path
import random, argparse, yaml, csv, traceback, shutil, hashlib
import concurrent.futures as fut
import pandas as pd

from SpatialAudio import spatial_foa
from SpatialCaps  import rewrite_spatial_caption
from json import loads

# ────────────────────────────────────────────────────────────
def gen_sample(row              : dict,
               split            : str,
               out_root         : Path,
               foa_root         : Path,
               suffix: str      = "",
               pair_id: str     = "",
               pair_type: str   = "",
               room_conf: dict  = None,
               stereo: bool     = "") -> dict:
    out_dir = out_root / f"{row['audiocap_id']}{suffix}"
    """
    1 本の mp3 を
      • FOA(+mic4) 生成
      • meta.yml -> GPT キャプション生成
      • caption.txt 保存
    """
    in_wav = Path(row["file"])              # すでに絶対パスにしてある
    if not in_wav.exists():
        return {"ok": False, "id": row["id"], "reason": "audio_missing"}

    out_dir = out_root / f"{row['audiocap_id']}{suffix}"

    try:
        spatial_foa(in_wav, out_dir, split=split,
                    room_conf=room_conf, stereo_out=stereo)
        # ─── Relocate FOA wav to unified folder ───
        foa_filename = f"{out_dir.name}.wav"
        foa_src = out_dir / 'foa.wav'
        foa_dst = foa_root / foa_filename
        foa_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(foa_src), str(foa_dst))
        meta = yaml.safe_load((out_dir / "meta.yml").read_text())
        cap  = rewrite_spatial_caption(row["caption"], meta)
        (out_dir / "caption.txt").write_text(cap, encoding="utf8")

        return dict(
            ok=True,
            foa_filename = foa_filename,
            caption      = cap,
            **{
                "meta/azimuth"    : meta["azimuth_deg"],
                "meta/elevation"  : meta["elevation_deg"],
                "meta/area"       : meta["area_m2"],
                "meta/t30"        : meta["fullband_T30_ms"],
                "room_id"         : meta["room_id"],
                "source_pos_x"    : meta["source_pos_xyz"][0],
                "source_pos_y"    : meta["source_pos_xyz"][1],
                "source_pos_z"    : meta["source_pos_xyz"][2],
                "fs"              : meta["fs"],
                "room_dim_x"      : meta["dims"][0],
                "room_dim_y"      : meta["dims"][1],
                "room_dim_z"      : meta["dims"][2],
                "alph"            : meta["alpha"],
                "split"           : meta["split"],
            }
        )

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "id": row["id"], "reason": str(e)}


# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--foa-root", type=Path, default=Path("foa_wavs"))
    ap.add_argument("--foa-dir", help="Directory to gather all FOA wav files (default: OUT_DIR/foa)")

    ap.add_argument("--spatial-parallel", type=int, default=None)
    ap.add_argument("--audio-parallel",   type=int, default=0)
    ap.add_argument("--stereo", default=False, type=bool)
    args = ap.parse_args()
    audio_dir = Path(args.audio_dir).resolve()
 
    # ── 入力 CSV を読み込み & ファイルパス列を作成
    
    df = pd.read_csv(args.csv)
    df["id"] = df["audiocap_id"]  
    if args.spatial_parallel == None:
        args.spatial_parallel = len(df)  # デフォルトは音声数と同じ

    print(f"🔍 {len(df)} rows in {args.csv}")
    df["file"] = df["audiocap_id"].astype(str).apply(
        lambda x: str(audio_dir / f"{x}.mp3"))

    rows = df.to_dict("records")
    random.shuffle(rows)

    # 出力ルート
    out_root = Path(args.out_dir) / args.split
    foa_root = Path(args.foa_dir) /args.split if args.foa_dir else out_root / "foa" /args.split
    out_root.mkdir(parents=True, exist_ok=True)

    man_file = Path(f"manifest_{args.split}.csv")
    header = [
        "foa_filename", "caption",
        "meta/azimuth", "meta/elevation", "meta/area", "meta/t30",
        "room_id", "source_pos_x", "source_pos_y", "source_pos_z",
        "fs", "room_dim_x", "room_dim_y", "room_dim_z", "alph", "split"
    ]
    write_header = not man_file.exists()
    # "a" モードでファイルを開いておく（以降はここに都度追記）
    fp = man_file.open("a", newline="", encoding="utf8")
    writer = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    futures = []
    with fut.ProcessPoolExecutor(args.workers) as ex:
        # ① 空間パラレル
        if args.spatial_parallel:
            pool_file = "Spatial_AudioCaps/room_pool_trainval.json" if args.split in ("train", "val") else "Spatial_AudioCaps/room_pool_test.json"
            room_pool = loads(Path(pool_file).read_text())
            max_sp = min(args.spatial_parallel, len(rows))
            if max_sp < args.spatial_parallel:
                print(f"⚠ spatial_parallel too large ({args.spatial_parallel}), reducing to {max_sp}")
            for i in range(max_sp):
                pair_id = f"SP{str(i).zfill(4)}"
                if not rows:
                    raise RuntimeError("Not enough rows remaining for spatial_parallel")
                r = rows.pop(0)
                for suf in ("A", "B"):
                    futures.append(
                        ex.submit(
                            gen_sample, r, args.split, out_root, foa_root,
                            suffix=f"_{suf}",
                            pair_id=pair_id,
                            pair_type="spatial",
                            stereo=args.stereo
                        )
                    )

        # ② 音源パラレル
        if args.audio_parallel:
            pool_file = "room_pool_trainval.json" if args.split in ("train", "val") else "room_pool_test.json"
            room_pool = loads(Path(pool_file).read_text())
            max_ap = min(args.audio_parallel, len(rows) // 2)
            if max_ap < args.audio_parallel:
                print(f"⚠ audio_parallel too large ({args.audio_parallel}), reducing to {max_ap}")
            for i in range(max_ap):
                pair_id = f"AP{str(i).zfill(4)}"
                room_conf = random.choice(room_pool)
                for _ in range(2):
                    if not rows:
                        raise RuntimeError("Not enough rows remaining for audio_parallel")
                    r = rows.pop(0)
                    futures.append(
                        ex.submit(
                            gen_sample, r, args.split, out_root, foa_root,
                            pair_id=pair_id,
                            pair_type="audio",
                            room_conf=room_conf,
                            stereo=args.stereo
                        )
                    )

        # ③ 残り
        for r in rows:
            futures.append(
                ex.submit(
                    gen_sample, r, args.split, out_root, foa_root,
                    stereo=args.stereo
                )
            )

        # ── as_completed で返ってくるたびに逐次書き込み ──
        for f in fut.as_completed(futures):
            res = f.result()
            if res.get("ok"):
                writer.writerow(res)
            else:
                reason = res.get("reason", "unknown")
                sample_id = res.get("id", "n/a")
                print(f"⏩ skip {sample_id} ({reason})")

    fp.close()
    print(f"✅ {args.split}: done. Manifest updated at {man_file}")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
