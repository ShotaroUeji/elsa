# test_spatial_foa.py
import math, tempfile, random, numpy as np, soundfile as sf
from pathlib import Path

# ---- あなたの SpatialAudio.py を import （同じフォルダに置く前提） ----
import SpatialAudio as SA

def _make_test_tone(fname: Path, fs=48000, freq=1000, dur_s=1.0):
    t = np.linspace(0, dur_s, int(fs*dur_s), endpoint=False)
    tone = np.sin(2*math.pi*freq*t).astype(np.float32)
    sf.write(fname, tone, fs)
    return fs

def test_foa_generation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_in   = tmp/'tone.wav'
        out_dir  = tmp/'out'
        fs       = _make_test_tone(wav_in)          # ① テスト入力

        # ② ミニマムな部屋設定（4×5×3 m, α=0.3）
        room_cfg = dict(
            dims      = [4.0, 5.0, 3.0],
            alpha     = 0.3,
            area_m2   = 4*5,
            T30_ms    = 400
        )

        SA.spatial_foa(
            in_wav    = wav_in,
            out_dir   = out_dir,
            split     = "train",
            room_conf = room_cfg,
            stereo_out= True,
        )

        # ③ FOA を読んで検証
        foa, fs2 = sf.read(out_dir/'foa.wav')
        assert fs2 == fs,         "FS mismatch"
        assert foa.shape[1] == 4, "FOA should have 4 channels"
        assert np.any(foa),       "FOA signal is all-zero?"

        # ④ ステレオも読めるか（任意）
        #stereo, _ = sf.read(out_dir/'stereo.wav')
        #assert stereo.shape[1] == 2, "Stereo should have 2 channels"
        #print("✅ FOA/stereo generation OK →", out_dir)

if __name__ == "__main__":
    test_foa_generation()
