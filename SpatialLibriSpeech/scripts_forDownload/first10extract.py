import pandas as pd

import soundfile as sf
import os

meta = pd.read_parquet("metadata.parquet")

# 10秒以上のものだけ
meta_10s = meta[(meta["audio_info/duration"] >= 10) & (meta["split"] == "test")]

# 保存先フォルダ
save_dir = "./takamichi09/SpatialLibriSpeech/first10sec_test"
os.makedirs(save_dir, exist_ok=True)

# 音源が存在するフォルダ
audio_dir = "./takamichi09/SpatialLibriSpeech/test"


# 新しいメタデータ格納用リスト
new_metadata = []
count = 0
for idx, row in meta_10s.iterrows():
    fname = f"{idx:06d}.flac"  # 6桁ゼロ埋めファイル名
    path = os.path.join(audio_dir, fname)
    if not os.path.exists(path):
        print(f"{path}: ファイルが存在しません。スキップ")
        continue

    # 音源読み込み
    audio, sr = sf.read(path)
    if audio.ndim != 2 or audio.shape[1] != 4:
        print(f"{fname}: 4ch(FOA)ではないのでスキップ")
        continue
    if sr != 16000:
        print(f"{fname}: サンプリングレートが16kHzでないのでスキップ")
        continue
        # 新しいメタデータ用辞書

    count +=1
    if count %2 ==0:
        # 10秒分だけ抽出
        audio_10s = audio[:160000, :]

        # 新ファイルパス（ファイル名は元のまま＝index番号.flac）
        out_path = os.path.join(save_dir, fname)
        sf.write(out_path, audio_10s, sr, format='FLAC')
        row_new = row.copy()
        row_new["audio_info/duration"] = 10.0  # 長さを10.0に
        new_metadata.append(row_new)


# DataFrame化（indexはnew_metadataの分だけ）
new_meta_df = pd.DataFrame(new_metadata)
new_meta_df.to_csv(os.path.join(save_dir, "first10sec_metadata_test.csv"), index=True)  # 必要ならindex=Falseに