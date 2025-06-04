# ===== dataloader_caption_only.py =====
import torch
from torch.utils.data import DataLoader
from foa_dataset_caption import FOADatasetWithCaptionOnly

def collate_fn(batch):
    """
    batch: List of tuples [(wav1, caption1), (wav2, caption2), ...]
      - wav_i     : Tensor (4, L_i)  ※ 各 i で L_i は異なる場合がある
      - caption_i : str

    これを以下の形式にまとめて返す:
      - foa_batch      : Tensor (B, 4, L_max)  ※ バッチ内最大長 L_max に右パディング
      - lengths        : Tensor (B,)            ※ 各 wav_i の元の長さ L_i
      - captions_batch : List[str]              ※ 長さ B のリスト
    """
    wav_list, captions_list = zip(*batch)

    # 1) 各 wav の長さを調べる
    lengths = torch.tensor([w.shape[1] for w in wav_list], dtype=torch.long)  # (B,)
    max_len = int(lengths.max().item())

    # 2) 右側ゼロパディングしてバッチ化
    #    wav_i は (4, L_i) の Tensor。pad_size = max_len - L_i
    padded_wavs = []
    for w in wav_list:
        L = w.shape[1]
        pad_size = max_len - L
        if pad_size > 0:
            # pad の引数は (左パッド, 右パッド) を指定。1 次元 (time) だけパディングしたいので (0, pad_size)。
            # 2 次元 (channels, time) の場合、pad の指定は (time_left, time_right, chan_left, chan_right) となるが、
            # torchaudio の場合は (pad_left, pad_right) を最後の次元に適用するので次のコードで OK。
            w_padded = torch.nn.functional.pad(w, (0, pad_size), mode="constant", value=0.0)
        else:
            w_padded = w
        padded_wavs.append(w_padded)

    # 3) stacked into (B, 4, max_len)
    foa_batch = torch.stack(padded_wavs, dim=0)  # (B, 4, max_len)

    # 4) キャプションは文字列リストとして返す
    captions_batch = list(captions_list)  # 長さ B の List[str]

    return foa_batch, lengths, captions_batch

def make_dataloader(audio_folder, metadata_csv, batch_size=8, num_workers=4, shuffle=True):
    """
    FOA 波形とキャプションのみを返す DataLoader を作成。
    """
    dataset = FOADatasetWithCaptionOnly(audio_folder=audio_folder, metadata_csv=metadata_csv, sr=16000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn
    )
    return loader

if __name__ == "__main__":
    audio_folder = "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/aa/test"
    metadata_csv = "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/manifest_a.csv"

    batch_size = 4
    num_workers = 2

    dataloader = make_dataloader(audio_folder, metadata_csv, batch_size, num_workers, shuffle=True)

    # デバッグ: 最初のバッチを取り出して形状を確認
    for foa_batch, lengths, captions_batch in dataloader:
        print("foa_batch.shape  :", foa_batch.shape)  # -> torch.Size([B, 4, L_max])
        print("lengths         :", lengths)             # -> tensor([L1, L2, ..., LB])
        print("captions_batch  :", captions_batch)      # -> ['caption1', 'caption2', ...]
        break
