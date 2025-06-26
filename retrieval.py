import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ユーザー定義モジュールをインポート
# このスクリプトが model/ や clap_helpers/ と同じ階層か、
# それらの親ディレクトリにあることを想定しています。
from model.audio_encoder import AudioEncoder
from model.text_encoder import TextEncoder
from model.audiocaps_dataset import AudioCapsDatasetForELSA
from model.train import collate_fn

# CLAPから流用したモジュールをインポート
from helper_metrics.metrics import get_retrieval_ranks, calculate_recall_at_k, calculate_median_rank

def get_audio_embeddings(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> (torch.Tensor, list):
    """音声埋め込みと、成功したサンプルの元のインデックスを返す"""
    model.eval()
    audio_embeddings = []
    successful_indices = [] # 成功したインデックスを保存するリスト
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding Audio"):
            # インデックスも受け取る
            I_act, I_rea, omni, _, idxs = batch_data
            
            if I_act is None:
                continue
            
            successful_indices.extend(idxs) # 成功したインデックスを追加
            
            I_act, I_rea, omni = I_act.to(device), I_rea.to(device), omni.to(device)
            a_emb = model(I_act, I_rea, omni)
            audio_embeddings.append(a_emb.cpu())
    
    if not audio_embeddings:
        raise RuntimeError("No valid audio files were processed.")
        
    return torch.cat(audio_embeddings, dim=0), successful_indices
def get_text_embeddings(captions: list, model: torch.nn.Module, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """キャプションのリストから全てのテキスト埋め込みを一括で計算"""
    model.eval()
    text_embeddings = []
    # テキストもバッチ処理するため、一時的なデータローダーを作成
    caption_loader = DataLoader(captions, batch_size=batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch_caps in tqdm(caption_loader, desc="Encoding Text"):
            # TextEncoderはテキストのリストを受け取る
            t_emb = model(list(batch_caps))
            text_embeddings.append(t_emb.cpu())
    return torch.cat(text_embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio-text retrieval (CLAP-style).")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--csv", required=True, help="Path to the AudioCaps metadata CSV.")
    parser.add_argument("--audio_dir", required=True, help="Path to the AudioCaps audio data directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. モデルのロード
    print("Loading user-defined models...")
    model_audio = AudioEncoder()
    model_text = TextEncoder()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_audio.load_state_dict(ckpt["audio"])
    model_text.load_state_dict(ckpt["text"])
    model_audio.to(device)
    model_text.to(device)
    print(f"Models loaded from epoch {ckpt.get('epoch', 'N/A')}.")

    # 2. データローダーの準備
    print("Preparing AudioCaps dataset...")
    val_ds = AudioCapsDatasetForELSA(audio_folder=args.audio_dir, metadata_csv=args.csv)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)

    # 3. 全ての埋め込みを一括計算
    all_captions_original = [val_ds.meta.iloc[i]['caption'] for i in range(len(val_ds))]
    
    # 音声埋め込みと「成功したインデックスのリスト」を取得
    audio_embeddings, valid_indices = get_audio_embeddings(val_loader, model_audio, device)

    # ★★★ ここでテキストデータを再同期する ★★★
    # 成功したインデックスに基づいて、キャプションのリストをフィルタリング
    valid_captions = [all_captions_original[i] for i in valid_indices]

    # フィルタリングされた有効なキャプションだけを使ってテキスト埋め込みを計算
    text_embeddings = get_text_embeddings(valid_captions, model_text, device, args.batch_size)
    
    print(f"Embeddings calculated and aligned: Audio {audio_embeddings.shape}, Text {text_embeddings.shape}")
    # 4. 巨大な類似度行列を計算
    print("Calculating similarity matrix...")
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1).to(device)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1).to(device)
    sim_matrix = text_embeddings @ audio_embeddings.T

    # 5. 指標計算 (clap_helpers.metrics を使用)
    print("Calculating metrics using CLAP helper functions...")
    t2a_ranks, a2t_ranks = get_retrieval_ranks(sim_matrix.cpu())
    
    k_values = [1, 5, 10, 20, 30, 40, 50,100]
    t2a_recall_metrics = calculate_recall_at_k(t2a_ranks, k_values)
    t2a_median_rank = calculate_median_rank(t2a_ranks)

    a2t_recall_metrics = calculate_recall_at_k(a2t_ranks, k_values)
    a2t_median_rank = calculate_median_rank(a2t_ranks)

    # 6. 結果表示
    print("\n--- Text-to-Audio Retrieval ---")
    for k, v in t2a_recall_metrics.items():
        print(f"Recall@{k.split('@')[1]}:  {v:.2f}%")
    print(f"Median Rank: {t2a_median_rank:.1f}")
    
    print("\n--- Audio-to-Text Retrieval ---")
    for k, v in a2t_recall_metrics.items():
        print(f"Recall@{k.split('@')[1]}:  {v:.2f}%")
    print(f"Median Rank: {a2t_median_rank:.1f}")
    print("---------------------------------\n")

if __name__ == "__main__":
    main()