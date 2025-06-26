import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
# 必要なカスタムモジュールをインポート
from model.audio_encoder import AudioEncoder
from model.foa_dataset_caption import FOADatasetWithIV
from model.train import collate_fn
import pandas as pd
import plotly.express as px
from umap import UMAP

# from sklearn.manifold import TSNE # t-SNEを使いたい場合

import pandas as pd
import plotly.express as px
from umap import UMAP
import numpy as np
import wandb # wandbをインポート

def visualize_embeddings(embeddings_512d, dataset):
    """ UMAPで次元削減し、様々なメタデータで色分けしたインタラクティブなプロットをwandbに記録する """
    
    print("\n--- Starting Visualization ---")
    
    # 1. UMAPで次元削減
    print("Running UMAP to reduce dimensions...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings_512d.numpy())
    
    # 2. プロット用のDataFrameを作成
    print("Preparing DataFrame for plotting...")
    plot_df = dataset.meta.copy()
    plot_df['umap_x'] = embeddings_2d[:, 0]
    plot_df['umap_y'] = embeddings_2d[:, 1]
    
    # hoverで表示する情報を定義（全プロットで共通化）
    hover_data_cols = ['caption', 'room_id', 'meta/azimuth', 'meta/elevation', 'meta/t30']

    # --- 可視化パターンA: 空間的特徴（方位角）で色分け ---
    print("Generating and logging plot colored by Azimuth...")
    fig_azimuth = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='meta/azimuth',
        color_continuous_scale=px.colors.diverging.Tealrose,
        hover_data=hover_data_cols,
        title='UMAP by Azimuth (Spatial Feature)'
    )
    # fig_azimuth.show() # ローカルで表示する場合はコメントを外す
    wandb.log({"UMAP by Azimuth": fig_azimuth})

    # --- 可視化パターンB: 音響的特徴（部屋ID）で色分け ---
    print("Generating and logging plot colored by Room ID...")
    fig_room = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='room_id',
        hover_data=hover_data_cols,
        title='UMAP by Room ID (Acoustic Feature)'
    )
    # fig_room.show()
    wandb.log({"UMAP by Room ID": fig_room})

    # --- ★★★ ここから追加パターン ★★★ ---

    # --- 可視化パターンC: 空間的特徴（仰角）で色分け ---
    print("Generating and logging plot colored by Elevation...")
    fig_elevation = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='meta/elevation',
        color_continuous_scale=px.colors.diverging.Picnic,
        hover_data=hover_data_cols,
        title='UMAP by Elevation (Spatial Feature)'
    )
    # fig_elevation.show()
    wandb.log({"UMAP by Elevation": fig_elevation})

    # --- 可視化パターンD: 音響的特徴（残響時間 t30）で色分け ---
    print("Generating and logging plot colored by Reverberation Time (t30)...")
    fig_t30 = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='meta/t30',
        color_continuous_scale='Viridis',
        hover_data=hover_data_cols,
        title='UMAP by Reverberation Time T30 (Acoustic Feature)'
    )
    # fig_t30.show()
    wandb.log({"UMAP by T30": fig_t30})

    # --- 可視化パターンE: 音響的特徴（部屋の面積）で色分け ---
    # 面積は値の範囲が広いため、対数を取ってから色分けすると見やすい
    plot_df['log_area'] = np.log10(plot_df['meta/area'] + 1e-6)
    print("Generating and logging plot colored by Room Area (log scale)...")
    fig_area = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='log_area',
        color_continuous_scale='Plasma',
        hover_data=hover_data_cols + ['meta/area'],
        title='UMAP by Room Area [log10] (Acoustic Feature)'
    )
    # fig_area.show()
    wandb.log({"UMAP by Room Area": fig_area})
    
    # --- 可視化パターンF: 空間的特徴（音源のX座標）で色分け ---
    print("Generating and logging plot colored by Source X Position...")
    fig_pos_x = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='source_pos_x',
        color_continuous_scale='Inferno',
        hover_data=hover_data_cols,
        title='UMAP by Source X-Position (Spatial Feature)'
    )
    # fig_pos_x.show()
    wandb.log({"UMAP by Source X-Pos": fig_pos_x})

    # --- 可視化パターンG: 空間的特徴（音源の方向カテゴリ）で色分け ---
    # 方位角を「左・正面・右」の3つのカテゴリに分けてみる
    def assign_4_directions(az):
        if -35.0 <= az <= 35.0:
            return 'front'
        elif 55.0 <= az <= 125.0:
            return 'right'
        elif -125.0 <= az <= -55.0:
            return 'left'
        elif az <= -145.0 and  145.0 <= az:
            return 'back'
     
        
    plot_df['direction_4cat'] = plot_df['meta/azimuth'].apply(assign_4_directions)
    print("Generating and logging plot colored by 4-Direction Category...")
    fig_direction_4 = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='direction_4cat',
        category_orders={"direction_4cat": ["front", "right", "left", "back"]}, # 表示順を固定
        hover_data=hover_data_cols,
        title='UMAP by 4-Direction Category (Spatial Feature)'
    )
    wandb.log({"UMAP by 4-Direction": fig_direction_4})

    # --- ★★★ 可視化パターンH: 拡張された意味的キーワードで色分け ★★★ ---
    # より多くのキーワードとカテゴリで分類する
    def find_keyword_extended(caption):
        caption_lower = caption.lower()
        # 優先度の高いものから順に判定
        if any(word in caption_lower for word in ['music', 'song', 'instrument', 'guitar', 'piano', 'drum', 'orchestra']):
            return "Music"
        if any(word in caption_lower for word in ['speak', 'vocal', 'man', 'woman', 'choir', 'laugh', 'shout', 'whisper']):
            return "Human Voice"
        if any(word in caption_lower for word in ['car', 'engine', 'vehicle', 'truck', 'bus', 'motorcycle', 'siren', 'horn']):
            return "Vehicle"
        if any(word in caption_lower for word in ['drill', 'engine', 'machine', 'tool', 'alarm', 'fan']):
            return "Machine/Alarm"
        if any(word in caption_lower for word in ['dog', 'cat', 'bird', 'bark', 'meow', 'chirp', 'insect']):
            return "Animal"
        if any(word in caption_lower for word in ['wind', 'rain', 'water', 'stream', 'thunder', 'fire', 'forest']):
            return "Nature/Ambient"
        if any(word in caption_lower for word in ['door', 'knock', 'keyboard', 'typing', 'footsteps', 'utensil', 'dishes']):
            return "Domestic/SFX"
        return "Other"

    plot_df['semantic_category_ext'] = plot_df['caption'].apply(find_keyword_extended)
    print("Generating and logging plot colored by Extended Semantic Keywords...")
    
    # カテゴリの出現頻度を計算して、凡例の表示順を制御する
    category_order = plot_df['semantic_category_ext'].value_counts().index.tolist()
    
    fig_semantic_ext = px.scatter(
        plot_df, x='umap_x', y='umap_y', color='semantic_category_ext',
        category_orders={"semantic_category_ext": category_order}, # 出現頻度順に
        hover_data=hover_data_cols + ['semantic_category_ext'],
        title='UMAP by Extended Semantic Keywords'
    )
    wandb.log({"UMAP by Extended Semantics": fig_semantic_ext})
def main():
    parser = argparse.ArgumentParser(description="Find top-k similar audio samples for a given query audio.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained AudioEncoder checkpoint.")
    parser.add_argument("--csv", required=True, help="Path to the SpatialAudioCaps manifest CSV (e.g., manifest_test.csv).")
    parser.add_argument("--audio_dir", required=True, help="Path to the SpatialAudioCaps FOA audio directory.")
    parser.add_argument("--query_index", type=int, required=True, help="Index of the query audio sample in the dataset (e.g., 0).")
    parser.add_argument("--top_k", type=int, default=10, help="Number of similar items to retrieve.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding.")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. モデルのロード
    print("Loading AudioEncoder model...")
    model_audio = AudioEncoder()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_audio.load_state_dict(ckpt["audio"])
    model_audio.to(device).eval()
    print(f"Model loaded from epoch {ckpt.get('epoch', 'N/A')}.")
    wandb.init(
        project="qualitative-analysis", # プロジェクト名
        name=f"query_{args.query_index}_epoch_{ckpt.get('epoch', 'N/A')}", # 実行名
        config=args # 引数を記録
    )
    # 2. データセットの準備
    print("Preparing SpatialAudioCaps dataset...")
    dataset = FOADatasetWithIV(audio_folder=args.audio_dir, metadata_csv=args.csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. 全ての音声の埋め込みとキャプションを一括で計算・取得
    print("Calculating embeddings for all audio files...")
    all_embeddings = []
    all_captions = []
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding Audio"):
            I_act, I_rea, omni, caps, _ = batch_data
            if I_act is None: continue
            
            I_act, I_rea, omni = I_act.to(device), I_rea.to(device), omni.to(device)
            a_emb = model_audio(I_act, I_rea, omni)
            all_embeddings.append(a_emb.cpu())
            all_captions.extend(caps)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=-1) # L2正規化

    # 4. クエリの選択と類似度計算
    if not (0 <= args.query_index < len(all_captions)):
        print(f"[Error] query_index must be between 0 and {len(all_captions) - 1}")
        return

    query_embedding = all_embeddings[args.query_index].unsqueeze(0) # (1, 512)
    query_caption = all_captions[args.query_index]

    print("\n" + "="*50)
    print(f"Query Audio (Index: {args.query_index})")
    print(f"Caption: '{query_caption}'")
    print("="*50 + "\n")

    # 全ての埋め込みとのコサイン類似度を計算
    similarities = query_embedding @ all_embeddings.T # (1, N)
    
    # 5. 上位K件を取得して表示
    # 自分自身（類似度1.0）が1位になるので、トップK+1件を取得して自分を除外する
    top_k_sims, top_k_indices = torch.topk(similarities[0], k=args.top_k + 1)

    print(f"--- Top {args.top_k} most similar audio samples ---")
    for i in range(1, args.top_k + 1):
        idx = top_k_indices[i].item()
        sim_score = top_k_sims[i].item()
        caption = all_captions[idx]
        print(f"Rank {i}:")
        print(f"  Similarity: {sim_score:.4f}")
        print(f"  Caption (Index {idx}): '{caption}'")
        print("-" * 20)

    visualize_embeddings(all_embeddings, dataset)

    wandb.finish()
    print("Successfully logged plots to wandb.")


if __name__ == "__main__":
    main()
    