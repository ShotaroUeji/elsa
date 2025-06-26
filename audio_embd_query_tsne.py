import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE # ★ UMAPからt-SNEに変更
import numpy as np
import wandb

# 必要なカスタムモジュールをインポート
from model.audio_encoder import AudioEncoder
from model.foa_dataset_caption import FOADatasetWithIV
from model.train import collate_fn

def visualize_embeddings_tsne(embeddings_512d, dataset, query_index, wandb_run_name):
    """ t-SNEで次元削減し、様々なメタデータで色分けしたプロットをwandbに記録する """
    
    print("\n--- Starting t-SNE Visualization ---")
    
    # wandbの初期化
    wandb.init(project="qualitative-analysis-tsne", name=wandb_run_name)
    
    # 1. t-SNEで次元削減
    print("Running t-SNE to reduce dimensions... (This may take several minutes)")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings_512d.numpy())
    
    # 2. プロット用のDataFrameを作成
    print("Preparing DataFrame for plotting...")
    plot_df = dataset.meta.copy()
    plot_df['tsne_x'] = embeddings_2d[:, 0]
    plot_df['tsne_y'] = embeddings_2d[:, 1]
    plot_df['caption'] = [dataset.meta.iloc[i]['caption'] for i in range(len(dataset))]
    
    hover_data_cols = ['caption', 'room_id', 'meta/azimuth', 'meta/elevation', 'meta/t30']

    # --- パターンA〜H: これまでの可視化をt-SNEで実行 ---
    # (コードが長くなるため、代表としてA, G, Hと、新しいIを詳細に記述します。他も同様の要領です)

    # --- パターンA: 空間的特徴（方位角）で色分け ---
    print("Logging plot colored by Azimuth...")
    fig_azimuth = px.scatter(
        plot_df, x='tsne_x', y='tsne_y', color='meta/azimuth',
        color_continuous_scale=px.colors.diverging.Tealrose,
        hover_data=hover_data_cols, title='t-SNE by Azimuth (Spatial Feature)'
    )
    wandb.log({"t-SNE by Azimuth": fig_azimuth})

    # --- パターンG: 4方向の方向カテゴリで色分け ---
    def assign_4_directions(azimuth):
        if -45 <= azimuth <= 45: return "Front"
        if 45 < azimuth <= 135: return "Right"
        if -135 <= azimuth < -45: return "Left"
        return "Back"
    plot_df['direction_4cat'] = plot_df['meta/azimuth'].apply(assign_4_directions)
    print("Logging plot colored by 4-Direction Category...")
    fig_direction_4 = px.scatter(
        plot_df, x='tsne_x', y='tsne_y', color='direction_4cat',
        category_orders={"direction_4cat": ["Left", "Front", "Right", "Back"]},
        hover_data=hover_data_cols, title='t-SNE by 4-Direction Category'
    )
    wandb.log({"t-SNE by 4-Direction": fig_direction_4})

    # --- パターンH: 拡張された意味的キーワードで色分け ---
    def find_keyword_extended(caption):
        # ... (前回の回答と同じ関数のため、内容は省略) ...
        caption_lower = caption.lower()
        if any(word in caption_lower for word in ['music', 'song', 'instrument', 'guitar', 'piano', 'drum', 'orchestra']): return "Music"
        if any(word in caption_lower for word in ['speak', 'vocal', 'man', 'woman', 'choir', 'laugh', 'shout', 'whisper']): return "Human Voice"
        if any(word in caption_lower for word in ['car', 'engine', 'vehicle', 'truck', 'bus', 'motorcycle', 'siren', 'horn']): return "Vehicle"
        if any(word in caption_lower for word in ['drill', 'engine', 'machine', 'tool', 'alarm', 'fan']): return "Machine/Alarm"
        if any(word in caption_lower for word in ['dog', 'cat', 'bird', 'bark', 'meow', 'chirp', 'insect']): return "Animal"
        if any(word in caption_lower for word in ['wind', 'rain', 'water', 'stream', 'thunder', 'fire', 'forest']): return "Nature/Ambient"
        if any(word in caption_lower for word in ['door', 'knock', 'keyboard', 'typing', 'footsteps', 'utensil', 'dishes']): return "Domestic/SFX"
        return "Other"
    plot_df['semantic_category_ext'] = plot_df['caption'].apply(find_keyword_extended)
    print("Logging plot colored by Extended Semantic Keywords...")
    category_order = plot_df['semantic_category_ext'].value_counts().index.tolist()
    fig_semantic_ext = px.scatter(
        plot_df, x='tsne_x', y='tsne_y', color='semantic_category_ext',
        category_orders={"semantic_category_ext": category_order},
        hover_data=hover_data_cols + ['semantic_category_ext'], title='t-SNE by Extended Semantic Keywords'
    )
    wandb.log({"t-SNE by Extended Semantics": fig_semantic_ext})
    
# visualize_embeddings_tsne 関数の最後の方、wandb.finish() の前にある
# 「パターンJ」のブロックを以下に差し替える

    # --- ★★★ 修正版: パターンJ: 音源とマイクの物理的な距離で色分け ★★★ ---
    print("Generating and logging plot colored by Source-Microphone Distance (Room Center)...")
    
    # マイク位置が部屋の中心(rx/2, ry/2, rz/2)であると仮定して距離を再計算
    mic_pos_x = plot_df['room_dim_x'] / 2
    mic_pos_y = plot_df['room_dim_y'] / 2
    mic_pos_z = plot_df['room_dim_z'] / 2
    
    plot_df['source_distance'] = np.sqrt(
        (plot_df['source_pos_x'] - mic_pos_x)**2 +
        (plot_df['source_pos_y'] - mic_pos_y)**2 +
        (plot_df['source_pos_z'] - mic_pos_z)**2
    )
    
    # hoverで表示する情報に物理距離も追加
    distance_hover_data = hover_data_cols + ['source_distance']
    
    fig_phys_dist = px.scatter(
        plot_df, x='tsne_x', y='tsne_y', color='source_distance',
        color_continuous_scale='Cividis', # 距離が見やすいカラースケール
        hover_data=distance_hover_data,
        labels={'color': 'Source-Mic Distance (m)'}, # カラーバーのラベル
        title='t-SNE by Physical Source-Microphone Distance (Mic at Room Center)'
    )
    wandb.log({"t-SNE by Source-Mic Distance": fig_phys_dist})
    wandb.finish()
    print("\nSuccessfully logged all t-SNE plots to wandb.")


def main():
    parser = argparse.ArgumentParser(description="Visualize audio embeddings using t-SNE.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained AudioEncoder checkpoint.")
    parser.add_argument("--csv", required=True, help="Path to the SpatialAudioCaps manifest CSV.")
    parser.add_argument("--audio_dir", required=True, help="Path to the SpatialAudioCaps FOA audio directory.")
    # パターンIのためにクエリインデックスを引数に追加
    parser.add_argument("--query_index", type=int, default=0, help="Index of the query audio for distance visualization.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. モデルロード
    print("Loading AudioEncoder model...")
    model_audio = AudioEncoder()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_audio.load_state_dict(ckpt["audio"])
    model_audio.to(device).eval()
    epoch = ckpt.get('epoch', 'N/A')
    print(f"Model loaded from epoch {epoch}.")

    # 2. データ準備
    print("Preparing SpatialAudioCaps dataset...")
    dataset = FOADatasetWithIV(audio_folder=args.audio_dir, metadata_csv=args.csv)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. 全埋め込み取得
    print("Calculating all embeddings...")
    all_embeddings = []
    # get_audio_embeddings関数を展開してここに記述
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding Audio"):
            I_act, I_rea, omni, _, _ = batch_data # インデックスは使わない
            if I_act is None: continue
            I_act, I_rea, omni = I_act.to(device), I_rea.to(device), omni.to(device)
            a_emb = model_audio(I_act, I_rea, omni)
            all_embeddings.append(a_emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # 4. 可視化関数の呼び出し
    run_name = f"epoch_{epoch}_query_{args.query_index}"
    visualize_embeddings_tsne(all_embeddings, dataset, args.query_index, run_name)

if __name__ == "__main__":
    main()