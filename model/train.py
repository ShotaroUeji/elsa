"""train.py (batch version)

Contrastive training (InfoNCE) between audio and text embeddings **in full batches**,
assuming *all inputs are already length‑aligned* by the dataset.

Dependencies
------------
* foa_dataset_caption.FOADatasetWithIV  (returns I_act, I_rea, omni, caption)
* AudioEncoder, TextEncoder  (each outputs 512‑dim)

Run example
-----------
python train.py \
  --csv /data/captions.csv \
  --audio_dir /data/foa_flac \
  --epochs 20 --batch 64 --lr 1e-4 --out ./ckpt
  takamichi-lab-pc09@takamichi-lab-pc09:~/elsa$ python3 model/train.py --csv Spatial_AudioCaps --audio_dir Spatial_AudioCaps/takamichi09/SpatialAudioCaps/foa --batch 64 --epochs 40 
"""
from __future__ import annotations

import wandb
import argparse, os, random, math, time
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------- Local modules ----------
from .foa_dataset_caption import FOADatasetWithIV
from .text_encoder import TextEncoder   # your TextEncoder implementation
from .audio_encoder import AudioEncoder       # your AudioEncoder implementation

# -------------------- Utils --------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def collate_fn(batch):
#     """Stack tensors so the model processes the whole batch at once.

#     Returns:
#         I_act (B, 3, F, T), I_rea (B, 3, F, T), omni (B, L), captions (list[str])
#     """
#     I_act, I_rea, omni, caps = zip(*batch)
#     return (
#         torch.stack(I_act, 0),
#         torch.stack(I_rea, 0),
#         torch.stack(omni, 0),
#         list(caps),
#     )

def collate_fn(batch):
    """Stack tensors so the model processes the whole batch at once."""
    # `None` を返すようにしたデータセットに対応するため、Noneをフィルタリング
    original_size = len(batch)
    batch = [b for b in batch if b is not None]
    
    if len(batch) < original_size:
        print(f"\n[Info] Skipped {original_size - len(batch)} corrupted files in a batch.")

    # バッチ内の全てのファイルが破損していた場合
    if not batch:
        return (None, None, None, [], None)

    I_act, I_rea, omni, caps, idxs = zip(*batch)
    return (
        torch.stack(I_act, 0),
        torch.stack(I_rea, 0),
        torch.stack(omni, 0),
        list(caps),
        list(idxs)
    )
def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


# -------------------- Training loop --------------------

#https://amaarora.github.io/posts/2023-03-11_Understanding_CLIP_part_2.html?utm_source=chatgpt.comを真似した
def get_ground_truth(device, num_logits) -> torch.Tensor:
    labels = torch.arange(num_logits, device=device, dtype=torch.long)
    return labels
#
def info_nce_loss(audio_embeds: torch.Tensor,
                  text_embeds: torch.Tensor,
                  logit_scale: torch.nn.Parameter) -> torch.Tensor:
    audio_embeds = l2_normalize(audio_embeds)
    text_embeds = l2_normalize(text_embeds)

    logits_per_audio = logit_scale*audio_embeds @ text_embeds.T
    logits_per_text = logit_scale*text_embeds @ audio_embeds.T

    targets = get_ground_truth(device=logits_per_audio.device, num_logits=logits_per_audio.shape[0])
    loss_a2t = F.cross_entropy(logits_per_audio, targets)
    loss_t2a = F.cross_entropy(logits_per_text, targets)
    total_loss =(loss_a2t + loss_t2a) /2
    return {"loss_a2t": loss_a2t, "loss_t2a": loss_t2a, "total_loss": total_loss}

# --------------------------- train -----------------------------------------

def val_one_epoch(model_a: AudioEncoder,
                 model_t: TextEncoder,
                 loader: DataLoader,
                 optimizer,
                 scaler,
                 device,
                 logit_scale: torch.nn.Parameter,
                 epoch: int,
                 ) -> None:
    model_a.eval(); model_t.eval()
    epoch_loss, epoch_a2t, epoch_t2a = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for step, (I_act, I_rea, omni, caps, _) in enumerate(loader):
            if I_act == None:
                continue
            I_act = I_act.to(device)
            I_rea = I_rea.to(device)
            omni  = omni.to(device)
            a_emb = model_a(I_act, I_rea, omni)    # (B,512)
            t_emb = model_t(caps)                   # (B,512)
            loss_dict = info_nce_loss(a_emb, t_emb, logit_scale)
            total_loss = loss_dict['total_loss']
            loss_a2t = loss_dict['loss_a2t']
            loss_t2a = loss_dict['loss_t2a']  
            epoch_loss += total_loss.item()
            epoch_a2t += loss_a2t.item()
            epoch_t2a += loss_t2a.item()
            count += 1 
    wandb.log({
    'val/epoch_loss': epoch_loss / count,
    'val/epoch_loss_a2t': epoch_a2t / count,
    'val/epoch_loss_t2a': epoch_t2a / count,
    'val/logit_scale': logit_scale.item(),
    'val/epoch': epoch,
    })

    return None
    
    

def train_one_epoch(model_a: AudioEncoder,
                    model_t: TextEncoder,
                    loader: DataLoader,
                    optim,
                    scaler: GradScaler,
                    device,
                    logit_scale: torch.nn.Parameter,
                    epoch: int,
                    log_interval: int):
    model_a.train(); model_t.train()
    
    # --- 変更点 1: 勾配累積のステップ数を設定 ---
    accumulation_steps = 32  # 2000 / 64 を切り上げて設定

    running_loss, running_a2t, running_t2a = 0.0, 0.0, 0.0
    epoch_loss, epoch_a2t, epoch_t2a = 0.0, 0.0, 0.0
    count = 0

    # --- 変更点 2: ループの前に勾配をゼロにする ---
    # ループ内で毎回ゼロにするのではなく、更新のタイミングでゼロにするため
    optim.zero_grad(set_to_none=True)

    for step, (I_act, I_rea, omni, caps, _) in enumerate(loader):
        if I_act == None:
            continue
        I_act = I_act.to(device)
        I_rea = I_rea.to(device)
        omni  = omni.to(device)
        
        # optim.zero_grad(set_to_none=True) # <-- この行をループの先頭に移動

        with autocast(enabled=device.type == "cuda"):
            a_emb = model_a(I_act, I_rea, omni)
            t_emb = model_t(caps)
            loss_dict = info_nce_loss(a_emb, t_emb, logit_scale)
            total_loss = loss_dict['total_loss']
            loss_a2t = loss_dict['loss_a2t']
            loss_t2a = loss_dict['loss_t2a']

            # --- 変更点 3: 損失を累積ステップ数で割る ---
            # 勾配がN倍になってしまうのを防ぐため、平均化する
            loss_to_backward = total_loss / accumulation_steps

        # 毎回勾配を計算して加算していく
        scaler.scale(loss_to_backward).backward()

        # --- 変更点 4: accumulation_steps ごとにモデルを更新 ---
        if (step + 1) % accumulation_steps == 0:
            # 累積した勾配を使ってモデルの重みを更新
            scaler.step(optim)
            # scalerを更新
            scaler.update()
            # 次の累積のために勾配をリセット
            optim.zero_grad(set_to_none=True)

        # running stats (この部分は変更なし)
        running_loss += total_loss.item(); running_a2t += loss_a2t.item(); running_t2a += loss_t2a.item()
        epoch_loss += total_loss.item()
        epoch_a2t += loss_a2t.item()
        epoch_t2a += loss_t2a.item()
        count += 1
    
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_a2t = running_a2t / log_interval
            avg_t2a = running_t2a / log_interval
            lr = optim.param_groups[0]['lr']
            wandb.log({
                'train/loss': avg_loss,
                'train/loss_a2t': avg_a2t,
                'train/loss_t2a': avg_t2a,
                'train/logit_scale': logit_scale.item(),
                'lr': lr
            })
            running_loss = running_a2t = running_t2a = 0.0
    # --- 変更点 5 (オプション): ループの最後に残った勾配で更新 ---
    # データセットの総ステップ数が accumulation_steps で割り切れない場合、
    # 最後の数ステップ分の勾配が無駄になるのを防ぐ
    if (len(loader) % accumulation_steps != 0):
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

    # ... (エポック単位のログ記録はそのまま)
    wandb.log({
        'train/epoch_loss': epoch_loss / count,
        'train/epoch_loss_a2t': epoch_a2t / count,
        'train/epoch_loss_t2a': epoch_t2a / count,
        'train/logit_scale': logit_scale.item(),
        'train/epoch': epoch
    })
    return epoch_loss / count

# 累積じゃないとき
# def train_one_epoch(model_a: AudioEncoder,
#                     model_t: TextEncoder,
#                     loader: DataLoader,
#                     optim,
#                     scaler: GradScaler,
#                     device,
#                     logit_scale: torch.nn.Parameter,
#                     epoch: int,
#                     log_interval: int):
#     model_a.train(); model_t.train()
#     running_loss, running_a2t, running_t2a = 0.0, 0.0, 0.0
#     epoch_loss =0
#     epoch_a2t = 0.0
#     epoch_t2a = 0.0
#     count = 0

    
#     for step, (I_act, I_rea, omni, caps) in enumerate(loader):
#         I_act = I_act.to(device)
#         I_rea = I_rea.to(device)
#         omni  = omni.to(device)
        
#         optim.zero_grad(set_to_none=True)
#         with autocast(enabled=device.type == "cuda"):
#             a_emb = model_a(I_act, I_rea, omni)    # (B,512)
#             t_emb = model_t(caps)                   # (B,512)
#             loss_dict = info_nce_loss(a_emb, t_emb, logit_scale)
#             total_loss = loss_dict['total_loss']
#             loss_a2t = loss_dict['loss_a2t']
#             loss_t2a = loss_dict['loss_t2a']
#         scaler.scale(total_loss).backward()
#         scaler.step(optim)
#         scaler.update()

#         # running stats
#         running_loss += total_loss.item(); running_a2t += loss_a2t.item(); running_t2a += loss_t2a.item()
#         epoch_loss += total_loss.item()
#         epoch_a2t += loss_a2t.item()
#         epoch_t2a += loss_t2a.item()
#         count += 1
    
#         if (step + 1) % log_interval == 0:
#             avg_loss = running_loss / log_interval
#             avg_a2t = running_a2t / log_interval
#             avg_t2a = running_t2a / log_interval
#             lr = optim.param_groups[0]['lr']
#             wandb.log({
#                 'train/loss': avg_loss,
#                 'train/loss_a2t': avg_a2t,
#                 'train/loss_t2a': avg_t2a,
#                 'train/logit_scale': logit_scale.item(),
#                 'lr': lr
#             })
#             running_loss = running_a2t = running_t2a = 0.0

#     wandb.log({
#         'train/epoch_loss': epoch_loss / count,
#         'train/epoch_loss_a2t': epoch_a2t / count,
#         'train/epoch_loss_t2a': epoch_t2a / count,
#         'train/logit_scale': logit_scale.item(),
#         'train/epoch': epoch
#     })
#     return epoch_loss / count
# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5 * 1e-5)
    parser.add_argument("--out", type=str, default="./ckpt")
    parser.add_argument("--wandb" ,type=str, default = "elsa")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to resume training from a checkpoint")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="WandB run ID to resume from, if any")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FOADatasetWithIV(f"{args.audio_dir}/train", f"{args.csv}/manifest_train.csv")  # labels cols fixed inside dataset
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, collate_fn=collate_fn, pin_memory=False, drop_last=True)
    
    val_ds = FOADatasetWithIV(f"{args.audio_dir}/val", f"{args.csv}/manifest_val.csv")  # labels cols fixed inside dataset
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=4, collate_fn=collate_fn, pin_memory=False, drop_last=False)
    

    model_audio = AudioEncoder().to(device)
    model_text = TextEncoder().to(device)

    # Shared learnable logit scale (same initial value as CLIP)
# これだけでリーフ・Tensor として GPU に乗る
    logit_scale    = nn.Parameter(torch.tensor(np.log(1/0.07)))

    optimizer = torch.optim.Adam(
        list(model_audio.parameters()) + list(model_text.parameters()) + [logit_scale],
        lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    use_wandb = args.wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb, 
                   config=dict(vars(args)), save_code=True,
                   resume = "allow" if args.wandb_id is not None else None,
                   id=args.wandb_id)
        wandb.watch([model_audio, model_text])

    best_loss = float("inf")
    log_interval =1
    resume_ckpt = args.resume is not None
    if resume_ckpt:
        ckpt = torch.load(args.resume, map_location="cuda:0")
        model_audio.load_state_dict(ckpt["audio"])
        model_text.load_state_dict(ckpt["text"])
        logit_scale.data = ckpt["logit_scale"]
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        print(f"Resuming from epoch {start_epoch}, best loss {best_loss:.4f}")
    else:start_epoch=1
    for epoch in range(start_epoch, args.epochs + 1):
        logit_scale.to(device)
        loss = train_one_epoch(model_audio, model_text, train_dl,
                              optimizer, scaler, device, logit_scale,epoch,log_interval)
        scheduler.step()
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={loss:.4f}")
        #print(f"[Epoch {epoch}/{args.epochs}] train_loss={12:.4f}")
        val_one_epoch(model_audio, model_text, val_dl, optimizer, scaler, device, logit_scale,
                        epoch)
        
        # Save checkpoint
        torch.save({
            "audio": model_audio.state_dict(),
            "text": model_text.state_dict(),
            "logit_scale": logit_scale,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, Path("ckpt/takamichi09/elsa_ckpt_batch2000") / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
