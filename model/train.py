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
from foa_dataset_caption import FOADatasetWithIV
from text_encoder import TextEncoder   # your TextEncoder implementation
from audio_encoder import AudioEncoder       # your AudioEncoder implementation

# -------------------- Utils --------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Stack tensors so the model processes the whole batch at once.

    Returns:
        I_act (B, 3, F, T), I_rea (B, 3, F, T), omni (B, L), captions (list[str])
    """
    I_act, I_rea, omni, caps = zip(*batch)
    return (
        torch.stack(I_act, 0),
        torch.stack(I_rea, 0),
        torch.stack(omni, 0),
        list(caps),
    )


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


def forward_audio(audio_encoder: AudioEncoder,
                  I_act: torch.Tensor,
                  I_rea: torch.Tensor,
                  omni: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    """Encode the *entire batch* of audio."""
    with autocast(enabled=device.type == "cuda"):
        return audio_encoder(I_act.to(device), I_rea.to(device), omni.to(device))  # (B,512)


def forward_text(text_encoder: TextEncoder,
                 captions: List[str],
                 device: torch.device) -> torch.Tensor:
    with autocast(enabled=device.type == "cuda"):
        return text_encoder(captions).to(device)   # (B,512)

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
    running_loss, running_a2t, running_t2a = 0.0, 0.0, 0.0
    epoch_loss =0
    count = 0
    print(f"count1:{count}")
    
    for step, (I_act, I_rea, omni, caps) in enumerate(loader):
        I_act = I_act.to(device)
        I_rea = I_rea.to(device)
        omni  = omni.to(device)
        
        optim.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            a_emb = model_a(I_act, I_rea, omni)    # (B,512)
            t_emb = model_t(caps)                   # (B,512)
            loss_dict = info_nce_loss(a_emb, t_emb, logit_scale)
            total_loss = loss_dict['total_loss']
            loss_a2t = loss_dict['loss_a2t']
            loss_t2a = loss_dict['loss_t2a']
        scaler.scale(total_loss).backward()
        scaler.step(optim)
        scaler.update()

        # running stats
        running_loss += total_loss.item(); running_a2t += loss_a2t.item(); running_t2a += loss_t2a.item()
        epoch_loss += total_loss.item()
        count += 1
        print(f"count2:{count}")
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_a2t = running_a2t / log_interval
            avg_t2a = running_t2a / log_interval
            wandb.log({
                'train/loss': avg_loss,
                'train/loss_a2t': avg_a2t,
                'train/loss_t2a': avg_t2a,
                'train/logit_scale': logit_scale.exp().item(),
                'train/step': epoch * len(loader) + step
            })
            running_loss = running_a2t = running_t2a = 0.0
    print(f"count3:{count}")
    wandb.log({
        'train/epoch_loss': epoch_loss / count,
        'train/epoch': epoch
    })
    return epoch_loss / count
# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", type=str, default="./ckpt")
    parser.add_argument("--wandb" ,type=str, default = "elsa")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FOADatasetWithIV(args.audio_dir, args.csv)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True)

    model_audio = AudioEncoder().to(device)
    model_text = TextEncoder().to(device)

    # Shared learnable logit scale (same initial value as CLIP)
# これだけでリーフ・Tensor として GPU に乗る
    logit_scale    = nn.Parameter(torch.tensor(np.log(1/0.07)))

    optimizer = torch.optim.Adam(
        list(model_audio.parameters()) + list(model_text.parameters()) + [logit_scale],
        lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    use_wandb = args.wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb, 
                   config=dict(vars(args)), save_code=True)
        wandb.watch([model_audio, model_text])

    best_loss = float("inf")
    log_interval =1
    for epoch in range(1, args.epochs + 1):
        logit_scale.to(device)
        loss = train_one_epoch(model_audio, model_text, dl,
                               optimizer, scaler, device, logit_scale,epoch,log_interval)
        scheduler.step()
        print(f"[Epoch {epoch}/{args.epochs}] loss={loss:.4f}")

        # Save checkpoint
        torch.save({
            "audio": model_audio.state_dict(),
            "text": model_text.state_dict(),
            "logit_scale": logit_scale,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, Path("/home/takamichi-lab-pc09/elsa/ckpt/takamichi09/elsa") / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
