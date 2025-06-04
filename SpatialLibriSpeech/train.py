

# ===== train.py =====
"""学習スクリプト
例:
$ python train.py --audio ./audio --meta metadata.csv \
                 --batch 32 --epochs 100 \
                 --labels speech/azimuth speech/elevation \
                          speech/distance room/volume \
                          acoustics/drr_db acoustics/t30_ms
"""
import argparse, wandb, torch
from torch.utils.data import DataLoader
from foa_dataset import FOALabeledDataset
from spatial_branch import SpatialAttributesBranch
from pytorch_lamb import Lamb
import torch.nn 
import numpy as np
import torch
from time import time
import os  # ファイル保存のためのモジュール（ファイル先頭で import してもOK）
from torch.amp import GradScaler, autocast
save_dir = "/home/takamichi-lab-pc09/SpatialLibriSpeech/takamichi09/SpatialLibriSpeech/check"
os.makedirs(save_dir, exist_ok=True)  # ← この行が重要！


parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--meta", required=True)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
# -------- 追加の引数 (例) --------
parser.add_argument("--micro_bs", type=int, default=256)   # 物理バッチ
parser.add_argument("--accum",    type=int, default=4)     # 何 step 溜める?
...
args = parser.parse_args()




wandb.init(project="spatial_attrs",id="y7v693gt", resume="must", config=vars(args))

#n_iter=len(args.audio) // (args.micro_bs*args.accum)
ds_train = FOALabeledDataset(f"{args.audio}/first10sec_train", f"{args.meta}/metadata_clean_train.csv")  # labels cols fixed inside dataset
loader_train = DataLoader(ds_train, batch_size=args.micro_bs, shuffle=True, num_workers=4, pin_memory=False)

ds_val = FOALabeledDataset(f"{args.audio}/first10sec_val", f"{args.meta}/first10sec_metadata_val.csv")  # labels cols fixed inside dataset
loader_val = DataLoader(ds_val, batch_size=args.micro_bs, shuffle=False, num_workers=4, pin_memory=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
# 1) モデル・オプティマイザを復元
model = SpatialAttributesBranch()
model.load_state_dict(torch.load("/home/takamichi-lab-pc09/SpatialLibriSpeech/takamichi09/SpatialLibriSpeech/check/ckpt_ep054.pt"))
model.to(device)
opt = Lamb(model.parameters(), lr=args.lr, weight_decay=0.01)
#opt.load_state_dict(torch.load("/home/takamichi-lab-pc09/SpatialLibriSpeech/takamichi09/SpatialLibriSpeech/check/ckpt_ep054.pt"))
mse_loss= torch.nn.MSELoss()
import torch

from math import pi


def cosine_loss(pred, true):
    cos_sim = torch.cos(pred- true)
    return -cos_sim.mean()  # batch平均

def calc_loss(y_pred, y_true):
    """
    y_pred, y_true : shape (B, 44)
    Returns
    -------
    total_loss (Tensor)                …  デフォルト
    または
    dict(total=..., az=..., el=...,    …  ret_dict=True のとき
         dist=..., vol=..., drr=..., t30=...)
    """

    # ------- 個別損失 -------
    loss_az   = cosine_loss(y_pred[:, 0],  y_true[:, 0])
    loss_el   = cosine_loss(y_pred[:, 1],  y_true[:, 1])
    loss_dist = mse_loss( y_pred[:, 2],  y_true[:, 2])
    loss_vol  = mse_loss( y_pred[:, 3],  y_true[:, 3])
    loss_drr  = mse_loss( y_pred[:, 4:24],  y_true[:, 4:24])
    loss_t30  = mse_loss( y_pred[:, 24:44], y_true[:, 24:44])

    # ------- 重み係数 -------
    λ_angle  = 1.0     # azimuth / elevation
    λ_scalar = 1.0     # distance / volume
    λ_drr    = 1.0    # 20 要素 → 1/20
    λ_t30    = 1.0    # 同上

    # ------- 合計 -------
    total = (λ_angle  * (loss_az + loss_el) +
             λ_scalar * (loss_dist + loss_vol) +
             λ_drr    * loss_drr +
             λ_t30    * loss_t30)

    return {
            "total": total,
            "az":    loss_az,
            "el":    loss_el,
            "dist":  loss_dist,
            "vol":   loss_vol,
            "drr":   loss_drr,
            "t30":   loss_t30,
        }

scaler = GradScaler()

# ----------------- epoch ループ -----------
steps_per_epoch = 527
offset_step = 54 * steps_per_epoch

global_step = offset_step  # 54 epoch までの学習済みステップ数
for epoch in range(55, args.epochs + 1):
    model.train()
    opt.zero_grad(set_to_none=True)

    # エポック集計用
    epoch_sum = {k: 0.0 for k in ["total","az","el","dist","vol","drr","t30"]}
    n_batch = 0

    for step, (I_act, I_rea, y_true) in enumerate(loader_train):
        global_step += 1
        I_act, I_rea, y_true = [t.to(device, non_blocking=True)
                                for t in (I_act, I_rea, y_true)]

        # ----- forward & loss -----
        with autocast(device_type="cuda"):
            y_pred = model(I_act, I_rea)
            loss_dict = calc_loss(y_pred, y_true)   # ← 個別 loss を返す
            loss = loss_dict["total"] / args.accum                 # 累積用に 1/accum

        # ----- backward (勾配を貯める) -----
        scaler.scale(loss).backward()

        # ----- accm 回ごとにパラメータ更新 -----
        if (step + 1) % args.accum == 0 or (step + 1) == len(loader_train):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        # ----- ❶ step ごと WandB -----
        wandb.log({f"train_loss_step/{k}": v for k, v in loss_dict.items()}, step=global_step)
                 # step=global_step)

        # ----- epoch 集計 -----
        for k in epoch_sum:
            epoch_sum[k] += loss_dict[k]
        n_batch += 1

    # ----- ❷ epoch ごと WandB -----
    avg_log = {f"train_loss_epoch/{k}": epoch_sum[k] / n_batch
               for k in epoch_sum}
    avg_log["epoch"] = epoch
    wandb.log(avg_log, step=global_step)
    model.eval()
    val_sum = {k:0.0 for k in ["total","az","el","dist","vol","drr","t30"]}
    val_batches = 0
    with torch.no_grad():
        for I_act, I_rea, y_true in loader_val:
            I_act, I_rea, y_true = [t.to(device, non_blocking=True) for t in (I_act, I_rea, y_true)]

            with autocast(device_type="cuda"):
                y_pred = model(I_act, I_rea)
                loss_dict = calc_loss(y_pred, y_true)

            for k in val_sum:
                val_sum[k] += loss_dict[k].item()
            val_batches += 1

    val_log= {f"val_loss_epoch/{k}": val_sum[k] / val_batches
               for k in val_sum}
    val_log["epoch"] = epoch
    wandb.log(val_log,step=global_step)  # ❸ val loss を WandB に記録

    # ----- 保存 & 進捗表示 -----
    torch.save(model.state_dict(), f"{save_dir}/ckpt_ep{epoch:03d}.pt")
    print(f"Epoch {epoch:03d} | " +
            " | ".join([f"train_{k}={avg_log[f'train_loss_epoch/{k}']:.4f}" for k in ['total','az','el','dist','vol','drr','t30']]) +
            " | " +
            " | ".join([f"val_{k}={val_log[f'val_loss_epoch/{k}']:.4f}" for k in ['total','az','el','dist','vol','drr','t30']]))


wandb.finish()  # WandB の終了