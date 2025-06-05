# # patch_ckpt.py
# import torch, os
# from transformers import ClapProcessor, ClapAudioModelWithProjection
# import torch
# ckpt = torch.load("/home/takamichi-lab-pc09/elsa/model/HTSAT-fullset-tiny-map=0.467.ckpt")
# print("checkpoint keys:", ckpt.keys())
# # たとえば 'state_dict' があるかどうか
# if "state_dict" in ckpt:
#     print("state_dict の中身キー例:", list(ckpt["state_dict"].keys()))


# processor = ClapProcessor.from_pretrained(ckpt)

# # from model.audio_encoder_lightning import HTSATHybridLightningModule

# # # 1) デバイス選択
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # 2) LightningModule の load_from_checkpoint を使って読み込み
# # lit_model = HTSATHybridLightningModule.load_from_checkpoint(
# #     checkpoint_path="HTSAT-fullset-tiny-map=0.467.ckpt",
# #     map_location=device
# # )



import torch
from transformers import ClapAudioModel, ClapConfig

# 1) チェックポイントから state_dict を取り出す
ckpt = "/home/takamichi-lab-pc09/elsa/model/HTSAT-fullset-tiny-map=0.467.ckpt"

# 2) Config を取得（HuggingFace から）
model= ClapAudioModel.from_pretrained("laion/clap-htsat-fused")

audio_ckpt = torch.load(ckpt, map_location='cpu', weights_only=True)
audio_ckpt = audio_ckpt['state_dict']
keys = list(audio_ckpt.keys())
for key in keys:
    if key.startswith('sed_model'):
        v = audio_ckpt.pop(key)
        audio_ckpt['audio_encoder.' + key[10:]] = v

model.load_state_dict(audio_ckpt, strict=False)
param_names = [n for n, p in model.named_parameters()]
for n in param_names:
    print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
# print("✅ strict=False でロードが通りました。")

# 4) state_dict をロード
# try:
#     model.load_state_dict(sd, strict=True)
#     print("✅ ローカル ckpt の重みが正常にロードされました。")
# except RuntimeError as e:
#    # print("⚠ strict=True でロードできませんでした。エラー:", e)
#    # print("  → strict=False で再度試行します。")
#     model.load_state_dict(sd, strict=False)
#     print("✅ strict=False でロードが通りました。")

