# patch_ckpt.py
import torch, os

SRC = "/home/takamichi-lab-pc09/elsa/model/HTSAT-fullset-tiny-map=0.467.ckpt"
assert os.path.exists(SRC), "アップロード済み ckpt が見つかりません"

ckpt = torch.load(SRC, map_location="cpu")
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# 消すべきキーは 2 パターンだけ
for k in ["text_branch.embeddings.position_ids",
          "module.text_branch.embeddings.position_ids"]:
    state.pop(k, None)

PATCH = SRC.replace(".ckpt", ".patched.ckpt")
torch.save({"state_dict": state}, PATCH)
print("patched ckpt ->", PATCH)
