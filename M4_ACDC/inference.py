"""推理可视化：用 save_weights/best_acdc.pth 对 ACDC 验证集切片做预测
生成 8 张「原图 + 真实标注(红) + 模型预测(绿)」对比图 inference_vis.png
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_acdc import TinyUnet
from acdc_dataset import ACDCDataset

ROOT = r"D:\M4_ACDC\training"
WEIGHT = r"D:\M4_ACDC\save_weights\best_acdc.pth"


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyUnet(in_ch=1).to(device)
    model.load_state_dict(torch.load(WEIGHT, map_location=device))
    model.eval()

    val_set = ACDCDataset(ROOT, train=False)
    print(f"验证集共 {len(val_set)} 张 2D 切片")

    candidates = [i for i in range(len(val_set)) if val_set[i][1].sum() > 500]
    pick = random.sample(candidates, 8)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    dices = []
    for ax, idx in zip(axes.flat, pick):
        img, mask = val_set[idx]
        out = model(img.unsqueeze(0).to(device))
        # 有的网络 forward 返回 OrderedDict（如带辅助输出的结构），真输出在 "out" 键里
        if isinstance(out, dict):
            out = out["out"]
        prob = torch.sigmoid(out)[0, 0].cpu()
        binary = (prob > 0.5).float()

        inter = (binary * mask).sum()
        dice = (2 * inter + 1) / (binary.sum() + mask.sum() + 1)
        dices.append(dice.item())

        ax.imshow(img[0], cmap="gray")
        ax.imshow(np.ma.masked_where(mask[0] == 0, mask[0]), cmap="autumn", alpha=0.4)
        ax.imshow(np.ma.masked_where(binary == 0, binary), cmap="cool", alpha=0.4)
        ax.set_title(f"idx={idx}  Dice={dice.item():.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(r"D:\M4_ACDC\inference_vis.png", dpi=150)
    print(f"已保存 D:\\M4_ACDC\\inference_vis.png，8 张切片平均 Dice = {np.mean(dices):.4f}")


if __name__ == "__main__":
    main()
