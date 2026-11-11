from acdc_dataset import ACDCDataset
import matplotlib.pyplot as plt

ds = ACDCDataset(r"D:\M4_ACDC\training", train=True)
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i in range(4):
    img, mask = ds[100]          # 同一个样本取 4 次，看增强是否随机
    axes[0, i].imshow(img[0], cmap="gray")
    axes[1, i].imshow(mask[0], cmap="gray")
    axes[0, i].set_title(f"try {i}")
for ax in axes.flat: ax.axis("off")
plt.tight_layout(); plt.savefig("check_aug.png", dpi=120)
print("ok")