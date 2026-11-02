import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from acdc_dataset import ACDCDataset
ds = ACDCDataset( r"D:\M4_ACDC\training", train=True)
loader = DataLoader(ds, batch_size=1, shuffle=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (img, mask) in enumerate(loader):
    if i >= 8: break
    ax = axes[i // 4][i % 4]
    ax.imshow(img[0, 0], cmap="gray")
    ax.imshow(mask[0, 0], cmap="Reds", alpha=0.4)  # 半透明红色盖上去
    ax.axis("off")
plt.savefig("acdc_vis.png", dpi=150)