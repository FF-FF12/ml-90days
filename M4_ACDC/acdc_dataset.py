import glob, os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
root = r"D:\M4_ACDC\training" 
class ACDCDataset(Dataset):
    def __init__(self, root, train=True):
        self.train_flag =train
        self.samples = []   # 每个元素 = (图片路径, 标签路径, 切片索引)
        patients = sorted(glob.glob(os.path.join(root, "patient*")))
        # 80/20 划分：前 80 训练，后 20 验证（ACDC 本身就按编号排好了）
        if train:
            patients = patients[:80]
        else:
            patients = patients[80:]

        for p in patients:
            for gt_path in glob.glob(os.path.join(p, "*_gt.nii.gz")):
                img_path = gt_path.replace("_gt", "")
                n_slices = nib.load(gt_path).shape[2]
                for s in range(n_slices):
                    self.samples.append((img_path, gt_path, s))
        print(f"共 {len(self.samples)} 张 2D 切片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, s = self.samples[idx]
        img = nib.load(img_path).get_fdata()[:, :, s]   # 只取第 s 层
        mask = nib.load(gt_path).get_fdata()[:, :, s]

        # 三标签 → 二分类：只留 LV（标签 3），其余全归背景
        mask = (mask == 3).astype(np.float32)

        # 归一化到 0-1
        img = img.astype(np.float32)
        img = img / (img.max() + 1e-8)

        # 训练集：随机水平翻转（img 和 mask 同步翻！）
        if self.train_flag and np.random.rand() > 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()

        # 加通道维 (H,W) → (1,H,W)，pad 到 8 的倍数
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        img = F.pad(img, (0, 256 - img.shape[2], 0, 256 - img.shape[1]))   # pad 到 256×256
        mask = F.pad(mask, (0, 256 - mask.shape[2], 0, 256 - mask.shape[1]))
        # ---------- 数据增强（只在训练集生效）----------
        if self.train_flag:
            # 随机水平翻转（心脏左右结构有语义，翻转是常见的几何增强）
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # 小角度旋转 ±10°，医学图像方向有语义，角度不能大
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)
            # 亮度微扰（只改 img，不改 mask！）
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                img = torch.clamp(img * factor, 0, 1)
        # ------------------------------------------------
        return img, mask

if __name__ == "__main__":
    ds = ACDCDataset(r"D:\M4_ACDC\training", train=True)
    img, mask = ds[0]
    print(img.shape, mask.shape, mask.sum())   # 应输出 [1,256,256] 和一个>0的数（该层LV像素数）