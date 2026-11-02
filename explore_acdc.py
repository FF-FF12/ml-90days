import nibabel as nib
import numpy as np
import glob, os

root = r"D:\M4_ACDC\training"   # 改成你的实际路径
patients = sorted(glob.glob(os.path.join(root, "patient*")))
print("病人数:", len(patients))

# 看第一个病人
p = patients[0]
for f in sorted(glob.glob(os.path.join(p, "*.nii.gz"))):
    img = nib.load(f)
    arr = img.get_fdata()
    print(os.path.basename(f), "形状:", arr.shape, "取值范围:", arr.min(), arr.max())

# 看标签有哪几类
gt = nib.load(glob.glob(os.path.join(p, "*_gt.nii.gz"))[0]).get_fdata()
print("标签类别:", np.unique(gt))   # 应该输出 [0. 1. 2. 3.]
