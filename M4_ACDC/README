# M4: ACDC 心脏磁共振左心室分割（PyTorch 入门实战）

用 2D U-Net（TinyUnet）对 ACDC 心脏 MRI 数据集进行左心室（LV）自动分割，
从零打通「数据探索 → Dataset 构建 → 可视化 → 训练 → 验证」完整深度学习管线。

> **最终结果：best val Dice = 0.8275**（100 epochs，RTX 3060 Laptop，约 1h38m）

## 项目背景

深度学习自学路线第 10 周（M4）动手任务：在 M3 驾驶场景分割的基础上，
迁移到医学影像场景，独立完成一个医学图像分割任务的全流程。

## 数据集

[ACDC（Automated Cardiac Diagnosis Challenge）](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

- 100 名患者 cine-MRI，`patientXXX_frameYY.nii.gz` + 对应 `_gt` 标注
- 标签：0 背景 / 1 右心室 RV / 2 心肌 MYO / 3 左心室 LV
- 本项目做**二分类**：`mask = (gt == 3)`，只分割左心室
- 每个 3D volume 按层切成 2D 切片，共 **1442 张**（前 80 病人训练 / 后 20 病人验证）
- 各病人尺寸不一，统一 `F.pad` 到 256×256

## 模型

**TinyUnet**（轻量版 U-Net，`in_ch=1` 灰度输入），损失函数为
**BCE + Dice Loss** 混合：

- BCE 负责逐像素分类
- Dice 直接优化重叠度，对抗前景占比小（类别不均衡）的问题

## 目录结构

```
M4_ACDC/
├── explore_acdc.py     # 数据探索：病人数量、尺寸、标签分布
├── acdc_dataset.py     # ACDCDataset：切片读取、二值化、归一化、pad
├── visualize.py        # 8 张 image/mask 叠加可视化
├── train_acdc.py       # 训练主脚本：训练循环 + 验证 + 保存最优权重
├── best_acdc.pth       # 训练好的权重（未上传，见 .gitignore）
├── train_acdc_log.txt  # 训练日志
└── images/             # 可视化结果
```

## 环境与运行

```bash
# Python 3.12 + PyTorch (CUDA 11.8)
pip install torch torchvision nibabel matplotlib --index-url https://download.pytorch.org/whl/cu118

# 1. 数据探索
python explore_acdc.py

# 2. 可视化标注
python visualize.py

# 3. 训练（-u 关闭缓冲，实时打印每个 epoch）
python -u train_acdc.py --epochs 100 2>&1 | Tee-Object train_acdc_log.txt
```

## 结果

| 指标 | 数值 |
|---|---|
| best val Dice | **0.8275** |
| 最终 val Dice | 0.8176 |
| 目标 | 0.70 ✅ 超额完成 |
| 训练时长 | 1h38m / 100 epochs（RTX 3060 Laptop 6GB） |
| 显存占用 | ~3.7 GB |

## 踩坑记录

1. **Dataset 忘存 flag**：`__init__` 里必须 `self.train_flag = train`，否则增强逻辑报 `AttributeError`
2. **翻转增强要同步**：img 和 mask 必须同时翻转，且用 `.copy()` 防止数组视图问题
3. **尺寸不一致**：不同病人 slice 尺寸不同，需 `F.pad` 到统一 256×256
4. **管道输出缓冲**：`| Tee-Object` 会触发 Python 输出缓冲，加 `-u` 才能实时看到每个 epoch 的日志
5. **mask 全 0 不是 bug**：不含 LV 的切片（如心脏边缘层）mask 本来就为空，属正常

## 学到什么

- 医学影像（NIfTI 格式）的读取与 3D→2D 切片处理
- 针对**类别不均衡**设计损失函数（BCE + Dice 互补）
- Dataset / DataLoader 规范写法与数据增强的"同变"原则
- 训练脚本的工程化：日志、最优权重保存、训练/验证分离
