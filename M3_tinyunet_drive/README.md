# M3 · 手写 TinyUnet 在 DRIVE 视网膜血管分割

从零实现轻量 U-Net（非调用现成库），接入标准分割训练流水线（Dice Loss + 混淆矩阵评估），在 DRIVE 数据集上训练 100 epochs。

## 结果

| 指标 | 数值 |
|---|---|
| **Dice coefficient** | **0.794** |
| 血管 IoU | 66.1 |
| 背景 IoU | 94.4 |
| mean IoU | 80.3 |
| global correct | 95.0 |
| 训练时长 | 2h11m（100 epochs，单卡） |

网络为自行设计的三层轻量结构（64/128/256 通道，转置卷积上采样），与仓库原版标准 U-Net（Dice 0.79）持平。

## 网络结构

```
输入 → DoubleConv(64) ──────────────────┐
     → pool → DoubleConv(128) ─────┐    │ 跳跃连接 cat(dim=1)
     → pool → DoubleConv(256)      │    │
       → ConvTranspose2d(2, stride=2) + cat → DoubleConv(128)
       → ConvTranspose2d(2, stride=2) + cat → DoubleConv(64) → 1×1 Conv
```

关键实现细节：forward 内先记录原始尺寸、pad 至 8 的倍数（3 次池化要求），输出再裁回原尺寸，支持任意尺寸输入推理。

## 调试记录（踩坑清单）

1. `.ipynb` 无法被 `import` → 模型类迁入 `.py` 模块
2. `create_model` 参数签名不匹配（`in_channels`/`base_c`）→ 与类定义逐字对齐
3. 仓库 loss 期望 `OrderedDict` 输出 → forward 适配接口约定
4. 文件顶层残留测试代码在 import 时执行 → 清理，自测代码用 `__main__` 保护
5. 同名文件多处重复导致改错文件 → 用 `模块.__file__` 验证真实加载路径
6. DRIVE 原图 565×584 不被 8 整除 → forward 内 pad-crop
7. 训练用终端而非 notebook：长任务不中断、可复现

## 复现

```
python train.py --epochs 100
```

- 环境：PyTorch + CUDA
- 数据：[DRIVE](https://drive.grand-challenge.org/)（视网膜血管分割公开基准）
- 训练日志见 `训练日志.txt`（注：终端缓冲区限制，自 epoch 51 起完整记录）
