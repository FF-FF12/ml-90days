# 第 11 周 · 对比实验台账（M4 调参实验）

> 目标：在 M4 基准（Dice 0.8275）上做**一次只改一个变量**的消融实验，
> 回答「哪个因素贡献了多少」。这是第 11 周的验收材料，也是第 12 周技术报告
> 「结果与分析」章节的原始素材。

## 实验环境（全部一致，保证可比）

| 项 | 值 |
|---|---|
| 数据集 | ACDC，前 80 病人训练 / 后 20 病人验证（460 张验证切片） |
| 模型 | TinyUnet（in_ch=1，结构同 M4 基准，未改） |
| 设备 | RTX 3060 Laptop 6GB |
| 基础损失 | **纯 Dice**（`dice_loss`，单一项，无 BCE 项）——基准即纯 Dice，故「拆损失」只需补跑纯 BCE 一组 |
| 评估指标 | val Dice（跨组只比 Dice，不比 loss，因各损失量级不同） |

## 总览表

| # | 实验 | 改动（唯一变量） | best val Dice | 对比基准 | 结论一句话 |
|---|---|---|---|---|---|
| 0 | 基准 | —（纯 Dice + lr 恒定 1e-3） | **0.8275** | — | 100ep，1h38m |
| 1 | +Cosine 调度 | 加 `CosineAnnealingLR(T_max=100, eta_min=1e-5)` | **0.8460** (ep74) | +0.0185 | 涨点集中在 ep61-74，后期波动明显变小 |
| 2 | +数据增强 | 随机水平翻转 + ±10° 旋转 + 亮度微扰（在 v2 基础上叠加） | **0.8594** (ep60) | +0.0319 | 1h47m，泛化进一步提升 |
| 3 | epochs 100→200 | 训练时长翻倍（T_max 同步 200） | ~0.8550 (ep158-173)【待精确】 | +0.0275 | 时间翻倍仅 +1 点 → 收敛接近饱和 |
| 4 | 纯 BCE | 损失换成 `BCEWithLogitsLoss`（在 v2 基础上改） | **0.8680** | +0.0405 | ⭐反直觉：纯 BCE 未退化，反而成为最优 |

## 各组详细记录

### 实验 0 · 基准
- 脚本：`train_acdc.py`；日志：`train_acdc_log.txt`；权重：`save_weights/best_acdc.pth`
- 配置：100 epochs，**纯 Dice loss**，lr 恒定 1e-3
- 结果：best val Dice = **0.8275**，最终 0.8176，1h38m
- 备注：M4 验收目标 0.70 ✅

### 实验 1 · +Cosine 学习率调度
- 脚本：`train_v2.py`；日志：`train_v2_log.txt`；权重：`save_weights/best_cosine.pth`
- 改动：优化器后加 `scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)`，
  **每个 epoch 末** `scheduler.step()`（不能放 batch 循环内）
- 结果：best = **0.8460**（ep74），1h39m，lr 0.001 → 0.000010
- 观察：涨点集中在 ep61-74（lr 2e-4→1e-4 区间）；ep74 后 25ep 无刷新，后期稳定性优于基准

### 实验 2 · +数据增强
- 脚本：`train_v3_aug.py`；日志：`train_v3_aug_log.txt`；权重：`save_weights/best_aug.pth`
- 改动：训练分支加「随机水平翻转（0.5）+ 小角度旋转 ±10°（0.5）+ 亮度微扰（0.3）」
  - img/mask 共用同一随机参数（翻转/旋转），亮度只改 img
  - 验证集分支**不加**增强
- 结果：best = **0.8594**（ep60），1h47m；ep61-99 稳定在 0.83-0.85 之间
- 观察：在 cosine 基础上再 +0.013，说明数据增强对 ACDC（仅 80 训练病人）仍有正收益

### 实验 3 · epochs 100 → 200
- 脚本：`train_v4_200ep.py`；日志：`train_v4_200ep_log.txt`；权重：`save_weights/best_200ep.pth`
- 改动：`--epochs 200`（`T_max=args.epochs` 自动同步，勿手写 100）
- 结果：best ≈ 0.8550（ep158-173 刷新），3h18m，ep199 = 0.8491
- 结论：时间翻倍仅 +~1 点 → 边际收益递减，瓶颈转向模型容量/数据量，不建议再加时长
- 查 best：`Select-String "保存最优" train_v4_200ep_log.txt | Select-Object -Last 1`

### 实验 4 · 纯 BCE（本次最有价值的发现）
- 脚本：`train_v5_bce.py`；日志：`train_v5_bce_log.txt`；权重：`save_weights/best_bce.pth`
- 改动：`loss = nn.BCEWithLogitsLoss()(out, masks)`（必须 `WithLogits`，不是 `BCELoss`）
- 结果：best = **0.8680**，1h52m，是全部实验的最高分
- ⭐ **反直觉结论**：常规预期是「纯 BCE 在类别不均衡时会退化到 0.6-0.75」，实际并未发生。
  归因：
  1. ACDC 的 LV 前景占比虽小，但未到 BCE 的退化阈值（心脏居中、LV 面积约占切片 5-15%）；
  2. BCE（sigmoid+log）的梯度比 Dice 更平滑——Dice 在训练初期前景与预测几乎不重叠时梯度噪声大，
     而 BCE 始终提供稳定梯度，配合 cosine 调度反而收敛得更好。
- ⚠️ 方法学说明：v5 是从 v2（cosine 版）复制的，故「纯 BCE vs 基准」里混入了 cosine 因素。
  **干净的对比是 v5 vs v2（同为 cosine，只差 loss）：0.8680 vs 0.8460，纯 BCE +0.022。**

### （已跳过）纯 Dice
- 基准（train_acdc.py）的 loss 本就只有 `dice_loss` 一项，**基准即纯 Dice 组**，
  无需再单独跑 `train_v6_dice.py`。纯 Dice 的代表成绩 = v2 的 0.8460（加 cosine 后）。

## 结论与后续

- **最优配置 = 纯 BCE + cosine（0.8680）**，其次 = 纯 Dice + cosine + 增强（0.8594）
- 最有价值的发现不是涨分，而是**推翻了「纯 BCE 必退化」的预设**：损失函数的选择取决于
  任务的前景占比与收敛稳定性，不能教条套用「医学分割必须用 Dice」
- 瓶颈判断：模型容量或数据量（200ep 收益递减已佐证）
- 第 12 周若继续：可选「模型加深（base 16→32）」，注意结构改动要与权重加载严格一致
- 第 12 周技术报告的「结果与分析」= 上面总览表 + 每组一句话归因，基本可直接搬运
