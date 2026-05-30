# PaSS-LION：图像作为 LION-Mamba SSM Selectivity Surface 的跨模态融合设计

- **日期**：2026-05-30
- **工作名**（临时）：PaSS-LION (Pixel-as-Selective-Signal LION)；备选 SSF-Mamba / CamSel-Mamba
- **基线**：LION-Mamba(S2) ── 仓库内已跑赢的 `bev_h / bev_h_t` 序列化方案（实现位于 `pcdet/models/backbones_3d/tip_lion/serialization.py`）
- **数据集**：KITTI 3D Object Detection（Car / Pedestrian / Cyclist）
- **定位**：顶会方法论文（CVPR/ICCV/NeurIPS 级），mechanism-driven，仅 KITTI + 充分 ablation 击穿
- **监督预算**：允许 ImageNet 预训练 image encoder（ResNet/Swin），不动任何额外标签
- **与 TIP-LION 关系**：TIP-LION (TIS/HGD/IGVG) 整体不再推进；TIS 的 `bev_h/bev_h_t` 序列化已被 LION-Mamba(S2) baseline 内含，HGD/IGVG 放弃。本文不做 "PaSS × TIP-LION 叠加" ablation

---

## TL;DR

**单点贡献**：把图像信息**仅**注入 LION-Mamba 主干 SSM 的两条 input-dependent selectivity 通道 —— **Δ**（"何时关注 / effective horizon"）与 **B**（"该 token 把什么注入隐状态"），**不动 A**（状态转移谱）与 **C**（输出读出）。这与所有现有跨模态融合范式正交（token-level cross-attn、BEV grid fusion、proposal-level painting、virtual point），并且在 Transformer 主干上无对应物 —— 是 Mamba 独有的"selectivity surface"。

**契约**：去掉 PaSS = bit-for-bit 等价于 LION-Mamba(S2) baseline。

---

## 目录

- [§1 整体架构与命名](#1-整体架构与命名)
- [§2 模块细节（PixelAlign / ΔBMod）](#2-模块细节pixelalign--δbmod)
- [§3 数据流 / Forward Pass / 与 LION-Mamba(S2) 的耦合点](#3-数据流--forward-pass--与-lion-mambas2-的耦合点)
- [§4 损失 / 训练策略 / 数值稳定性](#4-损失--训练策略--数值稳定性)
- [§5 实验设计 / Ablation / 论文卖点](#5-实验设计--ablation--论文卖点)
- [附录 A：实现落点](#附录-a实现落点)
- [附录 B：超参快查表](#附录-b超参快查表)
- [附录 C：设计决策追溯](#附录-c设计决策追溯)

---

## §1 整体架构与命名

### 1.1 单点贡献声明

把图像信息**仅注入 Mamba 的两条 input-dependent selectivity 通道（Δ、B）**，而**不进入 token 序列、不进入 BEV 后处理、不接 cross-attention**。这是与现有所有 image-LiDAR fusion（BEVFusion / TransFusion / VirtualSparse / DeepInteraction / SFD）正交的融合通道。

### 1.2 架构骨架

```
RGB image ──► ImageBranch (ResNet-FPN, ImageNet pretrained, FROZEN early epochs)
                │
                ▼  multi-scale image feature maps  {F_1, F_2, F_3}
                │
points ──► VFE ──► LION-Mamba(S2) backbone  ── BEV map ──► SECOND head
                │  │  │  │
                │  │  │  └─ Layer-4 Mamba block ◄── PixelAlign+ΔBMod (scale-matched)
                │  │  └──── Layer-3 Mamba block ◄── PixelAlign+ΔBMod
                │  └─────── Layer-2 Mamba block ◄── PixelAlign+ΔBMod
                └────────── Layer-1 Mamba block ◄── PixelAlign+ΔBMod
```

### 1.3 新增模块

| 模块 | 职责 | 输入 → 输出 |
|---|---|---|
| `ImageBranch` | 图像特征金字塔 | `[B,3,H,W]` → `{F_l ∈ [B,C_l,H_l,W_l]}` |
| `PixelAlign` | voxel 中心通过 KITTI calib 投影到图像平面 → bilinear 采样 multi-scale image feature → 拼成 per-voxel image vector；FOV 外 voxel 标 mask=0 | voxel coords + `{F_l}` → `[N, C_img]` + FOV mask |
| `ΔBMod` | per-voxel image vector → SSM 的 Δ 乘性修正 + B 低秩残差 | `[N, C_img]` → `(Δ_corr, B_corr)`，FOV 外强制 identity |

### 1.4 四条核心设计承诺

1. **LION-Mamba(S2) 主干保持 bit-for-bit 不变**（包括 `bev_h/bev_h_t` 序列化、PatchMerging、扫描方向、diffusion）。Ablation "去掉 PaSS = 我们的 strong baseline" 成立。
2. **FOV 外 voxel 不受影响**（KITTI 前向视角天然只覆盖部分 FOV，必须显式处理，不然评测时这部分 voxel 会被未训练的 modulator 污染）。
3. **图像分支可加可减**：去掉整个 PaSS 子图 → 直接退化为 LION-Mamba(S2)；这是 reviewer 必查的可移除性。
4. **逐层注入**（4 个 LION layer 都注入）作为主方案；选层注入作为 ablation 维度。

---

## §2 模块细节（PixelAlign / ΔBMod）

### 2.1 ImageBranch

- 默认 ResNet50-FPN，ImageNet 预训练；前 5 epoch 冻结，之后解冻末两个 stage（避免早期 KITTI 小数据 + LiDAR 信号干扰把 image weights 拖坏）。
- 输出 3 个 scale 的特征图 `{F_1, F_2, F_3}`，channel 各 256（FPN 默认）。
- KITTI 原始 1242×375 → padding 到 1248×384 输入。

### 2.2 PixelAlign（无参数）

对 LION 第 *l* 层的每个 voxel（坐标 `p ∈ R^3`，LiDAR 系）：

```
p_cam = R0_rect · Tr_velo_to_cam · [p; 1]        # 标准 KITTI 标定
(u·z, v·z, z) = P2 · p_cam                        # 投影到像素
FOV_mask = (0 ≤ u < W) ∧ (0 ≤ v < H) ∧ (z > 0)
v_img_l_s = bilinear_sample(F_s, normalize(u,v))  for s ∈ {1,2,3}
v_img_l   = MLP_proj( concat_s(v_img_l_s) ) ∈ R^{C_img=64}
```

- 每个 LION layer 都重新跑 PixelAlign（voxel 在 PatchMerging 后位置变了）。
- FOV 外 voxel 的 `v_img` 设为 0，并由 mask 在下游门控。
- FPN 三个 scale 与 LION 四个 layer 不硬绑定，靠 `MLP_proj` 学习权重。

### 2.3 ΔBMod（核心，每 LION layer 一个）

**原 LION-Mamba 的 SSM 状态更新**（token *i* 处）：

```
x_i = exp(Δ_i · A) · x_{i-1} + (Δ_i · B_i) · u_i
y_i = C_i · x_i
```

**注入图像修正（只动 Δ 和 B）**：

```
g_Δ_i = sigmoid( W_Δ · v_img_i )                       # ∈ (0,1)^D，乘性门
Δ'_i  = Δ_i ⊙ ( FOV_i · g_Δ_i + (1 - FOV_i) · 1 )     # FOV 外 = identity

α_i   = W_lr · v_img_i                                  # ∈ R^r, r=4
ΔB_i  = U · diag(α_i) · V^T                             # ∈ R^{D×N}, U/V 共享
B'_i  = B_i + FOV_i · ΔB_i

A、C 完全不变 ── 这是设计的纯净性所在
```

### 2.4 为什么刻意只动 Δ、B、不动 A、C（论文核心理论小节）

- **A** 是 HiPPO 衍生的状态转移谱，data-independent 是 Mamba / Mamba-2 的稳定性来源；让图像调 A 会动到 state-space 的几何，Mamba-2 已经验证 data-dependent A 会引入训练不稳。
- **C** 是输出读出矩阵；若让图像调 C，等价于让图像直接重定向 voxel 输出特征，本质退化为传统跨模态 feature blending，跟 "selectivity" 无关。
- **Δ**（"何时关注 / effective receptive field"）和 **B**（"该 token 把什么注入隐状态"）才是 input-dependent selectivity 的本体。只调这两个，可以严格说："图像只通过 SSM 的 selectivity 表面影响 LiDAR，而不进入其几何"。

### 2.5 参数预算

| 部件 | shape | params |
|---|---|---|
| `MLP_proj` (FPN 3-scale → 64) | (256·3) × 64 | ~49 K |
| `W_Δ` (64 → D=64) | 64 × 64 | ~4 K |
| `W_lr` (64 → r=4) | 64 × 4 | ~0.3 K |
| `U, V` (D × r) ×2 | shared | ~0.5 K |
| **每层** | | **~54 K** |
| **4 层合计** | | **~0.22 M** |
| `ImageBranch` (ResNet50-FPN) | | ~25 M（pretrained） |

主干新增 < 0.3M 参数；ImageBranch 是 pretrained 大头但与 fusion 设计无关，可在 ablation 里换 ResNet18 反证 "非容量赢"。

---

## §3 数据流 / Forward Pass / 与 LION-Mamba(S2) 的耦合点

### 3.1 整体一次 forward（伪代码）

```
1. ImageBranch 跑 1 次（per scene）
   image[B,3,384,1248] → ResNet50-FPN → {F_1,F_2,F_3}, channels 256

2. VFE → 得到初始 voxel features 与 coords

3. for l in {1,2,3,4}:                          # 4 个 LION layer
     # ── 在 LION layer 入口（在双方向序列化之前）一次性算好 v_img ──
     v_img_l, FOV_mask_l = PixelAlign(voxel_coords_l, {F_s})  # [N_l, 64], [N_l]

     # ── 走原 LION layer，仅在 Mamba 算子内做 Δ/B 修正 ──
     for direction in ['bev_h', 'bev_h_t']:    # 两方向序列化共享 v_img_l
         partition into groups of size G_l
         for each group:
             tokens = serialize(group, direction)
             Δ, B, A, C = SSM_param_proj(tokens)
             Δ', B' = ΔBMod_l(Δ, B, v_img[tokens], FOV[tokens])  # ← 注入点
             out = SelectiveScan(tokens, Δ', A, B', C)            # A,C 不动
         unmerge two-direction outputs

     PatchMerging3D → voxel_coords_{l+1}  (空间下采样, voxel 数减少)
```

### 3.2 五个关键耦合细节

1. **PixelAlign 每层只跑 1 次**（不是每方向各 1 次）。`bev_h` 和 `bev_h_t` 两方向共享同一份 `v_img`，因为图像信息与序列扫描方向无关。这点能省一半 image-sample 计算，也让方向消融更干净。

2. **注入点在 SelectiveScan 前**，不动 `SSM_param_proj` 自身（保留 Mamba selectivity 的输入路径不变），只在 Δ、B 出炉之后做 element-wise 门控 / 低秩残差。这样回退到 baseline 是切一个开关的事。

3. **每层 voxel coords 都变**（PatchMerging 后 spatial resolution 下采样），所以每层都重新 PixelAlign。FPN 三个 scale 跟 LION 四个 layer 的对应：layer-1 用 F_1（高分辨率）为主，layer-4 用 F_3（低分辨率）为主，靠 `MLP_proj` 学权重而不是硬绑定（保持简单）。

4. **训练 schedule**：
   - epoch 1-5：ImageBranch 冻结
   - epoch 6+：image backbone 末两个 stage 解冻，其余冻结；ΔBMod / MLP_proj 全程 trainable
   - 与现有 EMA 训练（commits `07bb5e9` / `395edf8`）兼容，EMA 同时跟踪 image 分支

5. **FOV-边界处理**：FOV mask 在 ΔBMod 内部消化 ——
   - eval 时 FOV 外的远距离 / 后向 voxel 不被未训练过的修正污染
   - ablation "如果把整个 PaSS 切掉" = 把 FOV mask 全置 0 = 严格等于 LION-Mamba(S2) baseline，这条等价性是 reviewer 必查的

### 3.3 张量 shape 速查（layer-1，KITTI 默认 voxel ~16000）

| 张量 | shape | 来自 |
|---|---|---|
| `voxel_feat_l1` | [16000, 64] | VFE/上层输出 |
| `voxel_coords_l1` | [16000, 4] (b, z, y, x) | LION 维护 |
| `v_img_l1` | [16000, 64] | PixelAlign |
| `FOV_mask_l1` | [16000] | PixelAlign |
| `Δ` | [N_group, G_l, D=64] | Mamba param proj |
| `Δ'` | [N_group, G_l, D] | ΔBMod 后 |
| `B` | [N_group, G_l, D, N=16] | Mamba param proj |
| `B'` | [N_group, G_l, D, N] | ΔBMod 后 |

---

## §4 损失 / 训练策略 / 数值稳定性

### 4.1 损失：完全沿用 baseline

- 主损失 = LION-Mamba(S2) + SECOND head 现有的 cls + reg + dir loss，**不加任何新 loss**。
- 不引入额外 2D 监督、不引入 image consistency loss、不引入 depth pseudo loss。
- 理由：与 "ImageNet 预训练 + 不动额外标签" 的预算承诺一致；同时迫使所有增益归因到 ΔBMod 自身。

### 4.2 Identity-at-init（关键稳定性设计）

让 PaSS 在 epoch 0 严格等价于 baseline：

| 参数 | 初始化 | 效果 |
|---|---|---|
| `W_Δ` 最后一层 bias | `+大值（≈3）` | sigmoid 输出 ≈ 1，`Δ' ≈ Δ`（identity 乘子） |
| `W_Δ` 最后一层 weight | 标准小高斯 | 初期对 Δ 几乎无修改 |
| `U, V`（低秩 B 残差） | `0` | `ΔB ≡ 0`，`B' = B`（identity 加法） |
| `MLP_proj` 末层 weight | 小尺度 0.01 init | 初期 `v_img` 接近常量，避免冷启动震荡 |

**Sanity check（必做）**：装上 PaSS 后跑 1 个 forward → 输出 logits 应与 baseline bit-equal（误差 < 1e-6）。如果不 bit-equal，说明 init 写错或注入点错。这条进 CI / smoke test。

### 4.3 Modulation warmup（前 2 epoch）

- 引入标量 `ρ(epoch) ∈ [0,1]`，cosine ramp from 0 → 1 over 2 epochs。
- `Δ' = Δ ⊙ (FOV·(1 + ρ·(g_Δ-1)) + (1-FOV))`
- `B' = B + ρ · FOV · ΔB`
- 让 LiDAR 主干先稳，再慢慢让图像介入。EMA 也同步 warmup。

### 4.4 优化器 / 学习率

| 参数组 | LR 倍率 | 备注 |
|---|---|---|
| LION-Mamba(S2) 主干 | 1× | 基线 LR 不变 |
| SECOND head | 1× | 基线 LR 不变 |
| `ImageBranch` (ResNet50-FPN) | 0.1× | pretrained，避免被打散 |
| `MLP_proj / ΔBMod` (新模块) | 1× | 主 LR；wd 与主干一致 |

继续用现有 cosine scheduler（commit `21e0d7d`）+ EMA（commit `07bb5e9 / 395edf8`）。

### 4.5 数值稳定性细节

- `v_img` 进 ΔBMod 前过一层 LayerNorm，输出再过一层 LayerNorm（标准 Mamba 设计风格）。
- B 的低秩残差幅度用 `||ΔB||_F ≤ τ` 软约束（spectral clamp via `tanh`），τ 取与原 B 同量级。
- 监控 `Δ'` 的分布：若 sigmoid 长期饱和到 0 或 1，说明梯度死掉，应升高 `W_Δ` 的 LR 或调 bias。进 grad-aux 日志（commit `2541afc` 已支持）。
- 混合精度：`SelectiveScan` 输入 cast 回 fp32 做 `Δ'` 的 exp（与原 Mamba 实现一致），不因为加 ΔBMod 而改精度策略。

### 4.6 与现有训练入口的衔接

- `tools/train.py` 不动（commit `07bb5e9` 已经接入 EMA），只在 model build 时多读 `ImageBranch` 的 pretrained 路径。
- `tools/run_kitti_experiment.sh` 已支持参数扩展（commit `2de8e5d`），新增 `--enable_pass` 切 PaSS。

---

## §5 实验设计 / Ablation / 论文卖点

### 5.1 论文主表（对照 baseline）

KITTI val/test，Car / Pedestrian / Cyclist 三类 × easy / moderate / hard，AP3D 与 APBEV。

| 组 | 方法 | 用途 |
|---|---|---|
| LiDAR-only baseline | LION-Mamba 官方版 | 起点 |
| LiDAR-only **strong** baseline | **LION-Mamba(S2)** | 真正 baseline |
| Painting 类 | PointPainting、PointAugmenting | 经典 fusion 对照 |
| Proposal-level fusion | CAT-Det、Frustum ConvNet | 经典对照 |
| Voxel-level fusion | SFD、Focals-Conv-F、VirtualSparse | 强 fusion 对照 |
| Token-level fusion | TransFusion、CMT、DeepInteraction | 顶会 SOTA fusion |
| BEV-level fusion | BEVFusion(MIT/PKU) | 顶会 SOTA fusion |
| **Ours** | **PaSS-LION** | 主方法 |

核心 claim：在 LION-Mamba(S2) 这条强 LiDAR-only 之上再叠 PaSS，跨过所有传统 fusion 路线。

### 5.2 Ablation 主表

| Ablation | 候选组 | 目的 / claim |
|---|---|---|
| **A. 调制通道**（核心） | `Δ` only / `B` only / `ΔB` (ours) / `ΔBC` / `ΔABC` | 证明 Δ+B 是 sweet spot；A 加上不稳；C 加上无收益 |
| **B. PaSS 模块剥离** | full / no `ΔBMod` / no `ImageBranch` / no `FOV mask` | 证明每个部件必要；no-mask 应轻微变差（远处污染） |
| **C. 注入层位** | L1 only / L4 only / 全 4 层 | 证明多层注入>单层；多 scale selectivity 都吃图像 |
| **D. Image encoder 容量** | ResNet18 / **ResNet50** / Swin-T | 证明不是 "image encoder 容量赢"（反 reviewer 质疑） |
| **E. 低秩 rank r** | 1 / 2 / **4** / 8 / 16 | 证明 rank=4 已饱和；不是参数量赢 |
| **F. Warmup** | 0 / 2 epoch / 5 epoch | 证明 identity-at-init 必要 |

注：早先讨论中考虑过加一组 "PaSS × TIP-LION 正交性" ablation；因 TIP-LION(TIS/HGD/IGVG) 整体不再推进，且 TIS 的 `bev_h / bev_h_t` 序列化已被 LION-Mamba(S2) baseline 内含，该组 ablation 取消。

### 5.3 机制诊断指标（非 AP）

| 指标 | 期望结果 | 说明 |
|---|---|---|
| FOV 内 / FOV 外 voxel AP 分离 | 增益**只**出现在 FOV 内 | 若 FOV 外也涨，说明 mask 没生效 |
| 距离分段 AP（0-20 / 20-40 / 40-70m） | 远距离段增益更大 | 远距离稀疏，最吃图像补全 |
| 类别增益排序 | Ped > Cyc > Car（预期） | 小目标 / 细长目标更依赖图像纹理 |
| `Δ'` entropy 分布 | 中段（避免全饱和） | 监控 selectivity 是否被有效用上 |
| `||ΔB||_F / ||B||_F` | 0.1 – 0.5 量级 | 修正与原信号同量级、未喧宾夺主 |
| Latency / 参数量 | < +5% 主干 latency；+0.3M 主干参数 | 反 "fusion 太重" 质疑 |

### 5.4 可视化（论文必带）

- **图 a**：ΔBMod 关闭 vs 开启时，远处 / 遮挡处一个 Car/Ped 的 SSM 隐状态 `x_t` 沿序列 trajectory 对比 → 证明图像确实改变了 selectivity 传播。
- **图 b**：把每个 voxel 的 `sigmoid(g_Δ)` 投回 BEV → 在前景目标处显著抬升、在地面 / 远处空白处接近 1 → 证明 "图像告诉 Mamba 该多关注哪些 voxel"。
- **图 c**：Q-Q 图，"PaSS 增益 vs voxel 在 image 上对应 patch 的显著性"，验证因果链。

### 5.5 论文 contribution 句式（拟 abstract）

> **C1.** 我们提出 **PaSS** —— 第一个把图像信息**仅**注入 LiDAR Mamba 主干 SSM 的 input-dependent selectivity 通道（Δ、B）、而**完全不动**状态转移谱（A）与输出读出（C）的跨模态融合范式。这与 token-level（cross-attention）、BEV-level（grid fusion）、proposal-level（feature painting）所有现有路线正交，并且在 Transformer 主干上无对应物。
>
> **C2.** 我们给出 "为什么只调 Δ、B" 的理论与实证论证：A 的 data-dependence 已被 Mamba-2 验证为训练不稳来源；C 的调制等价退化为传统跨模态混合。
>
> **C3.** 在 LION-Mamba(S2) 强 LiDAR-only baseline 之上叠 PaSS，仅引入 0.3M 新主干参数（不计 ImageBranch），在 KITTI 三类全档位均涨；远距离与 Ped/Cyc 增益尤其显著。
>
> **C4.** 完整 ablation 揭示了调制通道（Δ vs B vs ΔBC vs ΔABC）、注入层位、低秩 rank 与 image encoder 容量在 SSM-fusion 中的 mechanism scaling 规律。

### 5.6 训练成本 & 时间估计

实测起点：LION-Mamba(S2) baseline ~8 h / 单 4080 / e40。

**单实验**（KITTI e40，单 4080）：

| 配置 | 估算 wall-time | 倍率 | 说明 |
|---|---|---|---|
| baseline（LION-Mamba S2, 无 PaSS） | ~8 h | 1.0× | 实测 |
| PaSS-LION (ResNet18) | ~10-11 h | ~1.3× | ablation D 轻量 image encoder |
| **PaSS-LION (ResNet50, 默认)** | **~12-14 h** | **~1.5-1.75×** | **主方案** |
| PaSS-LION (Swin-T) | ~14-16 h | ~1.8-2.0× | ablation D 重量 image encoder |

增量构成：ImageBranch (ResNet50-FPN) +50 GFLOPs/image ≈ baseline 主干 30-60% 算力；ΔBMod 自身 < 1%（参考工程经验：BEVFusion/SFD/VirtualSparse 加 image 分支通常 +40-60% wall-time）。前 5 epoch ImageBranch 冻结可节约该段 ~10-15% 时间。

**完整 ablation**（主表 1 + ablation A-F 合计约 20-25 个训练，平均按 ~12 h/run）：

| 方案 | wall-time |
|---|---|
| 单 4080 | ~10-14 天（240-360 h） |
| 2× 4080 DDP（理想 ~1.8× 加速） | ~6-8 天 |
| 4× 4080 DDP | ~3-4 天 |

不需要 nuScenes / Waymo 的额外训练。

### 5.7 失败兜底

- **若主表增益 < 0.5 AP**：把 paper 重写为 "selective state modulation as a fusion principle"（mechanism study + 完整 ablation），仍可投 BMVC / WACV / T-PAMI。
- **若 Ped/Cyc 不涨**：可能 KITTI 远处 Ped 投影到 image 的 patch 太小，image feature 噪声主导；ablation D（换 Swin-T 或 high-res FPN）可救。
- **若 ΔB 都没动（饱和到 identity）**：说明 modulation warmup 太长 / LR 太小，调超参；若仍无变化，说明 SSM selectivity 对图像不敏感 —— 这是负结果，paper 转向 mechanism 解释。

---

## 附录 A：实现落点

| 改动面 | 文件 / 目录 | 改动内容 |
|---|---|---|
| 新增子模块 | `pcdet/models/backbones_3d/pass_lion/` | 新建：`image_branch.py`、`pixel_align.py`、`delta_b_mod.py` |
| LION backbone 插入点 | `pcdet/models/backbones_3d/lion_backbone_one_stride.py` | 每个 LION layer 入口加 `PixelAlign`；Mamba 算子调用处接收 `(Δ_corr, B_corr)` |
| 序列化复用 | `pcdet/models/backbones_3d/tip_lion/serialization.py` | **不动**，沿用 `bev_h / bev_h_t` |
| Detector 路由 | `pcdet/models/detectors/second_net.py` | forward 入口把 `batch_dict['images']` 路由到 backbone |
| KITTI 数据管线 | `tools/cfgs/dataset_configs/kitti_dataset.yaml` | 新 PaSS config 加 `GET_ITEM_LIST: ['images', 'points', 'calib_matricies']` |
| 训练入口 | `tools/train.py` | 不动；model build 时多读 ImageBranch pretrained 路径 |
| 启动脚本 | `tools/run_kitti_experiment.sh` | 新增 `--enable_pass` 切 PaSS |
| 配置文件 | `tools/cfgs/kitti_models/PaSS-LION/` | 新建主 config + ablation configs |
| Smoke test | `tools/smoke_test_pass_lion_bit_equal.py` | identity-at-init bit-equal 检查 |

## 附录 B：超参快查表

| 项 | 默认值 |
|---|---|
| ImageBranch | ResNet50-FPN, ImageNet pretrained |
| Image 输入尺寸 | 1248 × 384（KITTI padded） |
| FPN scale 数 | 3 |
| FPN channel | 256 |
| `C_img` (per-voxel image vector dim) | 64 |
| Mamba `D` | 64（与 LION 一致） |
| Mamba `N` | 16（state dim） |
| Low-rank `r`（B 残差） | 4 |
| ImageBranch 冻结 epoch | 1 - 5 |
| Modulation warmup | 2 epoch, cosine 0 → 1 |
| ImageBranch LR 倍率 | 0.1× |
| `MLP_proj / ΔBMod` LR 倍率 | 1× |
| `W_Δ` 末层 bias init | +3.0 |
| `U, V` init | 0 |

## 附录 C：设计决策追溯

逐条记录 brainstorming 过程中的关键决策与替代项，便于 reviewer 反问 / 后续修订。

| # | 决策 | 替代项 | 选定理由 |
|---|---|---|---|
| 1 | 论文定位 = 顶会方法论文 | 二区 / workshop / 不限定 | 推进强度匹配 LION-Mamba(S2) 已具备的 baseline 强度 |
| 2 | Narrative = 图像调制 SSM Δ/A/B/C 参数 | 跨模态统一序列 / 几何补全 / BEV 后融合 | 状态参数调制是 Mamba 独有 selectivity，Transformer 无对应物，narrative 最锐利 |
| 3 | 数据集 = 仅 KITTI | +nuScenes / +Waymo / 三套 | 走 mechanism-driven 路线；用充分 ablation 替代多数据集泛化论证 |
| 4 | 与 TIP-LION 关系 = 不再叠加 | 独立 / 合并一文 / 取代 / 正交可叠加 | 用户最终澄清：TIP-LION (TIS/HGD/IGVG) 整体不再推进；TIS 的 `bev_h/bev_h_t` 序列化已被吸收进 LION-Mamba(S2) baseline，HGD/IGVG 放弃。本文不再做 "PaSS × TIP-LION 叠加" ablation |
| 5 | 监督预算 = ImageNet pretrained，不动额外标签 | +2D box / +SAM / 从零训 image | 与 BEVFusion/TransFusion 一致，公平可比 |
| 6 | 调制通道 = ΔB（B 方案） | Δ only / Δ+A+B+C 全调 / B 为主 + A/C 作 ablation | Δ+B 是 sweet spot，A 不稳、C 退化；A/C 作为 ablation 出现而非主方法 |
| 7 | LION-Mamba(S2) 定义 | LION 官方 / 其他自家 baseline | 用户内部已跑赢的 `bev_h / bev_h_t` 序列化方案；strong baseline |
| 8 | FOV 外 voxel 处理 = 强制 identity | 学习一个 "FOV-out modulator" / 完全丢弃 | 保证可移除性 / 切 PaSS = baseline 严格成立 |
| 9 | 注入层位 = 4 层全注入 | 只 L1 / 只 L4 / 选层 | 主方案最大化 fusion 表面；选层作为 ablation |
| 10 | Identity-at-init + warmup | 直接训练 | 让 epoch 0 forward 与 baseline bit-equal，保训练曲线开头重合 |
