# PaSS-LION：图像作为 LION-Mamba SSM Selectivity Surface 的跨模态融合设计

- **日期**：2026-05-30
- **工作名**（临时）：PaSS-LION (Pixel-as-Selective-Signal LION)；备选 SSF-Mamba / CamSel-Mamba
- **基线**：LION-Mamba(S2) ── 仓库内已跑赢的 `bev_h / bev_h_t` 序列化方案（实现位于 `pcdet/models/backbones_3d/lion_improve/serialization.py`）
- **数据集**：KITTI 3D Object Detection（Car / Pedestrian / Cyclist）
- **定位**：顶会方法论文（CVPR/ICCV/NeurIPS 级），mechanism-driven，仅 KITTI + 充分 ablation 击穿
- **监督预算**：允许 ImageNet 预训练 image encoder（ResNet/Swin），不动任何额外标签
- **与 TIP-LION 关系**：TIP-LION (TIS/HGD/IGVG) 整体不再推进；TIS 的 `bev_h/bev_h_t` 序列化已被 LION-Mamba(S2) baseline 内含，HGD/IGVG 放弃。本文不做 "PaSS × TIP-LION 叠加" ablation

---

## TL;DR

**单点贡献**：把图像信息**仅**注入 LION-Mamba 主干 SSM 的两条 input-dependent selectivity 通道 —— **Δ**（"何时关注 / effective horizon"）与 **B**（"该 token 把什么注入隐状态"），**不动 A**（状态转移谱）与 **C**（输出读出）。这与所有现有跨模态融合范式正交（token-level cross-attn、BEV grid fusion、proposal-level painting、virtual point），并且在 Transformer 主干上无对应物 —— 是 Mamba 独有的"selectivity surface"。

**契约**：PaSS 关闭且保留原 fused 主干 = bit-for-bit 等价于 LION-Mamba(S2) baseline；PaSSMamba split identity 路径与 fused baseline 只要求数值等价（`allclose < 1e-6`），并通过 split-only baseline 隔离算子路径差异。

---

## 目录

- [§1 整体架构与命名](#1-整体架构与命名)
- [§2 模块细节（PixelAlign / ΔBMod）](#2-模块细节pixelalign--δbmod)
- [§3 数据流 / Forward Pass / 与 LION-Mamba(S2) 的耦合点](#3-数据流--forward-pass--与-lion-mambas2-的耦合点)
- [§4 损失 / 训练策略 / 数值稳定性](#4-损失--训练策略--数值稳定性)
- [§5 实验设计 / Ablation / 论文卖点](#5-实验设计--ablation--论文卖点)
- [§6 外部 review 修订（v2-v2.3，2026-05-30 后增）](#6-外部-review-修订v2-v232026-05-30-后增)
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
                │
                ├─ linear_1 LIONBlock: 4× LIONLayer ◄── PixelAlign per LIONLayer + ΔBMod stage-1
                ├─ linear_2 LIONBlock: 4× LIONLayer ◄── PixelAlign per LIONLayer + ΔBMod stage-2
                ├─ linear_3 LIONBlock: 4× LIONLayer ◄── PixelAlign per LIONLayer + ΔBMod stage-3
                ├─ linear_4 LIONBlock: 4× LIONLayer ◄── PixelAlign per LIONLayer + ΔBMod stage-4
                └─ linear_out LIONLayer               ◄── PixelAlign + ΔBMod out-stage
```

### 1.3 新增模块

| 模块 | 职责 | 输入 → 输出 |
|---|---|---|
| `ImageBranch` | 图像特征金字塔 | `[B,3,H,W]` → `{F_l ∈ [B,C_l,H_l,W_l]}` |
| `PixelAlign` | voxel 中心通过 KITTI calib 投影到图像平面 → bilinear 采样 multi-scale image feature → 拼成 per-voxel image vector；FOV 外 voxel 标 mask=0 | voxel coords + `{F_l}` → `[N, C_img]` + FOV mask |
| `ΔBMod` | per-voxel image vector → SSM 的 Δ 乘性修正 + B 低秩残差 | `[N, C_img]` → `(Δ_corr, B_corr)`，FOV 外强制 identity |

### 1.4 四条核心设计承诺

1. **PaSS 关闭时原 fused LION-Mamba(S2) 主干保持 bit-for-bit 不变**（包括 `bev_h/bev_h_t` 序列化、PatchMerging、扫描方向、diffusion）。PaSSMamba split 路径另以 `allclose < 1e-6` 做 identity smoke test，并在实验中单列 split-only baseline。
2. **FOV 外 voxel 不受影响**（KITTI 前向视角天然只覆盖部分 FOV，必须显式处理，不然评测时这部分 voxel 会被未训练的 modulator 污染）。
3. **图像分支可加可减**：去掉整个 PaSS 子图 → 直接退化为 LION-Mamba(S2)；这是 reviewer 必查的可移除性。
4. **P2 逐 LIONLayer 注入作为主方案**：17 个 LIONLayer 入口各跑一次 PixelAlign；ΔBMod / MLP_proj 用 5 套 stage 共享权重（4 个 LIONBlock + `linear_out`）；选层注入作为 ablation 维度。

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

- 每个 LIONLayer 入口都重新跑 PixelAlign（P2 默认，17 次/forward；voxel 在 PatchMerging / PatchExpanding 后位置会变）。
- FOV 外 voxel 的 `v_img` 设为 0，并由 mask 在下游门控。
- FPN 三个 scale 与 LION stage 不硬绑定，靠 stage 共享的 `MLP_proj` 学习权重。

### 2.3 ΔBMod（核心，5 套 stage 共享权重）

**原 LION-Mamba 的 SSM 状态更新**（token *i* 处）：

```
x_i = exp(Δ_i · A) · x_{i-1} + (Δ_i · B_i) · u_i
y_i = C_i · x_i
```

**注入图像修正（只动 Δ 和 B）**：

> **【v2 修订，详见 §6.1 / §6.2】** 下面 sigmoid 形式的 Δ 门、以及 `U, V` 同时零初始化的 B 残差，均已被外部 review 指出问题（identity 不严格、梯度死锁）。**实际实现请按 §6.1 / §6.2 的修订式**。

```
g_Δ_i = sigmoid( W_Δ · v_img_i )                       # ∈ (0,1)^D，乘性门     ← 已废弃，见 §6.1
Δ'_i  = Δ_i ⊙ ( FOV_i · g_Δ_i + (1 - FOV_i) · 1 )     # FOV 外 = identity

α_i   = W_lr · v_img_i                                  # ∈ R^r, r=4
ΔB_i  = U · diag(α_i) · V^T                             # ∈ R^{D×N}, U/V 共享   ← init 已修订，见 §6.2
B'_i  = B_i + FOV_i · ΔB_i

A、C 完全不变 ── 这是设计的纯净性所在
```

### 2.4 为什么刻意只动 Δ、B、不动 A、C（论文核心理论小节）

- **A** 是 HiPPO 衍生的状态转移谱，data-independent 是 Mamba / Mamba-2 的稳定性来源；让图像调 A 会动到 state-space 的几何，Mamba-2 已经验证 data-dependent A 会引入训练不稳。
- **C** 是输出读出矩阵；若让图像调 C，等价于让图像直接重定向 voxel 输出特征，本质退化为传统跨模态 feature blending，跟 "selectivity" 无关。
- **Δ**（"何时关注 / effective receptive field"）和 **B**（"该 token 把什么注入隐状态"）才是 input-dependent selectivity 的本体。只调这两个，可以严格说："图像只通过 SSM 的 selectivity 表面影响 LiDAR，而不进入其几何"。

### 2.5 参数预算

> **【v2.3 修订】** 此表采用 P2 默认颗粒度：PixelAlign 每个 LIONLayer 调一次（17 次/forward，无参数），`MLP_proj / ΔBMod` 参数按 5 套 stage 共享权重计算。D 维度为 `d_inner=2·d_model=128`；`V` 形状为 `N×r=16×r`。

| 部件 | shape | params |
|---|---|---|
| `MLP_proj` (FPN 3-scale → 64) | (256·3) × 64 | ~49 K |
| `W_Δ` (64 → d_inner=128) | 64 × 128 | ~8 K |
| `W_lr` (64 → r=4) | 64 × 4 | ~0.3 K |
| `U` (d_inner × r) | 128 × 4 | ~0.5 K |
| `V` (N × r) | 16 × 4 | ~0.06 K |
| **每套 stage 共享权重** | | **~58 K** |
| **5 套合计** | | **~0.29 M** |
| `ImageBranch` (ResNet50-FPN) | | ~25 M（pretrained） |

主干新增约 0.29M 参数；ImageBranch 是 pretrained 大头但与 fusion 设计无关，可在 ablation 里换 ResNet18 反证 "非容量赢"。

---

## §3 数据流 / Forward Pass / 与 LION-Mamba(S2) 的耦合点

### 3.1 整体一次 forward（伪代码）

```
1. ImageBranch 跑 1 次（per scene）
   image[B,3,384,1248] → ResNet50-FPN → {F_1,F_2,F_3}, channels 256

2. VFE → 得到初始 voxel features 与 coords

3. for stage in [linear_1, linear_2, linear_3, linear_4, linear_out]:
     delta_b_mod = ΔBMod[stage]                 # 5 套权重；stage 内共享

     for lion_layer in stage.LIONLayers:        # 4×(2 enc + 2 dec) + linear_out = 17
         # ── 在每个 LIONLayer 入口、双方向序列化之前算好 v_img ──
         v_img_l, FOV_mask_l = PixelAlign(voxel_coords_l, {F_s})  # [N_l, 64], [N_l]

         # ── 走原 LIONLayer，仅在 Mamba 算子内做 Δ/B 修正 ──
         for direction in lion_layer.directions:  # 同一 LIONLayer 的两方向共享 v_img_l
             partition into groups of size G_l
             for each group:
                 tokens = serialize(group, direction)
                 Δ, B, A, C = SSM_param_proj(tokens)
                 Δ', B' = delta_b_mod(Δ, B, v_img_l[tokens], FOV_mask_l[tokens])
                 out = SelectiveScan(tokens, Δ', A, B', C)          # A,C 不动
             unmerge two-direction outputs

         PatchMerging3D / PatchExpanding3D 可能更新 voxel_coords_l
```

### 3.2 五个关键耦合细节

1. **PixelAlign 每个 LIONLayer 只跑 1 次**（不是每方向各 1 次；P2 默认共 17 次/forward）。同一 LIONLayer 内 `bev_h` 和 `bev_h_t` 两方向共享同一份 `v_img`，因为图像信息与序列扫描方向无关。这点能省一半 image-sample 计算，也让方向消融更干净。

2. **注入点在 SelectiveScan 前**，不动 `SSM_param_proj` 自身（保留 Mamba selectivity 的输入路径不变），只在 Δ、B 出炉之后做 element-wise 门控 / 低秩残差。这样回退到 baseline 是切一个开关的事。

3. **跨 LIONLayer 的 voxel coords 可能变**（PatchMerging / PatchExpanding 会改空间分辨率），所以每个 LIONLayer 入口都重新 PixelAlign。FPN 三个 scale 与 5 套 stage 权重不硬绑定，靠 `MLP_proj` 学权重而不是写死对应关系（保持简单）。

4. **训练 schedule**：
   - epoch 1-5：ImageBranch 冻结
   - epoch 6+：image backbone 末两个 stage 解冻，其余冻结；ΔBMod / MLP_proj 全程 trainable
   - 与现有 EMA 训练（commits `07bb5e9` / `395edf8`）兼容，EMA 同时跟踪 image 分支

5. **FOV-边界处理**：FOV mask 在 ΔBMod 内部消化 ——
   - eval 时 FOV 外的远距离 / 后向 voxel 不被未训练过的修正污染
   - ablation "如果把整个 PaSS 切掉" = `pass_enabled=False`；原 fused 主干应 bit-for-bit 等于 LION-Mamba(S2) baseline，PaSSMamba split identity 路径应满足 `allclose < 1e-6`

### 3.3 张量 shape 速查（layer-1，KITTI 默认 voxel ~16000）

| 张量 | shape | 来自 |
|---|---|---|
| `voxel_feat_l1` | [16000, 64] | VFE/上层输出 |
| `voxel_coords_l1` | [16000, 4] (b, z, y, x) | LION 维护 |
| `v_img_l1` | [16000, 64] | PixelAlign |
| `FOV_mask_l1` | [16000] | PixelAlign |
| `Δ` | [N_group, G_l, d_inner=128] | Mamba param proj（v2.1：d_inner=2·d_model） |
| `Δ'` | [N_group, G_l, d_inner] | ΔBMod 后 |
| `B_raw` | [N_group, N=16, G_l] | Mamba param proj 原始 shared-over-channel B |
| `B_mod` | PaSS-off: [N_group, N, G_l]；PaSS-on: [N_group, d_inner, N, G_l] | ΔBMod 后；ρ=0 / PaSS-off 不 broadcast 到 variable-B |
| `z` | [N_group, G_l, d_inner] | Mamba 的 SiLU 输出门，**v2.1**：必须传给 selective_scan_fn，否则数值等价不成立 |

---

## §4 损失 / 训练策略 / 数值稳定性

### 4.1 损失：完全沿用 baseline

- 主损失 = LION-Mamba(S2) + SECOND head 现有的 cls + reg + dir loss，**不加任何新 loss**。
- 不引入额外 2D 监督、不引入 image consistency loss、不引入 depth pseudo loss。
- 理由：与 "ImageNet 预训练 + 不动额外标签" 的预算承诺一致；同时迫使所有增益归因到 ΔBMod 自身。

### 4.2 Identity-at-init（关键稳定性设计）

让 PaSS 在 epoch 0 于同一 split 路径内严格 identity；与原 fused baseline 的比较使用 `allclose < 1e-6`：

> **【v2 修订】** 下表第 1 行 (`sigmoid bias=+3`) 和第 3 行 (`U,V 都 0`) 在外部 review 中被指出无法达成严格 identity / 梯度死锁。**实际实现请按 §6.1 / §6.2 的修订表**。

| 参数 | 初始化 | 效果 |
|---|---|---|
| `W_Δ` 最后一层 bias | `+大值（≈3）` | sigmoid 输出 ≈ 1，`Δ' ≈ Δ`（identity 乘子） ← 不严格，sigmoid(3)≈0.95；废弃，见 §6.1 |
| `W_Δ` 最后一层 weight | 标准小高斯 | 初期对 Δ 几乎无修改 |
| `U, V`（低秩 B 残差） | `0` | `ΔB ≡ 0`，`B' = B`（identity 加法） ← 梯度三方全死；废弃，见 §6.2 |
| `MLP_proj` 末层 weight | 小尺度 0.01 init | 初期 `v_img` 接近常量，避免冷启动震荡 |

**Sanity check（必做）**：三条都进 CI / smoke test：(a) PaSS 关闭且走原 fused 主干 → 与 LION-Mamba(S2) baseline bit-for-bit；(b) PaSSMamba split、`pass_enabled=False` → 与 fused baseline `allclose < 1e-6`；(c) PaSSMamba split、identity init / `ρ=0` → 与 split-no-PaSS `allclose < 1e-6`。

### 4.3 Modulation warmup（前 2 epoch）

- 引入标量 `ρ(epoch) ∈ [0,1]`，cosine ramp from 0 → 1 over 2 epochs。
- `Δ' = Δ ⊙ exp( ρ · s · tanh(W_Δ · v_img) · FOV )`（v2.1 修订；ρ=0 时指数项=1，严格 identity）
- `B' = B + ρ · FOV · ΔB`（其中 `ΔB = U · diag(α) · Vᵀ`，`V=0` 初始）
- 让 LiDAR 主干先稳，再慢慢让图像介入。EMA 也同步 warmup。
- `ρ` 是 schedule 不是 learnable param，不进梯度图；ρ=0 时 PaSS 子图整体冻结，与 §6.2 梯度分析一致。

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
- 监控 `log(Δ'/Δ) = ρ·s·tanh(W_Δ·v_img)·FOV` 的分布（v2.1 修订；旧文档错按 sigmoid 写）：若长期饱和到 ±s（tanh 边缘），说明梯度死，应升高 `W_Δ` 的 LR 或降 `s`；若长期≈0，说明图像信号没用上，需查 `v_img` 是否全零或 `W_Δ` weight 死。进 grad-aux 日志（commit `2541afc` 已支持）。
- 混合精度：`SelectiveScan` 输入 cast 回 fp32 做 `Δ'` 的 exp（与原 Mamba 实现一致），不因为加 ΔBMod 而改精度策略。
- **【v2 补充】** LION 用 `mamba_ssm.Block`（双向 Mamba，走 fused `mamba_inner_fn`），Δ/B/C 在 CUDA kernel 内部从 `x_dbl = F.linear(conv1d_out, x_proj_weight)` 生成，**外部无法 hook**。要落地 ΔBMod 必须 fork `PaSSMamba` 走 `selective_scan_fn`（split 算子）路径。详见 §6.3。
- **【v2 补充】** KITTI 默认 augment（`random_world_rotation / scaling / flip` + `gt_sampling`）只改 `points` 与 `gt_boxes`，**不同步变换 calib/image**；`FOV_POINTS_ONLY` 又在 augment 之前按原始 calib 过滤。直接做 PixelAlign 会投影错位。详见 §6.4。

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
| **C. 注入层位** | `linear_1` only / `linear_4` only / `linear_out` off / **全 17 个 LIONLayer (P2)** | 证明多层注入>单层；多 scale selectivity 都吃图像 |
| **D. Image encoder 容量** | ResNet18 / **ResNet50** / Swin-T | 证明不是 "image encoder 容量赢"（反 reviewer 质疑） |
| **E. 低秩 rank r** | 1 / 2 / **4** / 8 / 16 | 证明 rank=4 已饱和；不是参数量赢 |
| **F. Warmup** | 0 / 2 epoch / 5 epoch | 证明 identity-at-init 必要 |
| **G. 算子路径归因**（v2.1 新增） | LION-Mamba(S2)-fused / LION-Mamba(S2)-split (PaSS off) / PaSS-LION-split | **把"PaSS 收益"与"fused→split 算子差异"分开**，详见 §6.5 |

注：早先讨论中考虑过加一组 "PaSS × TIP-LION 正交性" ablation；因 TIP-LION(TIS/HGD/IGVG) 整体不再推进，且 TIS 的 `bev_h / bev_h_t` 序列化已被 LION-Mamba(S2) baseline 内含，该组 ablation 取消。

### 5.3 机制诊断指标（非 AP）

| 指标 | 期望结果 | 说明 |
|---|---|---|
| FOV mask 诊断（默认 KITTI + 专用配置） | 默认 `FOV_POINTS_ONLY=True` 只做 mask 单测；专用 `FOV_POINTS_ONLY=False` 配置下，增益应集中在可投影到图像的点 / object 上 | 默认 KITTI 管线已在 augment 前过滤 FOV 外点，不能直接报告 "FOV 外 AP"；若专用配置里不可见区域也涨，说明 mask 或分组定义有问题 |
| 距离分段 AP（0-20 / 20-40 / 40-70m） | 远距离段增益更大 | 远距离稀疏，最吃图像补全 |
| 类别增益排序 | Ped > Cyc > Car（预期） | 小目标 / 细长目标更依赖图像纹理 |
| `Δ'` entropy 分布 | 中段（避免全饱和） | 监控 selectivity 是否被有效用上 |
| `||ΔB||_F / ||B||_F` | 0.1 – 0.5 量级 | 修正与原信号同量级、未喧宾夺主；若长期≈0，说明 B 低秩残差路径没用上，优先检查 `V` 分级解锁与 `ΔB` clamp，再给 `U/V/W_lr` 单独提高 LR |
| Latency / 参数量（v2.2 修订） | fused baseline → split-no-PaSS：**+50%-200% latency**（split tax，参见 §6.3 / §6.5）；split-no-PaSS → split-PaSS：**+5-10% latency**（PaSS modulation 本身轻）；主干新增 **~0.29M 参数**（P2，见 §2.5 + §6.6） | 反 "fusion 太重" 质疑，同时把 split tax 与 PaSS overhead 分开报 |

### 5.4 可视化（论文必带）

- **图 a**：ΔBMod 关闭 vs 开启时，远处 / 遮挡处一个 Car/Ped 的 SSM 隐状态 `x_t` 沿序列 trajectory 对比 → 证明图像确实改变了 selectivity 传播。
- **图 b**：把每个 voxel 的 `exp(s · tanh(W_Δ · v_img))`（v2.1 修订；旧文档错按 sigmoid 写）投回 BEV → 在前景目标处显著偏离 1（>1 或 <1，证明双向调制）、在地面 / 远处空白处接近 1 → 证明 "图像告诉 Mamba 该多关注 / 多忽略哪些 voxel"。
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

实测起点：LION-Mamba(S2) baseline（fused mamba_inner_fn）~8 h / 单 4080 / e40。

**单实验**（KITTI e40，单 4080）—— v2.2 修订：拆出 split tax，三栏并列：

| 配置 | 主干 Mamba | 图像分支 | 估算 wall-time | 倍率 | 说明 |
|---|---|---|---|---|---|
| **A. baseline-fused** | `mamba_inner_fn` | 无 | ~8 h | 1.0× | 已实测；v1 strong baseline |
| **B. baseline-split (PaSS off)** | `PaSSMamba` (selective_scan_fn) | 无 | ~12-24 h | **1.5-3×**（split tax） | **算子路径归因 baseline**，必须跑（§6.5） |
| **C. PaSS-LION (ResNet18)** | `PaSSMamba` | RN18-FPN | ~14-26 h | ~1.7-3.3× | ablation D 轻量 |
| **D. PaSS-LION (ResNet50)** ★主方案 | `PaSSMamba` | RN50-FPN | **~18-32 h** | **~2.3-4.0×** | 主方案；含 split tax + image branch |
| E. PaSS-LION (Swin-T) | `PaSSMamba` | Swin-T | ~22-36 h | ~2.8-4.5× | ablation D 重量 |

增量构成（相对 fused baseline）：

- **split tax**（A→B）：fused→split 在 1.5-3× 之间，需第一次跑后实测才能定。可能的 mitigation：只在 stage-1/2 用 PaSSMamba、stage-3/4 仍用 fused（混合架构 ablation）。
- **image branch**（B→C/D/E）：ResNet50-FPN +50 GFLOPs/image ≈ baseline 主干 30-60% 算力（参考 BEVFusion/SFD 经验）。前 5 epoch ImageBranch 冻结节约 ~10-15% 时间。
- **ΔBMod 本身**：< 1%，可忽略。

**完整 ablation**（主表 + ablation A-G 合计约 22-28 个训练，平均按 ~20 h/run，v2.2 修订：含 split tax）：

| 方案 | wall-time |
|---|---|
| 单 4080 | ~18-24 天（440-560 h） |
| 2× 4080 DDP（理想 ~1.8× 加速） | ~10-13 天 |
| 4× 4080 DDP | ~5-7 天 |

成本上调主要来自 split tax 落实 + 新增 G 组（split-only baseline）。若 split tax 实测 < 1.5×（乐观），总成本回落到原 v2 估算。不需要 nuScenes / Waymo 的额外训练。

### 5.7 失败兜底

- **若主表增益 < 0.5 AP**：把 paper 重写为 "selective state modulation as a fusion principle"（mechanism study + 完整 ablation），仍可投 BMVC / WACV / T-PAMI。
- **若 Ped/Cyc 不涨**：可能 KITTI 远处 Ped 投影到 image 的 patch 太小，image feature 噪声主导；ablation D（换 Swin-T 或 high-res FPN）可救。
- **若 ΔB 都没动（饱和到 identity）**：说明 modulation warmup 太长 / LR 太小，调超参；若仍无变化，说明 SSM selectivity 对图像不敏感 —— 这是负结果，paper 转向 mechanism 解释。

---

## §6 外部 review 修订（v2-v2.4，2026-05-30 后增）

外部研究员对 v1 设计提出 6 条意见。用户决策：第 1（新颖性）暂不收窄；第 6（仅 KITTI）维持。第 2、3、4、5 经源码核验后修订如下。

> **v2.1 changelog（第二轮 review 后）**：
> - §6.2：去掉"外加 zero-init 标量门 `g`"。`g=0` 与 `V=0` 叠加导致 4 个参数梯度全死。identity-at-init 由 `V=0` + warmup `ρ`（schedule）双保险。
> - §6.3：split 路径伪代码补 `z` gate（mamba 的 SiLU 输出门，原文 `xz.chunk → out_z = out·silu(z)`），backward branch (`xz.flip(-1)`) 的 `v_img/FOV` 也必须同步 `flip`。
> - §6.3：所有 D 标注从 `D=64` 改为 `D=d_inner=2·d_model=128`；`V` 形状从 `D×r` 改为 `N×r`（这影响 §2.5 参数预算、§3.3 shape 表、§6.2 形状表）。
> - §6.5（新增）：主 ablation 加 split-only baseline，把"算子路径差异"归因从"PaSS modulation 收益"隔离。
>
> **v2.2 changelog（第三轮 review 后）**：
> - §4.3 / §4.5 / §5.4：清除 sigmoid 残留，warmup 公式改为 `Δ' = Δ·exp(ρ·s·tanh(W_Δ·v)·FOV)`，监控改为 `log(Δ'/Δ)`，可视化改为 `exp(s·tanh(·))`。
> - §6.3：B 形状二分支（PaSS-off / ρ=0 → `(B, N, L)` shared-B；PaSS-on → `(B, d_inner, N, L)` variable-B），避免即便 ρ=0 也走不同 CUDA kernel 分支。
> - §5.3 / §5.6：latency 拆 split tax（fused→split 1.5-3×）+ PaSS overhead（+5-10%）；wall-time 表三栏并列。
> - §6.6（新增）：明确注入颗粒度候选（P1 stage 共享 / **P2 per LIONLayer 暂定** / P3 per Mamba block），v2.2 先暂定 P2。
> - 全文路径：`pcdet/models/backbones_3d/tip_lion/` → `pcdet/models/backbones_3d/lion_improve/`（代码已重命名）。
> - 引用：§6.6（未采纳意见）顺延为 §6.7；附录 C 旧条目里 "+ zero gate" 标记已清除。
>
> **v2.3 changelog（第四轮 review 后）**：
> - §1 / §2 / §3 / §5：把 P2 从"暂定 / 待确认"提升为默认主方案；正文统一为 17 次 PixelAlign/forward + 5 套 stage 共享 `MLP_proj / ΔBMod`，主干新增参数改为 **~0.29M**。
> - §6.3 / 附录 A：统一 B 形状二分支契约；PaSS-off / `ρ=0` 保持 `(B, N, L)` shared-over-channel B，PaSS-on 才升维到 `(B, d_inner, N, L)` variable-B。
> - §4.2 / 附录 A：收窄 identity smoke test 口径；原 fused PaSS-off 与 baseline 要求 bit-for-bit，PaSSMamba split identity 路径要求 `allclose < 1e-6`。
> - §5.3 / 附录 C：FOV 诊断改为默认 KITTI mask 单测 + `FOV_POINTS_ONLY=False` 专用配置，不再在默认 `FOV_POINTS_ONLY=True` 管线下直接报告 "FOV 外 AP"。
> - §6.6 / 附录 C：记录 P2 已确认，并把原 "需用户确认" checklist 改为已确认决策。
>
> **v2.4 changelog（集成验证后）**：
> - §6.3 / §6.5：PaSSMamba 已接入 LION backbone；真实 LION block replay 下 34 个 block forward 最大差 `9.53674e-07 < 1e-6`，block backward 检查 612 个梯度项，最大差约 `6.1e-4 < 1e-3`。端到端 encoded feature 差约 `4.3e-4`，视为多层 fp32 累积误差，不把端到端 `1e-6` 作为契约。
> - §6.4 / §5.3：新增 `FOV_POINTS_ONLY=False` 专用诊断，实测 fov_ratio 约 `0.44-0.51`，确认 FOV mask 分支被真实锻炼。
> - §6.4：新增 `test_pass_lion_aug_inverse_projection.py`，覆盖 KITTI world flip/rotation/scaling 累积 `lidar_aug_matrix` 后由 `PixelAlign._apply_inverse_aug` 逆回原始 LiDAR 系再投影；3 帧实测 inverse pixel error 最大 `0.645px < 1px`，不开 inverse 的负对照误差最小 `1508px`。
> - 附录 A：补充当前落地脚本名：`test_lion_backbone_pass_fused_equivalence.py`、`diagnose_lion_pass_fusion_train_step.py`、`test_pass_lion_kitti_projection_geometry.py`。

### 6.1 Δ 调制公式：sigmoid 门 → exp(tanh) 门

**问题**（review 第 2 点）：v1 的 `g_Δ = sigmoid(W_Δ·v + 3)`：

- 数值上 `sigmoid(3) ≈ 0.9526`，不是 1 —— "epoch 0 identity / `allclose < 1e-6`" 这条 contract **达不成**。即便 bias 拉到 +10，也只到 0.99995。
- 方向上 `sigmoid ∈ (0,1)`，只能让 `Δ' < Δ`（缩短 effective horizon），**无法增强**；丢一半假设空间。

**修订式**：

```
Δ'_i = Δ_i ⊙ exp( s · tanh( W_Δ · v_img_i ) )
```

| 参数 | 初始化 | 数值性质 |
|---|---|---|
| `W_Δ` 末层 weight | **零初始化** | `tanh(0)=0` → `exp(0)=1` → `Δ'=Δ` 严格 identity |
| `W_Δ` 末层 bias | 0 | 不需要 hack bias |
| `s`（scale） | 0.5，可学习标量 | `Δ'/Δ ∈ [exp(-s), exp(s)] = [0.61, 1.65]`，双向有界 |

`tanh` 的有界性 + `exp` 的对数空间，与 Mamba 原本对 Δ 走 `softplus → exp` 的数值域同源；不引入新的数值不稳。FOV mask 仍走硬开关：

```
Δ'_i = Δ_i ⊙ exp( s · tanh(W_Δ · v_img_i) · FOV_i )    # FOV_i ∈ {0,1}
```

FOV 外 `FOV_i=0` → 指数项 = 1，严格 identity。

### 6.2 B 低秩残差初始化：U/V 双零 → LoRA 风格不对称初始化

**问题**（review 第 3 点）：v1 的 `ΔB = U · diag(α) · Vᵀ`，`U, V` 同时零初始化。设 `dout` 为下游回传：

- `∂L/∂U ∝ dout · V · diag(α)` → `V=0` 时为 **0**
- `∂L/∂V ∝ doutᵀ · U · diag(α)` → `U=0` 时为 **0**
- `∂L/∂α ∝ Uᵀ · dout · V` → 同时为 **0**

三个梯度同时死，ΔB 永远学不起来。

**修订式**（LoRA 经典做法，**v2.1 去掉 zero-init scalar gate**）：

设 `ΔB ∈ R^{D×N}`（D=d_inner=128, N=d_state=16），分解为 `ΔB = U · diag(α) · Vᵀ`，则 `U ∈ R^{D×r}`、`V ∈ R^{N×r}`（v2.1 修订：v2 把 V 写成 D×r 是错的，不能拼出 D×N）。

| 参数 | 形状 | 初始化 | 效果 |
|---|---|---|---|
| `U` | `D × r` = 128×4 | `N(0, σ²)`, σ=0.01 | 非零小高斯 |
| `V` | `N × r` = 16×4 | **零** | 起步 `ΔB = 0`，identity 严格成立 |
| `W_lr`（产生 α） | 64 × r | 标准小高斯 | α 起步小但非零 |

**不再额外加 zero-init scalar gate**。原 v2 设计 `ΔB ← g · U·diag(α)·Vᵀ` + `g=0` + `V=0` 会让 4 个参数梯度同时归零（review 第二轮指出）：

```
∂L/∂g ∝ ⟨G, U·diag(α)·Vᵀ⟩ = 0   (V=0)
∂L/∂U ∝ g · G · V · diag(α) = 0  (g=0, V=0)
∂L/∂V ∝ g · Uᵀ · G · diag(α) = 0 (g=0)
∂L/∂α ∝ g · diag(Uᵀ · G · V) = 0 (g=0, V=0)
```

去掉 gate 后梯度检查：`V=0` 时 `∂L/∂V ∝ Uᵀ·G·diag(α)`，`U` 非零、`α` 非零 → **非零**，`V` 可学；下一步 `V≠0` 后 `U`、`α` 也开始获得非零梯度。

**identity-at-init 由 `V=0` 保证；渐进开启由 warmup `ρ`（schedule、非 learnable param）保证**：

```
B'_i = B_i + ρ(epoch) · FOV_i · ΔB_i      # ρ ∈ [0,1] cosine ramp, 不进梯度图
```

`ρ` 是 cosine schedule，前 2 epoch 从 0 → 1。它不是可学参数，所以不会引入梯度死锁；`ρ=0` 时严格不影响 backward（梯度路径上 `ρ · ΔB`，`∂/∂ΔB = ρ`，`ρ=0` → ΔB 梯度归零，但**此时 v_img 路径上的 W_lr, W_Δ 等也都不更新，等价于 PaSS 子图整体冻结，是想要的行为**）。这与"V=0 让 V 在第一步就有梯度"不矛盾 —— V 的梯度激活发生在 `ρ > 0` 之后。

### 6.3 实现风险：mamba_inner_fn fused 不可外部 hook

**源码核验**（`mamba_ssm-1.1.1/modules/mamba_simple.py` 与 `ops/selective_scan_interface.py`）：

1. LION 用的 `MambaBlock = mamba_ssm.Block` → 内部 `self.mamba = Mamba(...)` → `Mamba.forward` 调 `mamba_inner_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight, dt_proj_weight, A, None, None, D, delta_bias, delta_softplus=True)`。
2. `mamba_inner_fn` 是 fused CUDA `torch.autograd.Function`（`MambaInnerFn`）：内部
   ```
   conv1d_out  = causal_conv1d_cuda.causal_conv1d_fwd(...)
   x_dbl       = F.linear(conv1d_out, x_proj_weight)
   delta       = delta_proj_weight @ x_dbl[:, :dt_rank].t()
   B           = x_dbl[:, dt_rank:dt_rank+d_state]
   C           = x_dbl[:, -d_state:]
   out, _, out_z = selective_scan_cuda.fwd(conv1d_out, delta, A, B, C, D, z, delta_bias, ...)
   ```
   **Δ/B/C 全部在 fused kernel 内部生成，外部 Python 层完全拿不到**。"SelectiveScan 前插一下" 在 v1.1 不成立。
3. `mamba_simple.py:178/192` 调了**两次** `mamba_inner_fn`：正向 `xz` 与反向 `xz.flip([-1])`，各有独立的 `conv1d_b / x_proj_b / dt_proj_b / D_b / A_b_log`。LION 的 Mamba 是 **bidirectional**。
4. `d_inner = expand · d_model = 2·d_model`（mamba_simple.py:67）。v1 文档里 `D=64` 的 shape 假设要按 **d_inner=128** 重算（除非把 PaSS 调制点放在 `out_proj` 之前的 d_model 域，而不是 ssm 内部）。
5. `dt_proj.bias` 通过 inv_softplus 初始化成 `U(dt_min, dt_max)`（mamba_simple.py:107-115），所以 Δ 乘性修正必须乘在 `softplus(dt_proj_out + dt_proj.bias)` **之后**；乘在 softplus 之前数值含义错。

**修订实现路径**：fork 一个 `PaSSMamba`，复制 `mamba_simple.py:Mamba.forward` 主体，把 fused 调用替换为 split 算子（**v2.1 修订**：补 `z` gate；补 backward branch 的 `v_img/FOV` 同步 flip；标量域统一为 `d_inner=128`）：

```python
# === 一次 PaSSMamba.forward（输入 hidden_states: (B, L, d_model)，v_img/FOV: (B, L, *)）===

# 1) in_proj 拿 xz：与 mamba_simple.py:167-173 一致
xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
               "d (b l) -> b d l", l=L)                                          # (B, 2*d_inner, L)
if self.in_proj.bias is not None:
    xz = xz + rearrange(self.in_proj.bias, "d -> d 1")
x, z = xz.chunk(2, dim=1)                                                         # 各 (B, d_inner=128, L)

# 2) conv1d：与 fused 路径一致
conv1d_out = causal_conv1d_fn(x, rearrange(self.conv1d.weight, "d 1 w -> d w"),
                              self.conv1d.bias, activation="silu")                # (B, d_inner, L)

# 3) x_proj 拆出 dt/B/C raw
x_dbl   = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), self.x_proj.weight) # (B*L, dt_rank+2*d_state)
dt_raw  = x_dbl[:, :self.dt_rank]
B_raw   = rearrange(x_dbl[:, self.dt_rank:self.dt_rank+self.d_state],
                    "(b l) n -> b n l", l=L)                                       # (B, N=16, L)
C_raw   = rearrange(x_dbl[:, -self.d_state:], "(b l) n -> b n l", l=L)             # (B, N=16, L)

# 4) Δ：先线性、再 softplus（softplus 必须在 modulation 前；dt_proj.bias 已是 inv_softplus(uniform) init）
delta = rearrange(F.linear(dt_raw, self.dt_proj.weight) + self.dt_proj.bias,
                  "(b l) d -> b d l", l=L)                                         # (B, d_inner, L)
delta = F.softplus(delta)                                                          # (B, d_inner, L)

# 5) PaSS modulation —— 这里才拿得到 Δ、B
# v_img: (B, L, 64); FOV: (B, L); 都按 token 对齐
# self.rho ∈ [0,1] 是 warmup schedule，非 learnable
if pass_enabled and self.rho > 0:
    g_delta = self.W_Delta(v_img)                                                   # (B, L, d_inner)
    delta_mul = torch.exp(
        self.rho * self.s * torch.tanh(g_delta) * FOV.unsqueeze(-1)
    )                                                                               # (B, L, d_inner)
    delta_mod = delta * rearrange(delta_mul, "b l d -> b d l")                      # (B, d_inner, L)

    alpha = self.W_lr(v_img)                                                        # (B, L, r=4)
    # ΔB = U·diag(α)·Vᵀ，per-token：(d_inner, r) · (r,) · (r, N) → (d_inner, N)
    delta_B = torch.einsum("dr,blr,nr->bldn", self.U, alpha, self.V)                # (B, L, d_inner, N)
    delta_B = delta_B * FOV[..., None, None]                                         # FOV 外清零
    B_mod = B_raw.unsqueeze(1).expand(-1, self.d_inner, -1, -1) + self.rho * rearrange(
        delta_B, "b l d n -> b d n l"
    )                                                                               # (B, d_inner, N, L)
else:
    # 关闭 PaSS 或 ρ=0 时保留 shared-over-channel B，避免 identity 路径误走 variable-B 分支
    delta_mod = delta
    B_mod = B_raw                                                                   # (B, N, L)

# 6) selective_scan —— 必须传 z！且 delta_bias/softplus 都已合并
# selective_scan_fn 的 B 参数支持 (B, d_inner, N, L)（variable-B 形式）或 (B, N, L)
out_z = selective_scan_fn(conv1d_out, delta_mod, A, B_mod, C_raw, self.D.float(),
                          z=z,                          # <-- v2.1 必须传，否则丢 SiLU 输出门
                          delta_bias=None,              # 已合并进 delta
                          delta_softplus=False)         # 已 softplus 过
# 返回 out_z = silu(z) * out，对应原 fused 路径的 out_z

# === 7) 反方向：xz.flip(-1) 那一支 ===
# v2.1 修订：v_img、FOV 也必须沿 L 维同步 flip，否则 backward 支的 token 顺序与 modulation 错位
v_img_b = v_img.flip(1)                                                            # (B, L, 64)
FOV_b   = FOV.flip(1)
# ... 复用上面 5/6 逻辑，全部用 _b 后缀的反向权重（conv1d_b/x_proj_b/dt_proj_b/D_b/A_b_log）
# 最后 out_b.flip(-1) 再与 out_z 相加，符合 mamba_simple.py:206
```

**关键点**：

- 正反双向都要做（参数翻倍但保持 LION 双向语义）。如果只对正向做 PaSS 作为 ablation，必须明确写入消融表。
- 标量域：`d_model=64`（LION layer dim），`d_inner = expand·d_model = 128`，`N=d_state=16`，`dt_rank=⌈d_model/16⌉=4`。所有 modulation 在 `d_inner=128` 域。
- **B 形状二分支（v2.2 新增）**：`selective_scan_fn` 的 `B` 接受 `(B, N, L)`（shared-over-channel B）或 `(B, d_inner, N, L)`（variable B per channel）。fused `mamba_inner_fn` 内部 reshape 成 `(B, 1, N, L)`。**PaSS-off / ρ=0 时必须把 B_raw 以 `(B, N, L)` 形式传入，不要 broadcast 成 `(B, d_inner, N, L)`**，这样 identity 路径不会额外引入 variable-B 分支差异：
  ```python
  if pass_enabled and rho > 0:
      # variable-B 分支：(B, d_inner, N, L)
      B_mod = B_raw.unsqueeze(1).expand(-1, self.d_inner, -1, -1) + rho * rearrange(
          delta_B, "b l d n -> b d n l"
      )
  else:
      # shared-over-channel B 分支：(B, N, L)，避免 identity 路径误走 variable-B
      B_mod = B_raw  # (B, N, L)
  ```
- `dt_proj.bias` 已经通过 inv_softplus 初始化成 `U(dt_min, dt_max)`（mamba_simple.py:107-115），所以 `delta = F.linear(dt_raw, dt_proj.weight) + dt_proj.bias` 必须先做，再 `softplus`，再做 modulation。三步顺序错一个都会破坏 identity / `allclose` smoke test。

**前置工作（必做，加入实验前置 checklist）**：

- [x] 实现 `pcdet/models/backbones_3d/pass_lion/pass_mamba.py`，并做三类 smoke test：原 fused PaSS-off 与 baseline **bit-for-bit**；PaSSMamba split PaSS-off 与 fused baseline **allclose < 1e-6**；PaSSMamba split identity-init / `ρ=0` 与 split PaSS-off **allclose < 1e-6**。
- [x] LION backbone 集成对拍：`test_lion_backbone_pass_fused_equivalence.py` 用真实 KITTI batch replay 34 个 LION block；forward 最大差 `9.53674e-07`，block backward 最大梯度差 `< 1e-3`。端到端 encoded feature 差约 `4e-4`，只作为累计误差监控，不作为 `1e-6` 契约。
- [ ] Benchmark：fused `mamba_inner_fn` vs split `selective_scan_fn` 端到端 latency 与显存差距。预期 split 路径慢 1.5×–3×（因为少了 fused 路径的 conv1d + linear + scan 融合）。这是 PaSS 的**实现侧 tax**，必须在论文 §5.6 wall-time 表里如实反映。
- [ ] 评估"只在 layer-1 / layer-2 注入 PaSS（剩下用 fused Mamba）"作为 latency mitigation 的退路。

### 6.4 数据增强对齐：world augment 不动 calib

**源码核验**（`pcdet/datasets/augmentor/augmentor_utils.py` + `pcdet/datasets/kitti/kitti_dataset.py`）：

1. `global_rotation` (line 50-70)：`points = rotate_points_along_z(points, noise_rotation)`；`gt_boxes[:,0:3]` 同步旋转；`gt_boxes[:,6] += noise_rotation`。**不碰 calib、不碰 image**。
2. `global_scaling` (line 74-91)：`points[:,:3] *= noise_scale`；`gt_boxes[:,:6] *= noise_scale`。**不碰 calib、不碰 image**。
3. `random_flip_along_x/y` (line 8-49)：仅翻转 `points` 与 `gt_boxes`。**不碰 calib、不碰 image**。
4. `random_image_flip_horizontal` (line 113) 是唯一同步动 image+depth+gt_boxes+calib 的 augment，但默认 KITTI config **未启用**。
5. `kitti_dataset.py:1330` 的 `FOV_POINTS_ONLY=True` 在 `__getitem__` 内、在 augment 之前按**原始 calib** 过滤；之后 augment 把 `points` 旋转/缩放/翻转，calib 保持原始（`data_dict['calib']` 一直是 `get_calib(idx)` 返回值，未做任何变换）。
6. augmentor 不持久化 `noise_rotation / noise_scale / flip_x / flip_y` 到 `data_dict`（grep `aug_param / aug_matrix / world_rotation` 在 `pcdet/datasets/processor` / `pcdet/datasets/dataset.py` 中**无命中**）。

**结论**：直接做 PixelAlign，voxel 在 augment 后 LiDAR 系，calib 是原始的 → 投影到 image 必然错位。

**两段式修订方案**：

| 阶段 | 方案 | 改动量 | 性能代价 |
|---|---|---|---|
| **第一版（baseline + PaSS 跑通用）** | **方案 A：禁用所有 world augment 与 gt_sampling**，仅保留与相机几何无关的 augment | 0（改 config） | ~1-2 AP 下降；可控、可解释 |
| **第二版（论文最终版）** | **方案 B：augmentor 持久化 augment 参数到 `data_dict['lidar_aug_matrix']`**（4×4 SE3 + scale）；PixelAlign 投影前用 `aug_matrix^-1` 把 voxel 中心变回原始 LiDAR 系 | 改 `augmentor_utils.global_rotation/scaling/flip` + 加 `lidar_aug_matrix` 字段 + PixelAlign 读取 | 0；保全部 augment |

**方案 B 实现要点**：

```python
# augmentor_utils.global_rotation
data_dict['lidar_aug_matrix'] = R_z(noise_rotation) @ data_dict.get('lidar_aug_matrix', I_4)

# augmentor_utils.global_scaling
data_dict['lidar_aug_matrix'][:3, :3] *= noise_scale

# augmentor_utils.random_flip_along_y (例)
F = diag([1, -1, 1, 1])  # y 翻
data_dict['lidar_aug_matrix'] = F @ data_dict['lidar_aug_matrix']

# PixelAlign (训练时)
voxel_center_orig = (lidar_aug_matrix.inverse() @ voxel_center_aug_homog.T).T[:, :3]
p_cam = R0_rect @ Tr_velo_to_cam @ [voxel_center_orig; 1]
(u·z, v·z, z) = P2 @ p_cam
```

`gt_sampling` 比较复杂（粘贴的虚拟物体来自其他帧、其他 calib），第一版直接禁用；第二版要么仍禁用，要么在粘贴时附带源帧 calib，对粘贴 voxel 走源 calib 投影（工程量大，可不做）。

**实验前置 checklist**：

- [x] 写一个数值验证：从 dataloader 取一个 batch，对比 "PixelAlign 投影到 image 的像素坐标" vs "把 gt_boxes 中心投影"，两者应落在同一 2D 框内（误差 < 1 像素）。当前 `test_pass_lion_kitti_projection_geometry.py` 在 KITTI 5 帧上 max error 约 `0.06px`。
- [x] 写一个 augment-aware 数值验证：用真实 KITTI 点云经过 `DataAugmentor.random_world_flip/rotation/scaling`，检查 `lidar_aug_matrix` 可复现增强后坐标、`PixelAlign._apply_inverse_aug` 可恢复原始坐标，并对比 inverse 后投影与原始 calib 投影误差 < 1 像素。当前 `test_pass_lion_aug_inverse_projection.py` 在 KITTI 3 帧上 max inverse pixel error `0.645px`；不开 inverse 的负对照最小误差 `1508px`。
- [x] 写 `FOV_POINTS_ONLY=False` 专用诊断：`diagnose_lion_pass_fusion_train_step.py --include-non-fov-points` 应记录明显低于 1 的 `pass_fusion/*/fov_ratio`，并断言 outside-FOV ratio 足够大。当前 3 iter 实测 fov_ratio 约 `0.44-0.51`。
- [ ] 在论文 §5 ablation 加一行 "PaSS-LION (no world augment)" vs "PaSS-LION (with augment-aware projection)"，证明性能差异并归因。

### 6.5 算子路径归因隔离：必须加 split-only baseline

**问题**（review 第二轮第 4 点）：v2 设计里 PaSS-on 走 `selective_scan_fn`（split），baseline 仍走 `mamba_inner_fn`（fused）。即便 forward smoke test 通过 (`< 1e-6`)，**训练过程中两条算子路径的数值差异（融合 reduction 顺序、内部精度策略、backward 自定义梯度）足以让两者训练动力学漂移**。reviewer 可以合理质疑："PaSS 的 AP 增益来自 split kernel 的数值差异，而不是 PaSS modulation"。

**修订**：主表必须含三组：

| 组 | 主干 Mamba | PaSS | 用途 |
|---|---|---|---|
| **A. LION-Mamba(S2)-fused** | `mamba_inner_fn` | OFF | v1 strong baseline，与已有数据对齐 |
| **B. LION-Mamba(S2)-split** | `PaSSMamba` (split `selective_scan_fn`, **modulation 完全关闭**：W_Δ=0 + V=0 + ρ=0) | OFF | **算子路径归因隔离**：与 A 的差 = 纯算子差，与 C 的差 = 纯 PaSS 收益 |
| **C. PaSS-LION-split** | `PaSSMamba` | ON | 主方法 |

报告时同时报 **(A→B) 的 Δ**（算子差）与 **(B→C) 的 Δ**（PaSS 净收益）。

**实现要点**：

- B 组与 C 组共用一套 `PaSSMamba`，仅以 `pass_enabled=False` 切；`pass_enabled=False` 时 `delta_mod=delta`、`B_mod=B_raw`、`ρ=0`，保证除算子路径外其他完全等同。
- B 组训练 wall-time 应与 C 组接近（差 PaSS 子图的 ΔBMod 算力，~1% 主干）；与 A 组的差 = fused→split 的 latency tax。论文 §5.6 wall-time 表三栏并列。
- 如果 (A→B) 的 AP 差 > 0.5（split 比 fused 显著差或显著好），需要在论文文本明确披露并讨论；若差异 < 0.2 AP（在 KITTI 噪声内），可在脚注说明 "算子路径差异在噪声范围内"。
- **验证口径（v2.4）**：逐 block forward 以 `<1e-6` 作为契约；逐 block backward 以 `<1e-3` 作为真实 LION 输入下的 allclose 契约。端到端 34 block 串联后的 `encoded_features` 允许出现 `~1e-4` 量级累计漂移，只报告不作为失败条件。

### 6.6 注入颗粒度：PixelAlign / ΔBMod 究竟放在哪一层（v2.3 已确认 P2）

**问题**（review 第三轮第 1 点）：v1/v2 文档把 PaSS 写成 "4 stage 注入"，但代码实际结构（`pcdet/models/backbones_3d/lion_backbone_one_stride.py`）是：

```
LION backbone
├── linear_1: LIONBlock (depth=2)
│   ├── encoder[0]: LIONLayer  ← Mamba × 2 direction (coords₀)
│   ├── downsample[0]: PatchMerging3D  → coords 改变 (coords₁)
│   ├── encoder[1]: LIONLayer  ← Mamba × 2 direction (coords₁)
│   ├── downsample[1]: PatchMerging3D  → (coords₂)
│   ├── decoder[0]: LIONLayer  ← (coords₂)
│   ├── upsample[0]: PatchExpanding3D  → (coords_back)
│   ├── decoder[1]: LIONLayer
│   └── upsample[1]: PatchExpanding3D
├── dow1: PatchMerging3D (stage 间下采样)
├── linear_2: LIONBlock (depth=2)
├── dow2 / linear_3 / dow3 / linear_4 / dow4
└── linear_out: LIONLayer
```

实际 Mamba 调用在 `LIONLayer._forward_serial`（lion_backbone_one_stride.py:826-831）的 `block(x_features)`：同 LIONLayer 内 2 个 direction 共享 coords；跨 PatchMerging/PatchExpanding 后 coords 改变。

**总 LIONLayer 数**：4 stages × (encoder_depth + decoder_depth) + 1 linear_out = 4 × (2+2) + 1 = **17 个 LIONLayer**。

**三个颗粒度候选**：

| 方案 | PixelAlign 调用粒度 | ΔBMod 权重粒度 | 主干新增 params | v_img 对齐难度 | 备注 |
|---|---|---|---|---|---|
| **P1：stage 共享** | 5 次/forward（每 stage 入口 1 次；stage 内 LIONLayer 共享 v_img） | 5 套权重 | ~58K × 5 ≈ 0.29M | stage 内 coords 变了 v_img 不对齐 → **需在每个 PatchMerging 后用 scatter/interpolate 重投影 v_img**，工程复杂 | params 最少但代码改动最重 |
| **P2：per LIONLayer（默认）** | 17 次/forward（每 LIONLayer 入口 1 次） | 5 套权重（**stage 内共享、跨 stage 独立**） | ~58K × 5 ≈ 0.29M | 每次 PixelAlign 用当 LIONLayer 的真实 coords 投影 → **天然对齐** | params 与 P1 相同；正确性最稳；调用次数多但单次 ~17K voxel·bilinear，开销 < 1% |
| **P3：per Mamba block（per direction 独立）** | 17 × 2 = 34 次/forward | 5 × 2 = 10 套权重 | ~58K × 10 ≈ 0.58M | 同 P2 + direction 分支 | params 翻倍；ablation 维度增加；但同 LIONLayer 两 direction 共享 v_img 本来就合理，没必要分 |

**v2.3 默认方案 P2**：每个 LIONLayer 入口调一次 PixelAlign（17 次/forward，cache key = `id(coords)`，同 coords 复用）；ΔBMod / MLP_proj 权重 stage 内共享（5 套）。理由：

- coords 对齐是正确性问题，stage 共享 (P1) 需要 scatter 回旧 coords，工程复杂且引入插值误差
- 同 LIONLayer 内两 direction 共享 v_img 是物理合理的（图像信息与扫描方向无关，§3.2 第 1 条已论证）
- params 与 P1 相同、与 P3 减半，对"非参数量赢"叙事有利

**对参数预算的影响**（v2.3 已同步 §2.5）：每套 stage 共享权重 ~58K × 5 stage = **~0.29M 主干新增**；与 v2.1 的"~0.23M（4 stage）"接近，因为多加了 `linear_out` 那一组。

**已确认决策**：

- [x] P2 作为默认主方案，正文 §1 / §2 / §3 / §5 已按 17 次 PixelAlign + 5 套 stage 共享权重改写
- [x] ΔBMod / MLP_proj 权重共享粒度固定为 stage 内共享；全网共享 1 套与每 LIONLayer 独立可作为后续轻量 ablation，不进入默认实现

### 6.7 未采纳意见

| Review 项 | 用户决策 | 备注 |
|---|---|---|
| 1. claim 收窄（"第一个 SSM selectivity 融合范式" 太大） | **暂不收窄** | 现有 cross-modal Mamba 工作（DDHFusion / Height-Fidelity DGF / S3M3D）做的是 token-level cross-attn 或 BEV grid fusion，而 PaSS 是参数级 Δ/B 调制；narrative 角度仍是新的。S3M3D 已撤稿，进一步降低撞车风险。先把方法做通，论文 abstract 收窄留到投稿前一轮 |
| 6. 只用 KITTI 不够 | **维持 KITTI-only** | 现阶段目标是验证 mechanism，KITTI 已足；多数据集留到 follow-up 工作 |

---

## 附录 A：实现落点

| 改动面 | 文件 / 目录 | 改动内容 |
|---|---|---|
| 新增子模块 | `pcdet/models/backbones_3d/pass_lion/` | 新建：`image_branch.py`、`pixel_align.py`、`delta_b_mod.py`、**`pass_mamba.py`**（fork 自 `mamba_ssm.modules.mamba_simple.Mamba`，走 split `selective_scan_fn`，参见 §6.3） |
| LION backbone 插入点 | `pcdet/models/backbones_3d/lion_backbone_one_stride.py` | 每个 LIONLayer 入口加 `PixelAlign`（P2 默认，17 次/forward）；当 PaSS 启用时把 `MambaBlock` 替换为 `PaSSMamba`（fused→split） |
| 序列化复用 | `pcdet/models/backbones_3d/lion_improve/serialization.py` | **不动**，沿用 `bev_h / bev_h_t` |
| Detector 路由 | `pcdet/models/detectors/second_net.py` | forward 入口把 `batch_dict['images']` 路由到 backbone |
| **Augmentor 改造（§6.4 方案 B）** | `pcdet/datasets/augmentor/augmentor_utils.py` + `pcdet/datasets/augmentor/data_augmentor.py` | `global_rotation / global_scaling / random_flip_along_*` 把 augment 累积到 `data_dict['lidar_aug_matrix']`；PixelAlign 取 inverse 用 |
| KITTI 数据管线 | `tools/cfgs/dataset_configs/kitti_dataset.yaml` | 新 PaSS config 加 `GET_ITEM_LIST: ['images', 'points', 'calib_matricies']`；**第一版 disable `random_world_rotation/scaling/flip` + `gt_sampling`（§6.4 方案 A）** |
| 训练入口 | `tools/train.py` | 不动；model build 时多读 ImageBranch pretrained 路径 |
| 启动脚本 | `tools/run_kitti_experiment.sh` | 新增 `--enable_pass` 切 PaSS |
| 配置文件 | `tools/cfgs/kitti_models/PaSS-LION/` | 新建主 config + ablation configs |
| Smoke / equivalence test | `tools/test_pass_lion_ref_cuda_equivalence.py`、`tools/test_pass_lion_fused_split_equivalence.py`、`tools/test_lion_backbone_pass_fused_equivalence.py` | **(a)** CUDA selective_scan vs ref；**(b)** PaSSMamba split PaSS-off 与 fused baseline `allclose < 1e-6`；**(c)** LION backbone 34 block replay forward/backward 对拍 |
| **投影 / FOV 对齐验证脚本** | `tools/test_pass_lion_kitti_projection_geometry.py`、`tools/test_pass_lion_aug_inverse_projection.py`、`tools/diagnose_lion_pass_fusion_train_step.py --include-non-fov-points` | 对比 voxel→image 投影与 gt_boxes→image 投影误差；验证 world augment 累积 `lidar_aug_matrix` 后 PixelAlign inverse 投影仍 <1px；专用 `FOV_POINTS_ONLY=False` 跑法验证 FOV mask 分支 |

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
| ~~`W_Δ` 末层 bias init~~ | ~~+3.0~~ → **v2 废弃**：`W_Δ` 末层 weight = 0, bias = 0；用 `exp(s·tanh(·))` 门，s=0.5（参见 §6.1） |
| ~~`U, V` init~~ | ~~0~~ → **v2.1 修订**：`U ∈ R^{d_inner×r} = 128×4 ~ N(0, 0.01²)`、`V ∈ R^{N×r} = 16×4 = 0`，**不加 zero-init scalar gate**（与 V=0 叠加会让 4 个参数梯度全死，参见 §6.2 v2.1） |
| ΔBMod scale `s` (§6.1, 新增) | 0.5，可学习标量 |
| ~~ΔBMod outer gate `g`~~ | ~~0~~ → **v2.1 删除**：与 V=0 叠加导致梯度死锁；identity 由 V=0 + warmup ρ 双保险 |
| 第一版数据增强（§6.4 方案 A） | disable `random_world_rotation/scaling/flip`、`gt_sampling` |
| Mamba 算子路径（§6.3） | PaSS-on: split `selective_scan_fn`（**必须传 z**）；PaSS-off: 保留 fused `mamba_inner_fn`（baseline 不退化）；**ablation 必须加 LION-Mamba(S2)-split-only 一组**（§6.5） |

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
| 8 | FOV 外 voxel 处理 = 强制 identity | 学习一个 "FOV-out modulator" / 完全丢弃 | 保证可移除性；默认 KITTI `FOV_POINTS_ONLY=True` 下不直接报告 FOV-out AP，另用 mask 单测和 `FOV_POINTS_ONLY=False` 专用诊断配置验证 |
| 9 | 注入层位 = P2 全 17 个 LIONLayer | stage 共享 / per direction / 选层 | 主方案最大化 fusion 表面；每个 LIONLayer 入口重投影保证 coords 对齐；选层作为 ablation |
| 10 | Identity-at-init + warmup | 直接训练 | 让 split 路径 epoch 0 forward 与 split-no-PaSS 数值等价，并用 split-only baseline 隔离 fused→split 算子差异 |
| 11 (v2) | Δ 调制用 `exp(s·tanh(·))` | sigmoid 门 / softplus 门 / 直接加性 | sigmoid 不严格 identity 且单向；exp·tanh 严格 identity-at-init、双向有界、与 Mamba 的 `softplus·exp` 数值域一致（参见 §6.1） |
| 12 (v2) | B 残差用 LoRA 式 U(N(0,σ²))/V(0)（v2.1 进一步去掉 zero gate） | U/V 双零 / 单边随机 | 双零导致三方梯度死锁；LoRA 经典模式确保 `V=0` 时 `∂L/∂V ∝ Uᵀ·dout·diag(α)` 非零（参见 §6.2） |
| 13 (v2) | Fork `PaSSMamba` 走 `selective_scan_fn` | hook fused `mamba_inner_fn` / 改 mamba_ssm 源码 / 整个换 Mamba 实现 | fused kernel 在 CUDA 内生成 Δ/B/C，Python 层无法 hook；fork split 是唯一可行路径，代价是 ~1.5-3× latency tax，需在 §5.6 wall-time 表如实反映（参见 §6.3） |
| 14 (v2) | 数据增强对齐分两段：A 先禁用、B 持久化 aug_matrix | 不修 / 一步到位走 B / 完全放弃 PaSS 在 KITTI | 第一版用 A 把 PaSS 跑通、保对比可控；第二版用 B 恢复全 augment、追末段 AP（参见 §6.4） |
| 15 (v2.1) | B 残差不加 zero-init scalar gate | v2 的 `g · U·diag(α)·Vᵀ` + `g=0` + `V=0` | 4 个参数梯度同时归零；LoRA 经典模式只用 `V=0` + `U ~ N(0,σ²)`，identity-at-init 由 V=0 + warmup ρ（schedule, 非 learnable）双保险（参见 §6.2 v2.1） |
| 16 (v2.1) | PaSSMamba split 路径必须传 `z` 给 selective_scan_fn | v2 伪代码 z=None | mamba 的 SiLU 输出门 `out_z = silu(z) · out`；丢了 z 不可能数值等价（参见 §6.3 v2.1） |
| 17 (v2.1) | shape 域统一为 `d_inner = 2·d_model = 128`；`V ∈ R^{N×r}` | v2 写 D=64、V ∈ R^{D×r} | 两处都错：mamba 的 ssm 算在 d_inner 域；要拼出 ΔB ∈ R^{D×N} 必须 U: D×r, V: N×r（参见 §2.5、§3.3、§6.2 v2.1） |
| 18 (v2.1) | 主 ablation 加 split-only baseline | 只对比 fused baseline vs PaSS-split | fused→split 算子差异会污染归因；必须三栏并列（fused / split-no-PaSS / split-PaSS），把"算子差"与"PaSS 净收益"分开（参见 §6.5） |
| 19 (v2.2) | 全文清除 sigmoid 残留 | 只在 §6.1 写新公式、其他章节仍 sigmoid | §4.3 warmup、§4.5 监控、§5.4 可视化、§1.3 ΔBMod 行原 sigmoid 措辞会让实现者无所适从（参见 v2.2 changelog） |
| 20 (v2.2) | B 形状二分支（PaSS-off shared-B / PaSS-on variable-B） | 永远 broadcast 成 variable-B | 关闭 PaSS 或 ρ=0 时不应误走 variable-B 分支；否则会给 identity / allclose smoke test 引入无关算子差异（参见 §6.3 v2.2 关键点） |
| 21 (v2.2) | latency 报告拆 split tax + PaSS overhead 两段 | 只报"PaSS-LION 比 baseline 慢 N%" | 算子归因 ablation (§6.5) 要求把"fused→split tax"与"PaSS 真实开销"分开（参见 §5.3 / §5.6） |
| 22 (v2.2) | 注入颗粒度候选中暂定 P2（per LIONLayer + stage 内权重共享） | P1 stage 共享 / P3 per Mamba block | 同 LIONLayer 内 coords 不变、两 direction 共享 v_img 物理合理；P1 跨 PatchMerging 需 scatter 重投影工程复杂；P3 params 翻倍且 ablation 维度无意义；v2.2 阶段仍待用户确认（参见 §6.6 v2.2） |
| 23 (v2.2) | 文档路径统一 `lion_improve/` | 旧 `tip_lion/` | 代码已重命名；文档落后于代码会让实现者找不到文件（参见 v2.2 changelog） |
| 24 (v2.3) | P2 从"待确认"改为默认主方案并同步正文 | 继续只在 §6.6 保留注脚 / 回退 P1 / 升级 P3 | 用户确认直接处理前 4 个问题后，§1 / §2 / §3 / §5 / 附录 A 全部改为 17 次 PixelAlign/forward + 5 套 stage 共享权重；参数预算固定为 ~0.29M，P1/P3 仅保留为候选或后续 ablation |
| 25 (v2.4) | LION 集成等价契约以逐 block 为准 | 端到端 encoded feature 强行 `<1e-6` | 真实 34 block 串联会把 split/fused 的 fp32 归约误差累积到 `~1e-4`；逐 block forward `<1e-6`、逐 block backward `<1e-3` 更符合算子契约与训练现实 |
| 26 (v2.4) | 默认 KITTI 之外增加 `FOV_POINTS_ONLY=False` 诊断 | 只用默认 KITTI fov_ratio≈1 的训练 smoke | 默认管线在 augment 前过滤 FOV 外点，FOV mask 分支几乎休眠；专用诊断实测 fov_ratio 约 `0.44-0.51`，能验证 FOV 外 identity 安全性 |
| 27 (v2.4) | 测试脚本默认 cfg 路径以 repo root 解析 | 依赖调用者 CWD=仓库根 | CI/手工从 `tools/` 目录运行时旧默认路径会变成 `tools/tools/...`；脚本应自包含地定位仓库配置 |
| 28 (v2.4) | §6.4 augment-aware projection 必须单独验逆变换路径 | 只用 `training=False` 的基础 calib 投影测试 | `training=False` 不走 world augment，无法覆盖 `data_augmentor` 累积 `lidar_aug_matrix` 与 `PixelAlign._apply_inverse_aug`；新增真实 KITTI 点云 world flip/rotation/scaling 后的 inverse projection 对拍，并保留不开 inverse 的负对照 |
