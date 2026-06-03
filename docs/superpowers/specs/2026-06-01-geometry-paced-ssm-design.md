# Geo-LION：几何配速 State-Space（GeoMamba）—— 让 Mamba 主干对 3D→1D 序列化损失鲁棒

- **日期**：2026-06-01
- **工作名**（临时）：Geo-LION；算子 `GeoMamba` (Geometry-Paced State-space)；备选 `MetricLION`
- **基线**：**官方 LION-Mamba**（仓库 `second_with_lion_mamba_64dim` 配置，默认窗口扫描序）。注：`bev_h/bev_h_t`（旧称 "S2"）等序列化变体在本仓库实测为**负收益**，**不作基线**，仅出现在 §1.2 动机与 §5.2 鲁棒性扫描序集合中
- **数据集**：KITTI 3D Object Detection（Car / Pedestrian / Cyclist），先在 KITTI 验证算法有效性；nuScenes / Waymo 子集留作后续泛化验证
- **定位**：顶会方法论文，mechanism-driven，**纯 LiDAR**，仅 KITTI 起步 + 充分 ablation 击穿
- **与 PaSS-LION 关系**：本工作是**点云主干**侧的独立创新，与 PaSS-LION（图像融合）正交。**不 import `pass_lion` 任何代码、不共享贡献**；split 算子只是两条线共享的*底层技术*。本文**不做** "叠加 PaSS" 的实验。两者都作用在 Δ 通道、乘性、可组合，但几何配速纯 LiDAR-only 独立成立。

---

## TL;DR

**单点贡献**：把序列化后的体素看成**连续 3D 场的不规则采样**，用相邻 token 的 **cell-normalized 3D 间距** 给 SSM 递推"计时"——曲线**撕裂**（大 gap）→ 拉长有效 Δ → `exp(ΔA)→0` 遗忘前驱、隔离不相关区域；**紧邻**（gap≈1）→ 保持基线。

核心实现是 selective scan 前对 Δ 的**一行 log-domain 乘子**：

```
Δ'_k = Δ_k · exp( clamp( γ · log ĝ_k , −C_lo , C_hi ) ) ,   γ = nn.Parameter(0.)  无约束
```

`γ=0` → 严格恒等初始化（退化为 split baseline，allclose ≈ fused 官方 LION-Mamba）、≈0 新参、kernel 友好、一键回退。

**核心证据**：*scan-order 鲁棒性*——GeoMamba 显著**降低 order-induced AP 方差**、**缓解坏序的性能坍塌**，把"重排序打不赢 baseline"的负结果转成正面证据：序列化损失被从**算子根部**处理，而非靠换曲线。

**契约**：`γ=0` 的 GeoMamba 与 split baseline 严格恒等（`allclose < 1e-6` 算子级、端到端按多层 fp32 累积放宽）；split baseline 与 fused 官方 LION-Mamba 复用 PaSS 线已验的等价性框架。

---

## 目录

- [§1 动机与命题：重排序的天花板](#1-动机与命题重排序的天花板)
- [§2 机制：GeoMamba 几何配速](#2-机制geomamba-几何配速)
- [§3 架构落点与 plumbing](#3-架构落点与-plumbing)
- [§4 训练与数值稳定性](#4-训练与数值稳定性)
- [§5 实验设计 / Ablation / 卖点](#5-实验设计--ablation--卖点)
- [§6 贡献与失败兜底](#6-贡献与失败兜底)
- [附录 A：实现落点](#附录-a实现落点)
- [附录 B：关键不变量速查](#附录-b关键不变量速查)
- [附录 C：决策追溯](#附录-c决策追溯)
- [附录 D：重排序负结果 provenance](#附录-d重排序负结果-provenance)

---

## §1 动机与命题：重排序的天花板

### 1.1 问题

把 3D 体素压成 1D 序列喂给 Mamba/SSM，递推对"相邻两 token 在 3D 里到底是真邻居、还是跨场景跳变（空间填充曲线的**撕裂**）"完全无感——每一步都用同样由 Δ 驱动的衰减 `exp(Δ_k A)`。两个失效模式：

1. **撕裂污染 / 假连续**：曲线断点处，空间上毫不相关区域的状态被当成邻居灌进当前体素。
2. **局部性稀释**：真正的 3D 邻居被打散到序列远处，被 `exp(ΔA)` 衰减掉，真实局部结构被低估。

### 1.2 命题（论文 thesis）

> **1D 曲线穿过 3D 空间必然产生撕裂，局部性全局不可保。重排序只是挪动撕裂的位置、消不掉它，也根本没改变"算子对几何无感"这件事——所以序列层面存在天花板，修复必须进算子内部。**

仓库实证（动机直接引用）：已实现的全部序列化变体——几何序族（Hilbert/Morton/3D/band，含旧称 "S2" 的 `bev_h/bev_h_t`）与 `topology` 图热扩散序——在 KITTI 上相对**官方 LION-Mamba 均为负收益（打平或更弱）**，重排序收益已见顶。这正是命题预言的结果，构成本文动机。

### 1.3 为何是 Mamba 独有

**full self-attention** 没有"沿 1D 前驱递推传播状态"的问题（全对偶、置换等变，不依赖 1D 邻接）；**局部/窗口 Transformer**（如 3D 检测里的 SST/SWFormer）可能有几何分组问题，但那不是同一种 **recurrent state contamination**——它没有把状态沿单条 1D 链传播、因而被撕裂污染的机制。本贡献针对的正是后者，是 SSM/Mamba 因"线性递推 + 序列时间"而独有的机制空白（避免"Transformer 无此问题"的过满表述）。

---

## §2 机制：GeoMamba 几何配速

### 2.1 cell-normalized 间距（各向异性安全）

每个 LIONLayer（含 PatchMerging 后的 stage），由 coords + `voxel_size·stride` + `pc_range` 还原体素中心 `p∈R³`（复用 `topology_context._compute_positions_xyz` 的算法，`topology_context.py:146`）。按当前序重排、分组后得到每组 token 序列 `p_1..p_G`，逐 token 间距用**逐轴 cell 归一化**后再取范数：

```
g_raw_k = ‖ (p_k − p_{k-1}) / (voxel_size_xyz · stride_xyz) ‖₂      # k ≥ 2
```

- 各向异性安全：相邻格天然 ≈1，跨 stage / 非等距体素都不含糊。
- **实现等价note**：`pc_range` 偏移在差分里抵消，该式在 stage 网格中恰等于**整数坐标差的 L2 范数**。
- **assert 边界（transition 级）**：`ĝ_k` 是到前驱的边属性。**real-real edge**（两端都 real：`valid[k] & valid[k-1]`）满足 `g_raw ≥ 1`（不同整数坐标）；**padding edge**（任一端 padding：`~valid[k] | ~valid[k-1]`，**含"前驱是 padding、当前是 real"** —— 反向 scan 污染的关键边）`g_raw` 可为 0，**绕过 raw assert**、按 I5 设 `ĝ=G_max`；**组首**按 I2 设 `ĝ=1`。assert 须按 real-real edge 掩码做，否则会在 padding 上误触发。

### 2.2 配速公式（默认 A1：log-domain 幂律 + cap）

```
# ĝ_k 是"到前驱 k-1"的边属性；valid_dir 为该方向的 group valid mask（[num_groups, G]）
real_real_edge[:,1:] = valid_dir[:,1:] & valid_dir[:,:-1]        # 两端都 real
padding_edge[:,1:]   = (~valid_dir[:,1:]) | (~valid_dir[:,:-1])  # 任一端 padding（含"前驱 padding→real"，反向污染的关键边）
ĝ_k      = clamp( g_raw_k , 1 , G_max )            # G_max ≈ 64；仅在 real_real_edge 用 raw
ĝ_k[:,0]             = 1                            # I2 组首无前驱
ĝ_k[padding_edge]    = G_max                        # I5（覆盖，不走 raw）
log_g_k  = log( ĝ_k )                              # assert g_raw≥1 只在 real_real_edge 上
log_pace = clamp( γ · log_g_k , −C_lo , C_hi )     # C_lo, C_hi > 0；bound pace factor 不受无约束 γ 影响
Δ'_k     = Δ_k · exp( log_pace )                   # selective_scan 前对 post-softplus Δ 逐元素乘
```

selective scan 语义：`state_k = exp(Δ'_k·A)·state_{k-1} + Δ'_k·B·u_k`，`A=−exp(A_log)<0`。
- **撕裂**（`ĝ_k` 大）→ `Δ'_k` 拉长 → `exp(Δ'A)→0` → 遗忘前驱；
- **紧邻**（`ĝ_k≈1`）→ `Δ'_k≈Δ_k` → 保持基线。

### 2.3 五条硬不变量（实现契约）

| # | 不变量 | 理由 / 实现 |
|---|---|---|
| I1 | **恒等初始化**：`γ=0 → exp(0)=1 → Δ'≡Δ` | GeoMamba 退化为 split baseline；与 fused 官方 LION-Mamba 复用 PaSS 等价性测试 |
| I2 | **组首 reset = 因子 1**：`k=1` 设 `ĝ_1=1`（非 raw 0） | 防 `0^γ=0` 压没组首 `ΔBu` 输入项；组内状态本就 group 独立 reset，组首不需额外遗忘 |
| I3 | **反向 gap 按反向 token 序重算**（不翻 gap 向量） | gap 是"到前驱"的**边属性**；每方向 `pos_dir=pos.flip(1)` 后重算 `g_dir[:,1:]=‖Δ‖, g_dir[:,0]=reset`。正向 `[1,d12,d23,d34]` → 反向应为 `[1,d34,d23,d12]`，而非 `flip=[d34,d23,d12,1]` |
| I4 | **γ 无约束** `nn.Parameter(0.)` | softplus 无法精确 0（破坏 I1），ReLU/square 在 0 处导数=0（init 梯度死）。无约束下 `∂Δ'/∂γ=Δ·log_g`，init 时只有撕裂 token（`log_g>0`）给梯度。**"γ 实测学到正值"反过来成为对假设的证据** |
| I5 | **padding 走 `ĝ=G_max` 而非清 state** | padding（组尾真体素重复）在 backward 先被扫、状态会流入真实 token。**padding 必须用 occurrence 级 group mask 识别**（`build_group_tokens_from_mapping` 的 `valid`，非 per-token mask）。padding edge 是 **transition 级**：`padding_edge[:,1:]=(~valid_dir[:,1:])|(~valid_dir[:,:-1])`（**任一端 padding，含"前驱 padding→real"**，否则漏掉反向污染的关键边）→ 设 `ĝ=G_max`（强 reset）仍走同一公式 → `γ=0` 仍严格恒等、不破坏 baseline allclose。**禁止**额外清 state（破坏等价契约）。valid-mask 单独做消融（§5） |

### 2.4 双向与组边界

- LION 把特征 `x.features[indices][flat2win].view(-1, group_size, dim)`（`lion_backbone_one_stride.py:860-861`），**每个 group 是独立序列**，SSM state 与 causal conv 在 group 边界天然重置。
- 双向：正/反两方向各自从 `pos_g`/`pos_g.flip(1)` 重算 gap（见 I3），按 `pass_mamba.py:254-263` 的 flip 套路对齐。
- `pos` 的 gather 照搬 PaSS 的 v_img 路径（`lion_backbone_one_stride.py:865-867` 同构），但 `ĝ` 必须在 gather 后 per-block 现算（见 §3.2）。

### 2.5 配速形式消融族（机制深度）

| 形式 | 定义 | 用途 |
|---|---|---|
| **A1 幂律**（默认） | §2.2 | 主方案，kernel 友好 |
| **A2 解耦遗忘门** | 只改状态转移、不改输入增益：`state_k=a_k·exp(ΔA)state_{k-1}+ΔBu_k`，`a_k=exp(−λφ(ĝ))` | 走 `_scan_ref_fallback`，**仅作精度机制对照，不进 latency 公平对比** |
| **A3 纯度量时间** | `Δ_k[c] = scale[c] · ĝ_k`（**用 §2.2 预处理后的 `ĝ`、非 raw**：组首=1、padding=G_max、real-real=clamp(metric)，与 A1 一致，避免再次压没组首 `ΔBu` 或与 padding 逻辑打架）。**完全替代** dt_proj；主消融用 **learned positive per-channel 向量** `scale∈R^{d_inner}` 保留 channel timescale，scalar 作更弱子变体 | 证"选择性 × 几何"二者都要；公平对照 |
| 间距定义 | 欧氏 / 曲线弧长 / 图测地；`G_max` / `C_lo,C_hi` 取值 | 间距度量消融 |

---

## §3 架构落点与 plumbing

### 3.1 新增模块（clean-room，不依赖 pass_lion）

- `pcdet/models/backbones_3d/geo_lion/geo_mamba.py`：`GeoMamba`（双向 split Mamba，复制 mamba split 路径主体，**不含图像/PaSS 逻辑**），在 `delta = softplus(...)` 之后、`selective_scan_fn` 之前插入 §2.2 配速乘子。`GeoMambaBlock`（`mamba_ssm.Block` 兼容、`supports_geo_pacing=True`），经 `LinearOperatorMap` 注册，配置 `operator.NAME='GeoMamba'` 选用。
- `_scan_ref_fallback` 同款参考实现：用于 CPU 等价性单测与 A2 形式。

### 3.2 LIONLayer 线程（gap 是边属性 → 必须 per-block 算）

> **关键**：`_forward_serial` 里每个 block 用各自的 `indices = mappings[self.direction[i]]`（同一 LIONLayer 可含多个不同 order 的 block）。`pos` 是 **token 属性**（与 order 无关、layer 级算一次即可，可照搬 PaSS v_img 的 gather）；但 `ĝ` 是 **edge 属性**，依赖具体 order，**绝不能 layer 级只算一份**，必须进 block loop 后用当前 `indices` 现算。

- 新增 `_build_geo_positions(x, coord_stride, geo_params)`：**只返回 per-token `pos` [N,3]**（token 属性，layer 级一次）。**不返回 `ĝ`、也不返回 per-token valid mask**。
- **padding 必须用 occurrence 级 group mask**：per-token mask 经 `flat2win` 后全 True（padding 位填的是真 token 索引，见 `lion_backbone_one_stride.py:121-127`），无法识别 padding 重复。改用现成 `build_group_tokens_from_mapping(order, flat2win, coords, batch_size, group_size)`（`serialization_diagnostics.py:32-52`，已在 `lion_improve.__init__` 导出），它返回 `(group_tokens, valid)`，`valid∈{T/F}^{[num_groups,G]}`，padding occurrence 处为 False。
- block loop 内（与 `x_features` 同款 gather）：`pos_g = pos[indices][flat2win].view(-1, G, 3)`；同时取该 order 的 `valid` group mask；再按 §2.1–2.3 + I2/I3/I5 现算 `ĝ_fwd / ĝ_bwd`（正向用 `pos_g`/`valid`、反向用 `pos_g.flip(1)`/`valid.flip(1)` **各自重算**）。padding 用 **transition 级边掩码** `padding_edge[:,1:]=(~valid_dir[:,1:])|(~valid_dir[:,:-1])`（任一端 padding，**含前驱 padding→real**），不是 `~valid`。
- 备选签名：`_build_geo_features` 返回 `dict[order_name] -> (ĝ_fwd, ĝ_bwd)`，但仍须按 layer 实际用到的 `self.direction` 逐 order 算，且 padding 同样走 group 级 `valid`。
- `coord_stride` 已是 `_forward_serial` forward 形参、非 PaSS 专属（`lion_backbone_one_stride.py:851`），直接拿；`voxel_size / point_cloud_range` 在 backbone `__init__` 存好、向下线程，**保证非 PaSS 配置也能拿到**（`_build_topology_context` 当前未传几何参数，故不复用该调用点）。
- **测试必覆盖**：同一 LIONLayer 两个不同 order 的 block，其 `ĝ` 必须不同（防退化为 layer 级一份）。

### 3.3 张量 shape 速查（layer-1，KITTI 默认）

| 张量 | shape | 来自 |
|---|---|---|
| `voxel_feat` | [N, 64] | VFE/上层 |
| `pos` (per-token, layer 级) | [N, 3] | `_build_geo_positions` |
| `pos_g` (per-block, 按当前 order) | [num_groups, group_size, 3] | `pos[indices][flat2win]` |
| `valid` group mask (per-block) | [num_groups, group_size] (bool) | `build_group_tokens_from_mapping`，padding=False |
| `ĝ_fwd / ĝ_bwd` (per-block) | [num_groups, group_size] | block loop 内现算（§2.1–2.3），padding 用 transition 级 `padding_edge`（任一端 padding） |
| `Δ` (post-softplus) | [num_groups, d_inner, group_size] | GeoMamba |
| `Δ'` | 同上 | 配速后 |

---

## §4 训练与数值稳定性

### 4.1 恒等初始化链与等价性测试

- `γ=0 → GeoMamba ≡ split baseline`；split ≈ fused 官方 LION-Mamba 复用 PaSS 线已验框架（算子级 `allclose < 1e-6`，端到端按多层 fp32 累积放宽，不把端到端 1e-6 当契约）。
- clean-room 复用三条 smoke：(a) `γ=0` GeoMamba vs split-no-geo `allclose < 1e-6`；(b) split-no-geo vs fused baseline 复用 PaSS 口径；(c) ref-scan vs CUDA-scan 数值一致（含 **dtype 等价测试**）。

### 4.2 warmup / 数值 / dtype

- **不需要 PaSS 式 ρ warmup**：`γ=0` 即恒等且梯度只来自撕裂 token，无 LoRA 死梯度问题。
- **log-domain cap**（§2.2）：`G_max` 单独不足以约束 `64^γ`（γ 无约束）；`clamp(γ·log_g, −C_lo, C_hi)` 保 init 恒等、梯度不死、又防早期某次更新把 pace factor 拉到数千倍。
- **dtype**：pacing 在 **fp32 / log 域**计算，再按 scan kernel 期望 dtype 传入（不擅自改 kernel dtype 契约）；以实现为准并加 dtype 等价测试。
- **监控**：`γ` 轨迹与符号、`max pace factor`、`Δ'/Δ` 分布、撕裂边贡献（`log_g` 直方图）。

### 4.3 优化器

- 主干 / SECOND head LR 不变；`γ` 第一批实验走主 LR。
- **保留消融**：`γ` 用 0.1×LR、或 `weight_decay=0` vs 主 wd —— **不在主实现里写死优化器策略**。
- 继续现有 cosine scheduler + EMA（EMA 自然覆盖 `γ`）。

---

## §5 实验设计 / Ablation / 卖点

### 5.1 主表（KITTI val/test）

Car / Ped / Cyc × easy/mod/hard，AP3D + APBEV：

| 组 | 方法 |
|---|---|
| **baseline（唯一锚点）** | **官方 LION-Mamba**（默认窗口扫描序，fused） |
| split 归因 baseline | 官方 LION-Mamba-split（geo off）——隔离 split tax |
| 近期 LiDAR SOTA | Voxel-Mamba 等 |
| **Ours** | **Geo-LION (GeoMamba)** = 官方 LION-Mamba + 几何配速（同序、仅加 GeoMamba，apples-to-apples） |

> 注：`bev_h/bev_h_t`（旧 "S2"）等序列化变体为**负收益**，**不作基线**；它们出现在 §1.2 动机与 §5.2 鲁棒性扫描序集合。主表对比严格为"同主干同序、仅加/不加 GeoMamba"，无 strawman。

### 5.2 杀手图 H（核心证据）· scan-order 鲁棒性

`{官方默认序, bev_h, bev_z, transposed, h3d, 随机序}` × `{vanilla split Mamba vs GeoMamba}`。这一步恰好把你那批"负收益"序列化变体**复用为鲁棒性扫描序集合**——废料变证据。

- **batch-local 不变量（硬约束）**：**所有** order（含随机序）必须 **preserve batch-contiguous LION mapping semantics**——`build_geometry_order_from_coords` 的 `window_key` 已把 `batch` 编进最高位、天然 batch 连续；随机序只能**在每个 batch 内**生成 permutation，**禁止全局 permutation**（否则破坏 `flat2win/win2flat` 的 batch 分段、甚至把不同样本扫进同一 group）。
- **必须新增显式 batch-local 校验**：现有 `_validate_serialization_mappings`（`lion_backbone_one_stride.py:808-815`）**只查 inverse + permutation、不查 batch 连续**，batch-crossing 的全局 permutation 能照样通过——"过 VALIDATE_MAPPING" 是错误安全感。新增 `_validate_batch_local(order, coords, batch_size)`：`coords[order][:,0]` 必须按 batch **分段非降连续**、每段长度 = `bincount(coords[:,0])`；等价检查"每个 group 不跨 batch"。
- **收敛后的预期结论**（科学性收敛，不过满）：**"GeoMamba 显著降低 order-induced AP 方差，并减少坏序列的性能坍塌。"** GeoMamba 只能 reset 坏连接、**不能合成缺失的局部性**，故不预期"随机序追平好序"。
- **seed 协议**：随机序做 **3–5 个固定 permutation seed**（per-batch 生成），报告 mean/std；其余可比实验按**单一固定训练 seed**（仓库策略 `3407→42→1024→810`，`docs/experiment_seed_policy.md`），避免 seed 漂移污染结论。

### 5.3 消融

- **配速形式**：A1 幂律(ours) / A2 解耦遗忘门(ref-scan, 仅精度) / A3 纯度量时间 / none(baseline)。
- **γ 作用域**：标量 / per-channel / per-stage。
- **间距度量**：欧氏 / 弧长 / 图测地；`G_max`、`C_lo/C_hi`。
- **注入 stage**：early / late / all。
- **valid-mask 三组**（回答"收益是否 padding 假象"）：
  1. baseline split 原行为；
  2. GeoMamba + padding-related `ĝ=G_max`；
  3. GeoMamba + true valid-mask scan / ref-scan 对照。

### 5.4 机制诊断（非 AP）

| 指标 | 期望 |
|---|---|
| 距离分段 AP（0-20/20-40/40-70m） | 远距增益更大 |
| 类别增益排序 | Ped > Cyc > Car |
| 撕裂率 vs 增益相关 | 增益集中在高撕裂组 |
| `γ` 轨迹 / 符号 | 学到正值（验证假设） |

### 5.5 可视化

- **主文**：`retention = exp(Δ'·A)` 与 `pace = ĝ^γ` 投回 BEV 热图（应在目标边界 / 远稀疏 / 扫描断点处亮）——这两者无需 kernel 中间 state 即可取。
- **补充材料**：撕裂点 SSM state 轨迹 有/无 pacing 对比（CUDA kernel 不暴露中间 state，故走 **ref-scan 小样本**）。

### 5.6 Latency

三栏（**只列测量目标、不预承诺数字**）：fused baseline / split-no-geo（split tax）/ split-geo。报告 **end-to-end** 与 **operator-level** 两套。A2(ref-scan) **不进** latency 表。

---

## §6 贡献与失败兜底

### 6.1 Contribution（拟）

- **C1** 把"3D→1D 序列化损失对 SSM"诊断为**不规则采样**问题，论证重排序天花板（引用仓库负结果当动机）。
- **C2** 几何配速 SSM：用 cell-normalized 物理 3D 间距给递推计时；log-domain 幂律 + cap，恒等初始化、≈0 参、kernel 友好、一键回退。
- **C3** scan-order 鲁棒性（降低 order-induced 方差、缓解坏序坍塌）作为"从根部解决"的证据。
- **C4**（**结果驱动、不预承诺 SOTA**）：在**官方 LION-Mamba** 基线上稳定提升，并在 KITTI 上达到 **competitive / SOTA-level performance**；是否 SOTA 待官方 test + 同协议对比确认。后续 nuScenes/Waymo 子集证泛化。

### 6.2 诚实定位

time-aware Δ（不规则采样 SSM）在时序文献有雏形、点云 Mamba 有人加 xyz 嵌入；本文贡献在**诊断 + 解法 + 鲁棒性证据 + 检测落地**的组合，及"几何配速使 SSM 对序列化选择鲁棒"的命题与证据。

### 6.3 兜底

- 若主表增益 < 0.5 AP：转写成 "geometry-paced selective scan as a serialization-loss remedy" 的**机制研究**（鲁棒性图 + 完整消融自成一篇），投 **WACV / BMVC / 3DV**。T-PAMI 需补 nuScenes/Waymo + 理论分析 + 更大规模鲁棒性证据才现实，不作为轻量兜底。
- 若 Ped/Cyc 不涨：查远处小目标 cell-normalized gap 是否饱和到 `G_max`；调 `G_max` / 间距度量。
- 若 `γ` 学不动或学到 ≈0：说明 SSM 对几何配速不敏感 —— 负结果转 mechanism 解释。

---

## 附录 A：实现落点

| 文件 | 内容 |
|---|---|
| `pcdet/models/backbones_3d/geo_lion/geo_mamba.py` | `GeoMamba` / `GeoMambaBlock`，clean-room split Mamba + §2.2 配速 + `_scan_ref_fallback` |
| `pcdet/models/backbones_3d/geo_lion/geo_pace.py` | cell-normalized gap、log-domain pace、I1–I5 不变量、双向重算 |
| `pcdet/models/backbones_3d/lion_backbone_one_stride.py` | `LIONLayer._build_geo_positions`（**只返回 per-token pos**；group valid mask 由 `build_group_tokens_from_mapping` 在 loop 内取）；`_forward_serial` block loop 内 per-order 现算 `ĝ` + `padding_edge`；新增 `_validate_batch_local`；backbone 存 `voxel_size/pc_range` |
| `tools/cfgs/kitti_models/geo_lion/` | `operator.NAME='GeoMamba'`、`GEO_PACING` 配置块、scan-order 鲁棒性扫描配置 |
| `tools/test_geo_mamba_identity_equivalence.py` | I1 `γ=0` 等价、dtype 等价 |
| `tools/test_geo_pace_backward_alignment.py` | I3 反向 gap `[1,d34,d23,d12]` 对齐 |
| `tools/test_geo_pace_group_padding.py` | I2 组首 reset、I5 padding edge=G_max（含"前驱 padding→real"边）、assert 仅 real-real edge |
| `tools/test_geo_pace_multiorder_gap.py` | Finding 1：同一 layer 两个不同 order 的 block，`ĝ` 必须不同 |
| `tools/test_geo_order_batch_local.py` | Finding 2：随机序 per-batch 生成；断言 `_validate_batch_local`（分段连续 + 段长=bincount + group 不跨 batch）；构造一个 batch-crossing order 必须被拒 |

## 附录 B：关键不变量速查

- I1 `γ=0` 严格恒等；I2 组首 `ĝ=1`；I3 反向重算（不翻 gap 向量）；I4 `γ` 无约束 init 0；I5 padding `ĝ=G_max`（不清 state）。
- `g_raw ≥ 1` **仅对 real-real edge**（`valid[k]&valid[k-1]`）成立 → `log_g ≥ 0`（assert 按该掩码）；组首 `ĝ=1`；**padding edge**（`~valid[k]|~valid[k-1]`，任一端 padding，含前驱 padding→real）`ĝ=G_max` 绕过 raw assert。
- pacing 在 fp32/log 域算，cap `clamp(γ·log_g, −C_lo, C_hi)`。

## 附录 C：决策追溯

- 机制选型：A 几何配速（用户锁定）；B 图结构状态接力（kernel 重、风险高，列 future work）；C 几何条件化 B/C（降级为消融通道）。
- 数据集：KITTI 起步（用户决策），nuScenes/Waymo 子集后续。
- 独立性：与 PaSS-LION 正交、不共享代码/贡献、不做叠加实验（用户决策）。
- 基线修正：去掉 "LION-Mamba(S2)" 强基线（它本身是负收益序列化变体），唯一基线改为**官方 LION-Mamba**，主表 apples-to-apples（同序仅加/不加 GeoMamba）；负收益序列化变体复用为 §5.2 鲁棒性扫描序集合（用户指出）。
- 外部 review（四轮）并入：log-domain + cap、cell-normalized 各向异性、反向 gap 重算、γ 无约束、padding via G_max、随机序预期收敛 + seed/std、A2 不进 latency、A3 清晰定义、可视化 retention/pace、latency 不预承诺、SOTA 语言结果驱动、兜底 venue 现实化、per-block gap（边属性）、随机序 batch-local、assert 仅 real-real、Transformer 表述收紧、负结果 provenance 表、A3 per-channel。
- **第四轮**：padding 改用 **occurrence 级 group mask**（`build_group_tokens_from_mapping`，`serialization_diagnostics.py:32-52`；per-token mask 经 flat2win 全 True 不可用）；新增显式 `_validate_batch_local`（`VALIDATE_MAPPING` 只查 permutation、不查 batch 连续）；A3 用预处理后 `ĝ` 而非 raw（组首/padding 一致）；附录目录补 D 并调序为 A→B→C→D。
- **第五轮**：padding 从 token 级 `~valid` 升级为 **transition 级 `padding_edge`**（`(~valid[k])|(~valid[k-1])`，**含"前驱 padding→real"**——反向污染的关键边，原 `~valid` 漏标）；assert 仅在 real-real edge（`valid[k]&valid[k-1]`）；附录 A 同步 `_build_geo_positions` 只返回 pos。

## 附录 D：重排序负结果 provenance

§1.2 的"重排序天花板"是核心动机，须有可追溯证据（动机证据链），否则论文写作时证据链不硬。下表 schema **待从已有 run 日志回填**（不是新跑实验）：

| order | config 路径 | seed | epoch / init | Car mod AP3D Δ | Ped mod AP3D Δ | Cyc mod AP3D Δ | 日志/eval 路径 |
|---|---|---|---|---|---|---|---|
| bev_h (旧"S2") | | | | | | | |
| bev_h_t | | | | | | | |
| bev_z / Morton | | | | | | | |
| h3d / z3d (3D) | | | | | | | |
| range_/height_ band | | | | | | | |
| topology (图热扩散) | | | | | | | |

- Δ 相对**官方 LION-Mamba** 同 seed/同协议。
- 填表后若发现某序其实为正收益，则 §1.2 动机与 §5.2 序集合需相应修订（诚实优先）。
