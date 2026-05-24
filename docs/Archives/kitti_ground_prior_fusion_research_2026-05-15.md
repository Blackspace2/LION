# KITTI 上基于 linefit 的地面先验融合研究与方案建议

日期：2026-05-15

## 1. 问题定义

目标是在当前 `LION + KITTI` 检测链路中，把 `linefit` 得到的地面分割结果变成一个真正进入检测模型的先验，而不是只停留在数据增强或可视化层面。

你关心的约束很明确：

- 方案要模块化，容易做消融
- 第一版最好稳，优先追求一个可靠的 `+1` 点级别增益
- 如果改到 Mamba，最好能从 selective SSM 的数学结构说清楚
- 尽量少改现有训练主链路

---

## 2. 当前代码现状

当前 KITTI 配置本质上是：

`DynamicVoxelVFE -> LION3DBackboneOneStride -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle`

对应文件：

- [second_with_lion_mamba_64dim.yaml](/root/project/LION/tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml)
- [dynamic_voxel_vfe.py](/root/project/LION/pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py)
- [lion_backbone_one_stride.py](/root/project/LION/pcdet/models/backbones_3d/lion_backbone_one_stride.py)
- [height_compression.py](/root/project/LION/pcdet/models/backbones_2d/map_to_bev/height_compression.py)

一个关键事实：

- `road_plane` 现在会被 [kitti_dataset.py](/root/project/LION/pcdet/datasets/kitti/kitti_dataset.py) 读入
- 但它主要只在 [database_sampler.py](/root/project/LION/pcdet/datasets/augmentor/database_sampler.py) 的 `USE_ROAD_PLANE` 路径里使用
- 也就是说，当前地面信息并没有进入 `VFE / LION / BEV backbone / dense head` 的主特征流

这意味着：如果后面我们把 ground prior 注入主干，实验归因会比较干净。

---

## 3. 文献给出的三条主结论

### 3.1 地面不是“无用背景”，而是强几何先验

- `MonoGround` 证明了 ground plane 作为几何约束可以改善 3D 定位，虽然它是单目方法，但结论对 KITTI 很有启发：地面先验本质上是在帮模型缩小合法 3D 解空间。  
  来源：CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Qin_MonoGround_Detecting_Monocular_3D_Objects_From_the_Ground_CVPR_2022_paper.html

- `3D Object Detection on Point Clouds using Local Ground-aware and Adaptive Representation of scenes' surface` 明确说明：局部地面表征优于单一全局平面。  
  对我们很重要，因为这意味着不要只给模型一个 `plane.txt`，而应该优先构建 local ground surface map。  
  来源：arXiv 2020  
  https://arxiv.org/abs/2002.00336

- `Can We Remove the Ground?` 的反向实验发现，现有 3D 检测器对目标下方和周围的地面点依赖很强。  
  这说明 ground 不只是可有可无的背景，而是 detector 已经在隐式使用的上下文。  
  来源：arXiv 2024  
  https://arxiv.org/abs/2410.00582

### 3.2 先验最稳的融合方式，通常是 side branch / conditional modulation

- `HyperDet3D` 说明 scene-conditioned prior 可以通过动态调制 detector 参数来带来收益。  
  这和我们想做的 `ground-conditioned` 思路非常一致。  
  来源：CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_HyperDet3D_Learning_a_Scene-Conditioned_3D_Object_Detector_CVPR_2022_paper.html

- `HDNET` 说明额外的几何/map prior 分支可以稳定提升 3D 检测。  
  类比上，它支持我们做 `BEV ground adapter` 这种 sidecar 设计。  
  来源：arXiv / CoRL spotlight  
  https://arxiv.org/abs/2012.11704

### 3.3 对 Mamba 来说，最合理的改法不是“全模型重写”，而是 hybrid 或调制

- `Mamba` 的关键点是 selective SSM 的参数可依赖输入。  
  工程上可推导出：如果地面先验能作为额外条件进入输入相关项，就可以做 `ground-conditioned selective scan`。  
  来源：arXiv 2023/2024  
  https://arxiv.org/abs/2312.00752

- `MambaVision` 说明视觉场景里，Mamba 与 attention 的混合比“纯 Mamba 替代一切”更稳。  
  所以第一版不建议直接重写整个 LION 主干。  
  来源：CVPR 2025  
  https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html

- `CrossMamba` 指出 Mamba 原生不擅长不同序列之间的依赖。  
  所以如果以后要做 `ground tokens -> object queries`，应当把它设计成显式 cross-style 模块，而不是假设普通 Mamba 自然就会学到。  
  来源：ICASSP 2025 / arXiv  
  https://arxiv.org/abs/2409.04803

- `3DET-Mamba` 已经在 3D 检测里证明了 Query-aware Mamba 是一条成立路线，但它更适合作为第二阶段研究，而不是你现在在 KITTI 上追求稳定增益的首发版本。  
  来源：NeurIPS 2024  
  https://openreview.net/forum?id=iOleSlC80F

---

## 4. 方案分层：从最稳到最重

下面按“侵入性从低到高、适合先拿增益到适合写论文”的顺序给方案。

## 4.1 方案 A：Point/Voxel 级地面几何特征

### 核心想法

给每个点增加一组相对地面的几何量，而不是只用绝对 `x,y,z,intensity`：

- `h_rel = z - z_ground(x, y)`：相对地面高度
- `d_ground`：到局部地面表面的距离
- `g_mask`：该点是否被 `linefit` 判为地面
- `g_valid`：所在 BEV cell 是否有可靠地面估计

然后让 `DynamicVoxelVFE` 和原有 point features 一起聚合。

### 为什么它可能有效

- KITTI 的三类目标都有稳定的“离地高度”统计规律
- 绝对 `z` 受路面坡度、车身俯仰、安装高度影响更大
- `h_rel` 更接近类别稳定几何量

### 建议实现

- 不要只用全局 `plane.txt`
- 用 `linefit` 的 ground points 构建局部 BEV 地面高程图 `z_ground(x, y)`
- 再回填到每个点上生成 `h_rel`

### 插入点

- dataset / processor：为点增加额外维度
- [dynamic_voxel_vfe.py](/root/project/LION/pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py)：直接接受新的 `points[:, 1:]`

### 适合做的消融

- `xyz+i` baseline
- `+ h_rel`
- `+ h_rel + g_mask`
- `+ h_rel + d_ground + g_valid`
- `h_rel` 替换绝对 `z` vs 与绝对 `z` 并存

### 风险

- 如果地面估计错误，硬二值 `g_mask` 可能伤到路沿、坡道、上下坡场景
- 所以更建议把 `g_mask` 当 soft feature，而不是 hard filter

### 评价

- 工程复杂度：低
- 消融友好：很高
- 预期收益：中等
- 论文新意：一般

---

## 4.2 方案 B：BEV Ground Adapter

### 结论先行

这是我最推荐你先做的版本。

### 核心想法

并行构建一组 ground BEV maps，然后作为 side branch 注入 `HeightCompression` 后的 `spatial_features`：

`F_bev' = F_bev + alpha * Adapter(G_bev)`

其中：

- `F_bev` 是原来的 BEV 特征
- `G_bev` 是 ground prior maps
- `alpha` 初始化为 `0`
- 或者让 `Adapter` 最后一层卷积 zero-init

这样一开始模型行为几乎等于旧模型，训练更稳，便于从已有 checkpoint 热启动。

### 推荐 ground maps

- `ground_height`
- `ground_density`
- `ground_valid_mask`
- `mean_height_residual`
- `max_clearance`
- `ground_ratio_in_cell`

### 为什么这条路最适合当前代码

- 你的 KITTI 主链路是 anchor-based，不是 query-based
- `HeightCompression` 后已经是标准 `B,C,H,W` 特征，最容易加 sidecar
- 旧权重可直接加载
- 开关最清晰，最适合做 ablation

### 插入点

- [height_compression.py](/root/project/LION/pcdet/models/backbones_2d/map_to_bev/height_compression.py) 后
- 或 [base_bev_backbone.py](/root/project/LION/pcdet/models/backbones_2d/base_bev_backbone.py) 前

### 建议的模块形态

- `1x1 conv -> 3x3 conv -> 1x1 conv`
- 或 very small UNet-like adapter
- 最后一层 zero-init

### 适合做的消融

- `+ ground_height only`
- `+ ground_height + valid_mask`
- `+ all ground maps`
- `concat fusion` vs `residual add fusion`
- `zero-init` vs `non-zero-init`

### 评价

- 工程复杂度：低到中
- 消融友好：最高
- 预期收益：高
- 论文新意：中等，但非常稳

---

## 4.3 方案 C：Ground-Conditioned Mamba

### 核心想法

这里不是重写 selective scan CUDA，而是在 `LIONLayer` 前后做 ground-conditioned 调制。

基于 `Mamba` 的输入相关 selective 参数，可以做如下工程推导：

- `Delta_t = softplus(W_delta x_t + U_delta g_t)`
- `B_t = W_b x_t + U_b g_t`
- `C_t = W_c x_t + U_c g_t`

这里：

- `x_t` 是当前 voxel token
- `g_t` 是与该 voxel 对齐的地面描述子

这不是论文原文直接给出的现成模块，而是基于 selective SSM 结构做的合理扩展推导。

### 为什么它有研究价值

- 数学上和 Mamba 的“输入驱动状态更新”是对齐的
- 比简单 concat 更像真正的 `ground-conditioned dynamics`
- 论文叙事会更完整

### 为什么不建议先上这一版

- 当前仓库里的 `LIONLayer` 直接调用 `MambaBlock`
- 如果你一开始就动 scan 内核或 block 内部，训练不稳定性、调参成本、debug 成本都会明显升高

### 更稳妥的第一版写法

先做 wrapper，而不是改内核：

- `x_t' = x_t + sigmoid(W_g g_t) * P(x_t)`
- 再把 `x_t'` 送给原始 `MambaBlock`

或者：

- `y_t = Mamba(x_t)`
- `y_t' = y_t + Gate(g_t, y_t)`

### 插入点

- [lion_backbone_one_stride.py](/root/project/LION/pcdet/models/backbones_3d/lion_backbone_one_stride.py) 的 `LIONLayer.forward`

### 适合做的消融

- pre-Mamba gate
- post-Mamba gate
- `Delta` only modulation
- `Delta+B+C` full modulation
- local `g_t` vs BEV pooled `g_t`

### 评价

- 工程复杂度：中到高
- 消融友好：中等
- 预期收益：中到高
- 论文新意：高

---

## 4.4 方案 D：Ground Memory / Query Sidecar

### 核心想法

如果你后续愿意从 anchor head 走向更显式的对象级解码，可以增加一个 sidecar query 模块：

1. 从 dense predictions 或 top-k BEV peaks 初始化 object queries
2. 从 ground BEV map 中抽取 ground tokens / ground memory
3. 先做 `query <- ground memory`
4. 再做 `query <- scene BEV/LiDAR features`
5. 输出 refine boxes 或并行 confidence

### 为什么这条路合理

- `CrossMamba` 和 `3DET-Mamba` 都支持“跨序列交互”和“query-aware Mamba”是值得做的
- 但这更像第二篇工作，或第一篇里的扩展实验

### 为什么不适合现在先做

- 当前 KITTI 主头是 `AnchorHeadSingle`
- 这条线会把问题从“ground fusion”升级成“detection head redesign”

### 评价

- 工程复杂度：高
- 消融友好：一般
- 预期收益：不确定但潜力高
- 论文新意：很高

---

## 4.5 方案 E：辅助监督，不改变主推理结构

### 可以加的辅助目标

- voxel / BEV groundness prediction
- `h_rel` 回归
- `ground surface reconstruction`
- `box_bottom_to_ground consistency loss`

### 作用

- 帮 backbone 学到更强地面几何表征
- 可作为 A/B/C 的附加项，而不是单独主方案

### 风险

- 如果只加辅助损失、不改推理融合，增益可能偏小

---

## 5. 推荐优先级

如果目标是先在 KITTI 上拿一个可靠增益，我建议顺序如下：

### 第一阶段：先做 B

先上 `BEV Ground Adapter`。

原因：

- 最小侵入
- 最容易复用旧权重
- 最容易做 clean ablation
- 最像 “先证明 ground prior 有用”

### 第二阶段：再做 A

在 B 有效后，再加 `h_rel / d_ground / g_mask` 的 point-level 特征。

原因：

- 这样可以区分 “地面作为 BEV 上下文有效” 还是 “地面作为 point 几何特征有效”

### 第三阶段：最后做 C

如果 B/A 已经证明地面先验有效，再做 `ground-conditioned Mamba`。

原因：

- 这时你的论文叙事会完整很多：
  - 先验本身有效
  - 简单融合有效
  - 再进一步，条件化 selective dynamics 更有效

---

## 6. 我对你当前项目的具体建议

## 6.1 先别把“plane 文件”当最终先验

`plane.txt` 适合：

- `USE_ROAD_PLANE` 数据增强
- 构造全局高度参考

但真正喂进 detector 时，更建议使用 `linefit` 的局部地面表面图，而不是只有单一平面。

## 6.2 第一版建议产物

离线缓存以下内容到 KITTI：

- 每帧 `ground_mask` 或 ground-labeled points
- 每帧 `ground_bev_maps`
- 可选：每点 `h_rel / d_ground / g_mask`

建议缓存目录类似：

- `training/linefit_ground/<frame>.npz`
- `training/linefit_ground_bev/<frame>.npz`

## 6.3 第一版推荐真正开做的模块

### 推荐 R1

`GroundBEVAdapter`

- 输入：`ground_bev_maps`
- 融合位置：`HeightCompression` 后
- 融合方式：zero-init residual

### 推荐 R2

`GroundAwarePointFeatures`

- 输入：`h_rel, d_ground, g_mask`
- 融合位置：`DynamicVoxelVFE`

### 推荐 R3

`GroundConditionedLIONGate`

- 输入：voxel 对齐的 `g_t`
- 融合位置：`LIONLayer` 前后

---

## 7. 最值得做的消融矩阵

建议至少做下面这组：

| ID | 方案 | 改动位置 | 预期作用 |
|---|---|---|---|
| B0 | baseline | 无 | 当前 LION-KITTI |
| B1 | `+ plane only` | augmentation only | 验证仅增强是否够 |
| B2 | `+ BEV ground height` | BEV | 最小先验注入 |
| B3 | `+ BEV full ground maps` | BEV | 验证 richer prior |
| P1 | `+ h_rel` | point/VFE | 验证相对高度是否有效 |
| P2 | `+ h_rel + g_mask` | point/VFE | 验证显式地面标签 |
| M1 | `+ post-Mamba gate` | LION | 最轻 Mamba 条件化 |
| M2 | `+ pre-Mamba gate` | LION | 对比调制位置 |
| M3 | `+ BEV adapter + Mamba gate` | BEV + LION | 验证是否互补 |

---

## 8. 最终建议

一句话版本：

先不要急着把 Mamba 改“重”。先用 `linefit` 生成局部地面 BEV 先验，做一个 zero-init 的 `BEV Ground Adapter`，这是最适合在 KITTI 上先验证“地面先验有效性”的路线；如果它真的能稳定涨点，再往 `point-level relative height` 和 `ground-conditioned Mamba` 推进。

如果只让我选一个“最可能先拿到稳定 +1”的版本，我选：

`linefit local ground surface -> BEV maps -> zero-init Ground Adapter after HeightCompression`

它的优点是：

- 与当前代码最匹配
- 最容易开关消融
- 最容易从旧 checkpoint 平滑开始
- 最容易把“ground prior 本身有效”这件事先证明出来

---

## 9. 参考链接

- Mamba: https://arxiv.org/abs/2312.00752
- MambaVision: https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html
- CrossMamba: https://arxiv.org/abs/2409.04803
- 3DET-Mamba: https://openreview.net/forum?id=iOleSlC80F
- HyperDet3D: https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_HyperDet3D_Learning_a_Scene-Conditioned_3D_Object_Detector_CVPR_2022_paper.html
- MonoGround: https://openaccess.thecvf.com/content/CVPR2022/html/Qin_MonoGround_Detecting_Monocular_3D_Objects_From_the_Ground_CVPR_2022_paper.html
- Local Ground-aware Surface Representation: https://arxiv.org/abs/2002.00336
- Can We Remove the Ground?: https://arxiv.org/abs/2410.00582
- HDNET: https://arxiv.org/abs/2012.11704
