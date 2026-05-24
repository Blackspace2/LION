# LION 中引入地面先验交互的系统调研与设计建议

> 检索与整理日期：2026-05-13  
> 面向问题：在 `LION + V2X/点云检测` 中，能否把地面先验做成类似 Transformer 中 `Object Query <- Ground <- Self <- LiDAR` 的交互链路；以及这件事在 Mamba/LION 框架下怎样做更合理、更有效。

---

## 0. 先给结论

一句话结论：

**能做，而且值得做；但最优路线不是把 LION/Mamba 强行改成“纯 Transformer 式 query 网络”，而是做成“LION/Mamba 主干 + Ground Memory / Query 轻量交互头”的混合结构。**

更具体地说：

1. 你的想法在抽象上是对的。  
   地面先验属于一种**低熵、稳定、全局几何约束**，它非常适合先去约束 object-level decoding，而不是一上来就深度侵入整个 3D backbone。

2. 对当前仓库和你正在跑的 V2X/KITTI 链路来说，**最值得做的版本不是“全量换成 query detector”**，而是：
   - 保留当前 `LION + SECOND/AnchorHeadSingle` 主链路；
   - 新增一个 **Ground-aware Query Sidecar / Refinement Head**；
   - 让 query 先和 Ground Memory 交互，再和局部 LiDAR/BEV 特征交互，用它做二次精炼或并行预测。

3. 如果你想要更“论文感”的创新点，最强的方向不是“有没有注意力”，而是：
   - **Ground-conditioned Query Refinement**
   - **Ground-conditioned Selective Scan**
   - **Plane-aligned / Ground-aware Serialization**
   - **Dual-stream Ground-Scene Mamba**

4. 如果只问“和 Mamba 是否冲突”，答案是否定的。  
   **Mamba 负责低成本长程传播，Attention/Query 负责对象级显式检索。**  
   这两者在高维抽象上是互补而不是互斥。

---

## 1. 研究方法与证据等级

本次结论来自三类证据：

### 1.1 本地代码核查

已核查的本地实现包括：

- `pcdet/models/backbones_3d/lion_backbone_one_stride.py`
- `pcdet/models/dense_heads/transfusion_head.py`
- `pcdet/models/model_utils/transfusion_utils.py`
- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml`
- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml`

### 1.2 外部主源

优先使用以下主源：

- arXiv / OpenReview / NeurIPS / CVPR 官方页面
- 官方 GitHub 仓库
- 官方项目页 / 官方研究页

### 1.3 关于 `grok-search`

本轮已尝试调用 `grok-search`，但接口没有在可接受时间内返回稳定结果，因此没有把它作为主证据源。  
后续判断主要基于网页检索 + 主源交叉核查完成。

### 1.4 证据等级约定

- `已验证`：本地源码或论文/官方仓库明确支持
- `强推断`：多个来源一致，但需要你真正实现后实验验证
- `弱推断`：思路合理，但目前更偏研究建议

---

## 2. 先澄清一个非常关键的工程事实

你现在最常用的 V2X/KITTI 训练链路，**并不是 query-based detector**。

### 2.1 当前 V2X/KITTI 配置是什么

在本地配置里：

- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml`
- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml`

都明确写的是：

- `MODEL.NAME: SECONDNet`
- `DENSE_HEAD.NAME: AnchorHeadSingle`

也就是说，你当前主链路本质上是：

`DynamicVoxelVFE -> LION3DBackboneOneStride -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle`

这不是 DETR/TransFusion 那种显式 `Object Query` 解码框架。

### 2.2 LION 仓库里有没有 query detector

有。

本地仓库包含：

- `pcdet/models/dense_heads/transfusion_head.py`

而且部分 nuScenes 配置确实在用：

- `tools/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml`
- `DENSE_HEAD.NAME: TransFusionHead`

所以，**“LION 中做 Object Query 与 Ground 交互”在仓库能力上不是不可能，但对你当前 V2X/KITTI 分支不是原生路径，而是需要补一个 query 分支或者迁移 head。**

这个事实会直接决定方案排序。

---

## 3. 本地代码给出的结构性启发

## 3.1 LION 主干在做什么

从 `pcdet/models/backbones_3d/lion_backbone_one_stride.py` 看，LION 的核心思路是：

1. 稀疏体素特征被组织为窗口序列；
2. 通过 `FlattenedWindowMapping` 做 `x/y` 两个方向的序列化；
3. `LIONLayer` 内部把每个 group 的序列送给线性算子；
4. 该线性算子可以是：
   - `Mamba`
   - `RWKV`
   - `RetNet`
   - `xLSTM`
   - `TTT`

也就是说，LION 的核心不是 query 解码，而是：

**把稀疏 3D 体素交互问题转化为“局部窗口内的大组序列建模问题”。**

### 3.1.1 一个重要点

`LIONLayer` 中使用的是不同方向的序列顺序，例如 `direction=['x', 'y']`，这说明：

- LION 已经默认承认：**序列顺序会影响 3D 几何表达**
- 因此，后面如果做 `ground-aware serialization`，它在 LION 里是有天然落点的

这点很关键。

## 3.2 本地 TransFusionHead 在做什么

从 `pcdet/models/dense_heads/transfusion_head.py` 和 `pcdet/models/model_utils/transfusion_utils.py` 看，当前 `TransFusionHead` 的解码链路大致是：

1. 从 dense heatmap 取 top-k proposal 作为 query 初始化；
2. 给 query 加类别编码；
3. 为每个 query 在局部窗口里收集 key/value 特征；
4. 经过一个 `TransformerDecoderLayer`：
   - 先 self-attn
   - 再 cross-attn
   - 再 FFN

这说明两件事：

1. 你提出的“Ground -> Self -> LiDAR”顺序，在这个头里是可以非常自然地插进去的；
2. 本仓库已经有 query-based 交互代码，不需要从零发明。

## 3.3 一个实际工程判断

如果你要做“地面先验与 Object Query 交互”，**最现实的切点不是先改 LION backbone，而是先改 TransFusionHead 或模仿它写一个 sidecar query head**。

原因很简单：

- backbone 改动影响全局训练稳定性
- head 改动更局部，验证更快
- ground prior 本质上更像“约束对象解码”而不是“替代场景主干”

这属于 `已验证 + 强推断`。

---

## 4. Mamba 与 Transformer 的高维抽象对应关系

你说“我不太了解 Mamba，能不能引用 Transformer 那种套路把地面先验搞进去”，这个问题本身抓得很准。

关键不在于名字，而在于**信息交互机制**。

## 4.1 Transformer 的本质

Transformer 中的注意力，本质是：

- `Query` 指定“我想找什么”
- `Key` 表示“我这里有什么”
- `Value` 表示“我能提供什么”

所以 Transformer 非常适合：

- 对象级检索
- 稀疏交互
- 可解释的选择性融合

## 4.2 Mamba 的本质

官方 Mamba 实现和 HuggingFace `modeling_mamba.py` 都强调：

- `A, D` 是输入无关的
- `Δ, B, C` 是输入相关的

这意味着 Mamba 的核心不是显式 token-to-token 匹配，而是：

**输入驱动的状态更新与记忆选择**

换句话说：

- Transformer 更像显式内容寻址
- Mamba 更像选择性状态流

## 4.3 它们在 3D 检测里各自擅长什么

### Transformer / Query 更擅长

- 对象级交互
- 稀疏实体之间的显式关系建模
- “我这个 query 该看哪块区域”的可解释融合

### Mamba / SSM 更擅长

- 长序列
- 线性复杂度上下文传播
- 低成本地把大范围几何上下文扫过去

## 4.4 为什么这不是二选一

`MambaVision`（2024 arXiv，2025 CVPR）和 `HybridTM`（2025 arXiv / IROS 2025）都给出一个很清晰的共同结论：

**Mamba 和 Attention 的混合往往比谁完全替代谁更有效。**

官方/主源信息：

- `MambaVision` 明确指出：在后层加入若干 self-attention block 可以明显增强长程空间依赖建模能力。
- `HybridTM` 明确提出：Transformer 与 Mamba 的互补性值得在更细粒度上结合。

因此，如果你要把地面先验引入 LION，我认为高概率正确的路子不是：

- “把所有交互都换成 Mamba”

而是：

- “让 Mamba 负责场景级传播”
- “让 query/attention 负责地面到对象的显式注入”

这属于 `已验证 + 强推断`。

---

## 5. 一个关键研究事实：跨序列交互本来就是 Mamba 的薄弱项

这恰好解释了为什么你的“Ground -> Query”构想更适合放在 head，而不是一上来就塞进 backbone。

`Cross-attention Inspired Selective State Space Models for Target Sound Extraction`（arXiv:2409.04803）明确指出：

- Mamba 对单序列建模很强
- 但它原生不擅长像 cross-attention 那样捕获**不同序列之间**的依赖

论文提出 `CrossMamba`，本质上就是想把“query-key-value 式跨序列交互”重新嫁接回 SSM。

对你这个问题的直接启发是：

1. `Ground tokens` 和 `Object queries` 是两条不同语义序列；
2. 它们之间的耦合更接近 cross-attention 问题；
3. 所以你要么：
   - 继续保留 cross-attention
   - 要么引入类似 `CrossMamba` 的跨序列 SSM 机制

也就是说，**Ground -> Query 这一步天然不是“纯 Mamba 最舒服的地带”**。

因此我的排序是：

### 第一选择

`Ground Cross-Attn -> Query Self-Attn / Mamba -> LiDAR Cross-Attn`

### 第二选择

`Ground CrossMamba -> Query Mamba -> LiDAR Cross-Attn`

### 不建议作为第一版

直接把 ground 信息硬塞进 LION 全主干，希望它自己学会对象级解码

这是 `强推断`，但我认为很可靠。

---

## 6. 相关方向调研：对你最有价值的外部工作

下面不是泛泛列论文，而是按“对你这个问题有多直接”的顺序来整理。

## 6.1 Query 协作 / Query 交互范式

### 6.1.1 QUEST: Query Stream for Practical Cooperative Perception

来源：

- arXiv: https://arxiv.org/abs/2308.01804
- 官方页：https://github.com/leofansq/QUEST

价值：

- 它明确把 **query 当作合作感知中的交互载体**
- 核心优势不是只看性能，而是：
  - 交互更可解释
  - 传输更灵活
  - 对丢包更鲁棒

对你的启发：

- 你不一定非要把“地面先验”变成 dense map 再和整张 BEV 做大规模融合；
- 完全可以把地面摘要压缩成 **Ground Queries / Ground Memory Tokens**；
- 然后只让对象 queries 去读它们

这和你提出的直觉非常一致。

### 6.1.2 CoopDETR: A Unified Cooperative Perception Framework for 3D Detection via Object Query

来源：

- arXiv: https://arxiv.org/abs/2502.19313

价值：

- 更进一步把 object query 作为 3D cooperative perception 的核心交互单位
- 说明 query-based object-level fusion 不是小众玩法，而是正向趋势

对你的启发：

- “Ground prior 先和 object query 交互”在研究叙事上是顺的；
- 你可以把它表述成：**从 agent-level query cooperation 迁移到 geometry prior-level query cooperation**

这是一个不错的论文包装角度。

## 6.2 Mamba 在协同感知里的落点

### 6.2.1 CoMamba

来源：

- arXiv: https://arxiv.org/abs/2409.10699
- 代码：https://github.com/taco-group/CoMamba
- 项目页：https://taco-group.github.io/CoMamba/

已核查源码结论：

- `point_pillar_opv2v_comamba.py` 里，Mamba 主要被放到 **fusion_net**
- `mamba_V2.py` 里的 `MambaFusionEncoder` 主要在做 **多车 / 多 agent BEV 特征融合**
- 它更像一个 **attention-free feature fusion block**

这意味着：

**CoMamba 证明了 “Mamba 适合做大规模 BEV / 多实体特征融合”，但它并没有证明 “Mamba 最适合做 object query 解码”。**

对你的启发：

- 地面先验如果要进入 Mamba，更适合先作为一种 **feature memory / fusion branch**
- 而不是一上来取代 query decoder

### 6.2.2 SparseCoop（2025）

来源：

- arXiv: https://arxiv.org/abs/2512.06838

价值：

- 它强调 `kinematic-grounded queries`
- 这里的 “grounded” 不是地面语义，而是几何/运动状态约束

对你的启发：

- object query 最终能否稳定，不只看 appearance feature；
- query 若被显式赋予几何结构（位置、速度、高度、对地关系），会更容易学

这和你要注入地面先验的目标非常一致。

## 6.3 Mamba 在点云/3D 检测中的代表方法

### 6.3.1 PointMamba / Point Cloud Mamba / Point Mamba

来源：

- PointMamba: https://arxiv.org/abs/2402.10739
- Point Cloud Mamba: https://arxiv.org/abs/2403.00762
- Point Mamba: https://arxiv.org/abs/2403.06467

共同结论：

- Mamba 在点云里真正的关键问题不是“能不能跑”
- 而是：
  - 如何序列化
  - 如何保留局部几何
  - 如何减少 3D 到 1D 的结构损失

对你的启发：

- 如果你引入 ground prior，最自然的一个方向不是直接多加一层注意力；
- 而是让 ground prior 参与 **序列化规则、位置编码、局部几何编码**

这是“更 Mamba 味”的创新点。

### 6.3.2 Voxel Mamba

来源：

- arXiv: https://arxiv.org/abs/2406.10700
- 代码：https://github.com/gwenzhang/Voxel-Mamba

已核查源码结论：

- 使用 Hilbert curve 模板进行 3D 序列化
- 主干是 `Dual-scale State Space Models Block`
- 没有走 query 解码路线
- 重点放在：
  - group-free serialization
  - dual-scale Mamba
  - 通过序列化策略尽量保住 3D spatial proximity

对你的启发：

1. **地面先验完全可以作用在“序列顺序”上**
2. 例如：
   - plane-aligned serialization
   - height-band serialization
   - ray-aligned + ground-first serialization

也就是后面我会重点建议的 `Ground-aware Serialization` 方向。

### 6.3.3 3DET-Mamba

来源：

- NeurIPS 2024 PDF / OpenReview 摘要可检索

关键点：

- 它不只是 backbone Mamba
- 它还提出了 **Query-aware Mamba module**

这非常关键，因为它说明：

**“Query 和 Mamba 可以共存，而且 query-aware 的 SSM 解码在 3D 检测里是有人认真做过的。”**

对你的启发：

- 你提出的 `Object Queries -> Ground -> Self -> LiDAR` 并不违背 Mamba 路线；
- 反而可以把它理解为：
  - 在 query-aware decoding 中引入 ground-conditioned memory

### 6.3.4 MambaDETR

来源：

- arXiv: https://arxiv.org/abs/2411.13628

关键点：

- 用 SSM 处理 query-based temporal modeling
- 说明 query 不是只能和 attention 绑定，也可以和 Mamba 绑定

对你的启发：

- 未来如果你觉得自注意力太重，可以把 query 序列内部的 self-interaction 改成 Mamba；
- 但 `Ground -> Query` 这一步仍然更建议保留显式 cross 机制

## 6.4 Mamba-Transformer 混合架构的结论

### 6.4.1 MambaVision

来源：

- arXiv: https://arxiv.org/abs/2407.08083
- NVIDIA Research: https://research.nvidia.com/publication/2025-06_mambavision-hybrid-mamba-transformer-vision-backbone
- 代码：https://github.com/NVlabs/MambaVision

核心启发：

- 在视觉任务中，纯 Mamba 不一定总是最优
- 在后层加入 attention blocks 能显著增强空间依赖建模

对你的问题的翻译是：

- 如果你要把地面先验作为一种更强的空间结构约束，
- 那么在靠近检测头的位置加入少量 attention 很可能比“全纯 Mamba”更值

### 6.4.2 HybridTM

来源：

- arXiv: https://arxiv.org/abs/2507.18575

核心启发：

- 它强调更细粒度的内层混合
- 不是粗暴地“前几层 Mamba，后几层 Transformer”

对你的启发：

- 你可以把 Ground interaction 做成一个 **内嵌的轻量混合单元**
- 不必把整个检测头/主干完全替换

### 6.4.3 A2Mamba

来源：

- arXiv: https://arxiv.org/abs/2507.16624

核心启发：

- 把 attention maps 作为对 SSM hidden states 的增强信号

对你的启发：

- 地面信息不一定非得作为 token；
- 也可以先形成一张 `ground attention map / ground confidence map`，
- 再去调制 Mamba/BEV feature flow

这给了你一个比 token 交互更轻量的备选方向。

## 6.5 地面先验本身的价值：不是只为了 gt sampling

### 6.5.1 Det6D

来源：

- arXiv: https://arxiv.org/abs/2207.09412

结论：

- 传统 3D detection 广泛依赖 flat-world assumption
- 非平路面会明显破坏检测
- local ground constraint 对姿态估计和鲁棒性确实有帮助

这说明地面先验绝不是“锦上添花”。

### 6.5.2 Local Ground-aware Surface Representation

来源：

- arXiv: https://arxiv.org/abs/2002.00336

结论：

- 相比单一全局平面，局部自适应地面表示更准确

对你的启发：

- 如果 DAIR-V2X 基础设施场景足够稳定，scene-level plane 是一个高性价比起点；
- 但真要做更强模型，后面应该从单一平面升级到：
  - local plane
  - slope field
  - roughness field
  - ground confidence map

---

## 7. 从这些证据反推：你在 LION 里最应该做什么

## 7.1 核心判断

### 判断 A

**“Ground -> Query -> LiDAR” 这条信息流是合理的。**

原因：

- 地面是稳定几何先验
- query 是对象级表示
- 先让 query 被地面约束，再去吸收高频 LiDAR 细节，符合 coarse-to-fine 逻辑

### 判断 B

**第一版不要把 ground prior 直接深侵入整个 LION backbone。**

原因：

- 你当前 V2X 主链路不是 query detector
- LION 主干本来就较复杂
- 地面先验更像解码约束，而不是替代场景主干的主信息源

### 判断 C

**最优首发方案是“Ground Memory + Query Sidecar / Refinement”**。

原因：

- 与当前工程形态兼容
- 与外部研究趋势一致
- 创新点集中且可控
- 论文叙事清楚

这三个判断都属于 `强推断`，但我认为是当前最稳的路线。

---

## 8. 方案空间排序

下面按“研究价值 / 工程可行性 / 与当前仓库兼容性”综合排序。

## 8.1 方案 S1：Ground-aware Query Sidecar（最推荐）

### 方案描述

在当前 `LION + AnchorHeadSingle` 主链路外，再新增一个轻量 query 分支：

1. 用当前 dense head 先产生候选中心/候选框；
2. 从 top-k 候选生成 pseudo object queries；
3. 构建 `Ground Memory Tokens`；
4. 执行：
   - `Q <- Cross(GroundMemory)`
   - `Q <- Self(Q)` 或 `Q <- Query-Mamba(Q)`
   - `Q <- Cross(Local LiDAR/BEV Features)`
5. 输出 refinement 偏移、score refine，或者并行的第二预测头

### 为什么它最适合你

1. 不要求你把当前 SECOND/anchor pipeline 推倒重来
2. query 思路得以落地
3. ground prior 被放到对象级显式交互里，解释性最好
4. 后续可以无缝替换里面的 self-attn 为 Mamba

### Ground Memory 应该包含什么

建议最少包含以下地面相关描述：

- `d_plane`：点/体素到地平面的有符号距离
- `h_above_ground`：对地高度
- `ground_confidence`：来自 linefit/局部统计的地面置信度
- `ground_density`：地面点密度
- `roughness`：局部粗糙度
- `slope`：局部坡度或法向变化

这些量比单一 `is_ground` 更有信息。

### Query 怎么来

因为你当前是 anchor 头，不是 query 头，所以有两个可选来源：

#### 方案 S1-A

从 dense classification heatmap / objectness map 取 top-k 作为 pseudo query

#### 方案 S1-B

从 NMS 前高分 anchors 取 top-k，做一个 query embedding

我更推荐 S1-A，因为更轻，更接近 `TransFusionHead` 的思路。

### 我对这个方案的评价

- 创新性：高
- 工程可做性：高
- 与当前仓库兼容性：高
- 论文叙事清晰度：高

这是我认为最值得先做的。

## 8.2 方案 S2：迁移到 TransFusionHead，再做 Ground-aware Decoder

### 方案描述

直接把当前 V2X/KITTI 分支从 `AnchorHeadSingle` 迁移成 `TransFusionHead` 风格：

- query 初始化
- ground cross
- self
- lidar cross

### 优点

- 你的原始想法最原汁原味
- 结构最干净
- 论文表达最直接

### 缺点

- 工程迁移成本更高
- 训练稳定性、标注形式、target assigner 都可能要重调
- 对当前 V2X/KITTI 分支来说不是小修小补

### 我对这个方案的评价

- 创新性：高
- 工程可做性：中
- 与当前仓库兼容性：中

适合第二阶段，而不适合第一枪。

## 8.3 方案 S3：Ground-conditioned Selective Scan（更“纯 Mamba”）

### 方案描述

不显式构建 query decoder，而是直接把地面先验注入 LION/Mamba 的 selective scan。

可做的三个子方向：

#### S3-1 Ground-conditioned Positional Descriptor

把当前 LION 的位置编码从纯 `(x,y,z)` 扩展为：

- `(x,y,z,d_plane,h_above_ground,slope,roughness,ground_conf)`

这是最低风险版。

#### S3-2 Ground-conditioned Selective Parameters

用 ground prior 去调制 Mamba 的输入投影，间接影响 `Δ, B, C` 对状态更新的选择性。

直觉上等价于：

- 地面附近的 token 该“记住什么”
- 高于地面的 token 该“忘掉什么”

#### S3-3 Ground-aware Scan Ordering

改变序列化顺序，让 scan 更符合几何结构，例如：

- ground-first then near-ground objects
- same-height-band scan
- plane-aligned scan
- ray-aligned + ground band scan

### 优点

- 更像真正的 “Mamba 版地面先验融合”
- 研究味道很强

### 缺点

- 训练风险更高
- 排查问题更难
- 可解释性不如 query 侧交互直接

### 我对这个方案的评价

- 创新性：很高
- 工程可做性：中低
- 作为第一阶段：不推荐
- 作为第二篇/第二阶段：很值得

## 8.4 方案 S4：Ground Map / Ground Attention Map 调制 BEV（最保守）

### 方案描述

不引入 query，只在当前 `BEV backbone -> dense head` 里增加一条地面分支：

- 生成 `ground feature map`
- 用它去门控或调制 `spatial_features_2d`

比如：

- `F' = F + Gate(F, G) * G`
- 或 `F' = Conv([F, G])`

### 优点

- 改动最小
- 训练最稳

### 缺点

- 创新性相对一般
- 很难把你最初“Object Query 与 Ground 交互”的思想表达完整

### 我对这个方案的评价

- 适合作为消融对照
- 不建议作为主创新点

---

## 9. 如果按你的原始想法来，最合理的模块形态是什么

我给出一个我认为最平衡的设计：

## 9.1 推荐模块：Ground-Conditioned Query Refinement Block

定义：

### 输入

- `Q0`：来自 dense heatmap / anchors 的 top-k pseudo queries
- `G`：Ground Memory Tokens
- `L`：局部 LiDAR/BEV 特征 token

### 计算顺序

1. `Q1 = CrossAttn(Q0, G)`
2. `Q2 = SelfAttn(Q1)` 或 `QueryMamba(Q1)`
3. `Q3 = CrossAttn(Q2, L)`
4. `Q4 = FFN(Q3)`
5. 输出：
   - score refine
   - center refine
   - size/orientation refine

### 为什么这个顺序好

#### 先 Ground

先让 query 知道：

- 它离地多高
- 该不该贴地
- 当前地面是否平缓

这样可抑制：

- 飘框
- 埋地
- 低矮类与地面噪声混淆

#### 再 Self

让 query 之间做互斥、去重、类别关系协调。

#### 最后 LiDAR

再去读取高频几何和边界细节。

这条链路在检测逻辑上是顺的。

## 9.2 如果你坚持更 “Mamba”

那么把第 2 步的 `SelfAttn(Q1)` 改成 `QueryMamba(Q1)` 是最自然的。

原因：

- query 序列通常不长
- 但 Mamba 可以更便宜地做多层交互
- 这一步的风险也比把 cross-attn 全部替换掉小

所以我的建议是：

### 第一版

`Ground CrossAttn -> Query SelfAttn -> LiDAR CrossAttn`

### 第二版

`Ground CrossAttn -> QueryMamba -> LiDAR CrossAttn`

### 第三版

再考虑 `Ground CrossMamba`

---

## 10. Ground Memory 到底该怎么构建

这是成败关键之一。

## 10.1 不建议只用“一个场景级平面参数”

scene plane 很适合：

- gt sampling
- 基础几何约束

但不太适合直接做强表达的交互，因为信息量太小。

## 10.2 我建议的三层表达

### 第 1 层：全局平面

由 `linefit -> ground points -> plane fitting` 得到：

- 平面法向
- 平面截距

作用：

- 全局对地高度
- 先验对齐

### 第 2 层：局部地面统计

在 BEV 网格或 voxel 上构建：

- ground occupancy
- ground confidence
- local slope
- local roughness
- local density

作用：

- 解决单平面无法表达局部起伏的问题

### 第 3 层：query-conditioned selection

不是所有地面 token 都给每个 query 看，而是：

- 只取 query 周围一定半径内的 ground tokens
- 或者按地面置信度 / 几何相关性 top-k

这能显著减少噪声与计算量。

## 10.3 一个很有研究价值的点

可以借鉴 `Where2comm` 的空间置信思路，把 Ground Memory 做成：

- `where-to-read-ground`

而不是：

- `read-all-ground`

换句话说，让 query 自己决定该关注哪些地面区域。

这个点我认为是一个不错的小创新。

---

## 11. 更适合 LION/Mamba 的创新点，不只是“加一个 attention”

如果只加 attention，能做，但不够“LION/Mamba 化”。  
真正更贴合这套框架的创新，我认为有下面四个。

## 11.1 创新点 A：Ground-aware Serialization

### 核心思想

当前很多 Mamba 3D 方法都在解决一个核心矛盾：

- 3D 结构是空间的
- Mamba 读的是 1D 序列

那么，地面先验最自然的一种用法就是：

**让序列更尊重地面几何。**

### 可做形式

- 先按 height band 排序，再按 x/y
- 先 ground，再 near-ground object，再 higher object
- 沿地平面切向方向扫描
- 借鉴 `RayMamba` 的 ray-aligned 思路，把序列化与传感器观测几何结合

### 为什么它值得

- 很 Mamba
- 不只是“多一个特征”
- 能直接作用于状态传播路径

这是我认为最有味道的一个方向。

## 11.2 创新点 B：Ground-conditioned Positional Encoding

LION 当前已经有 position embedding。  
你完全可以把它升级成：

- 3D absolute position
- relative-to-ground position
- local slope code
- roughness code

即把“位置”从欧式坐标扩展为“几何语义位置”。

这是一个低风险高回报点。

## 11.3 创新点 C：Ground-conditioned State Update

更进一步，可以让地面先验通过门控影响 selective scan：

- 地面 token 更偏长期记忆
- 明显离地的 object token 更偏短期突显
- 噪声区域更强抑制

这相当于让 Mamba 的“记忆策略”变得地形感知。

## 11.4 创新点 D：Dual-stream Ground-Scene Mamba

做两条流：

- `Ground Stream`
- `Scene/Object Stream`

再通过轻量门控融合。

这和很多双流结构的直觉一致，但比直接拼接更优雅。

---

## 12. 我对“先和 Ground 混合编码后的特征 cross，再 self，再和点云 cross”的具体评价

这个链路本身，我给出如下判断。

## 12.1 哪部分最强

最强的是第一步：

**Query 先和 Ground 交互**

因为这是你区别于普通 query detector 的核心。

## 12.2 哪部分可替换

第二步：

- `SelfAttn`

这一步最适合尝试替换为：

- `Query-Mamba`

因为这一步本质上是 query 序列内部交互，Mamba 完全有机会胜任。

## 12.3 哪部分不建议过早替换

第三步：

- `Query <- Cross(LiDAR Features)`

这一步短期内不建议去掉显式 cross-attn。  
原因是它最直接、最稳、最容易训。

## 12.4 所以最推荐的混合顺序是

### 推荐版 V1

`Ground CrossAttn -> Query SelfAttn -> LiDAR CrossAttn`

### 推荐版 V2

`Ground CrossAttn -> QueryMamba -> LiDAR CrossAttn`

### 进阶版 V3

`Ground CrossMamba -> QueryMamba -> LiDAR CrossAttn`

### 高风险版 V4

`Ground-conditioned LION backbone + minimal query refinement`

---

## 13. 如何和你现有的 linefit/plane 方案真正打通

你前面已经提出：

- 用 `linefit` 快速提地面
- 场景级提一次 plane
- 用于 gt sampling

这个思路我仍然认为非常对。  
但如果要进入模型，建议把它从“单一平面文件”升级成“Ground Representation Pipeline”。

## 13.1 第一步：保留 scene-level plane

作用：

- gt sampling
- 基础对地高度
- 全局几何基准

## 13.2 第二步：生成 per-frame / per-BEV ground descriptors

建议为每帧额外缓存：

- `plane_distance_map`
- `ground_conf_map`
- `roughness_map`
- `density_map`

这些图可以离线生成，训练时直接读。

## 13.3 第三步：构造 Ground Memory

做法例如：

1. 把 `ground_conf_map` 与 `spatial_features_2d` 融合
2. 按置信度 / query 相关性选 top-k cell
3. 投影成 `Ground Tokens`

## 13.4 第四步：接 query refinement

这才真正实现你最初的想法，而不是只停留在 gt sampling。

---

## 14. 一个对当前 V2X/LION 分支最现实的实现路线

如果让我替你拍板技术路线，我会这么排。

## 14.1 第一阶段：低风险验证

目标：

验证“地面先验进入网络”是否真的有收益

做法：

1. 完成 `linefit + plane fit`
2. 先启用 `gt sampling` 的 `USE_ROAD_PLANE`
3. 同时生成简单 ground BEV descriptors
4. 在当前 anchor head 前加一个轻量 ground gating 分支

意义：

- 验证地面信息总体有没有用
- 不引入 query 迁移成本

## 14.2 第二阶段：主创新

目标：

实现你想要的 query-ground 交互

做法：

1. 基于当前 dense heatmap / objectness 取 top-k pseudo queries
2. 实现 `Ground-Conditioned Query Refinement Block`
3. 用 refine head 做二次预测

意义：

- 能保住你当前 V2X-LION 主链路
- 能把 object query 思想真的用起来

## 14.3 第三阶段：更 Mamba 化

目标：

把“只是加注意力”升级成“Ground-aware Mamba”

做法：

1. Query self-interaction 改成 QueryMamba
2. 加入 ground-conditioned positional descriptor
3. 尝试 ground-aware serialization / scan ordering

---

## 15. 我认为最有价值的创新点清单

按我个人排序如下。

## 15.1 最推荐：Ground-Conditioned Query Refinement

原因：

- 贴合你的原始想法
- 与当前代码兼容
- 论文表达清楚
- 好做消融

## 15.2 第二推荐：Ground-aware Serialization

原因：

- 非常贴 Mamba
- 不落俗套
- 能和 LION/Voxel-Mamba/RayMamba 这条线形成强关联

## 15.3 第三推荐：Ground-conditioned Positional Descriptor

原因：

- 成本低
- 稳定
- 很容易成为有效增强

## 15.4 第四推荐：Ground-conditioned QueryMamba

原因：

- 研究味浓
- 有机会把 query 与 Mamba 真正接起来
- 但应放在第二阶段

## 15.5 第五推荐：Dual-stream Ground-Scene Mamba

原因：

- 很有研究价值
- 但第一版工程复杂度高

---

## 16. 我最终的建议

如果目标是“既要有研究价值，又要在你当前 LION/V2X 工程链路里尽快落地”，我建议你不要直接做“完整 TransFusion 化”，也不要第一枪就去改 selective scan 内核。

我建议的主线是：

### 主线方案

1. 用 `linefit + plane fitting` 完成 scene-level ground extraction  
2. 将 scene plane 升级为 frame-level / BEV-level ground descriptors  
3. 在当前 `SECONDNet + LION + AnchorHeadSingle` 基础上新增 `Ground-aware Query Sidecar`  
4. 执行：
   - `Pseudo Query Init`
   - `Query <- Ground Memory`
   - `Query <- Self / QueryMamba`
   - `Query <- Local LiDAR/BEV`
   - `Refine`
5. 稳定后，再探索：
   - ground-aware serialization
   - ground-conditioned position/state update

### 为什么是这条线

- 它不否定你对 Object Query 的直觉
- 它尊重当前 LION 的真实工程结构
- 它利用了 Mamba 和 Attention 的互补性
- 它最容易做出一条可讲、可复现、可扩展的研究路线

---

## 17. 最后一句判断

如果你问我：

> “在 LION 里把地面先验做成交互，而不是只用于 gt sampling，这件事值不值得？”

我的回答是：

**值得，而且我认为这是比“仅仅补 planes 文件”更有研究价值的一步。**

但如果你问：

> “第一枪该打在哪？”

我的回答是：

**不要先改最深的 Mamba 内核；先在检测头侧做 `Ground Memory -> Query` 的显式交互。**  
这是当前这套仓库、这条 V2X/KITTI 分支、以及你想追求的创新性之间，最平衡的方案。

---

## 18. 参考资料

### 18.1 本地代码

- `pcdet/models/backbones_3d/lion_backbone_one_stride.py`
- `pcdet/models/dense_heads/transfusion_head.py`
- `pcdet/models/model_utils/transfusion_utils.py`
- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml`
- `tools/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml`

### 18.2 论文与官方页面

- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-ND: https://arxiv.org/abs/2402.05892
- MambaVision: https://arxiv.org/abs/2407.08083
- NVIDIA Research on MambaVision: https://research.nvidia.com/publication/2025-06_mambavision-hybrid-mamba-transformer-vision-backbone
- HybridTM: https://arxiv.org/abs/2507.18575
- A2Mamba: https://arxiv.org/abs/2507.16624
- CrossMamba: https://arxiv.org/abs/2409.04803
- PointMamba: https://arxiv.org/abs/2402.10739
- Point Cloud Mamba: https://arxiv.org/abs/2403.00762
- Point Mamba: https://arxiv.org/abs/2403.06467
- Voxel Mamba: https://arxiv.org/abs/2406.10700
- 3DET-Mamba: NeurIPS 2024 official PDF / OpenReview 可检索
- MambaDETR: https://arxiv.org/abs/2411.13628
- CoMamba: https://arxiv.org/abs/2409.10699
- QUEST: https://arxiv.org/abs/2308.01804
- CoopDETR: https://arxiv.org/abs/2502.19313
- SparseCoop: https://arxiv.org/abs/2512.06838
- Det6D: https://arxiv.org/abs/2207.09412
- Local Ground-aware Surface Representation: https://arxiv.org/abs/2002.00336
- RayMamba: https://arxiv.org/abs/2604.02903

### 18.3 官方代码仓库

- LION: https://github.com/happinesslz/LION
- Voxel-Mamba: https://github.com/gwenzhang/Voxel-Mamba
- CoMamba: https://github.com/taco-group/CoMamba
- MambaVision: https://github.com/NVlabs/MambaVision
- PointMamba: https://github.com/LMD0311/PointMamba
- QUEST: https://github.com/leofansq/QUEST

