# Lightweight Ground Defect Guidance

## 1. 背景

当前 `ground_guided_diffusion` 的做法，是把地面先验直接加到 sparse routing score 上：

```text
guided_score = learned_response + alpha * prior_bias
```

这条路线的问题是：

- 它直接改了 backbone 的决策路径
- 很容易把 baseline 已经学到的好解拉偏
- 对 `Car/Cyclist` 可能有局部收益，但对 `Pedestrian` 容易造成伤害
- `ground region prior` 太粗，容易把整片地面背景一起抬高

因此，更稳妥的创新路线不是“让 ground prior 直接决定选谁”，而是：

- 保留 baseline 主干决策
- 在输入点上注入最少量的地面几何特征
- 增加一个轻量的 BEV 地面缺陷分支
- 把“地面缺陷 / 接触异常”作为辅助几何线索反馈给检测分支

核心假设是：

- linefit 可以给出粗地面分割
- 目标一般位于地面上方，并与地面发生接触
- 点云中的地面观测通常是不完整的、局部的、稀疏的
- 目标会在局部可观测地面上形成缺口、扰动、边界异常或近地抬升

因此，与其把“地面区域”当成正先验，不如把“局部地面缺陷和接触异常”当成更接近目标的几何线索。

## 2. 创新点定义

建议把创新点收敛为：

**Lightweight Ground Defect Guidance for LiDAR 3D Object Detection**

一句话描述：

> 利用 linefit 导出的粗地面点级先验，在原始点上注入少量地面几何特征，并在 BEV 空间中构建一个轻量的地面缺陷引导分支，将由目标引起的地面缺陷、边界扰动和接触异常作为辅助几何线索，以残差式方式增强检测特征。

与当前版本的本质区别：

- 不是 `ground-region bias`
- 不是 `hard routing`
- 不是“先验替模型做决定”
- 而是“先验提供一个局部、轻量的几何参考”

## 3. 总体结构

整体网络分成两条分支：

1. 检测主干分支
   - 维持当前正常的 LION voxelization + 3D backbone + BEV backbone + dense head
   - 不再让地面先验直接改 top-k routing score

2. Ground defect 分支
   - 从 linefit 结果和点云中提取地面相关 BEV 几何图
   - 显式输出“地面缺陷 / 目标接触异常”响应图
   - 通过残差门控方式增强主 BEV feature
   - 分支必须轻量，不能明显增加显存和推理成本

推荐的高层流程：

```text
raw points
  -> add 3 ground-aware point features
  -> voxelization
  -> shared voxel / sparse encoding
  -> main LION branch -> F_bev
  -> lightweight ground branch -> D_defect, F_ground
  -> residual gated fusion
  -> enhanced F_bev'
  -> detection head
```

## 4. 输入先验设计

### 4.1 点级附加特征

在原始点云上只增加 3 个附加特征，再正常体素化：

- `is_ground`
  - linefit 预测该点是否为地面点
- `delta_z_to_ground`
  - 点到局部地面的高度差

其中最关键的是：

- `delta_z_to_ground`

因为这个量能区分：

- 地面点
- 近地非地面点
- 真正高于地面的目标点

这比单纯提供 `is_ground` 更有用。

### 4.2 显存与计算影响

这 3 个点级附加特征预计不会明显影响显存，原因是：

- 它们只是在原始点级输入上增加了 3 列
- 体素化之后，真正送入 backbone 的仍然是固定维度的 voxel / pillar feature
- 当前 `DynamicVoxelVFE` 的主要开销在于：
  - 点级临时特征拼接
  - PFN / VFE 激活
  - 后续 sparse backbone 激活
- 增加 3 个输入特征，只会扩大第一层 VFE 的输入宽度，不会改变后续主干的通道数

因此：

- 参数量会略增
- 点级前处理显存会略增
- 但总体显存主开销仍然在 sparse backbone 和 BEV backbone 上

结论：

> 在原始点云上增加 `is_ground`、`delta_z_to_ground`、`local_ground_height`，通常不会成为主要显存瓶颈。

### 4.3 地面 BEV 几何图

linefit 只能给到点级“是否是地面”的粗结果，而且点云本身对地面的覆盖并不完整。  
因此，Ground 分支不能假设场景地面是完整可观测的，而应把它当成：

- 稀疏的
- 局部有效的
- 带缺失的
- 只在部分区域可信的

建议从点云和 linefit 结果生成一组轻量地面相关 BEV map：

- `G_obs`
  - 观测到的地面占据图
- `H_g`
  - 地面高度图
- `H_res`
  - 离地高度残差统计图
- `N_ng`
  - 近地非地面点计数图
- `B_g`
  - 地面边界图
- `M_valid_ground`
  - 当前区域是否存在可用地面观测

这里不建议只保留一个 ground mask。  
真正对检测有帮助的是：

- 稀疏地面观测
- 局部地面高度
- 近地异常
- 边界扰动

后续 Ground 分支和 defect loss 都只应在有效区域内起作用，避免把“没看到地面”和“地面异常”混为一谈。

## 5. Ground 分支设计

Ground 分支不应做成一个重的重建网络，而应收敛成一个轻量的 defect-aware BEV 分支。

### 5.1 设计原则

必须满足：

- 分支轻量化
- 只使用少量输入通道
- 不依赖完整地面覆盖
- 不引入大的解码器
- 不成为新的显存主开销

推荐结构：

- 2 到 3 层小型 2D Conv block
- 通道数控制在 `16 -> 32 -> 32`
- 输出：
  - `D_defect`
  - `F_ground`

### 5.2 Defect / contact anomaly prediction

预测：

- `D_defect`

含义：

- 当前局部是否存在“地面可观测结构被破坏”的现象
- 这种破坏是否像由目标 footprint / contact 引起

这个图本质上应该响应：

- 地面缺失
- 接触边缘
- 局部抬升
- 近地障碍
- footprint 轮廓

它不是普通 anomaly map，而是**与目标几何相关的 defect map**。

## 6. 融合方式

不建议：

- 直接用 ground prior 去改 sparse score
- 直接 hard select voxel
- 直接替换主干 feature

建议采用残差门控融合：

```text
F_bev' = F_bev + gate(D_defect, M_valid_ground, H_res) * P(F_ground)
```

其中：

- `F_bev`
  - 主检测分支 BEV feature
- `F_ground`
  - ground 分支中间 feature
- `P(.)`
  - 轻量投影层
- `gate(.)`
  - 一个小型卷积门控头，输出 `[0, 1]`

关键设计：

- `gate` 初始值应接近 `0`
- 整体以 residual 方式注入
- 默认行为尽量接近 baseline

这样做的优点：

- 不破坏 baseline 的原始信息流
- 只在地面几何线索有用时增强主 BEV 表征
- 更容易训练稳定

## 7. 监督设计

建议采用“主任务监督 + 一个低权重辅助监督”的最简版本。

### 7.1 Object-footprint defect loss

根据 3D GT box 投影到 BEV，生成：

- footprint mask
- center heatmap
- contact band

然后把这些 supervision 用在 `D_defect` 上，使它更偏向学习“与目标有关的地面异常”，而不是泛化为任意缺洞都是前景。

建议：

- footprint 区域给予正响应
- 接触边界给予更高权重
- 仅在 `M_valid_ground` 区域内计算
- 整体 loss 权重设低，避免压过原检测目标

### 7.2 主检测监督

最终检测头仍然使用原有 detection loss，不必大改。

总 loss 可以写成：

```text
L = L_det + lambda_def * L_def
```

其中：

- `L_det`
  - 原始检测损失
- `L_def`
  - defect / contact 监督损失

建议初始权重：

- `lambda_def = 0.05 ~ 0.2`

更推荐先从：

- `lambda_def = 0.1`

开始。

## 8. 为什么这条路线比当前 guided 更稳

当前 guided 的主要问题是：

- 先验直接进入选点分数
- 一旦先验有偏差，就会误导主干
- 对小目标和稀疏区域尤其危险

而 Lightweight Ground Defect Guidance 更稳的原因是：

1. 它不直接篡改 backbone 的离散路由
2. 它只提供辅助几何上下文
3. 它强调的是“局部结构破坏”而不是“大块区域偏置”
4. 残差式融合可以天然保 baseline
5. 辅助 loss 很轻，不容易压过主任务

一句话概括：

> 当前 guided 是让先验替模型做决定；新方案是让先验以轻量、局部、残差的方式给模型提供几何参考。

## 9. 实现路径

建议按下面顺序推进：

### Stage A: 点级地面特征增强

只在输入点上增加：

- `is_ground`
- `delta_z_to_ground`
- `local_ground_height`

其余主干完全不改。

目标：

- 先验证地面几何信息作为输入特征是否有正收益

### Stage B: Ground auxiliary branch without fusion

增加 ground 分支，但先只做辅助任务：

- `D_defect`

暂时不与主 BEV feature 融合，只加辅助损失。

目标：

- 验证轻量 ground defect 分支是否能提升 backbone 表征

### Stage C: Residual gated BEV fusion

在确认 Stage B 不伤 baseline 后，再把 `F_ground` 以 residual gate 方式融入 `F_bev`。

目标：

- 验证地面分支提供的几何上下文是否能进一步提高检测效果
