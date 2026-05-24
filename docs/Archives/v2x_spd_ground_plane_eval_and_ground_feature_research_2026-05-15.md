# V2X-SPD Ground Plane, Large-ROI Eval, and Ground-Feature Research

日期: 2026-05-15

## 1. 任务范围

本次任务包含四部分:

1. 在原始 `V2X-SPD` 数据上用 `linefit` 生成 KITTI `plane` 文件, 并在 `strict3` 上通过软链接复用
2. 验证 `USE_ROAD_PLANE: True` 的数据链路和短续训 smoke
3. 对指定 scratch run 的所有编号 checkpoint 在大 ROI `[0, -60, -3, 140, 60, 2]` 下跑验证, 输出到 `eval_large_range`
4. 研究如何在现有 `LION-V2X` 的 `80 epoch` 训练基础上无感引入地面特征, 兼顾提点和论文叙事

---

## 2. Plane 生成

### 2.1 关键约束

`plane` 文件必须符合 OpenPCDet 的 KITTI 约定:

- 文件格式:

```txt
# Matrix
WIDTH 4
HEIGHT 1
a b c d
```

- 系数 `[a, b, c, d]` 表示平面方程:

```txt
a*x + b*y + c*z + d = 0
```

- 坐标系必须是 `rectified camera coordinate`
- 法向需要满足 KITTI 读入约定, 即归一化后 `b < 0`

这点非常重要。不能直接把 LiDAR 坐标系下拟合出的平面参数写进 KITTI `planes/*.txt`。

### 2.2 实现流程

脚本:

- [tools/generate_kitti_scene_planes.py](/root/project/LION/tools/generate_kitti_scene_planes.py)

输入配置:

- `linefit` 配置: `/root/project/LION/plugins/groundSeg/linefit/assets/config.toml`

核心流程:

1. 从 `strict3` 的 sample id 回溯到原始 `V2X-SPD/{training,validation}/{scene}` 的真实场景
2. 每个 scene 均匀采样若干帧
3. 用 `linefit` 在 LiDAR 点云上分出 ground points
4. 用标定把 ground points 从 LiDAR 坐标变到 `rect camera` 坐标
5. 在 `rect camera` 坐标系下做 scene-level plane fitting
6. 归一化并强制法向满足 `b < 0`
7. 将同一个 scene-level plane 写入该 scene 下所有帧的 `planes/<frame>.txt`
8. 在 `strict3` 侧通过扁平化软链接复用

### 2.3 产物布局

真实 plane 文件写在原始数据上:

- `/root/autodl-tmp/V2X-SPD/training/<scene>/planes/<frame>.txt`
- `/root/autodl-tmp/V2X-SPD/validation/<scene>/planes/<frame>.txt`

扁平化软链接:

- `/root/autodl-tmp/V2X-SPD-KITTI/shared/planes/<scene>_<frame>.txt`

`strict3` 训练目录复用方式:

- `/root/autodl-tmp/V2X-SPD-KITTI/strict3/training/planes -> ../../shared/planes`

### 2.4 生成结果

- scene 数: `57`
- plane 文件数: `9425`
- `shared/planes` 软链接数: `9425`

真实 plane 文件总占用:

- `755,035 bytes`
- `737.34 KiB`
- `0.72 MiB`

---

## 3. Plane 正确性验证

### 3.1 几何验证

做法:

1. 在全数据中抽样 12 帧
2. 对每帧再次用 `linefit` 取 ground points
3. 把点变到 `rect camera` 坐标
4. 计算这些 ground points 到对应 scene-level plane 的绝对距离

结果:

- median of median abs distance: `0.04194 m`
- median p95 abs distance: `0.14523 m`
- max sampled p95 abs distance: `0.21273 m`

同时抽样中所有 plane 都满足 `plane_b < 0`, 与 KITTI 读入逻辑一致。

这个结果说明:

- 拟合坐标系是对的
- plane 方向约定是对的
- scene-level plane 对当前路侧场景足够稳定

### 3.2 `USE_ROAD_PLANE` 数据链路 smoke

脚本:

- [tools/smoke_test_use_road_plane.py](/root/project/LION/tools/smoke_test_use_road_plane.py)

它做了两件事:

1. 强制打开 `USE_ROAD_PLANE=True`, 跑 dataloader 多个 batch, 统计 `put_boxes_on_road_planes` 是否真的被调用
2. 从 `checkpoint_epoch_80.pth` 恢复, 跑一个极短的训练 smoke, 验证续训链路是否稳定

结果摘要:

- data batches: `8`
- `put_boxes_on_road_planes` data-side calls: `8`
- resumed train steps: `2`
- `put_boxes_on_road_planes` train-side calls: `2`
- loss 有限, grad norm 有限

输出 JSON:

- `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/use_road_plane_smoke.json`

结论:

- `plane` 文件可被 loader 正常读取
- `USE_ROAD_PLANE: True` 的 gt sampling 链路已被实际触发
- 短续训 smoke 正常

---

## 4. Large-ROI Eval

评测目标:

- 大 ROI: `[0, -60, -3, 140, 60, 2]`
- checkpoint 范围: `5, 10, 15, ..., 80`
- 不包含 `latest_model.pth`
- 输出目录:
  `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/eval_large_range`

### 4.1 为兼容大 ROI 做的修正

脚本:

- [tools/run_eval_v2x_spd_strict3_smallrange_large_roi.sh](/root/project/LION/tools/run_eval_v2x_spd_strict3_smallrange_large_roi.sh)

修正点:

1. 使用 fully expanded 的 run-side cfg, 避免 nested `_BASE_CONFIG_` 没有完全展开
2. 扩大 `POINT_CLOUD_RANGE` 到 `[0, -60, -3, 140, 60, 2]`
3. 为了保持与训练时一致的 z-bin 数, 将 eval `VOXEL_SIZE.z` 从 `0.125` 调整到 `0.15625`
4. 因为 `tools/test.py` 会按临时 YAML 的 `TAG` 把原始 `result.pkl` 写到 `run/tmp/...`, 脚本增加了结果回收逻辑, 再统一复制到 `eval_large_range/...`
5. `test.py` 内部 KITTI 评测在当前 `py38 + numba` 环境下会因为 CUDA context 问题退出非零, 但 `result.pkl` 已经落盘, 后续官方 AP 统一交给 `py310` 的独立脚本处理

### 4.2 最终结果

已完成全部 `16` 个 checkpoint:

- `5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80`

以 `AP_R40` 的 `3D / Moderate` 为例:

- best macro moderate 3D AP epoch: `20`
- best macro moderate 3D AP: `7.9716`
- best Car moderate 3D AP epoch: `20`, value: `22.3090`
- best Pedestrian moderate 3D AP epoch: `30`, value: `0.6242`
- best Cyclist moderate 3D AP epoch: `40`, value: `3.7221`

`epoch 80` 的 moderate 3D AP:

- Car: `11.7384`
- Pedestrian: `0.0850`
- Cyclist: `0.8434`
- macro moderate 3D AP: `4.2223`

从这组大 ROI 结果看:

1. `epoch 20` 是最强 checkpoint, 明显优于后续多数 epoch
2. 车类在大 ROI 下的峰值提升主要集中在中前期 checkpoint
3. Pedestrian 和 Cyclist 在大 ROI 下波动较大, 说明这个扩展范围对小目标更敏感
4. `epoch 80` 不是大 ROI 下的最优点, 后续如果做大范围部署, 不应默认沿用最后一个 checkpoint

表格图像:

- `3D AP_R40`: `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/eval_large_range/eval_summary_3d_R40.png`
- `BEV AP_R40`: `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/eval_large_range/eval_summary_bev_R40.png`
- `BBox AP_R40`: `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/eval_large_range/eval_summary_bbox_R40.png`

---

## 5. 如何无感加入地面特征

### 5.1 当前模型现实约束

当前训练配置本质上是:

`DynamicVoxelVFE -> LION3DBackboneOneStride -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle`

也就是说:

- 当前不是 query-based detector
- 当前最稳妥的扩展点不是硬改成 DETR 式解码
- 更适合做“低侵入 sidecar ground branch”或者“零初始化 residual fusion”

### 5.2 结论先行

如果目标是:

- 尽量复用现有 `epoch 80` 权重
- 尽量少改训练链路
- 尽量容易讲故事
- 尽量先拿到稳定增益

最推荐的路线不是先改 LION selective scan, 而是:

1. 先保留 scene-level plane 用于 `USE_ROAD_PLANE` 和全局高度先验
2. 再增加一个 zero-init 的 BEV ground adapter, 让老模型可以无缝加载
3. 如果第二阶段有效, 再尝试点级 `h_above_ground / d_plane / ground_conf`
4. 最后才考虑更重的 Ground-Scene Mamba 或 query sidecar

### 5.3 推荐方案 R1: Zero-init BEV Ground Adapter

这是我最推荐的首发方案。

做法:

1. 保持现有 backbone 和 dense head 不动
2. 在 dataset / processor 中额外构建一张或多张 ground descriptor BEV map
3. 在 `BaseBEVBackbone` 前或后加一个极轻量的 residual adapter:

```txt
bev_out = bev_old + alpha * GroundAdapter(ground_maps)
```

其中:

- `alpha` 初始化为 `0`
- 或 `GroundAdapter` 最后一层卷积初始化为 `0`

这样做的好处:

- checkpoint 可以完整加载
- 训练开始时网络行为和老模型几乎一致
- 续训是“无感”的
- 随着 finetune, 模型逐步学会利用 ground 信息

推荐的 ground maps:

- `plane_distance_map`
- `height_above_plane_map`
- `ground_confidence_map`
- `ground_density_map`
- `roughness_map`

论文叙事也很顺:

- 路侧传感器安装固定, ground prior 稳定
- scene-level plane 提供全局几何参考
- local descriptor map 弥补单平面无法表达的局部起伏和路缘信息
- zero-init residual fusion 让旧模型平滑吸收新先验

### 5.4 推荐方案 R2: 点级地面特征, 但做零影响初始化

第二推荐方案是给点直接追加:

- `d_plane`
- `h_above_ground`
- `ground_conf`

当前代码支持这条路:

- [pcdet/datasets/processor/point_feature_encoder.py](/root/project/LION/pcdet/datasets/processor/point_feature_encoder.py)
- [pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py](/root/project/LION/pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py)

因为 `PointFeatureEncoder` 可以扩展 `src_feature_list / used_feature_list`, `DynamicVoxelVFE` 也会透明接收更多点特征。

但这条路有一个代价:

- 第一层 PFN 输入通道数会变化
- 直接加载老 checkpoint 会有 shape mismatch

解决办法:

1. 保留老权重
2. 只对新增输入通道对应的权重做零初始化
3. 继续从 `epoch 80` 续训

这样网络初始时等价于“忽略新增地面特征”, 训练过程中再逐步学会使用它们。

这也是“无感续训”的一种实现方式, 但工程风险高于 R1。

这里要注意一个实现细节:

- 当前 [pcdet/models/detectors/detector3d_template.py](/root/project/LION/pcdet/models/detectors/detector3d_template.py:374) 的 `load_params_from_file(..., strict=False)` 对 shape mismatch 的层会整层跳过
- 它不会自动做“旧通道沿用, 新通道补零”的部分加载

所以如果采用 R2, 最稳妥的做法是:

1. 先构造新模型
2. 把旧 checkpoint 的第一层 PFN `linear.weight` 复制到前半部分
3. 新增输入通道对应的列显式置零
4. 其余 shape 相同层照常加载

这样才是真正意义上的“平滑续训”。

还有一个坐标系提醒:

- KITTI `plane` 文件里的 `[a, b, c, d]` 是 `rect camera` 坐标系
- 当前模型输入点 `points[:, 1:4]` 是 LiDAR 坐标系

所以如果后续要构造点级:

- `d_plane`
- `h_above_ground`

不能直接用:

```txt
a * x_lidar + b * y_lidar + c * z_lidar + d
```

正确做法有两种:

1. 先把点从 LiDAR 变到 `rect camera`, 再算平面距离
2. 或先把平面从 `rect camera` 变到 LiDAR 坐标系, 再在 LiDAR 点上计算

若 `x_cam = T_cam<-lidar * x_lidar`, 则平面参数满足:

```txt
pi_lidar = T_cam<-lidar^T * pi_cam
```

这里如果坐标系处理错了, 地面特征会全部失真。

### 5.5 推荐方案 R3: Ground-Scene Mamba / Query Sidecar

这条路线更有论文味, 但不建议作为第一版。

可以分成两类:

1. Ground-Scene Mamba
   - 在 LION 旁边增加一条 ground memory / ground token 分支
   - 再与 scene BEV 特征做轻量融合

2. Query Sidecar
   - 单独加一个小的 query refinement 分支
   - 用 ground memory 先 refine proposal, 再和 scene feature 交互

这类方法更适合在:

- 先证明 R1 或 R2 有效之后
- 再作为“从几何先验到结构交互”的升级故事

### 5.6 为什么不建议第一版直接改 selective scan

原因很实际:

1. selective scan 改动大, 回归风险高
2. 当前目标是复用 `80 epoch` 训练结果, 不是从头换 backbone
3. 先证明“ground prior 有用”比先证明“更纯 Mamba”更重要

如果第一版就改 scan, 一旦增益不稳定, 很难判断问题来自:

- ground prior 本身无效
- 还是 scan 设计/训练稳定性出了问题

### 5.7 最推荐的实验顺序

建议按下面顺序走:

1. Baseline-resume:
   - 只打开 `USE_ROAD_PLANE`
   - 不改模型
   - 看续训是否已经有收益

2. R1:
   - 加 `BEV ground adapter`
   - zero-init residual fusion
   - 从 `epoch 80` 继续训练

3. R2:
   - 加点级 `d_plane / h_above_ground / ground_conf`
   - 第一层 PFN 对新增通道零初始化
   - 再从 `epoch 80` 继续训练

4. R3:
   - 如果前两步有效, 再做 Ground-Scene Mamba 或 query sidecar

### 5.8 建议的实验矩阵

如果目标是“有效提点 + 容易讲故事”, 我建议实验矩阵不要一上来铺太大, 而是按下面顺序推进:

1. `E0: Baseline resume`
   - 不加任何新特征
   - 只从 `epoch 80` 再续几轮
   - 作用: 排除“纯续训自己就会涨”的干扰

2. `E1: Road-plane only`
   - 打开 `USE_ROAD_PLANE`
   - 模型结构不变
   - 作用: 证明仅利用地面先验改善 GT sampling 就已经有收益

3. `E2: Global plane + BEV adapter`
   - 只用 scene-level plane 生成简单地面图
   - zero-init residual fusion
   - 作用: 证明廉价全局几何先验可以被现有 LION 平滑吸收

4. `E3: Global plane + local descriptors`
   - 在 E2 基础上增加 `ground_confidence / roughness / density`
   - 作用: 证明“单平面有用, 但局部地面描述更强”

5. `E4: Point-level ground features`
   - 增加 `d_plane / h_above_ground / ground_conf`
   - 做第一层 PFN 的 checkpoint surgery
   - 作用: 证明细粒度点级几何编码还能继续提升

6. `E5: Ground-Scene Mamba or query sidecar`
   - 只在 E3/E4 证明有效后再做
   - 作用: 作为更强但更重的结构创新版本

最重要的是:

- `E1 -> E2 -> E3`

这三步已经足够形成一条非常完整的故事线:

`ground prior exists -> global plane helps -> local ground descriptors help more`

### 5.9 最适合写故事的主线

我建议论文主线这样讲:

1. `Roadside geometry is stable`
   - 路侧感知设备安装固定, scene-level ground prior 稳定且廉价

2. `A single plane is useful but insufficient`
   - 单平面能提供强全局几何约束
   - 但无法表达局部坡度、路缘和粗糙度

3. `Ground descriptors are a low-cost, high-value prior`
   - 从 plane 和 ground points 可以提炼出局部几何描述
   - 不引入昂贵标注
   - 不引入重型外部模型

4. `Residual ground fusion enables seamless continuation from a strong baseline`
   - 基于 zero-init adapter, 可以从已有 `80 epoch` checkpoint 无感续训
   - 这是工程上最有说服力的一点

5. `Ground-aware LION is an incremental path, not a disruptive rewrite`
   - 先 ground adapter
   - 再点级特征
   - 最后 Ground-Scene Mamba

### 5.10 建议重点汇报的指标

除了常规 KITTI AP_R40, 我建议额外做三类切片统计:

1. 距离切片
   - 近距离 / 中距离 / 远距离
   - 路侧场景下远距离目标更容易受高度和地面几何影响

2. 高度切片
   - 近地目标 vs 相对离地更高目标
   - 有助于证明 ground prior 主要改善的是近地几何判断

3. 场景切片
   - 平整场景 vs 有坡度/路缘/粗糙路面的场景
   - 有助于支持“local descriptors 比 single plane 更强”的论点

如果这些切片上能观察到更稳定的收益, 故事会明显更完整。

### 5.11 一个很关键的工程提醒

如果后续 ground feature 需要在推理阶段也可用, 目前 loader 还需要补一刀:

- [pcdet/datasets/kitti/kitti_dataset.py](/root/project/LION/pcdet/datasets/kitti/kitti_dataset.py:372)

当前 `road_plane` 只在 `if 'annos' in info:` 分支里注入 `input_dict`。

这意味着:

- train/val 有标注时没问题
- 但没有 `annos` 的 test/inference 场景, plane 不会进入 sample

如果 ground feature 真要作为模型输入, 需要把 plane 的读取挪到 annotation 分支之外。

---

## 6. 外部研究依据

以下主源对本次方案判断最有帮助:

1. 3DET-Mamba
   - https://papers.neurips.cc/paper_files/paper/2024/file/547108084f0c2af39b956f8eadb75d1b-Paper-Conference.pdf
   - 结论: query-aware Mamba 比 naive Mamba decoder 更适合处理 query 与 scene feature 的关系

2. CoMamba
   - https://arxiv.org/abs/2409.10699
   - 结论: 在 V2X / cooperative perception 里, Mamba 很适合做大范围特征融合与实时建模

3. MambaVision
   - https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf
   - 结论: 混合少量 self-attention 的 hybrid 结构, 往往比纯 Mamba 更稳

4. MonoGAE
   - https://arxiv.org/abs/2310.00400
   - 结论: 路侧固定安装使 ground plane 成为稳定强先验

5. BEVHeight
   - https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVHeight_A_Robust_Framework_for_Vision-Based_Roadside_3D_Object_Detection_CVPR_2023_paper.pdf
   - 结论: 在路侧场景里, height-to-ground 比 camera-centric depth 更稳, 更利于鲁棒建模

6. MonoGround
   - https://arxiv.org/abs/2206.07372
   - 结论: ground plane 可以作为强几何先验改善 3D 定位

7. Det6D
   - https://arxiv.org/abs/2207.09412
   - 结论: ground-aware 模块可以做成 plug-and-play, 并显著增强复杂地形鲁棒性

8. Local Ground-aware and Adaptive Representation
   - https://arxiv.org/abs/2002.00336
   - 结论: 局部 ground representation 比单一 whole-scene plane 更准确, 对点云 3D 检测有价值

---

## 7. 最终产物

本次任务的主要产物如下:

1. plane 生成脚本
   - [tools/generate_kitti_scene_planes.py](/root/project/LION/tools/generate_kitti_scene_planes.py)

2. `USE_ROAD_PLANE` smoke 脚本
   - [tools/smoke_test_use_road_plane.py](/root/project/LION/tools/smoke_test_use_road_plane.py)

3. 大 ROI eval 脚本
   - [tools/run_eval_v2x_spd_strict3_smallrange_large_roi.sh](/root/project/LION/tools/run_eval_v2x_spd_strict3_smallrange_large_roi.sh)

4. 大 ROI eval 输出目录
   - `/root/project/LION/run/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_strict3_smallrange_trainvox16000/v2x_spd_strict3_smallrange_lion_mamba_fp32_bs4_trainvox16000_scratch/eval_large_range`

5. 表格图像
   - `eval_summary_3d_R40.png`
   - `eval_summary_bev_R40.png`
   - `eval_summary_bbox_R40.png`

6. 总结文档
   - [docs/v2x_spd_ground_plane_eval_and_ground_feature_research_2026-05-15.md](/root/project/LION/docs/v2x_spd_ground_plane_eval_and_ground_feature_research_2026-05-15.md)
