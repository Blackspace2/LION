# V2X-SPD 与 KITTI 的训练配置差异分析

本文用于指导在 `/root/autodl-tmp/V2X-SPD` 上训练 LION-KITTI 风格模型。统计对象为当前可用的 V2X-SPD `trainval`，并与 `/root/autodl-tmp/kitti-offical` 的 KITTI `trainval` 对比。

## 数据结构差异

- KITTI-official 已经是 OpenPCDet/LION 可直接读取的扁平结构：
  - `training/velodyne/*.bin`
  - `training/image_2/*.png`
  - `training/label_2/*.txt`
  - `training/calib/*.txt`
  - 已有 `kitti_infos_*.pkl` 与 gt database
- V2X-SPD 当前是 sequence 分组结构：
  - `training/{scene}/velodyne/*.pcd`
  - `training/{scene}/image_02/*.jpg`
  - `training/{scene}/label_02_split/*.txt`
  - `training/{scene}/calib/*.txt`
- 因此 V2X-SPD 不能只换 yaml 直接复用 `KittiDataset`，需要先转换成 LION/KITTI-like 扁平格式，或新增专用 Dataset。

## 划分与规模

- KITTI trainval: 7481 帧
- V2X-SPD trainval: 9425 帧
- V2X-SPD 当前有效划分已经排除 README 中说明的坏标定 sequence。

每帧点数：

- KITTI:
  - p50: 120229
  - p90: 124796
  - mean: 119225
- V2X-SPD:
  - p50: 56186
  - p90: 64463
  - mean: 55472

结论：V2X-SPD 每帧点数约为 KITTI 的 46%，但这不代表训练更轻，因为它的目标更远、更密，合理 range 会比 KITTI 大很多。

## 目标密度

每帧目标数：

- KITTI:
  - p50: 5
  - p90: 11
  - p99: 16
  - max: 22
  - mean: 5.42
- V2X-SPD:
  - p50: 13
  - p90: 31
  - p99: 40
  - max: 49
  - mean: 15.48

结论：V2X-SPD 是高目标密度数据，训练时 anchor assign、NMS、`MAX_NUMBER_OF_VOXELS` 和 gt sampling 都不能照搬 KITTI 的直觉。

## 类别分布

KITTI trainval 原始类别：

- Car: 28742
- Pedestrian: 4487
- Cyclist: 1627
- Van: 2914
- Truck: 1094
- Tram: 511
- Misc: 973
- Person_sitting: 222

V2X-SPD trainval 原始类别：

- Car: 89676
- Van: 11230
- Truck: 6564
- Bus: 2886
- Motorcyclist: 20216
- Cyclist: 9885
- Pedestrian: 5450

建议第一版做三类 merged baseline：

- `Car = Car + Van + Truck + Bus`
- `Pedestrian = Pedestrian`
- `Cyclist = Cyclist + Motorcyclist`

原因：

- 如果只保留 strict KITTI 三类，会丢掉 40896 个有效标注。
- V2X-SPD 中 `Motorcyclist` 数量超过 `Cyclist`，直接丢掉会让骑行类学习严重偏置。
- `Van/Truck/Bus` 如果不合并进车辆类，会在训练中形成大量近似车辆形态的背景噪声。

## 目标距离与 range

KITTI 目标距离明显近：

- KITTI Car 距离 p50/p90/p95: 27.02 / 52.93 / 60.51 m
- KITTI Pedestrian 距离 p50/p90/p95: 15.92 / 33.72 / 40.15 m
- KITTI Cyclist 距离 p50/p90/p95: 23.12 / 45.82 / 53.78 m

V2X-SPD 目标明显更远：

- V2X-SPD Car 距离 p50/p90/p95: 60.58 / 128.84 / 154.09 m
- V2X-SPD Pedestrian 距离 p50/p90/p95: 82.96 / 107.90 / 114.16 m
- V2X-SPD Cyclist 距离 p50/p90/p95: 76.01 / 103.81 / 113.44 m
- V2X-SPD Truck p90: 173.05 m

候选 range 对 merged3 目标中心覆盖率：

- KITTI 默认 `[0, -40, -3, 70.4, 40, 1]`: 58.14%
- `[0, -50, -4, 100, 50, 3]`: 82.33%
- `[0, -60, -4, 120, 60, 3]`: 89.39%
- `[0, -70, -4, 160, 70, 3]`: 95.82%
- `[0, -80, -4, 200, 80, 3]`: 100.00%

建议第一版使用：

```yaml
POINT_CLOUD_RANGE: [0, -70, -4, 160, 70, 3]
```

理由：

- 比 KITTI 默认多覆盖约 37.7% 的目标。
- 相比 200m range，显存/计算更稳。
- 对 merged3 三类整体覆盖 95.8%，已经适合作为稳定 baseline。

如果后续业务强依赖 160-200m 超远目标，再单独做 long-range 版本。

## anchor 尺寸建议

KITTI 默认 anchor：

- Car: `[3.9, 1.6, 1.56]`
- Pedestrian: `[0.8, 0.6, 1.73]`
- Cyclist: `[1.76, 0.6, 1.73]`

V2X-SPD merged3 统计：

- Car merged:
  - mean l/w/h: 4.835 / 2.020 / 1.773
  - p50 l/w/h: 4.332 / 1.935 / 1.587
  - p90 l/w/h: 5.262 / 2.261 / 2.222
- Pedestrian:
  - mean l/w/h: 0.540 / 0.578 / 1.644
  - p50 l/w/h: 0.515 / 0.577 / 1.646
- Cyclist merged:
  - mean l/w/h: 1.774 / 0.710 / 1.541
  - p50 l/w/h: 1.786 / 0.709 / 1.557

建议第一版 anchor：

```yaml
Car: [[4.8, 2.0, 1.75]]
Pedestrian: [[0.55, 0.6, 1.65]]
Cyclist: [[1.8, 0.72, 1.55]]
```

如果 Car 合并了 Bus/Truck，单一 Car anchor 会对大车不友好，但先保持三类 head 更稳。后续可以考虑 Car 多 anchor：

```yaml
Car: [[4.3, 1.9, 1.6], [7.0, 2.7, 3.1], [11.5, 2.9, 3.25]]
```

但多 anchor 会增加 dense head 负担，第一版不建议直接上。

## voxel 与显存估算

V2X-SPD 的点云是 `binary_compressed` PCD。当前环境没有 `pypcd/open3d`，我用自写 LZF 解析器做了抽样校验；第一次快速统计时发现 LZF offset 字节顺序写错会导致 intensity 和空间统计异常，后续已经修正。为了避免把解码器性能误差带入配置，第一版显存估算不再依赖大规模 voxel unique 统计，而基于以下稳定事实：

- V2X-SPD 每帧点数约 KITTI 的 46%。
- 但 V2X-SPD 合理 range 至少应扩到 160m，BEV 网格面积明显大于 KITTI 默认 range。
- V2X-SPD 每帧目标数约 KITTI 的 2.85 倍，dense head 和 target assign 压力更大。

建议第一版仍然使用 KITTI-LION 已验证过的较粗 voxel：

```yaml
VOXEL_SIZE: [0.2, 0.2, 0.125]
MAX_NUMBER_OF_VOXELS:
  train: 40000
  test: 60000
```

32G 单卡 fp32 起步建议：

```yaml
BATCH_SIZE_PER_GPU: 4
NUM_WORKERS: 8
```

更激进可试：

```yaml
BATCH_SIZE_PER_GPU: 6
```

不建议第一轮直接 fp32 batch size 8，因为 range 比 KITTI 大很多，目标数也明显更多。等转换后的 `.bin` 和 pkl 生成后，再用真实 dataloader 在 GPU 上测 `MAX_NUMBER_OF_VOXELS=40000/60000` 是否需要调整。

## gt sampling 建议

V2X-SPD 目标非常密，第一版不建议照搬 KITTI 的 gt sampling 强度。

建议第一轮：

- 先关闭 gt_sampling，确认训练稳定性。
- 或者弱采样：

```yaml
SAMPLE_GROUPS: ['Car:5', 'Pedestrian:5', 'Cyclist:5']
LIMIT_WHOLE_SCENE: True
```

稳定后再逐步增强。

## 训练超参建议

第一版稳定配置：

```yaml
OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 4
  NUM_EPOCHS: 80
  OPTIMIZER: adam_onecycle
  LR: 0.0015
  WEIGHT_DECAY: 0.01
  GRAD_NORM_CLIP: 2.0
```

原因：

- V2X-SPD 有 README 明确提示的时空不完全对齐问题。
- 目标距离远、box 噪声更大。
- 之前 KITTI fp16 出现过 NaN/梯度爆炸，V2X-SPD 第一版应该优先 fp32 稳定。

## 转换时必须注意

- frame id 需要扁平化，例如 `0000_000061`，避免不同 sequence 下同名帧冲突。
- `.pcd` 要转换成 `.bin`。
- label 要转换成 LION/OpenPCDet 标准 KITTI detection 格式，不能把 tracking 格式前缀直接喂给 `object3d_kitti.py`。
- PCD intensity 在修正 LZF 解码后范围正常，约为 `[0, 1]`。转换时应保持 `float32` intensity，不要二次除以 255。
- 转换后的 `ImageSets/train.txt` 和 `val.txt` 应该按帧级 id 写入，而不是 sequence id。

## 推荐第一版配置方向

```yaml
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

POINT_CLOUD_RANGE: [0, -70, -4, 160, 70, 3]

VOXEL_SIZE: [0.2, 0.2, 0.125]
MAX_NUMBER_OF_VOXELS:
  train: 40000
  test: 60000

ANCHORS:
  Car: [[4.8, 2.0, 1.75]]
  Pedestrian: [[0.55, 0.6, 1.65]]
  Cyclist: [[1.8, 0.72, 1.55]]

OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 4
  LR: 0.0015
  GRAD_NORM_CLIP: 2.0
```

后续 GPU 到位后，先做以下 smoke test：

1. 转换后随机读取 100 帧，检查 points/gt_boxes 坐标是否同系。
2. 生成 pkl 和 gt database。
3. CPU dataloader smoke test。
4. GPU 单 batch forward/backward。
5. fp32 batch size 4 跑 1 epoch。
6. 如果显存稳定，再试 batch size 6。
