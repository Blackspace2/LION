# V2X-SPD merge3 使用 LION-KITTI 训练备忘

本文记录当前数据状态，以及后续 GPU 到位后从 `merge3` 数据集开始训练 LION-KITTI 风格模型还缺哪些步骤。

## 当前已经完成

V2X-SPD 已经转换成 KITTI detection 风格目录：

```text
/root/autodl-tmp/V2X-SPD-KITTI/
  velodyne_bin/
  shared/
    image_2/
    calib/
    ImageSets/
  merge3/
    training/
      velodyne -> ../../velodyne_bin
      image_2  -> ../../shared/image_2
      calib    -> ../../shared/calib
      label_2/
    ImageSets -> ../shared/ImageSets
```

`merge3` 的类别映射已经在标签转换阶段完成：

```text
Car = Car + Van + Truck + Bus
Pedestrian = Pedestrian
Cyclist = Cyclist + Motorcyclist
```

因此现在不需要新增 Dataset 模块，也不需要重写 LION 网络。后续继续复用现有的 `KittiDataset` 和 `second_with_lion_mamba_64dim` 训练链路即可。

## pkl 和 gt database 状态

以下训练前索引文件已经生成完成：

```text
kitti_infos_train.pkl     6685 帧，约 24M
kitti_infos_val.pkl       2740 帧，约 11M
kitti_infos_trainval.pkl  9425 帧，约 35M
kitti_dbinfos_train.pkl   训练集 gt database 索引，约 28M
gt_database/              训练集采样目标点云，约 684M
```

类别统计：

```text
train:
  Car: 79282
  Cyclist: 18847
  Pedestrian: 2606

val:
  Car: 31074
  Cyclist: 11254
  Pedestrian: 2844

trainval:
  Car: 110356
  Cyclist: 30101
  Pedestrian: 5450
```

`kitti_dbinfos_train.pkl` 与 `gt_database/` 的目标数和 train split 标注数一致：

```text
Car: 79282
Cyclist: 18847
Pedestrian: 2606
gt_database 文件数: 100735
```

已经做过一次 `KittiDataset` 单帧 smoke test，可以正常加载 pkl、点云、GT box，并生成 voxel：

```text
dataset_len: 6685
frame_id: 0000_000000
points_shape: (53392, 4)
gt_boxes_shape: (18, 8)
voxels_shape: (34282, 5, 4)
```

## 如需重新生成 pkl 和 gt database

当前已有脚本：

```text
tools/create_v2x_spd_kitti_infos.py
```

建议命令：

```bash
cd /root/project/LION

python tools/create_v2x_spd_kitti_infos.py \
  --data-root /root/autodl-tmp/V2X-SPD-KITTI/merge3 \
  --workers 12
```

该脚本默认适配 V2X-SPD LiDAR-only 训练：

```text
FOV_POINTS_ONLY = False
count_inside_pts = False
```

如果 CPU 负载不高或耗时异常，可以先降到：

```bash
python tools/create_v2x_spd_kitti_infos.py \
  --data-root /root/autodl-tmp/V2X-SPD-KITTI/merge3 \
  --workers 8
```

生成完成后检查：

```bash
ls -lh /root/autodl-tmp/V2X-SPD-KITTI/merge3/*.pkl
ls -lh /root/autodl-tmp/V2X-SPD-KITTI/merge3/gt_database | head
```

预期：

```text
kitti_infos_train.pkl     6685 帧
kitti_infos_val.pkl       2740 帧
kitti_infos_trainval.pkl  9425 帧
kitti_dbinfos_train.pkl   训练集 gt database 索引
gt_database/              训练集采样目标点云
```

## merge3 训练配置状态

V2X-SPD merge3 专用训练配置文件已经补充完成：

```text
tools/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml
```

该配置基于 `tools/cfgs/kitti_models/second_with_lion_mamba_64dim.yaml`，已经把数据路径、点云范围、anchor、voxel 数量、学习率、batch size 等改成适合 V2X-SPD 的第一版保守参数。

第一版建议重点参数：

```yaml
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
  _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
  DATA_PATH: /root/autodl-tmp/V2X-SPD-KITTI/merge3
  FOV_POINTS_ONLY: False
  POINT_CLOUD_RANGE: [0, -65, -4, 140, 65, 3]
```

原因：

- 当前模型是纯点云 LION，不需要图像特征。
- V2X-SPD 的 2D 投影标定不完全精确，关闭 `FOV_POINTS_ONLY` 可以减少图像标定误差对点云过滤的影响。
- V2X-SPD 目标距离明显比 KITTI 更远，不能继续使用 KITTI 默认 `70.4m` range。

推荐 voxel：

```yaml
VOXEL_SIZE: [0.2, 0.2, 0.25]
MAX_NUMBER_OF_VOXELS:
  train: 40000
  test: 60000
```

说明：V2X-SPD 的 z range 为 `[-4, 3]`，如果继续使用 KITTI 的 `z=0.125`，z 方向 voxel 数会从 KITTI 的 32 增加到 56，导致 `HeightCompression` 后 BEV channel 从 128 变成 256，而当前 2D BEV backbone 仍按 128 channel 构建。第一版为了保持 LION-KITTI 主干结构和显存压力稳定，使用 `z=0.25`。

推荐 anchor：

```yaml
Car: [[4.8, 2.0, 1.75]]
Pedestrian: [[0.55, 0.6, 1.65]]
Cyclist: [[1.8, 0.72, 1.55]]
```

推荐优化参数：

```yaml
OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 4
  NUM_EPOCHS: 80
  LR: 0.0015
  GRAD_NORM_CLIP: 2.0
```

32G GPU + fp32 建议先从 batch size 4 起步。稳定后再尝试 batch size 6。

## 第三步：正式训练

配置文件补完并确认 pkl 已存在后，再启动训练。

建议此次输出目录放到：

```text
/root/project/LION/run
```

训练命令可以沿用之前 KITTI 单卡脚本的风格，只替换配置文件和实验 tag。示例：

```bash
cd /root/project/LION

python tools/train.py \
  --cfg_file tools/cfgs/kitti_models/second_with_lion_mamba_64dim_v2x_spd_merge3.yaml \
  --batch_size 4 \
  --workers 8 \
  --extra_tag v2x_spd_merge3_lion_mamba_fp32_bs4 \
  --output_dir /root/project/LION/run
```

如果后续继续使用“训练结束后再单独评估”的稳定流程，可以按之前 KITTI 的方式再写一个 bash 脚本，把训练和 AP 计算拆开。

## TensorBoard

另开终端：

```bash
tensorboard \
  --logdir /root/autodl-tmp/LION_output \
  --port 6006
```

## 注意事项

- 训练接口已经对齐，不需要新增 Dataset 模块。
- pkl 和 gt database 是训练前必须补齐的索引文件，不是重新转换标签。
- 第一版建议 fp32，不建议直接 fp16，因为之前 KITTI 训练已经出现过梯度爆炸、NaN、skip 等问题。
- 第一版建议关闭或弱化 gt sampling；如果配置里启用 gt sampling，必须确保 `kitti_dbinfos_train.pkl` 和 `gt_database/` 已经生成。
- 如果 AP 评估结果不理想，优先检查 range、anchor、gt sampling 强度和学习率，不要先怀疑数据接口。
