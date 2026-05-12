import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# OpenPCDet 核心模块
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu

class DummyKITTIDataset(DatasetTemplate):
    """
    伪造数据集：用于模拟 KITTI 格式的数据流，验证模型前向与反向传播。
    继承 DatasetTemplate 以自动复用 config 中定义的数据处理 pipeline (如体素化 transform_points_to_voxels)。
    """
    def __init__(self, dataset_cfg, class_names, training=True):
        super().__init__(
            dataset_cfg=dataset_cfg, 
            class_names=class_names, 
            training=training, 
            root_path=None, 
            logger=None
        )
        self.num_samples = 4  # 设定一个极小的 Epoch 大小用于测试

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 1. 生成符合 KITTI point_cloud_range [0, -40, -3, 70.4, 40, 1] 的随机点云
        num_points = 16384  # 模拟每帧 1.6 万个点
        x = np.random.uniform(0.1, 70.0, num_points)
        y = np.random.uniform(-39.0, 39.0, num_points)
        z = np.random.uniform(-2.9, 0.9, num_points)
        intensity = np.random.uniform(0.0, 1.0, num_points)
        points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)

        # 2. 生成随机 GT Boxes [x, y, z, dx, dy, dz, heading, class_label]
        # label: 1=Car, 2=Pedestrian, 3=Cyclist
        gt_boxes = np.array([
            [20.0,  5.0, -1.0, 4.0, 1.6, 1.5,  0.0,  1],
            [10.0, -5.0, -1.0, 0.8, 0.6, 1.7,  1.57, 2],
            [30.0,  0.0, -1.0, 1.8, 0.6, 1.7,  0.0,  3]
        ], dtype=np.float32)
        # 生成对应的类别名字符串
        label_to_name = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
        gt_names = np.array([label_to_name[int(l)] for l in gt_boxes[:, -1]])

        input_dict = {
            'frame_id': str(index),
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names
        }

        # 3. 将原始数据送入 OpenPCDet 的 DataProcessor 进行体素化等处理
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def main():
    print("=== [1] 初始化配置 ===")
    # 指定 LION-Mamba 的配置文件
    cfg_file = "cfgs/kitti_models/second_with_lion_mamba_64dim.yaml"
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"找不到配置文件: {cfg_file}。请确保你在 tools 目录下运行此脚本。")
    
    cfg_from_yaml_file(cfg_file, cfg)
    
    # 【关键】禁用数据增强：防止因为找不到真实的 KITTI GT Database (.pkl) 导致报错
    if 'DATA_AUGMENTOR' in cfg.DATA_CONFIG:
        # 将所有配置好的增强项全部禁用
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
            aug.NAME for aug in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST
        ]

    print("=== [2] 构建 Dummy Dataset 与 DataLoader ===")
    dataset = DummyKITTIDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=2,          # 单卡测试 Batch Size 设置为 2
        pin_memory=True, 
        num_workers=2,
        shuffle=True, 
        collate_fn=dataset.collate_batch,
        drop_last=False
    )

    print("=== [3] 构建 LION 3D 检测模型 ===")
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n模型结构已加载，准备执行单卡前向/反向传播测试...")
    
    # ------------------ 训练流程验证 ------------------
    print("\n=== [4] 测试 Training Pipeline ===")
    model.train()
    for batch_idx, batch_dict in enumerate(dataloader):
        # 将数据加载到 CUDA:0
        load_data_to_gpu(batch_dict)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播 (在 train 模式下，模型会返回 loss)
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        loss = ret_dict['loss']
        
        print(f"  [Train] Batch {batch_idx}: Forward 成功. Loss: {loss.item():.4f}")
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        print(f"  [Train] Batch {batch_idx}: Backward 成功. 参数已更新.")
        
        break # 验证一次即可

    # ------------------ 验证流程验证 ------------------
    print("\n=== [5] 测试 Inference/Eval Pipeline ===")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)
            
            # 评估模式下，模型会通过 DenseHead 执行 NMS 并返回后处理的预测字典
            pred_dicts, recall_dicts = model(batch_dict)
            
            print(f"  [Eval] Batch {batch_idx}: Inference 成功.")
            print(f"  [Eval] 检测到目标的数量 (当前Batch第1帧): {len(pred_dicts[0]['pred_boxes'])}")
            print(f"  [Eval] 预测框 Shape: {pred_dicts[0]['pred_boxes'].shape}")
            break

    print("\n=== 🎯 测试通过：环境、数据预处理、模型计算图、损失计算与推理后处理全部走通！ ===")

if __name__ == '__main__':
    main()