import argparse
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir)

# 将项目根目录添加到 Python 模块搜索路径
if src_dir not in sys.path:
    sys.path.append(src_dir)

import torch
import hydra
import laspy
import numpy as np
from src.utils import init_config
from src.transforms import instantiate_datamodule_transforms
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.data import Data, NAG, InstanceData
from src.datasets.qc_dataset import QC_NUM_CLASSES
from src.utils.color import to_float_rgb


def read_single_point_cloud(raw_cloud_path: str) -> Data:
    """Read a single LAS file and convert it to a torch_geometric Data object.

    Args:
        raw_cloud_path: Path to the LAS file

    Returns:
        torch_geometric.data.Data: Point cloud data with features and labels
    """
    # Create an empty Data object
    data = Data()

    # Read the LAS file
    las = laspy.read(raw_cloud_path)

    pos = torch.stack([
        torch.tensor(np.ascontiguousarray(las[axis]))
        for axis in ["X", "Y", "Z"]], dim=-1)
    pos *= las.header.scales
    pos_offset = pos[0]
    data.pos = (pos - pos_offset).float()
    data.pos_offset = pos_offset

    # Populate data with point RGB colors
    if hasattr(las, "rgb"):
        # RGB stored in uint16 lives in [0, 65535]
        data.rgb = to_float_rgb(torch.stack([
            torch.FloatTensor(np.ascontiguousarray(las[axis].astype('float32') / 65535))
            for axis in ["red", "green", "blue"]], dim=-1))

    # Extract intensity and convert to float [0-1]
    if hasattr(las, 'intensity'):
        # Heuristic to bring the intensity distribution in [0, 1]
        data.intensity = torch.FloatTensor(
            np.ascontiguousarray(las['intensity'].astype('float32'))
        ).clip(min=0, max=600) / 600

    # Extract classification
    if hasattr(las, 'classification'):
        data.y = torch.LongTensor(las.classification)
    else:
        # For unlabeled data, use the maximum label value to indicate unlabeled
        data.y = torch.full((data.pos.shape[0],), QC_NUM_CLASSES, dtype=torch.long)

    # Extract instance labels
    if hasattr(las, 'InsClass'):
        idx = torch.arange(data.num_points)
        ins_class = np.copy(las.InsClass)
        obj = torch.LongTensor(ins_class)
        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)
        y = torch.LongTensor(las.classification)
        data.obj = InstanceData(idx, obj, count, y, dense=True)
    else:
        # Handle the case where instance labels are not available
        data.obj = None

    return data


def save_panoptic_predictions_to_las(input_path, output_path, semantic_preds, instance_preds):
    """Save panoptic predictions (semantic + instance) to LAS file

    Args:
        input_path: Input LAS file path
        output_path: Output LAS file path
        semantic_preds: Semantic class predictions
        instance_preds: Instance ID predictions
    """
    print(f"Saving panoptic predictions to: {output_path}")

    # Read original LAS file
    input_las = laspy.read(input_path)

    # Create a new LAS file
    output_las = laspy.LasData(input_las.header)

    # Copy all point data
    for dimension in input_las.point_format.dimension_names:
        setattr(output_las, dimension, getattr(input_las, dimension))

    # Save semantic predictions as classification
    output_las.classification = semantic_preds.cpu().numpy().astype(np.uint8)

    # Save instance predictions as user_data or point_source_id
    instance_data = instance_preds.cpu().numpy()

    output_las.add_extra_dim(laspy.ExtraBytesParams('ins_class', type=np.uint8))

    output_las.ins_class = instance_data.astype(np.uint8)

    # Save LAS file
    output_las.write(output_path)
    print(f"Panoptic predictions saved to: {output_path}")


def find_las_files(input_dirs, processed_dirs=None, output_dir=None):
    """Find all LAS files that need to be processed."""
    las_files = []
    processed_files = set()

    # 收集已处理的文件名
    if processed_dirs:
        for proc_dir in processed_dirs:
            proc_path = Path(proc_dir)
            if proc_path.exists():
                for las_file in proc_path.glob("**/*.las"):
                    processed_files.add(las_file.stem)

    # 收集输出目录中已存在的文件名
    if output_dir:
        output_path = Path(output_dir)
        if output_path.exists():
            for las_file in output_path.glob("**/*.las"):
                processed_files.add(las_file.stem)

    # 查找需要处理的文件
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            for las_file in input_path.glob("**/*.las"):
                if las_file.stem not in processed_files:
                    las_files.append(las_file)
                else:
                    print(f"Skipping already processed file: {las_file}")

    return las_files


def batch_predict(args):
    # 下载并准备预训练权重
    ckpt_path = args.weights
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Weights file not found at: {ckpt_path}.")

    print(f"Using weights from: {ckpt_path}")

    # 加载配置
    print("Loading QC experiment configuration")
    cfg = init_config(overrides=["experiment=panoptic/qc"])

    # 移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # 初始化并应用transforms
    print("Initializing transforms")
    transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)

    # 实例化模型
    print("Instantiating model")
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(ckpt_path)
    model = model.eval().to(device)

    # 查找需要处理的文件
    las_files = find_las_files(
        input_dirs=args.input_dirs,
        processed_dirs=args.processed_dirs,
        output_dir=args.output_dir
    )

    print(f"Found {len(las_files)} files to process")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个文件
    for i, las_file in enumerate(las_files, 1):
        print(f"\nProcessing file {i}/{len(las_files)}: {las_file}")

        try:
            # 读取点云数据
            data = read_single_point_cloud(str(las_file))
            print(f"Loaded point cloud with {data.num_points} points")

            # 应用transforms
            nag = transforms_dict['pre_transform'](data)
            nag = nag.to(device)
            nag = transforms_dict['on_device_test_transform'](nag)

            # 运行推理
            with torch.no_grad():
                output = model(nag)

            # 获取预测
            raw_y, raw_instance_ids, raw_obj_pred = output.full_res_panoptic_pred(
                super_index_level0_to_level1=nag[0].super_index,
                sub_level0_to_raw=nag[0].sub if hasattr(nag[0], 'sub') else None
            )

            # 保存结果
            output_file = output_dir / f"{las_file.stem}_predicted.las"
            save_panoptic_predictions_to_las(
                str(las_file),
                str(output_file),
                raw_y,
                raw_instance_ids
            )

        except Exception as e:
            print(f"Error processing {las_file}: {e}")
            continue

    print(f"\nBatch processing completed! Processed {len(las_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Batch predict LAS files")
    parser.add_argument("--input_dirs", nargs="+",
                        default=['/mnt/d/qicheng/qc'],
                        help="Input directories containing LAS files")
    parser.add_argument("--output_dir",
                        default="/mnt/d/qicheng/pred",
                        help="Output directory for predictions")
    parser.add_argument("--processed_dirs", nargs="*", default=['area1', 'area2', 'area3', 'area4', 'area5', 'area6'],
                        help="Directories containing already processed files to skip")
    parser.add_argument("--weights",
                        default="logs/train/runs/2025-07-16_22-22-14/checkpoints/epoch_299.ckpt",
                        help="Path to model weights")
    parser.add_argument("--cpu", action="store_true",
                        help="Force use CPU instead of GPU")

    args = parser.parse_args()
    batch_predict(args)


if __name__ == "__main__":
    main()