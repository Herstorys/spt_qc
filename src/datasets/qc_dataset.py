# src/datasets/las_dataset.py
import os
from typing import List

import numpy as np
import laspy
from src.data import Data, InstanceData
import torch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.datasets import BaseDataset
from src.utils.color import to_float_rgb


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


QC_NUM_CLASSES = 2  # Number of classes in the dataset
THING_CLASSES = [1]
QC_STUFF_CLASSES = [0]
# Define your ID to train ID mapping
# ID2TRAINID = np.asarray([0, 1, 2, 3])


class QcDataset(BaseDataset):
    """Custom dataset for LAS point cloud files."""
    @property
    def class_names(self) -> List[str]:
        """Return the class names for the dataset."""
        return [
            'other',
            'tree',
        ]

    @property
    def num_classes(self) -> int:
        return QC_NUM_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization."""
        return [
            [255, 255, 0],  # other - yellow
            [0, 255, 0],  # tree - green
        ]

    @property
    def stuff_classes(self) -> List[int]:
        return QC_STUFF_CLASSES

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
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

        # Extract point coordinates
        # pos = torch.from_numpy(np.vstack((las.x, las.y, las.z)).T).float()
        #
        # # Store position and offset
        # pos_offset = pos[0].clone()
        # data.pos = pos - pos_offset
        # data.pos_offset = pos_offset

        # Apply the scale provided by the LAS header
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
            data.y = torch.full((data.pos.shape[0],), self.num_classes, dtype=torch.long)

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

    @property
    def all_base_cloud_ids(self):
        """Define the cloud IDs for train, validation, and test sets."""
        # List your LAS files in each split
        # Example: If your directory structure is like:
        # data/las_dataset/raw/train/cloud1.las, cloud2.las, ...
        # data/las_dataset/raw/val/cloud3.las, ...
        # data/las_dataset/raw/test/cloud4.las, ...

        # train_ids = [f.split('.')[0] for f in os.listdir(os.path.join(self.raw_dir, 'train'))
        #              if f.endswith('.las')]
        # val_ids = [f.split('.')[0] for f in os.listdir(os.path.join(self.raw_dir, 'val'))
        #            if f.endswith('.las')]
        # test_ids = [f.split('.')[0] for f in os.listdir(os.path.join(self.raw_dir, 'test'))
        #             if f.endswith('.las')]
        #
        # print('train:', train_ids)
        # print('val:', val_ids)
        # print('test:', test_ids)
        # data= {
        #     'train': train_ids,
        #     'val': val_ids,
        #     'test': test_ids
        # }
        data = {
            'train': ['area2','area4','area5'], # , 'area7', 'area8', 'area9'
            'val': ['area6'],
            'test': ['area3']
        }

        return data

    @property
    def all_cloud_ids(self) -> List[str]:
        return self.all_base_cloud_ids

    def download_dataset(self) -> None:
        pass

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in self.cloud_ids, return the
        path (relative to self.raw_dir) of the corresponding raw cloud.
        """
        if id in self.all_cloud_ids['train']:
            stage = 'train'
        elif id in self.all_cloud_ids['val']:
            stage = 'val'
        elif id in self.all_cloud_ids['test']:
            stage = 'test'
        else:
            raise ValueError(f"Unknown cloud id '{id}'")
        return os.path.join(stage, self.id_to_base_id(id) + '.las')

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input processed path."""
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            os.path.splitext(processed_path)[0].split(os.sep)[-3:]

        # Remove any tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = os.path.join(self.raw_dir, stage, base_cloud_id + '.las')

        return raw_path

    @property
    def raw_file_structure(self) -> str:
        """Define the structure of raw files in the dataset."""
        return f"""
    {self.root}/
        └── raw/
            ├── train/
            │   └── {{cloud_id}}.las
            ├── val/
            │   └── {{cloud_id}}.las
            └── test/
                └── {{cloud_id}}.las
        """