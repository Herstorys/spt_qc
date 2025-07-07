# src/datamodules/las_datamodule.py
import logging
from src.datamodules.base import BaseDataModule
from src.datasets.qc_dataset import QcDataset

log = logging.getLogger(__name__)


class QcDataModule(BaseDataModule):
    """DataModule for LAS point cloud dataset."""

    _DATASET_CLASS = QcDataset
    _MINIDATASET_CLASS = QcDataset
