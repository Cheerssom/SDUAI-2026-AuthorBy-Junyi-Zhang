"""Centralized configuration — replaces hardcoded paths.

Usage:
    from config import CFG
    data_path = CFG.data_root / "cifar-100-python"
"""
import os
from pathlib import Path


class Config:
    def __init__(self):
        # Data roots (override via environment variables)
        self.data_root = Path(os.environ.get("JLT_DATA",
                                             r"G:\科研新\data"))
        self.baseline_dir = Path(os.environ.get(
            "JLT_BASELINE", r"E:\机器学习\大作业"))
        self.output_dir = Path(os.environ.get("JLT_OUTPUT",
                                              r"G:\科研新\JLT-Head"))
        self.device = os.environ.get("JLT_DEVICE",
                                     "cuda" if torch_cuda() else "cpu")

    @property
    def cifar_root(self):
        return self.data_root / "cifar-100-python"

    @property
    def runs_dir(self):
        return self.output_dir / "runs"

    @property
    def cache_dir(self):
        return self.output_dir / "cache"

    @property
    def results_dir(self):
        return self.output_dir / "experiments"


def torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


CFG = Config()
