import os 
from dataclasses import dataclass, field
from typing import List

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelConfig:
    d_model: int=256
    n_layers: int=6
    d_state: int=16
    d_conv: int=4
    expand: int=2
    num_classes: int=58
    n_gmm_components: int=10
    max_stroke_segments: int=32
    dropout: float=0.1


@dataclass
class TrainingConfig:
    epochs: int=50
    batch_size: int=256
    lr: float=1e-3
    weight_decay: float=1e-4
    grad_clip_norm: float=1.0
    early_stopping_patience: int=10
    lambda_cls: float=1.0
    lambda_gen: float=1.0
    lambda_pen: float=0.5
    lr_min: float=1e-5
    curriculum_warmup_epochs: int=10
    curriculum_start_ratio: float=1.0
    curriculum_end_ratio: float=0.1
    stroke_permute_prob: float=0.3
    use_amp: bool=True
    num_workers: int=4
    

@dataclass
class DataConfig:
    data_dir: str="quickdraw_preprocessed/sketchrnn_npz"
    normalize:  bool=True
    scale_factor: float=1.0


@dataclass
class PathConfig:
    results_dir: str="results"
    checkpoints_dir: str="results/checkpoints"
    log_dir: str="results/log"
    figures_dir: str="results/figures"
    metrics_dir: str="results/metrics"
    export_dir: str="results/export"
