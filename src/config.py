"""
SketchMamba - Configuration
All hyperparameters, paths, and training settings in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_classes: int = 58
    n_gmm_components: int = 10
    max_stroke_segments: int = 32  # max stroke index for segment embedding
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10

    # Loss weights
    lambda_cls: float = 1.0
    lambda_gen: float = 1.0
    lambda_pen: float = 0.5

    # Cosine annealing
    lr_min: float = 1e-5

    # Curriculum learning
    curriculum_warmup_epochs: int = 10
    curriculum_start_ratio: float = 1.0
    curriculum_end_ratio: float = 0.1

    # Augmentation
    stroke_permute_prob: float = 0.3

    # Mixed precision
    use_amp: bool = True

    # Workers
    num_workers: int = 4


@dataclass
class DataConfig:
    """Data paths and preprocessing settings."""
    data_dir: str = "quickdraw_preprocessed/sketchrnn_npz"
    normalize: bool = True
    # Will be computed from training data and saved with the checkpoint
    scale_factor: float = 1.0


@dataclass
class PathConfig:
    """Output directory structure (all relative to project root)."""
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"
    figures_dir: str = "results/figures"
    metrics_dir: str = "results/metrics"
    export_dir: str = "results/export"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int = 42
    device: str = "cuda"

    def create_dirs(self):
        """Create all output directories."""
        root = get_project_root()
        for dir_name in [
            self.paths.results_dir,
            self.paths.checkpoint_dir,
            self.paths.log_dir,
            self.paths.figures_dir,
            self.paths.metrics_dir,
            self.paths.export_dir,
        ]:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct Config from a dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            paths=PathConfig(**d.get("paths", {})),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
        )
