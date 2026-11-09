# epihydra/__init__.py

__version__ = "0.1.0"
__author__ = "Barry Ryan"

from .epihydra import (
    VocabWrapper,
    AgeDataset,
    MultiClockDataset,
    AttentionPoolingHead,
    AgePredictor,
    MultiClockPredictor,
    load_backbone,
)

from .train import (
    build_dataloaders,
    build_multiclock_dataloaders,
    train_one_epoch,
    train_one_epoch_multiclock,
    evaluate,
    evaluate_multiclock,
)