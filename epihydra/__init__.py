# epihydra/__init__.py

__version__ = "0.1.0"
__author__ = "Barry Ryan"

# Core model — the main thing users will import
from .epihydra import EpiHydra  # or whatever the main class is called

# Training and inference entry points
from .train import train
from .run import run
from .run_multiclock import run_multiclock
