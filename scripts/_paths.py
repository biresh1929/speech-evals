"""Where the studies read data and write outputs. Set SPEECH_EVALS_ROOT to use another folder; the default is this
repo's workspace/ (laid out as the original working folder: voice_dataset/, rumik_samples/, code/models/)."""
import os
from pathlib import Path

DATA_ROOT = os.environ.get("SPEECH_EVALS_ROOT", str(Path(__file__).resolve().parents[1] / "workspace")).replace("\\", "/")
