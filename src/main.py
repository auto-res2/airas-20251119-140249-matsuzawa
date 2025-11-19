"""src/main.py
Hydra orchestrator – launches a single training job as a subprocess.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="../config", config_name="config")
def main(_cfg):
    # Original CWD before Hydra changes directory
    root = Path(hydra.utils.get_original_cwd())

    overrides = HydraConfig.get().overrides.task  # exact CLI overrides
    cmd = ["python", "-u", "-m", "src.train", *overrides]
    print("Executing:", " ".join(cmd))

    subprocess.run(cmd, cwd=str(root), env=os.environ.copy(), check=True)

if __name__ == "__main__":
    main()