"""src/main.py
Hydra orchestrator – launches a single training job as a subprocess.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    # Original CWD before Hydra changes directory
    root = Path(hydra.utils.get_original_cwd())

    overrides = HydraConfig.get().overrides.task  # exact CLI overrides

    # Add config group override if not present
    run_id = cfg.run
    has_runs_override = any(o.startswith("runs@run=") or o.startswith("+runs@run=") for o in overrides)
    if not has_runs_override:
        overrides = [f"+runs@run={run_id}"] + list(overrides)

    cmd = ["python", "-u", "-m", "src.train", *overrides]
    print("Executing:", " ".join(cmd))

    subprocess.run(cmd, cwd=str(root), env=os.environ.copy(), check=True)

if __name__ == "__main__":
    main()