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
    # Extract run_id - cfg.run could be a string or object with run_id field
    if isinstance(cfg.run, str):
        run_id = cfg.run
    else:
        run_id = cfg.run.run_id

    has_runs_override = any(o.startswith("runs@run=") or o.startswith("+runs@run=") for o in overrides)

    # Convert run= overrides to runs@run= to use config group
    new_overrides = []
    for o in overrides:
        if o.startswith("run="):
            # Extract the run_id from run=<value> and convert to runs@run=<value>
            run_value = o.split("=", 1)[1]
            new_overrides.append(f"runs@run={run_value}")
        else:
            new_overrides.append(o)
    overrides = new_overrides

    # Only add runs@run= if not already specified
    has_runs_override = any(o.startswith("runs@run=") or o.startswith("+runs@run=") for o in overrides)
    if not has_runs_override:
        overrides = [f"runs@run={run_id}"] + list(overrides)

    cmd = ["python", "-u", "-m", "src.train", *overrides]
    print("Executing:", " ".join(cmd))

    subprocess.run(cmd, cwd=str(root), env=os.environ.copy(), check=True)

if __name__ == "__main__":
    main()