"""
Convenience launcher for hyperparameter sweeps via Hydra Optuna Sweeper.

Wraps the verbose --multirun command into a simple CLI so you can still type:
  python sweep.py --domain bacteria

Requirements (install once):
  pip install hydra-optuna-sweeper hydra-submitit-launcher

Usage
-----
  # Local sweep — saves to bacteria_sweep.db automatically:
  python sweep.py --domain bacteria

  # Override number of trials:
  python sweep.py --domain microplastic --n-trials 50

  # Custom storage path (default is ./<domain>_sweep.db):
  python sweep.py --domain mlrod --storage sqlite:////scratch/user/mlrod_sweep.db

  # SLURM: one job per trial — set your partition name:
  python sweep.py --domain bacteria --slurm --partition gpu

  # SLURM with custom resource overrides:
  python sweep.py --domain bacteria --slurm --partition gpu \\
      --slurm-args "hydra.launcher.timeout_min=240 hydra.launcher.mem_gb=64"

Equivalent raw Hydra commands (if you prefer):
  python main.py --multirun domain=bacteria hydra/sweeper=optuna_bacteria mode=train
  python main.py --multirun domain=bacteria hydra/sweeper=optuna_bacteria mode=train \\
      hydra.sweeper.storage=sqlite:///bacteria_sweep.db \\
      hydra/launcher=submitit_slurm hydra.launcher.partition=gpu
"""
from __future__ import annotations

import argparse
import subprocess
import sys

DOMAINS = ["microplastic", "bacteria", "mlrod", "pollen", "textile"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a Hydra Optuna hyperparameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain",     required=True, choices=DOMAINS)
    parser.add_argument("--n-trials",   type=int,   default=None,
                        help="Override n_trials from sweeper config")
    parser.add_argument("--storage",    default=None,
                        help="Optuna storage URL (default: sqlite:///<domain>_sweep.db)")
    parser.add_argument("--study-name", default=None,
                        help="Override study name")
    parser.add_argument("--n-jobs",     type=int,   default=None,
                        help="Parallel trials (needs shared --storage)")
    parser.add_argument("--slurm",      action="store_true",
                        help="Submit each trial as a separate SLURM job")
    parser.add_argument("--partition",  default=None,
                        help="SLURM partition (required with --slurm)")
    parser.add_argument("--slurm-args", default="",
                        help="Extra Hydra overrides for the SLURM launcher")
    args = parser.parse_args()

    if args.slurm and not args.partition:
        parser.error("--partition is required when using --slurm")

    # Default to the sweeps/ directory so the file is on the persisted volume
    # inside Docker and readable by the optuna-dashboard service.
    import os
    sweeps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps")
    os.makedirs(sweeps_dir, exist_ok=True)
    storage = args.storage or f"sqlite:///{sweeps_dir}/{args.domain}_sweep.db"

    cmd = [
        sys.executable, "main.py",
        "--multirun",
        f"domain={args.domain}",
        f"hydra/sweeper=optuna_{args.domain}",
        "mode=train",
    ]

    if args.n_trials is not None:
        cmd.append(f"hydra.sweeper.n_trials={args.n_trials}")
    cmd.append(f"hydra.sweeper.storage={storage!r}")
    if args.study_name is not None:
        cmd.append(f"hydra.sweeper.study_name={args.study_name}")
    if args.n_jobs is not None:
        cmd.append(f"hydra.sweeper.n_jobs={args.n_jobs}")

    if args.slurm:
        cmd.append("hydra/launcher=submitit_slurm")
        cmd.append(f"hydra.launcher.partition={args.partition}")
        if args.slurm_args:
            cmd.extend(args.slurm_args.split())

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
