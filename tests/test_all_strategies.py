"""
Smoke test: run all 6 training strategies for 1 epoch each.

Usage:
    python -m tests.test_all_strategies
    python -m tests.test_all_strategies --epochs 2 --num-workers 2
"""
import argparse
import copy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Maps strategy name -> (config file, training module)
STRATEGIES = [
    ("Baseline",         "strategy1_baseline.yaml",  "src.training.train_baseline"),
    ("Heavy Augment",    "stategy2_augmented.yaml",   "src.training.train_augmented"),
    ("Curriculum",       "strategy3_curriculum.yaml", "src.training.train_curriculum"),
    ("Self-Supervised",  "strategy4_ssl.yaml",        "src.training.train_ssl"),
    ("Hard Neg Mining",  "strategy5_hard_neg.yaml",   "src.training.train_hard_neg"),
    ("Multi-Task",       "strategy6_multitask.yaml",  "src.training.train_multitask"),
]


def make_test_config(base_config: dict, epochs: int, num_workers: int, save_dir: str) -> dict:
    """Return a copy of the config patched for a quick smoke test."""
    cfg = copy.deepcopy(base_config)
    cfg["num_epochs"] = epochs
    cfg["batch_size"] = min(cfg.get("batch_size", 32), 8)
    cfg["num_workers"] = num_workers
    cfg["save_dir"] = save_dir

    # SSL: patch both phases
    if "phase_1" in cfg:
        cfg["phase_1"]["num_epochs"] = epochs
        cfg["phase_1"]["batch_size"] = 8
    if "phase_2" in cfg:
        cfg["phase_2"]["num_epochs"] = epochs
        cfg["phase_2"]["batch_size"] = 8
        cfg["phase_2"]["num_workers"] = num_workers
        cfg["phase_2"]["freeze_backbone_epochs"] = 0  # skip freeze when only 1 epoch

    # Curriculum: collapse all phases to start at epoch 0 so epoch 0 is always covered
    if "curriculum" in cfg:
        all_manips = []
        for phase in cfg["curriculum"].values():
            all_manips.extend(phase["manipulations"])
        all_manips = list(dict.fromkeys(all_manips))  # dedupe, preserve order
        cfg["curriculum"] = {
            "phase_1": {"epochs": [0, max(epochs - 1, 0)], "manipulations": all_manips}
        }

    return cfg


def run_strategy(name: str, config_file: str, module: str,
                 epochs: int, num_workers: int, tmpdir: Path) -> tuple[bool, float, str]:
    """Write a patched config and run the training module. Returns (passed, elapsed, error)."""
    config_path = CONFIGS_DIR / config_file
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    save_dir = str(tmpdir / name.replace(" ", "_").lower())
    test_cfg = make_test_config(base_cfg, epochs, num_workers, save_dir)

    tmp_cfg_path = tmpdir / f"test_{config_file}"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(test_cfg, f)

    # Patch the config path the module reads via env-level monkeypatch:
    # we symlink the original config name to our tmp file so the module
    # picks it up transparently.
    original_cfg = CONFIGS_DIR / config_file
    backup = CONFIGS_DIR / (config_file + ".bak")
    original_cfg.rename(backup)
    try:
        tmp_cfg_path.replace(original_cfg)  # atomic on same fs; fallback below
    except Exception:
        import shutil
        shutil.copy(str(tmp_cfg_path), str(original_cfg))

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0
        passed = result.returncode == 0
        error = "" if passed else (result.stderr[-2000:] if result.stderr else result.stdout[-2000:])
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        passed = False
        error = "TIMEOUT after 600s"
    finally:
        # Restore original config
        original_cfg.unlink(missing_ok=True)
        backup.rename(original_cfg)

    return passed, elapsed, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per strategy (default: 1)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--strategies", nargs="+", metavar="N", type=int,
                        help="Run only these strategy numbers 1-6 (default: all)")
    args = parser.parse_args()

    selected = set(args.strategies) if args.strategies else set(range(1, 7))
    strategies_to_run = [(i + 1, *s) for i, s in enumerate(STRATEGIES) if (i + 1) in selected]

    print("=" * 60)
    print(f"Smoke Test: {len(strategies_to_run)} strategies × {args.epochs} epoch(s)")
    print("=" * 60)

    results = []
    with tempfile.TemporaryDirectory(prefix="deepfake_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        for num, name, config_file, module in strategies_to_run:
            print(f"\n[{num}/6] {name} ...", flush=True)
            passed, elapsed, error = run_strategy(
                name, config_file, module, args.epochs, args.num_workers, tmpdir
            )
            status = "PASS" if passed else "FAIL"
            print(f"      {status}  ({elapsed:.1f}s)")
            if not passed:
                print(f"      Error tail:\n{error}")
            results.append((num, name, passed, elapsed))

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    all_passed = True
    for num, name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Strategy {num} ({name:<18}): {status}  ({elapsed:.1f}s)")
        all_passed = all_passed and passed

    print("=" * 60)
    if all_passed:
        print("All strategies passed.")
    else:
        print("Some strategies FAILED. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
