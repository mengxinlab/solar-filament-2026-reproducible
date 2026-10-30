"""Explicit pipeline commands; existing run directories are never overwritten."""
import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "classical-cv", "neural-cv", "finalize", "infer"])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/MAGFiLO_1.0_Kaggle_2026"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--classical-run", type=Path)
    parser.add_argument("--neural-run", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    from . import pipeline as p
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.data_root, args.output)
    elif args.command == "classical-cv":
        p.classical_cv(args.data_root, args.processed, args.config, args.output, args.workers)
    elif args.command == "neural-cv":
        p.neural_cv(args.data_root, args.processed, args.config, args.output)
    elif args.command == "finalize":
        p.finalize(args.data_root, args.processed, args.classical_run, args.neural_run, args.output)
    elif args.command == "infer":
        assert not args.output.exists()
        p.make_submission(args.data_root, args.model, args.output, args.device)


if __name__ == "__main__":
    main()
