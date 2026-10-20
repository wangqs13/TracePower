"""Run the experiments and save their figures."""
import argparse
import json
from pathlib import Path

from experiments import run_experiments
from plot import draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "1", "2"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    configs = {name: json.loads((root / "configs" / f"{name}.json").read_text()) for name in ("figure1", "figure2")}
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    results = run_experiments(configs, output, args.workers, args.figure)
    draw(results, output)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
