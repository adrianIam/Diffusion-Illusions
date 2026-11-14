"""Prepare a bounded comparison; run GPU jobs only with explicit --run.

python scripts/compare_flippy.py --output-root /content/comparison
python scripts/compare_flippy.py --output-root /content/comparison --run --limit 2
python scripts/compare_flippy.py --output-root /content/comparison --summarize
"""
import argparse
import csv
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.flippy import FlippyConfig, run, write_json


def prepare(root, iterations=1000, max_seconds=None):
    suite = json.loads((ROOT / "configs/flippy_evaluation.json").read_text())
    plans = []
    for pair in suite["pairs"]:
        for seed in suite["seeds"]:
            for backend in ("sd1", "sdxl"):
                name = f"{pair['name']}-seed{seed}-{backend}"
                config = FlippyConfig(
                    backend=backend, prompt_a=pair["prompt_a"], prompt_b=pair["prompt_b"], seed=seed,
                    iterations=iterations, max_seconds=max_seconds,
                    guidance_scale=60. if backend == "sd1" else 20.,
                    output_dir=str((root / "trials" / name).resolve()),
                )
                path = root / "configs" / f"{name}.json"
                settings = asdict(config)
                if path.exists() and FlippyConfig(**json.loads(path.read_text())) != config:
                    raise ValueError(f"Existing configuration differs: {path}; choose a new output root")
                write_json(path, settings)
                plans.append((name, pair["name"], config, path))
    ratings_path = root / "ratings.csv"
    if not ratings_path.exists():
        with ratings_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run", "pair", "seed", "backend", "upright_recognition_0_4",
                             "flipped_recognition_0_4", "aesthetics_0_4", "balance_0_4", "notes"])
            writer.writerows([name, pair, c.seed, c.backend, "", "", "", "", ""] for name, pair, c, _ in plans)
    return plans


def summarize(root):
    rows = []
    for path in sorted((root / "trials").glob("*/manifest.json")):
        data = json.loads(path.read_text())
        settings, metrics = data["config"], data["metrics"]
        rows.append(dict(
            run=path.parent.name, backend=settings["backend"], seed=settings["seed"],
            status=data["status"], iterations=data["completed_iterations"],
            model_size=settings["model_size"], image_size=settings["image_size"],
            guidance_scale=settings["guidance_scale"], view_sampling=settings["view_sampling"],
            unet_evaluations=sum(metrics["view_steps"]), optimization_seconds=metrics["elapsed_seconds"],
            load_seconds=sum(s["load_seconds"] for s in metrics["sessions"]),
            peak_allocated_bytes=metrics["peak_allocated_bytes"], peak_reserved_bytes=metrics["peak_reserved_bytes"],
            gpu=(data["environment"]["gpu"] or {}).get("name"), output_dir=str(path.parent),
        ))
    if not rows:
        print("No trial manifests found; quality and performance are pending.")
        return
    with (root / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measured runs to {root / 'measurements.csv'}; human ratings remain separate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--run", action="store_true", help="Execute generated jobs on the attached GPU")
    parser.add_argument("--limit", type=int, default=12, help="Maximum jobs in this invocation")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_root)
        return
    if args.limit < 1:
        parser.error("--limit must be positive")
    plans = prepare(args.output_root, args.iterations, args.max_seconds)
    print(f"Prepared {len(plans)} explicit configurations in {args.output_root / 'configs'}")
    if not args.run:
        print("No GPU jobs started. Review the settings; add --run to execute them.")
        return
    import torch
    if not torch.cuda.is_available():
        parser.error("GPU required; use a Colab GPU runtime")
    for name, _, config, _ in plans[:args.limit]:
        folder = Path(config.output_dir)
        manifest = folder / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["status"] == "complete":
            print("Already complete:", name)
            continue
        resume = folder / "checkpoint.pt" if args.resume_existing else None
        print("Running:", name, flush=True)
        run(config, resume=resume if resume and resume.exists() else None)
        gc.collect()
        torch.cuda.empty_cache()
    summarize(args.output_root)


if __name__ == "__main__":
    main()
