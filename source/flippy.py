"""Resumable two-view image optimization, independent of SD1/SDXL conditioning.

Run from the repository root: python -m source.flippy --smoke --backend sdxl
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

import numpy as np
from PIL import Image
import torch
from torch import nn


@dataclass
class FlippyConfig:
    backend: str = "sdxl"
    model_id: str = ""
    revision: str | None = None
    variant: str | None = None
    prompt_a: str = "an intricate detailed hb pencil sketch of a giraffe head"
    prompt_b: str = "an intricate detailed hb pencil sketch of a penguin"
    negative_prompt: str = ""
    seed: int = 0
    image_size: int = 256
    model_size: int | None = None
    hidden_dim: int = 256
    num_features: int = 128
    fourier_scale: float = 10.
    iterations: int = 1000
    learning_rate: float = 1e-4
    guidance_scale: float = 20.
    noise_coef: float = .1
    min_timestep: float = .02
    max_timestep: float = .98
    view_weights: tuple = (1., 1.)
    view_sampling: str = "random"
    dtype: str = "auto"
    vae_dtype: str = "float32"
    offload_text_encoders: bool = True
    attention_slicing: bool = False
    save_interval: int = 50
    max_seconds: float | None = None
    output_dir: str = "untracked/flippy-sdxl"

    def __post_init__(self):
        if self.backend not in {"sd1", "sdxl"}:
            raise ValueError("backend must be sd1 or sdxl")
        if not self.model_id:
            self.model_id = ("stabilityai/stable-diffusion-xl-base-1.0" if self.backend == "sdxl"
                             else "CompVis/stable-diffusion-v1-4")
        if self.model_size is None:
            self.model_size = 1024 if self.backend == "sdxl" else 512
        for name in ("image_size", "model_size", "hidden_dim", "num_features", "iterations", "save_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_size < 64 or self.model_size % 8:
            raise ValueError("model_size must be a multiple of 8 and at least 64")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "guidance_scale", "noise_coef", "fourier_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.min_timestep < self.max_timestep < 1:
            raise ValueError("Require 0 <= min_timestep < max_timestep < 1")
        self.view_weights = tuple(self.view_weights)
        if (len(self.view_weights) != 2 or
                any(not math.isfinite(w) or w < 0 for w in self.view_weights) or
                sum(self.view_weights) <= 0):
            raise ValueError("view_weights must contain two finite nonnegative values with positive sum")
        if self.view_sampling not in {"random", "both"}:
            raise ValueError("view_sampling must be random or both")
        if self.dtype not in {"auto", "float16", "float32", "bfloat16"}:
            raise ValueError("Unsupported UNet dtype")
        if self.vae_dtype not in {"float32", "bfloat16"}:
            raise ValueError("VAE must use float32 or explicitly supported bfloat16")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")
        if not self.prompt_a.strip() or not self.prompt_b.strip():
            raise ValueError("Both prompts must be nonempty")
        if not self.output_dir:
            raise ValueError("output_dir must be nonempty")

    @classmethod
    def smoke(cls, backend="sdxl", **overrides):
        values = dict(backend=backend, iterations=4, image_size=64, model_size=512,
                      hidden_dim=64, num_features=32, save_interval=2, view_sampling="both",
                      guidance_scale=20. if backend == "sdxl" else 60.,
                      output_dir=f"untracked/flippy-{backend}-smoke")
        values.update(overrides)
        return cls(**values)


class FourierImage(nn.Module):
    """Dependency-light adaptation of learnable_textures.LearnableImageFourier.

    Same square UV grid, frequencies, 1x1 MLP, BatchNorm and sigmoid. Buffers make
    frequencies/features persistent without presenting them to the optimizer.
    Original code: Copyright (c) 2023 Ryan Burgert; see learnable_textures.py.
    """
    def __init__(self, size, hidden_dim=256, num_features=128, scale=10.):
        super().__init__()
        # Keep the original initialization order for the SD1 regression check.
        y = np.linspace(0, 1, size, endpoint=False)
        grid = torch.tensor(np.stack(np.meshgrid(y, y), -1)).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous()
        frequencies = torch.randn(2, num_features) * scale
        phases = (grid.permute(0, 2, 3, 1).reshape(-1, 2) @ frequencies)
        phases = phases.view(1, size, size, num_features).permute(0, 3, 1, 2) * (2 * torch.pi)
        self.register_buffer("features", torch.cat([phases.sin(), phases.cos()], dim=1))
        h = hidden_dim
        self.model = nn.Sequential(
            nn.Conv2d(num_features * 2, h, 1), nn.ReLU(), nn.BatchNorm2d(h),
            nn.Conv2d(h, h, 1), nn.ReLU(), nn.BatchNorm2d(h),
            nn.Conv2d(h, h, 1), nn.ReLU(), nn.BatchNorm2d(h),
            nn.Conv2d(h, 3, 1), nn.Sigmoid(),
        )

    def forward(self):
        return self.model(self.features)  # [1,3,H,W]


def rotate_view(image, view):
    if view not in (0, 1):
        raise ValueError("view must be 0 (upright) or 1 (180 degrees)")
    return image if view == 0 else image.rot90(2, dims=(-2, -1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of CUDA devices")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    data = json.dumps(value, indent=2, default=str, allow_nan=False).encode()
    atomic_save(path, lambda stream: stream.write(data))


def resume_settings(config):
    settings = asdict(config)
    # Budgets may increase, and outputs may move from local disk to Drive.
    for key in ("iterations", "max_seconds", "save_interval", "output_dir"):
        settings.pop(key)
    return settings


def environment_metadata(device):
    versions = {}
    for name in ("torch", "numpy", "Pillow", "diffusers", "transformers", "accelerate",
                 "huggingface-hub", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = dict(python=platform.python_version(), packages=versions, git_commit=commit,
                device=str(device), torch_cuda=torch.version.cuda, gpu=None)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = dict(name=props.name, total_vram_bytes=props.total_memory,
                           capability=list(torch.cuda.get_device_capability(device)))
    return info


def checkpoint(path, config, image, optimizer, iteration, metrics, identity):
    payload = dict(format_version=1, config=asdict(config), image=image.state_dict(),
                   optimizer=optimizer.state_dict(), iteration=iteration, rng=rng_state(),
                   metrics=metrics, identity=identity)
    atomic_save(path, lambda stream: torch.save(payload, stream))


def load_checkpoint(path, config, image, optimizer, identity):
    # Only load a checkpoint you produced/trust: optimizer/RNG serialization uses pickle.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    previous = FlippyConfig(**saved["config"])
    before, after = resume_settings(previous), resume_settings(config)
    mismatches = [key for key in before if before[key] != after[key]]
    if mismatches:
        raise ValueError("Incompatible resume settings: " + ", ".join(mismatches))
    if saved["identity"] != identity:
        raise ValueError("Resume model/scheduler/dtype/device/dependency identity differs")
    if config.iterations < saved["iteration"]:
        raise ValueError("iterations must be at least the saved iteration")
    image.load_state_dict(saved["image"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    return saved["iteration"], saved["metrics"]


@torch.no_grad()
def export_views(image, folder):
    was_training = image.training
    image.eval()  # Preview must not update BatchNorm buffers or consume RNG.
    try:
        rgb = image().squeeze(0).permute(1, 2, 0).float().cpu().clamp(0, 1).numpy()
    finally:
        image.train(was_training)
    upright = Image.fromarray((rgb * 255).round().astype(np.uint8))
    flipped = upright.transpose(Image.Transpose.ROTATE_180)
    comparison = Image.new("RGB", (upright.width * 2, upright.height))
    comparison.paste(upright, (0, 0))
    comparison.paste(flipped, (upright.width, 0))
    for filename, value in (("image.png", upright), ("upright.png", upright),
                            ("flipped.png", flipped), ("comparison.png", comparison)):
        atomic_save(Path(folder) / filename, lambda stream, im=value: im.save(stream, format="PNG"))
    return comparison


def run(config, resume=None, callback=None, backend=None):
    """Run a real GPU experiment, or injected fake backend for CPU checks.

    max_seconds is a per-invocation optimization budget (model loading excluded).
    iterations is the total target across resumes. No setting changes on OOM.
    """
    from source.illusion_backends import DiffusionSDSBackend

    config = FlippyConfig(**asdict(config))  # Revalidate notebook edits.
    if backend is None and not torch.cuda.is_available():
        raise RuntimeError("GPU required. In Colab select Runtime > Change runtime type > GPU, then reconnect.")
    device = torch.device("cuda:0") if backend is None else backend.device
    folder = Path(config.output_dir).expanduser().resolve()
    if folder.exists() and any(folder.iterdir()) and resume is None:
        raise ValueError("Output directory is nonempty; choose a new directory or resume its checkpoint")
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "checkpoint.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    load_started = time.perf_counter()
    try:
        backend = backend if backend is not None else DiffusionSDSBackend(config, device=device)
        backend.prepare((config.prompt_a, config.prompt_b), config.negative_prompt)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "GPU memory exhausted while loading models or caching prompts. Optimization has not started; "
            "existing checkpoints are untouched. Try a fresh runtime or a GPU with more VRAM. "
            "For a NEW smaller experiment use the smoke preset (512 model input, 64 image, 64 hidden, 32 features)."
        ) from exc
    if getattr(backend, "resolved_revision", None):
        # Persist an immutable revision, making later resumes independent of Hub main.
        config.revision = backend.resolved_revision
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    seed_all(config.seed)  # Model-loading internals cannot perturb image initialization.
    image = FourierImage(config.image_size, config.hidden_dim, config.num_features, config.fourier_scale).to(device)
    optimizer = torch.optim.SGD(image.parameters(), lr=config.learning_rate)
    environment = environment_metadata(device)
    backend_info = backend.metadata()
    # A default/branch revision is pinned to its resolved SHA in saved settings.
    # The requested spelling can then change on resume without changing weights.
    identity_backend = {key: value for key, value in backend_info.items() if key != "requested_revision"}
    identity = dict(backend=identity_backend, device=environment["device"], gpu=environment["gpu"],
                    packages=environment["packages"], torch_cuda=environment["torch_cuda"])
    metrics = dict(elapsed_seconds=0., load_seconds=load_seconds, view_steps=[0, 0],
                   peak_allocated_bytes=0, peak_reserved_bytes=0, last_gradient_norm=None,
                   image_parameters_changed=False, frozen_model_gradients_absent=True,
                   sessions=[])
    iteration = 0
    if resume:
        iteration, metrics = load_checkpoint(resume, config, image, optimizer, identity)
    initial_parameters = [p.detach().clone() for p in image.parameters()]
    completed_before = iteration
    elapsed_before = metrics["elapsed_seconds"]
    started = time.perf_counter()
    session_record = dict(started_utc=datetime.now(timezone.utc).isoformat(), load_seconds=load_seconds,
                          start_iteration=iteration)
    metrics["sessions"].append(session_record)
    weights = [2 * w / sum(config.view_weights) for w in config.view_weights]

    def update_metrics():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics["peak_allocated_bytes"] = max(metrics["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device))
            metrics["peak_reserved_bytes"] = max(metrics["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device))
        metrics["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        metrics["image_parameters_changed"] = metrics["image_parameters_changed"] or any(
            not torch.equal(p.detach(), old) for p, old in zip(image.parameters(), initial_parameters))
        metrics["frozen_model_gradients_absent"] = all(
            p.grad is None and not p.requires_grad for module in backend.model_modules() for p in module.parameters())
        session_record.update(end_iteration=iteration,
                              optimization_seconds=metrics["elapsed_seconds"] - elapsed_before)

    def manifest(status, error=None):
        write_json(folder / "manifest.json", dict(
            status=status, error=error, completed_iterations=iteration,
            config=asdict(config), environment=environment, backend=backend_info, metrics=metrics,
            quality_evaluation="pending human assessment; execution is not evidence of illusion quality",
        ))

    # An OOM during the first iteration still leaves a complete initialization checkpoint.
    checkpoint(checkpoint_path, config, image, optimizer, iteration, metrics, identity)
    manifest("running")
    try:
        while iteration < config.iterations:
            optimizer.zero_grad(set_to_none=True)
            views = [random.randrange(2)] if config.view_sampling == "random" else [0, 1]
            for view in views:
                if weights[view] == 0:
                    continue
                # Recompute the image per view to release the VAE graph between calls.
                loss = backend.loss(rotate_view(image(), view), view, weights[view])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite SDS surrogate")
                loss.backward()
                metrics["view_steps"][view] += 1
            gradients = [p.grad for p in image.parameters() if p.grad is not None]
            if gradients:
                if not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Nonfinite image parameter gradient")
                norm = sum(g.detach().float().square().sum() for g in gradients).sqrt().item()
                if norm == 0:
                    raise FloatingPointError("Zero image parameter gradient")
                metrics["last_gradient_norm"] = norm
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in image.parameters()):
                    raise FloatingPointError("Nonfinite image parameters after optimizer step")
            iteration += 1
            budget_reached = config.max_seconds is not None and time.perf_counter() - started >= config.max_seconds
            if iteration % config.save_interval == 0 or iteration == config.iterations or budget_reached:
                update_metrics()
                if not metrics["frozen_model_gradients_absent"]:
                    raise RuntimeError("Pretrained parameters must remain frozen and have no gradients")
                preview = export_views(image, folder)
                # Archive periodic views as well as updating the latest comparison.
                atomic_save(folder / f"preview-{iteration:06d}.png", lambda stream: preview.save(stream, format="PNG"))
                checkpoint(checkpoint_path, config, image, optimizer, iteration, metrics, identity)
                manifest("running")
                if callback:
                    callback(iteration, preview, dict(metrics))
                else:
                    print(f"{iteration}/{config.iterations} | {metrics['elapsed_seconds']:.1f}s | "
                          f"gradient {metrics['last_gradient_norm']} | views {metrics['view_steps']}", flush=True)
            if budget_reached:
                break
    except (Exception, KeyboardInterrupt) as exc:
        # Do not replace the last complete checkpoint with a partially applied step.
        message = str(exc) or type(exc).__name__
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            message += (". Last complete checkpoint preserved. Start a NEW experiment with image_size=64, "
                        "hidden_dim=64, num_features=32, model_size=512; try attention_slicing=True. "
                        "Lower model_size is exploratory. Changed settings cannot resume this checkpoint.")
        manifest("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", message)
        raise RuntimeError(f"{message}\nLast complete checkpoint: {checkpoint_path}") from exc
    update_metrics()
    export_views(image, folder)
    checkpoint(checkpoint_path, config, image, optimizer, iteration, metrics, identity)
    status = "complete" if iteration == config.iterations else "budget_reached"
    if iteration > completed_before and not metrics["image_parameters_changed"]:
        manifest("failed", "Image parameters did not change")
        raise RuntimeError("Image parameters did not change")
    manifest(status)
    return dict(output_dir=str(folder), checkpoint=str(checkpoint_path), iteration=iteration,
                status=status, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON FlippyConfig settings")
    parser.add_argument("--backend", choices=["sd1", "sdxl"], default="sdxl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint produced by this runner")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.config:
        config = FlippyConfig(**json.loads(args.config.read_text()))
    elif args.smoke:
        config = FlippyConfig.smoke(args.backend)
    else:
        config = FlippyConfig(backend=args.backend, guidance_scale=60. if args.backend == "sd1" else 20.)
    if args.output_dir:
        config.output_dir = args.output_dir
    print(json.dumps(run(config, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
