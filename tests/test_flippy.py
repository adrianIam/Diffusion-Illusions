"""CPU contract tests. No pretrained weights or model downloads are used."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from source.flippy import (FlippyConfig, FourierImage, atomic_save, checkpoint,
                          export_views, load_checkpoint, restore_rng, rng_state,
                          rotate_view, run, seed_all)
from source.illusion_backends import (DiffusionSDSBackend, assemble_conditioning,
                                     prediction_to_epsilon)

torch.set_num_threads(1)


class FakeVAE(nn.Module):
    def __init__(self, scaling=.13025):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(scaling_factor=scaling, force_upcast=True)

    def encode(self, pixels):
        encoded = self.conv(torch.nn.functional.avg_pool2d(pixels, 8))
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: encoded))


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)
        self.config = SimpleNamespace(addition_time_embed_dim=2)
        self.add_embedding = SimpleNamespace(linear_1=SimpleNamespace(in_features=20))
        self.calls = []

    def forward(self, latents, timestep, encoder_hidden_states, added_cond_kwargs=None):
        self.calls.append((encoder_hidden_states.detach().clone(), added_cond_kwargs))
        value = encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None]
        return UNet2DConditionOutput(sample=self.conv(latents) + value)


class FakePipe:
    def __init__(self, backend="sdxl", prediction_type="epsilon"):
        self.unet, self.vae = FakeUNet(), FakeVAE(.13025 if backend == "sdxl" else .18215)
        self.text_encoder = nn.Linear(2, 2)
        self.text_encoder_2 = nn.Linear(2, 2) if backend == "sdxl" else None
        if self.text_encoder_2 is not None:
            self.text_encoder_2.config = SimpleNamespace(projection_dim=8)
        self.scheduler = DDPMScheduler(num_train_timesteps=100, prediction_type=prediction_type,
                                       beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear")
        self.backend = backend

    def encode_prompt(self, prompt, negative_prompt, **kwargs):
        batch = len(prompt)
        # Distinct signs AND prompt indices detect accidental CFG/view interleaving.
        pos = torch.stack([torch.full((77, 16 if self.backend == "sdxl" else 768), i + 1.) for i in range(batch)])
        neg = -pos
        if self.backend == "sd1":
            return pos, neg
        pooled = torch.stack([torch.full((8,), 10. + i) for i in range(batch)])
        return pos, neg, pooled, -pooled

    _get_add_time_ids = StableDiffusionXLPipeline._get_add_time_ids


def tiny_config(tmp_path, **kwargs):
    defaults = dict(model_size=64, image_size=8, hidden_dim=8, num_features=4,
                    iterations=4, save_interval=2, dtype="float32", view_sampling="both",
                    output_dir=str(tmp_path / "run"))
    defaults.update(kwargs)
    return FlippyConfig(**defaults)


@pytest.mark.parametrize("settings", [
    {"backend": "flux"}, {"image_size": 0}, {"iterations": True}, {"model_size": 65},
    {"seed": -1}, {"seed": 2**32}, {"guidance_scale": float("nan")},
    {"learning_rate": 0}, {"min_timestep": .99, "max_timestep": .5},
    {"max_timestep": 1}, {"view_weights": [0, 0]}, {"view_weights": [1, -1]},
    {"view_weights": [1, float("inf")]}, {"max_seconds": float("inf")},
    {"prompt_a": " "}, {"vae_dtype": "float16"},
])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        FlippyConfig(**settings)


def test_resolution_presets():
    assert FlippyConfig().model_size == 1024
    assert FlippyConfig(backend="sd1").model_size == 512
    assert FlippyConfig.smoke().model_size == 512
    assert FlippyConfig.smoke().view_sampling == "both"


def test_rotation_gradient():
    image = torch.arange(18.).reshape(1, 3, 2, 3).requires_grad_()
    flipped = rotate_view(image, 1)
    assert torch.equal(flipped, image.flip(-1, -2))
    assert torch.equal(rotate_view(flipped, 1), image)
    flipped.sum().backward()
    assert torch.equal(image.grad, torch.ones_like(image))


def test_conditioning_order():
    pos = torch.arange(24.).reshape(2, 3, 4).requires_grad_()
    neg, pooled, times = -pos, torch.arange(16.).reshape(2, 8), torch.arange(6.).reshape(1, 6)
    for i, cond in enumerate(assemble_conditioning(pos, neg, pooled, -pooled, times)):
        assert torch.equal(cond.tokens[0], neg[i])
        assert torch.equal(cond.tokens[1], pos[i])
        assert torch.equal(cond.pooled[0], -pooled[i])
        assert torch.equal(cond.pooled[1], pooled[i])
        assert torch.equal(cond.time_ids, times.repeat(2, 1))
        assert not cond.tokens.requires_grad


@pytest.mark.parametrize("kind", ["epsilon", "v_prediction"])
def test_prediction_conversion(kind):
    epsilon, clean, alpha = torch.randn(4), torch.randn(4), torch.tensor(.7)
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * epsilon
    prediction = epsilon if kind == "epsilon" else alpha.sqrt() * epsilon - (1 - alpha).sqrt() * clean
    torch.testing.assert_close(prediction_to_epsilon(prediction, noisy, alpha, kind), epsilon)
    with pytest.raises(ValueError, match="prediction_type"):
        prediction_to_epsilon(prediction, noisy, alpha, "sample")


@pytest.mark.parametrize("backend", ["sd1", "sdxl"])
@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_sds_gradients_and_conditioning(tmp_path, backend, prediction_type):
    config = tiny_config(tmp_path, backend=backend)
    engine = DiffusionSDSBackend(config, FakePipe(backend, prediction_type), device="cpu")
    conditioning = engine.prepare([config.prompt_a, config.prompt_b], "negative")
    assert engine.min_step == 2 and engine.max_step == 98
    assert engine.scaling_factor == (.13025 if backend == "sdxl" else .18215)
    pixels = torch.rand(1, 3, 8, 8, requires_grad=True)
    for view in [0, 1]:
        engine.loss(rotate_view(pixels, view), view, timestep=50).backward()
    assert torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for m in engine.model_modules() for p in m.parameters())
    assert len(engine.unet.calls) == 2
    if backend == "sd1":
        assert conditioning[0].tokens.shape == (2, 77, 768)
        assert conditioning[0].pooled is None
    else:
        assert conditioning[0].time_ids[0].tolist() == [64., 64., 0., 0., 64., 64.]


def test_sds_weight_is_applied_once(tmp_path):
    config = tiny_config(tmp_path)
    engine = DiffusionSDSBackend(config, FakePipe(), device="cpu")
    engine.prepare([config.prompt_a, config.prompt_b], "")
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    state = rng_state()
    first = torch.autograd.grad(engine.loss(image, 0, weight=1, timestep=50), image)[0]
    restore_rng(state)
    second = torch.autograd.grad(engine.loss(image, 0, weight=.25, timestep=50), image)[0]
    torch.testing.assert_close(second, first * .25)


def test_original_fourier_representation_and_gradients():
    # Execute only the four original definitions to avoid its unrelated rp/einops dependencies.
    tree = ast.parse(Path("source/learnable_textures.py").read_text())
    names = {"GaussianFourierFeatureTransform", "get_uv_grid", "LearnableImage", "LearnableImageFourier"}
    subset = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
    scope = dict(torch=torch, nn=nn, np=np)
    exec(compile(subset, "source/learnable_textures.py", "exec"), scope)
    seed_all(17)
    old = scope["LearnableImageFourier"](height=8, width=8, hidden_dim=8, num_features=4)
    seed_all(17)
    new = FourierImage(8, hidden_dim=8, num_features=4)
    torch.testing.assert_close(new.features, old.features, rtol=0, atol=0)
    old_output, new_output = old(), new().squeeze(0)
    torch.testing.assert_close(old_output, new_output, rtol=0, atol=0)
    old_output.square().sum().backward()
    new_output.square().sum().backward()
    for old_param, new_param in zip(old.model.parameters(), new.model.parameters()):
        torch.testing.assert_close(old_param.grad, new_param.grad, rtol=0, atol=0)


def test_old_train_step_sds_gradient(tmp_path):
    # Compare the historical explicit backward and the new surrogate at identical RNG/timestep.
    tree = ast.parse(Path("source/stable_diffusion.py").read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StableDiffusion")
    methods = {"train_step", "encode_imgs", "add_noise", "remove_noise", "predict_noise"}
    klass.body = [node for node in klass.body if getattr(node, "name", None) in methods]
    scope = dict(torch=torch, nn=nn, F=torch.nn.functional, Optional=__import__("typing").Optional)
    exec(compile(ast.Module(body=[klass], type_ignores=[]), "legacy-sds", "exec"), scope)
    config = tiny_config(tmp_path, backend="sd1", model_size=512)
    engine = DiffusionSDSBackend(config, FakePipe("sd1"), device="cpu")
    engine.prepare([config.prompt_a, config.prompt_b], "")
    legacy = scope["StableDiffusion"]()
    legacy.device, legacy.vae, legacy.unet = engine.device, engine.vae, engine.unet
    legacy.scheduler, legacy.alphas = engine.scheduler, engine.alphas
    legacy.num_train_timesteps = 100
    pixels = torch.rand(1, 3, 8, 8, requires_grad=True)
    state = rng_state()
    legacy.train_step(engine.conditioning[0].tokens, pixels, guidance_scale=config.guidance_scale,
                      t=torch.tensor([50]), noise_coef=config.noise_coef)
    expected = pixels.grad.clone()
    pixels.grad = None
    restore_rng(state)
    engine.loss(pixels, 0, timestep=50).backward()
    torch.testing.assert_close(pixels.grad, expected)


def random_values():
    return random.random(), np.random.rand(), torch.rand(3)


def test_checkpoint_roundtrip_and_mismatch(tmp_path):
    config = tiny_config(tmp_path)
    image = FourierImage(8, 8, 4)
    optim = torch.optim.SGD(image.parameters(), lr=config.learning_rate, momentum=.5)
    image().sum().backward()
    optim.step()
    path = tmp_path / "checkpoint.pt"
    checkpoint(path, config, image, optim, 2, {"example": 5}, {"model": "fake"})
    expected_rng = random_values()
    new_image = FourierImage(8, 8, 4)
    new_optim = torch.optim.SGD(new_image.parameters(), lr=.5, momentum=.5)
    iteration, metrics = load_checkpoint(path, replace(config, iterations=8), new_image, new_optim, {"model": "fake"})
    assert iteration == 2 and metrics == {"example": 5}
    assert new_optim.param_groups[0]["lr"] == config.learning_rate
    for key, value in image.state_dict().items():
        torch.testing.assert_close(value, new_image.state_dict()[key], rtol=0, atol=0)
    actual_rng = random_values()
    assert actual_rng[:2] == expected_rng[:2]
    torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
    with pytest.raises(ValueError, match="prompt_a"):
        load_checkpoint(path, replace(config, prompt_a="changed"), new_image, new_optim, {"model": "fake"})
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, config, new_image, new_optim, {"model": "other"})


def test_atomic_save_preserves_previous_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous")
    def failed_writer(stream):
        stream.write(b"partial")
        raise OSError("disk failure")
    with pytest.raises(OSError):
        atomic_save(path, failed_writer)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_preview_does_not_change_buffers_or_rng(tmp_path):
    image = FourierImage(8, 8, 4)
    buffers = {name: value.clone() for name, value in image.named_buffers()}
    state = torch.get_rng_state().clone()
    export_views(image, tmp_path)
    assert image.training
    assert torch.equal(torch.get_rng_state(), state)
    for name, value in image.named_buffers():
        assert torch.equal(value, buffers[name])
    from PIL import Image
    upright = np.asarray(Image.open(tmp_path / "upright.png"))
    flipped = np.asarray(Image.open(tmp_path / "flipped.png"))
    assert np.array_equal(flipped, upright[::-1, ::-1])


def test_runner_resume_matches_uninterrupted_cpu(tmp_path):
    # This establishes deterministic CPU fake-component resume, not GPU determinism.
    config = tiny_config(tmp_path, seed=42)
    seed_all(123)
    engine = DiffusionSDSBackend(config, FakePipe(), device="cpu")
    result = run(config, backend=engine)
    full = torch.load(result["checkpoint"], weights_only=False)
    split_config = replace(config, iterations=2, output_dir=str(tmp_path / "split"))
    first = run(split_config, backend=engine)
    resumed = run(replace(split_config, iterations=4), resume=first["checkpoint"], backend=engine)
    split = torch.load(resumed["checkpoint"], weights_only=False)
    for name, value in full["image"].items():
        torch.testing.assert_close(value, split["image"][name], rtol=0, atol=0)
    assert resumed["iteration"] == 4
    assert resumed["metrics"]["view_steps"] == [4, 4]
    assert resumed["metrics"]["image_parameters_changed"]
    manifest = json.loads((Path(resumed["output_dir"]) / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["metrics"]["frozen_model_gradients_absent"]


def test_failure_preserves_last_complete_checkpoint(tmp_path):
    config = tiny_config(tmp_path)
    engine = DiffusionSDSBackend(config, FakePipe(), device="cpu")
    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated OOM")
    engine.loss = fail
    with pytest.raises(RuntimeError, match="Last complete checkpoint preserved"):
        run(config, backend=engine)
    saved = torch.load(Path(config.output_dir) / "checkpoint.pt", weights_only=False)
    assert saved["iteration"] == 0
    assert json.loads((Path(config.output_dir) / "manifest.json").read_text())["status"] == "failed"


def test_resume_pins_resolved_revision(tmp_path):
    config = tiny_config(tmp_path, iterations=2)
    sha = "a" * 40
    seed_all(123)
    engine = DiffusionSDSBackend(config, FakePipe(), device="cpu")
    engine.resolved_revision = sha
    first = run(config, backend=engine)
    saved = torch.load(first["checkpoint"], weights_only=False)
    assert saved["config"]["revision"] == sha
    resumed_config = FlippyConfig(**saved["config"])
    resumed_config.iterations = 4
    seed_all(123)
    second_engine = DiffusionSDSBackend(resumed_config, FakePipe(), device="cpu")
    second_engine.resolved_revision = sha
    second = run(resumed_config, resume=first["checkpoint"], backend=second_engine)
    assert second["iteration"] == 4


def test_no_real_cpu_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU required"):
        run(tiny_config(tmp_path))


def test_offline_real_sdxl_components(tmp_path):
    """Exercise installed Diffusers/Transformers APIs with tiny random weights.

    This tests integration and gradients, not the real pretrained SDXL model.
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), model_max_length=77)
    text_config = CLIPTextConfig(vocab_size=3, hidden_size=8, intermediate_size=16,
                                 projection_dim=8, num_hidden_layers=2, num_attention_heads=2,
                                 max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(8, 16), cross_attention_dim=16, attention_head_dim=2,
        norm_num_groups=4, addition_embed_type="text_time", addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=20,
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 16, 16, 16), norm_num_groups=4,
        sample_size=64, scaling_factor=.13025, force_upcast=True,
    )
    pipe = StableDiffusionXLPipeline(
        vae=vae, unet=unet, text_encoder=CLIPTextModel(text_config),
        text_encoder_2=CLIPTextModelWithProjection(text_config), tokenizer=tokenizer,
        tokenizer_2=tokenizer, scheduler=DDPMScheduler(num_train_timesteps=100),
        add_watermarker=False,
    )
    config = tiny_config(tmp_path, prompt_a="a", prompt_b="a")
    engine = DiffusionSDSBackend(config, pipe=pipe, device="cpu")
    engine.prepare(["a", "a"], "")
    assert engine.conditioning[0].tokens.shape == (2, 77, 16)
    assert engine.conditioning[0].pooled.shape == (2, 8)
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    engine.loss(image, 0, timestep=50).backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert all(p.grad is None for module in engine.model_modules() for p in module.parameters())
