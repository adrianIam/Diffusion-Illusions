"""Small SDS backend boundary; the historical stable_diffusion API stays intact.

The scheduler here represents the *training forward marginal*, not an inference
Euler/PNDM trajectory. DDPMScheduler.from_config preserves the model's betas and
prediction type while providing alpha-cumprod-based add_noise.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Conditioning:
    # Always [negative batch, positive batch], including every added condition.
    tokens: torch.Tensor
    pooled: torch.Tensor | None = None
    time_ids: torch.Tensor | None = None

    def unet_kwargs(self):
        result = {"encoder_hidden_states": self.tokens}
        if self.pooled is not None:
            result["added_cond_kwargs"] = {
                "text_embeds": self.pooled, "time_ids": self.time_ids,
            }
        return result


def assemble_conditioning(positive, negative, pooled=None, negative_pooled=None,
                          time_ids=None):
    """Split cached prompts into one-view classifier-free guidance batches."""
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError("Positive and negative token shapes must match [B,L,D]")
    if pooled is not None:
        if negative_pooled is None or pooled.shape != negative_pooled.shape:
            raise ValueError("Positive and negative pooled shapes must match")
        if len(pooled) != len(positive) or time_ids is None or time_ids.shape != (1, 6):
            raise ValueError("SDXL requires pooled embeddings and six size/crop values")
    result = []
    for i in range(len(positive)):
        result.append(Conditioning(
            tokens=torch.cat([negative[i:i+1], positive[i:i+1]]).detach(),
            pooled=(torch.cat([negative_pooled[i:i+1], pooled[i:i+1]]).detach()
                    if pooled is not None else None),
            time_ids=time_ids.repeat(2, 1).detach() if time_ids is not None else None,
        ))
    return result


def prediction_to_epsilon(prediction, noisy_latents, alpha, prediction_type):
    if prediction_type == "epsilon":
        return prediction
    if prediction_type == "v_prediction":
        return alpha.sqrt() * prediction + (1 - alpha).sqrt() * noisy_latents
    raise ValueError(f"Unsupported prediction_type={prediction_type!r}; expected epsilon or v_prediction")


class DiffusionSDSBackend:
    """Frozen SD1/SDXL models, cached conditioning, differentiable VAE encoding."""

    def __init__(self, config, pipe=None, device="cuda"):
        from diffusers import DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

        self.config = config
        self.requested_revision = config.revision
        self.device = torch.device(device)
        dtype_name = config.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if self.device.type == "cuda" else "float32"
        self.dtype = getattr(torch, dtype_name)
        self.vae_dtype = getattr(torch, config.vae_dtype)
        if self.device.type != "cuda" and self.dtype != torch.float32:
            raise ValueError("CPU backend checks require dtype=float32")
        if self.dtype == torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; select float16 or float32")

        self.weight_variant = config.variant
        if (self.weight_variant is None and self.dtype != torch.float32 and
                config.model_id == "stabilityai/stable-diffusion-xl-base-1.0"):
            self.weight_variant = "fp16"
        if pipe is None:
            # Resolve an immutable Hub revision when possible, so the manifest and
            # a resumed session identify the actual weights rather than a moving main.
            from huggingface_hub import model_info
            try:
                self.resolved_revision = model_info(config.model_id, revision=config.revision,
                                                    timeout=15).sha
            except (OSError, ValueError):
                # An explicit immutable SHA can also be loaded from a local Hub cache.
                revision = config.revision or ""
                self.resolved_revision = (revision if len(revision) == 40 and
                                          all(c in "0123456789abcdef" for c in revision.lower()) else None)
            pipeline_cls = StableDiffusionXLPipeline if config.backend == "sdxl" else StableDiffusionPipeline
            options = dict(torch_dtype=self.dtype, use_safetensors=True, low_cpu_mem_usage=True)
            if self.weight_variant:
                options["variant"] = self.weight_variant
            if self.resolved_revision or config.revision:
                options["revision"] = self.resolved_revision or config.revision
            if config.backend == "sd1":
                options.update(safety_checker=None, requires_safety_checker=False)
            pipe = pipeline_cls.from_pretrained(config.model_id, **options)
        else:
            self.resolved_revision = getattr(pipe.unet.config, "_commit_hash", None)
        self.pipe = pipe
        self.unet = pipe.unet.to(device=self.device, dtype=self.dtype)
        # The original SDXL VAE has force_upcast=True. Never blanket-cast it to fp16.
        if getattr(pipe.vae.config, "force_upcast", False) and self.vae_dtype != torch.float32:
            raise ValueError("This VAE requires vae_dtype=float32 (force_upcast=True)")
        self.vae = pipe.vae.to(device=self.device, dtype=self.vae_dtype)
        for module in self.model_modules():
            module.requires_grad_(False).eval()
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.prediction_type = self.scheduler.config.prediction_type
        prediction_to_epsilon(torch.tensor(0.), torch.tensor(0.), torch.tensor(.5), self.prediction_type)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)
        count = self.scheduler.config.num_train_timesteps
        self.min_step = max(0, int(count * config.min_timestep))
        self.max_step = min(count - 1, int(count * config.max_timestep))
        if self.min_step > self.max_step:
            raise ValueError("No timesteps in selected interval")
        self.scaling_factor = float(self.vae.config.scaling_factor)
        if not 0 < self.scaling_factor < 10:
            raise ValueError("Invalid VAE scaling_factor")
        if config.attention_slicing:
            # SDPA is already memory efficient. Slicing is an explicit alternative.
            self.unet.set_attention_slice("auto")
        self.conditioning = None

    def model_modules(self):
        return [m for m in (self.unet, self.vae, self.pipe.text_encoder,
                            getattr(self.pipe, "text_encoder_2", None)) if m is not None]

    def prepare(self, prompts, negative_prompt):
        encoders = [self.pipe.text_encoder, getattr(self.pipe, "text_encoder_2", None)]
        for encoder in encoders:
            if encoder is not None:
                encoder.to(device=self.device, dtype=self.dtype)
        try:
            with torch.no_grad():
                encoded = self.pipe.encode_prompt(
                    prompt=list(prompts), negative_prompt=[negative_prompt] * len(prompts),
                    device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True,
                )
                if self.config.backend == "sdxl":
                    pos, neg, pooled, neg_pooled = encoded
                    size = (self.config.model_size, self.config.model_size)
                    # Version-pinned Diffusers helper checks UNet addition embedding dimensions.
                    time_ids = self.pipe._get_add_time_ids(
                        size, (0, 0), size, dtype=pos.dtype,
                        text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim,
                    ).to(self.device)
                    self.conditioning = assemble_conditioning(pos, neg, pooled, neg_pooled, time_ids)
                else:
                    pos, neg = encoded
                    self.conditioning = assemble_conditioning(pos, neg)
        finally:
            if self.config.offload_text_encoders:
                for encoder in encoders:
                    if encoder is not None:
                        encoder.to("cpu")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return self.conditioning

    def loss(self, image, view, weight=1., timestep=None):
        """Surrogate whose derivative matches the original summed latent gradient.

        grad = weight * noise_coef * (1-alpha_t) * (eps_cfg - sampled_noise).
        There is no pixel/latent averaging and no second multiplication by w(t).
        The scalar surrogate value is not a quality metric.
        """
        if self.conditioning is None:
            raise RuntimeError("Call prepare before loss")
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError("Expected one RGB image [1,3,H,W]")
        resized = F.interpolate(image.to(self.vae_dtype),
                                (self.config.model_size, self.config.model_size),
                                mode="bilinear", align_corners=False)
        # Requires autograd with respect to pixels, despite frozen VAE parameters.
        latents = self.vae.encode(resized * 2 - 1).latent_dist.sample() * self.scaling_factor
        latents = latents.float()
        if timestep is None:
            timestep = torch.randint(self.min_step, self.max_step + 1, (1,), device=self.device)
        else:
            timestep = torch.as_tensor(timestep, device=self.device, dtype=torch.long).reshape(1)
        if not 0 <= int(timestep.item()) < len(self.alphas):
            raise ValueError("Timestep outside scheduler training range")
        alpha = self.alphas[timestep].reshape(1, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            noisy = self.scheduler.add_noise(latents, noise, timestep)
            # DDPM training marginal already has unit input scaling.
            model_input = torch.cat([noisy, noisy]).to(self.dtype)
            prediction = self.unet(model_input, timestep,
                                   **self.conditioning[view].unet_kwargs()).sample.float()
            prediction = prediction_to_epsilon(prediction, torch.cat([noisy, noisy]),
                                                alpha, self.prediction_type)
            uncond, cond = prediction.chunk(2)
            guided = uncond + self.config.guidance_scale * (cond - uncond)
            gradient = weight * self.config.noise_coef * (1 - alpha) * (guided - noise)
        if not torch.isfinite(latents).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite VAE latents or SDS gradient; try float32 components")
        return (latents * gradient.detach()).sum()

    def metadata(self):
        return {
            "model_id": self.config.model_id,
            "requested_revision": self.requested_revision,
            "weight_variant": self.weight_variant,
            "resolved_revision": self.resolved_revision,
            "dtype": str(self.dtype), "vae_dtype": str(self.vae_dtype),
            "vae_scaling_factor": self.scaling_factor,
            # Exclude private loader bookkeeping (including unordered default-key
            # lists) from resume identity; preserve every public scheduler option.
            "source_scheduler": {k: v for k, v in self.pipe.scheduler.config.items() if not k.startswith("_")},
            "sds_scheduler": {k: v for k, v in self.scheduler.config.items() if not k.startswith("_")},
            "min_step": self.min_step, "max_step": self.max_step,
            "conditioning_shapes": [
                {"tokens": list(c.tokens.shape),
                 "pooled": list(c.pooled.shape) if c.pooled is not None else None,
                 "time_ids": list(c.time_ids.shape) if c.time_ids is not None else None}
                for c in self.conditioning or []
            ],
        }
