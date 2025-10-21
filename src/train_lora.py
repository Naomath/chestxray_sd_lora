
import argparse, os, math, yaml
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer

from .dataset import ChestXrayDataset
from .utils import set_seed

logger = get_logger(__name__)

@dataclass
class TrainConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 256
    train_batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 5e-5
    max_train_steps: int = 20000
    checkpointing_steps: int = 1000
    mixed_precision: str = "bf16"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    num_workers: int = 8
    seed: int = 42
    enable_sdp: bool = True
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    training_mode: str = "lora"  # "lora" or "full"
    sample_prompts: List[str] = field(default_factory=lambda: ["chest x-ray, normal"])
    samples_per_prompt: int = 1
    sample_num_inference_steps: int = 30
    sample_guidance_scale: float = 7.5
    sample_seed: Optional[int] = None

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--config', type=str, default=None)
    ap.add_argument('--model_id', type=str, default=None)
    ap.add_argument('--resolution', type=int, default=None)
    ap.add_argument('--train_batch_size', type=int, default=None)
    ap.add_argument('--grad_accum_steps', type=int, default=None)
    ap.add_argument('--learning_rate', type=float, default=None)
    ap.add_argument('--max_train_steps', type=int, default=None)
    ap.add_argument('--checkpointing_steps', type=int, default=None)
    ap.add_argument('--mixed_precision', type=str, default=None, choices=['no','fp16','bf16'])
    ap.add_argument('--lora_rank', type=int, default=None)
    ap.add_argument('--lora_alpha', type=int, default=None)
    ap.add_argument('--lora_dropout', type=float, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--enable_sdp', action='store_true')
    ap.add_argument('--no_sdp', action='store_true')
    ap.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
    ap.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    ap.add_argument('--wandb_project', type=str, default=None, help='W&B project name.')
    ap.add_argument('--wandb_run_name', type=str, default=None, help='Optional W&B run name.')
    ap.add_argument('--wandb_entity', type=str, default=None, help='Optional W&B entity.')
    ap.add_argument('--wandb_group', type=str, default=None, help='Optional W&B group.')
    ap.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated W&B tags.')
    ap.add_argument('--sample_prompts', type=str, default=None, help='Comma-separated prompts for checkpoint sampling.')
    ap.add_argument('--samples_per_prompt', type=int, default=None, help='Number of images to sample per prompt at checkpoints.')
    ap.add_argument('--sample_num_inference_steps', type=int, default=None, help='Inference steps for checkpoint sampling.')
    ap.add_argument('--sample_guidance_scale', type=float, default=None, help='Guidance scale for checkpoint sampling.')
    ap.add_argument('--sample_seed', type=int, default=None, help='Seed offset for checkpoint sampling (defaults to training seed).')
    ap.add_argument('--training_mode', type=str, default=None, choices=['lora', 'full'], help='Training strategy: LoRA adapters or full fine-tuning.')
    return ap.parse_args()

def load_config(args):
    # defaults
    cfg = TrainConfig().__dict__
    # yaml overrides
    if args.config:
        with open(args.config, 'r') as f:
            y = yaml.safe_load(f)
        cfg.update(y or {})
    # cli overrides
    for k in cfg.keys():
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    # handle sdp flags
    if args.enable_sdp:
        cfg['enable_sdp'] = True
    if args.no_sdp:
        cfg['enable_sdp'] = False
    if getattr(args, 'wandb', False):
        cfg['wandb_enabled'] = True
    if getattr(args, 'no_wandb', False):
        cfg['wandb_enabled'] = False
    tags = cfg.get('wandb_tags')
    if isinstance(tags, str):
        cfg['wandb_tags'] = [t.strip() for t in tags.split(',') if t.strip()]
    prompts = cfg.get('sample_prompts')
    if isinstance(prompts, str):
        cfg['sample_prompts'] = [t.strip() for t in prompts.split(',') if t.strip()]
    if not cfg.get('sample_prompts'):
        cfg['sample_prompts'] = ["chest x-ray, normal"]
    mode = str(cfg.get('training_mode', 'lora')).lower()
    if mode not in ('lora', 'full'):
        logger.warning(f"Unknown training_mode '{mode}', defaulting to 'lora'.")
        mode = 'lora'
    cfg['training_mode'] = mode
    try:
        cfg['samples_per_prompt'] = max(1, int(cfg.get('samples_per_prompt', 1)))
    except (TypeError, ValueError):
        cfg['samples_per_prompt'] = 1
    return TrainConfig(**cfg)

def main():
    args = get_args()
    cfg = load_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision=cfg.mixed_precision,
    )
    logger.info(accelerator.state, main_process_only=False)

    wandb_module = None
    if cfg.wandb_enabled:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging is enabled but the 'wandb' package is not installed. "
                "Install it with `pip install wandb` or disable W&B logging."
            ) from exc

        project_name = cfg.wandb_project or "chestxray_sd_lora"
        wandb_kwargs = {"project": project_name}
        if cfg.wandb_run_name:
            wandb_kwargs["name"] = cfg.wandb_run_name
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        if cfg.wandb_group:
            wandb_kwargs["group"] = cfg.wandb_group
        if cfg.wandb_tags:
            wandb_kwargs["tags"] = cfg.wandb_tags
        accelerator.init_trackers(project_name, config=dict(cfg.__dict__), init_kwargs={"wandb": wandb_kwargs})

    set_seed(cfg.seed + accelerator.process_index)

    # Enable SDP attention (PyTorch 2.x) for speed/memory if requested
    if cfg.enable_sdp:
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

    # Load pretrained components
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_id, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder='unet')

    # Freeze VAE & text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA processors to UNet attention
    train_lora = cfg.training_mode == "lora"
    if train_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet.requires_grad_(False)
        unet.add_adapter(lora_config)
    else:
        unet.requires_grad_(True)
    
    # Dataset & Dataloader
    dataset = ChestXrayDataset(args.dataset_dir, resolution=cfg.resolution)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")

    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # checkpoint sampling helpers
    sample_prompts = list(cfg.sample_prompts or ["chest x-ray, normal"])
    samples_per_prompt = max(1, cfg.samples_per_prompt)
    sample_pipeline = None
    def generate_and_log_samples(step: int, weights_dir: str):
        nonlocal sample_pipeline
        if not (cfg.wandb_enabled and wandb_module is not None and accelerator.is_main_process):
            return
        pipeline_dtype = torch.float32 if accelerator.device.type == "cpu" else dtype
        if sample_pipeline is None:
            sample_pipeline = StableDiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=pipeline_dtype,
                safety_checker=None,
            )
            sample_pipeline.set_progress_bar_config(disable=True)
            sample_pipeline = sample_pipeline.to(accelerator.device)
        if train_lora:
            sample_pipeline.unet.load_attn_procs(weights_dir)
        else:
            finetuned_unet = UNet2DConditionModel.from_pretrained(
                weights_dir,
                torch_dtype=pipeline_dtype,
            )
            sample_pipeline.unet = finetuned_unet.to(accelerator.device)
        generator_device = accelerator.device if accelerator.device.type != "meta" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        base_seed = cfg.sample_seed if cfg.sample_seed is not None else cfg.seed
        if base_seed is not None:
            generator.manual_seed(base_seed + step)
        with torch.inference_mode():
            outputs = sample_pipeline(
                sample_prompts,
                num_inference_steps=cfg.sample_num_inference_steps,
                guidance_scale=cfg.sample_guidance_scale,
                num_images_per_prompt=samples_per_prompt,
                generator=generator,
            )
        images = outputs.images
        samples_dir = os.path.join(weights_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        wandb_images = []
        for idx, img in enumerate(images):
            prompt_idx = idx // samples_per_prompt if samples_per_prompt else 0
            sample_idx = idx % samples_per_prompt if samples_per_prompt else idx
            prompt = sample_prompts[prompt_idx]
            filename = f"step{step:06d}_prompt{prompt_idx:02d}_img{sample_idx:02d}.png"
            img_path = os.path.join(samples_dir, filename)
            img.save(img_path)
            wandb_images.append(
                wandb_module.Image(
                    img,
                    caption=f"{prompt} [step {step}, sample {sample_idx}]",
                )
            )
        if wandb_images:
            accelerator.log({"checkpoint/samples": wandb_images}, step=step)

    # Optimizer（LoRAパラメータのみ）
    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Prepare models（順序は LoRA → optimizer 作成 → prepare でOK）
    unet, optimizer, train_loader, text_encoder, vae = accelerator.prepare(
        unet, optimizer, train_loader, text_encoder, vae
    )
    # tokenizer は CPU のままでOK

    global_step = 0
    unet.train()

    # Save config
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'config_used.yaml'), 'w') as f:
            yaml.safe_dump(cfg.__dict__, f)

    for epoch in range(10_000_000):  # infinite until max steps
        for batch in train_loader:
            with accelerator.accumulate(unet):
                pixel_values = batch['pixel_values'].to(dtype=vae.dtype, device=accelerator.device)
                prompts = [batch['prompt'][0]] * pixel_values.shape[0] if isinstance(batch['prompt'], list) else [batch['prompt']] * pixel_values.shape[0]

                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings (frozen)
                text_inputs = tokenizer(prompts, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
                text_input_ids = text_inputs.input_ids.to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # Loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction='mean')

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if cfg.wandb_enabled:
                    loss_tensor = loss.detach()
                    gathered = accelerator.gather(loss_tensor)
                    loss_value = gathered.mean().item()
                    if accelerator.is_main_process:
                        logs = {
                            "train/loss": loss_value,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        }
                        accelerator.log(logs, step=global_step)
                if accelerator.is_main_process and cfg.checkpointing_steps and global_step % cfg.checkpointing_steps == 0:
                    # Save LoRA weights
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    weight_subdir = 'lora_unet' if train_lora else 'unet'
                    save_path = os.path.join(save_dir, weight_subdir)
                    unwrapped = accelerator.unwrap_model(unet)
                    if train_lora:
                        unwrapped.save_attn_procs(save_path)
                    else:
                        unwrapped.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_dir}")
                    if cfg.wandb_enabled:
                        accelerator.log({"checkpoint/step": global_step}, step=global_step)
                        generate_and_log_samples(global_step, save_path)

            if global_step >= cfg.max_train_steps:
                break

        if global_step >= cfg.max_train_steps:
            break

    # Final save
    if accelerator.is_main_process:
        weight_subdir = 'lora_unet' if train_lora else 'unet'
        final_dir = os.path.join(args.output_dir, weight_subdir)
        unwrapped = accelerator.unwrap_model(unet)
        if train_lora:
            unwrapped.save_attn_procs(final_dir)
        else:
            unwrapped.save_pretrained(final_dir)
        logger.info(f"Training complete. Weights saved to {final_dir}")
        if cfg.wandb_enabled:
            generate_and_log_samples(global_step, final_dir)

    accelerator.wait_for_everyone()
    if cfg.wandb_enabled:
        accelerator.end_training()

if __name__ == '__main__':
    main()
