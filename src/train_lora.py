
import argparse, os, math, yaml
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
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
    return TrainConfig(**cfg)

def main():
    args = get_args()
    cfg = load_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps, mixed_precision=cfg.mixed_precision)
    logger.info(accelerator.state, main_process_only=False)

    if cfg.wandb_enabled:
        try:
            import wandb  # noqa: F401
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
    lora_rank = cfg.lora_rank
    lora_alpha = cfg.lora_alpha
    lora_dropout = cfg.lora_dropout

    def _set_lora(unet):
        lora_attn_procs = {}
        for name, module in unet.named_modules():
            if hasattr(module, 'set_processor'):
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=module.to_q.in_features, rank=lora_rank, network_alpha=lora_alpha)
        unet.set_attn_processor(lora_attn_procs)

    _set_lora(unet)
    attn_procs = AttnProcsLayers(unet.attn_processors)

    # Dataset & Dataloader
    dataset = ChestXrayDataset(args.dataset_dir, resolution=cfg.resolution)
    train_loader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model_id, subfolder='scheduler')

    # Optimizer
    if accelerator.mixed_precision == 'fp16':
        dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    optimizer = torch.optim.AdamW(attn_procs.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # Prepare models
    unet, optimizer, train_loader, text_encoder, vae = accelerator.prepare(unet, optimizer, train_loader, text_encoder, vae)
    # tokenizer on CPU

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
                    unet.module.save_attn_procs(os.path.join(save_dir, 'lora_unet'))
                    logger.info(f"Saved checkpoint to {save_dir}")
                    if cfg.wandb_enabled:
                        accelerator.log({"checkpoint/step": global_step}, step=global_step)

            if global_step >= cfg.max_train_steps:
                break

        if global_step >= cfg.max_train_steps:
            break

    # Final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, 'lora_unet')
        unet.module.save_attn_procs(final_dir)
        logger.info(f"Training complete. LoRA weights saved to {final_dir}")

    accelerator.wait_for_everyone()
    if cfg.wandb_enabled:
        accelerator.end_training()

if __name__ == '__main__':
    main()
