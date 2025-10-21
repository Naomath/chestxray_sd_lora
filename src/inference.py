
import argparse, os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def to_grayscale_rgb(img: Image.Image):
    # Convert result to grayscale and back to RGB for single-channel look
    return Image.fromarray((Image.fromarray((img.convert('L'))).convert('RGB')))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', type=str, default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--lora_path', type=str, required=True, help='Path to LoRA UNet weights saved with save_attn_procs')
    ap.add_argument('--out_dir', type=str, default='./samples')
    ap.add_argument('--num_images', type=int, default=4)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--prompt', type=str, default='chest x-ray, normal')
    ap.add_argument('--guidance_scale', type=float, default=5.0)
    ap.add_argument('--num_inference_steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grayscale_out', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    # load LoRA into UNet
    pipe.unet.load_attn_procs(args.lora_path)

    pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
    pipe.safety_checker = None  # disable for medical domain to avoid false positives

    generator = torch.Generator(device=pipe.device.type).manual_seed(args.seed)

    for i in range(args.num_images):
        img = pipe(
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            height=args.height, width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images[0]

        if args.grayscale_out:
            img = img.convert('L')

        img.save(os.path.join(args.out_dir, f"sample_{i:03d}.png"))

    print(f"Saved {args.num_images} images to {args.out_dir}")

if __name__ == '__main__':
    main()
