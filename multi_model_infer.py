from diffusers import DiffusionPipeline
import torch

prompt = prompt = "portrait of azzy cat swimming underwater""portrait of azzy cat swimming underwater"
num_samples = 6
guidance_scale = 8
num_inference_steps = 50

# Paths to all checkpoints.
# Note: Google Colab has a hard time displaying more than 4 rows at a time.
model_list = [
    # 'dreambooth-concept/checkpoint-200',
    'dreambooth-concept/checkpoint-400',
    # 'dreambooth-concept/checkpoint-600',
    'dreambooth-concept/checkpoint-800',
    # 'dreambooth-concept/checkpoint-1000',
    'dreambooth-concept/checkpoint-1200',
    # 'dreambooth-concept/checkpoint-1400',
    'dreambooth-concept/checkpoint-1600',
]

# output images
all_images = []
for model in model_list:
    # Setup pipeline for checkpoint
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")

    # Set the seed to compare checkpoints.
    generator = torch.Generator(device="cuda").manual_seed(42)

    # Generate images & add to output
    images = pipe(
        prompt, 
        num_images_per_prompt=num_samples, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
        generator=generator,
    ).images
    all_images.extend(images)

# Display all in a grid
grid = image_grid(all_images, len(model_list), num_samples)
grid