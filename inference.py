from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
import torch
import os

gen = torch.Generator()
gen.manual_seed(42)
model_choices = ['basic', 'prior_preserving', 'text_and_unet']
# defining hyperparameters
output_dir = "inference_outputs"
check_point = 2500
model_idx = -1
model_name = 'segioperez'

# creating save folder
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

curr_save_sub = os.path.join(output_dir, model_name)
if not os.path.exists(curr_save_sub):
    os.mkdir(curr_save_sub)

check_point_sub_dir = os.path.join(curr_save_sub, str(check_point))
if not os.path.exists(check_point_sub_dir):
    os.mkdir(check_point_sub_dir)


model_types = [model_choices[model_idx]]
for model_type in model_types:
    model_id = f"{model_name}_{model_type}"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")

    text_encoder = CLIPTextModel.from_pretrained(f"segioperez_text_and_unet/checkpoint-{check_point}",subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained(f"segioperez_text_and_unet/checkpoint-{check_point}", subfolder="unet", torch_dtype=torch.float16).to("cuda")
    pipe.text_encoder = text_encoder
    pipe.unet = unet
    prompt = "A  portrait profile picture of (((sksjeff man))) as a well-dressed man wearing sunglasses in a business suit, clicking profile picture with sunglasses, ultra realistic, 4k Resolution, stunning, office background, highly detailed, HD quality image."
    neg_prompt = "(((wide shot))), (cropped head), naked, nude, mole, bindi, moles bad framing, out of frame, deformed, cripple, old, fat, ugly, poor, missing arm, additional arms, additional legs, additional head, additional face, multiple people, group of people, dyed hair, black and white, grayscale"

    images = pipe(prompt, num_inference_steps=25, guidance_scale=7, num_images_per_prompt=4, generator=gen, negative_prompt=neg_prompt).images

    i=0
    for image in images:
        image.save(f"{check_point_sub_dir}/{i}_{model_type}.png")
        i+=1


import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel

def generate_images(seed, model_name, check_point, model_type, prompt, neg_prompt, output_dir="inference_outputs"):
    # Initialize the random number generator with a seed
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    curr_save_sub = os.path.join(output_dir, model_name)
    os.makedirs(curr_save_sub, exist_ok=True)
    check_point_sub_dir = os.path.join(curr_save_sub, str(check_point))
    os.makedirs(check_point_sub_dir, exist_ok=True)
    
    # Load the models and set them to CUDA
    pipe = StableDiffusionPipeline.from_pretrained(
        f"{model_name}_{model_type}",
        scheduler=DPMSolverMultistepScheduler.from_pretrained(f"{model_name}_{model_type}", subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")
    text_encoder = CLIPTextModel.from_pretrained(f"{model_name}_text_and_unet/checkpoint-{check_point}", subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained(f"{model_name}_text_and_unet/checkpoint-{check_point}", subfolder="unet", torch_dtype=torch.float16).to("cuda")
    pipe.text_encoder = text_encoder
    pipe.unet = unet
    
    # Generate images based on the prompt
    images = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=25, guidance_scale=7, num_images_per_prompt=4, generator=gen).images
    
    # Save generated images to disk
    for i, image in enumerate(images):
        image.save(f"{check_point_sub_dir}/{i}_{model_type}.png")

prompt="A portrait profile picture of a well-dressed man wearing sunglasses in a business suit, ultra realistic, 4k Resolution, stunning, office background, highly detailed, HD quality image."
neg_prompt="Wide shot, cropped head, naked, nude, bad framing, out of frame, deformed, old, fat, ugly, poor, missing arm, additional arms, additional legs, additional head, additional face, multiple people, group of people, dyed hair, black and white, grayscale"

generate_images(
    seed=gen.initial_seed(),
    model_name=model_name,
    check_point=check_point,
    model_type=model_choices[model_idx],
    prompt=prompt,
    neg_prompt=neg_prompt,
    output_dir=output_dir
)
