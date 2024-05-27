import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_images(prompt, num_images, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_id = "stabilityai/stable-diffusion-2-1"
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    name_offset = 5420
    # Generate the specified number of images
    index = 0
    for _ in range(num_images):
        images = pipe(prompt).images
        for j, image in enumerate(images):
            image.save(os.path.join(output_folder, f"image_{index+1+name_offset}.png"))
            index += 1
        
        if index >= num_images:
            break

# Example usage
prompt = "dog, photo, photo realistic, realistic, dog photo, realistic background, photograph"
num_images = 12500 - 5420
output_folder = "data/sd_2_1_dogs_2500"

generate_images(prompt, num_images, output_folder)