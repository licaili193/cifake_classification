import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_images(prompt, modifiers, negative_prompt, num_images, batch_size, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_id = "stabilityai/stable-diffusion-2-1"
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.load_lora_weights("data/lora_result", weight_name="pytorch_lora_weights.safetensors")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    name_offset = 140
    # Generate the specified number of images
    index = 0
    while index < num_images:
        current_modifier = modifiers[(index // 4) % len(modifiers)]
        full_prompt = f"{prompt}, {current_modifier}"
        
        images = pipe(full_prompt, negative_prompt=negative_prompt, num_images_per_prompt=batch_size, cross_attention_kwargs={"scale": -0.4}).images
        for j, image in enumerate(images):
            if index >= num_images:
                break
            image.save(os.path.join(output_folder, f"image_{index+1+name_offset}.png"))
            index += 1

# Example usage
fixed_prompt = "a photograph of a dog"
modifiers = ["indoors", "outdoor", "walking", "running", "sleeping", "eating", "jumping", "sitting"]
negative_prompt = "f4ke, blurry, distorted, low qualiaty, low resolution, pixelated, noisy, abstract, surreal, unrealistic, cartoonish, animated, illustrated, painted, drawn, sketch, sketchy, sketch-like, sketchbook, doodle"
num_images = 500-140
batch_size = 4
output_folder = "data/sd_2_1_lora_dogs_with_modifiers_no_trigger_negative_3"

generate_images(fixed_prompt, modifiers, negative_prompt, num_images, batch_size, output_folder)
