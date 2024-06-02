from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("tuxmon_lora", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A tuxemon with blue eyes").images[0]
image.save("tuxmon.png")