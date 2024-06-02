from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")
image = pipeline("A tuxemon with blue eyes").images[0]
image.save("tuxmon_2.png")