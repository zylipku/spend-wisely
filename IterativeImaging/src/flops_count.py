import torch
from lightning.fabric.utilities import measure_flops

from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline, DDIMPipeline

device = torch.device('cuda')

ckpt_pipeline = DDPMPipeline.from_pretrained('hf_pipelines/ddpm-cifar10-32')

# aliasing
unet = ckpt_pipeline.unet
unet = unet.to(device)

x = torch.randn(1, 3, 32, 32).to(device)
t = torch.rand(1).to(device)
print(measure_flops(unet, lambda: unet(x, t)))
