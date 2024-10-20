import time
from pathlib import Path

import numpy as np
import rerun as rr

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.unet_models import UNet
from my_image_diffusion.utils import get_images_dataloader, save_images


def train():
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    print(f"using {device=}")
    image_size = 64

    model = UNet(device=device).to(device)
    checkpoint_path = "/home/peter/Documents/my_diffusion/.cadence/cache/id8161a212e2f54951899eBca6196d84ed/10687/outputs/model_115.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    noise_steps = 100
    diffusion = Diffusion(shape=(3, image_size, image_size), device=device, noise_steps=noise_steps)

    rr.init("viz_image_diffusion")
    rr.connect()

    model.eval()
    n_samples = 1
    with torch.no_grad():
        x = torch.randn((n_samples,) + diffusion.shape).to(diffusion.device)
        i = 99
        while True:
            t = (torch.ones(n_samples) * i).long().to(diffusion.device)
            predicted_noise = model(x, t)

            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                beta) * noise
            image_np = torch.clamp(x.squeeze() * 255, 0, 255).cpu().numpy().transpose(1, 2, 0)

            rr.log("sampling_process/image", rr.Image(image_np))

            if i > 1:
                i -= 1


if __name__ == '__main__':
    train()
