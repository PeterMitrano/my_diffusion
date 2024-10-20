import socket
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
    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    print(f"using {device=}")
    image_size = 64
    batch_size = 8
    dataloader = get_images_dataloader(dataset_path, image_size, batch_size)

    model = UNet(device=device).to(device)
    checkpoint_path = "/home/peter/Documents/my_diffusion/.cadence/cache/id8161a212e2f54951899eBca6196d84ed/10687/outputs/model_115.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    opt = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    noise_steps = 300
    diffusion = Diffusion(shape=(3, image_size, image_size),
                          device=device,
                          noise_steps=noise_steps,
                          beta_start=3e-5,
                          beta_end=0.1)
    tb_writer = SummaryWriter()
    l = len(dataloader)

    rr.init("image_diffusion")
    hostname = socket.gethostname()
    if hostname == 'T15':
        rr.connect()
    else:
        rr.save(f"train_image_diffusion.rrd")

    rr.set_time_sequence("step", 0)

    for epoch in range(120):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            global_step = epoch * l + i
            rr.set_time_sequence("step", global_step)
            tb_writer.add_images("train/Images", images, global_step=global_step)

            t = diffusion.sample_timestamps(batch_size)
            images = images.to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            rr.log("mse", rr.Scalar(loss.item()))

        if epoch % 5 == 0 and epoch > 2:
            sampled_images, sampling_process_images = diffusion.sample_images(model, n_samples=4)
            save_images(sampled_images, results_dir / f"sampled_{epoch}.png")
            tb_writer.add_images("Sampled", sampled_images, global_step=epoch * l)
            for sampled_image in sampled_images:
                rr.log("sampled/image", rr.Image(np.transpose(sampled_image.cpu().numpy(), (1, 2, 0))))
            torch.save(model.state_dict(), models_dir / f"latest_model.pt")


if __name__ == '__main__':
    train()
