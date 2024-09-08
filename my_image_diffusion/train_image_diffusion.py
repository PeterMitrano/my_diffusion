import copy
from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.my_unet import UNet, EMA
from my_image_diffusion.utils import get_images_dataloader, save_images, find_latest_checkpoint


def train():
    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    device = 'cpu'
    image_size = 64
    batch_size = 9
    dataloader = get_images_dataloader(dataset_path, image_size, batch_size)

    model = UNet(device=device).to(device)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # ckpt = find_latest_checkpoint(models_dir)
    # model_state = torch.load(ckpt, weights_only=True)
    # model.load_state_dict(model_state)

    opt = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    diffusion = Diffusion(shape=(3, image_size, image_size), device=device)
    tb_writer = SummaryWriter()
    l = len(dataloader)

    for epoch in range(251):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            tb_writer.add_images("train/Images", images, global_step=epoch * l + i)

            t = diffusion.sample_timestamps(batch_size)
            images = images.to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 5 == 0 and epoch > 0:
            sampled_images = diffusion.sample(ema_model, n=8)
            save_images(sampled_images, results_dir / f"sampled_{epoch}.png")
            tb_writer.add_images("Sampled", sampled_images, global_step=epoch * l)
            torch.save(model.state_dict(), models_dir / f"model_{epoch}.pt")


if __name__ == '__main__':
    train()
