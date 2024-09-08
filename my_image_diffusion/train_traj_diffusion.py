from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.my_unet import MyToyMLP
from my_image_diffusion.utils import save_images, find_latest_checkpoint, TrajDataset, ToyDataset


def train():
    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    batch_size = 32
    # dataset = TrajDataset(dataset_path)
    dataset = ToyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MyToyMLP()

    opt = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    diffusion = Diffusion(shape=(1, 1, 1))
    tb_writer = SummaryWriter()
    l = len(dataloader)

    for epoch in range(251):
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            tb_writer.add_images("train/Images", images, global_step=epoch * l + i)

            t = diffusion.sample_timestamps(batch_size)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            samples = diffusion.sample(model, n=1_000)
            plt.figure()
            plt.hist(samples)
            plt.savefig(results_dir / "samples_hist.png")
            plt.show()

            # sampled_images = diffusion.sample(model, n=4)
            # save_images(sampled_images, results_dir / f"sampled_{epoch}.png")
            # tb_writer.add_images("Sampled", sampled_images, global_step=epoch)
            # torch.save(model.state_dict(), models_dir / f"model_{epoch}.pt")


if __name__ == '__main__':
    train()
