from pathlib import Path
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.my_unet import MyToyMLP, UNet
from my_image_diffusion.utils import save_images, find_latest_checkpoint, TrajDataset, ToyDataset


def train():
    np.set_printoptions(suppress=True, precision=4, linewidth=220)
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=220)

    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    rr.init('train_traj_diffusion')
    rr.connect()

    batch_size = 1024
    # dataset = TrajDataset(dataset_path)
    dataset = ToyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MyToyMLP()
    # model = UNet(c_in=1, c_out=1)

    opt = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    tb_writer = SummaryWriter()
    l = len(dataloader)

    diffusion = Diffusion(shape=(1, 1, 1), noise_steps=1000, beta_start=5e-5, beta_end=0.01)
    diffusion.viz_fwd_diffusion(next(iter(dataloader)).squeeze())
    
    # import wandb
    # wandb.init(project="my_diffusion")

    for epoch in range(251):
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            t = diffusion.sample_timestamps(batch_size)
            x_t, noise = diffusion.noise_images(images, t)

            # plt.figure()
            # plt.title("noise")
            # plt.hist(noise.squeeze().flatten().numpy())
            # plt.show()

            # for test_t in range(0, diffusion.noise_steps, 10):
            #     test_t_arr = torch.ones_like(t) * test_t
            #     x_t, noise = diffusion.noise_images(images, test_t_arr)
            #     rr.log("t", rr.Scalar(test_t))
            #     x_t = x_t.squeeze().flatten().numpy()
            #     x_t_counts = np.histogram(x_t, bins=100, range=(-3.5, 3.5))[0]
            #     rr.log("x_t", rr.BarChart(x_t_counts))

            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            samples = diffusion.sample(model, noise_steps=diffusion.noise_steps)
            samples = np.squeeze(samples.numpy())
            plt.figure()
            plt.hist(samples)
            plt.savefig(results_dir / "samples_hist.png")
            plt.show()

            # sampled_images = diffusion.sample(model, n=4)
            # sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
            # sampled_images = (sampled_images * 255).type(torch.uint8)
            # save_images(sampled_images, results_dir / f"sampled_{epoch}.png")
            # tb_writer.add_images("Sampled", sampled_images, global_step=epoch)
            # torch.save(model.state_dict(), models_dir / f"model_{epoch}.pt")



if __name__ == '__main__':
    train()
