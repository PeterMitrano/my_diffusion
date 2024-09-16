from pathlib import Path
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.models_v1 import DiffusionModelWithResNet
from my_image_diffusion.my_unet import MyToyMLP, MyToyMLP2, UNet, DiffusionModelWithAttention
from my_image_diffusion.utils import save_images, find_latest_checkpoint, ToyDataset


def train():
    np.set_printoptions(suppress=True, precision=4, linewidth=220)
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=220)

    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    batch_size = 128
    dataset = ToyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model = MyToyMLP()
    # model = MyToyMLP2()
    # model = DiffusionModelWithAttention()
    model = DiffusionModelWithResNet()
    # model = UNet(c_in=1, c_out=1)

    opt = optim.AdamW(model.parameters(), lr=1e-5)
    mse = nn.MSELoss()
    tb_writer = SummaryWriter()
    l = len(dataloader)

    # plot the toy data as histogram
    plt.figure()
    plt.title("Training Data Histogram")
    plt.hist(dataset.data, bins=25, color='r')
    plt.xlim([-1, 1])
    fig_path = results_dir / "toy_data_hist.png"
    plt.savefig(fig_path)
    plt.savefig(fig_path)
    log_fig(tb_writer, "training_dataset_hist", fig_path, 0)
    plt.close()

    diffusion = Diffusion(shape=(1,), noise_steps=100, beta_start=1e-4, beta_end=0.02)
    diffusion.viz_fwd_diffusion(next(iter(dataloader)).squeeze())

    for epoch in range(30):
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            t = diffusion.sample_timestamps(batch_size)

            x_t, noise = diffusion.noise_scalar(images, t)

            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 2 == 0:
            n_test_samples =1_000
            samples, all_samples = diffusion.sample_scalar(model, n_samples=n_test_samples)
            plt.figure()
            plt.hist(dataset.data[:n_test_samples], bins=25, color='r', alpha=0.5)
            plt.hist(samples, color='b', bins=25, alpha=0.5)
            plt.xlim([-1, 1])
            fig_path = results_dir / "samples_hist.png"
            plt.savefig(fig_path)
            log_fig(tb_writer, "samples_hist", fig_path, epoch)
            plt.close()

            # visualize the reverse diffusion process which takes samples from a unit gaussian and
            # moves them back to the original distribution
            plt.figure()
            for sample in all_samples.T:
                plt.plot(sample, alpha=0.05, c='k')
            fig_path = results_dir / "sampling_process.png"
            plt.savefig(fig_path)
            log_fig(tb_writer, "sampling_process", fig_path, epoch)
            plt.close()


    samples, _ = diffusion.sample_scalar(model, n_samples=1_000)
    samples = np.squeeze(samples.numpy())
    plt.figure()
    plt.hist(samples, color='b', bins=25)
    plt.xlim([-1, 1])
    plt.show()


def log_fig(tb_writer, tb_name, fig_path, epoch):
    samples_hist_img = np.asarray(Image.open(fig_path))
    samples_hist_img = samples_hist_img[..., :3]
    samples_hist_img = samples_hist_img.transpose(2, 0, 1)
    tb_writer.add_image(tb_name, samples_hist_img, global_step=epoch)


if __name__ == '__main__':
    train()
