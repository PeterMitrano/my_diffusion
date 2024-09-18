from pathlib import Path
from scipy.stats import chisquare, power_divergence
import numpy as np
import matplotlib.pyplot as plt

import torch
import wandb
from PIL import Image
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.my_unet import DiffusionModelWithAttention
from my_image_diffusion.utils import ToyDataset


def train():
    np.set_printoptions(suppress=True, precision=4, linewidth=220)
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=220)

    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_cls = DiffusionModelWithAttention
    config = {
        'lr': 1e-5,
        'batch_size': 128,
        'noise_steps': 45,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'model_cls': model_cls.__name__,
        "epochs": 30,
        "n_test_samples": 1_000,
        "model_kwargs": {
            "h1": 128,
            "h2": 128,
            "h3": 512,
            "use_layer_norm": True,
            "time_emb_dim": 128,
        }
    }

    dataset = ToyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = model_cls(config['model_kwargs'])

    opt = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss()
    l = len(dataloader)

    wandb.init(project="my_diffusion")
    wandb.config.update(config)

    plt.figure()
    plt.title("Training Data Histogram")
    plt.hist(dataset.data, bins=25, color='r')
    plt.xlim([-1, 1])
    fig_name = "toy_data_hist"
    wandb_save_fig(fig_name, results_dir)

    diffusion = Diffusion(shape=(1,), noise_steps=config['noise_steps'], beta_start=config['beta_start'],
                          beta_end=config['beta_end'])
    fwd_viz_samples, _, _ = diffusion.fwd_diffusion(next(iter(dataloader)).squeeze())
    plt.figure()
    # plot each sample in xs as a line with low opacity
    for x_i in fwd_viz_samples.T:
        plt.plot(x_i, alpha=0.05, c='k')
    plt.title("Forward diffusion process")
    fig_name = "fwd_diffusion"
    wandb_save_fig(fig_name, results_dir)

    plt.figure()
    plt.title("fwd distribution final")
    plt.hist(fwd_viz_samples[-1], bins=25)
    plt.xlim([-1, 1])
    fig_name = "fwd_distribution_final"
    wandb_save_fig(fig_name, results_dir)

    for epoch in range(config['epochs']):
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            t = diffusion.sample_timestamps(config['batch_size'])

            x_t, noise = diffusion.noise_scalar(images, t)

            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(EPOCH=epoch, MSE=loss.item())
            wandb.log({"MSE": loss.item(), "epoch": epoch})

        if epoch % 2 == 0:
            test_samples, all_test_samples = diffusion.sample_scalar(model, n_samples=config['n_test_samples'])
            train_samples = dataset.data[:config['n_test_samples']]
            plt.figure()
            bins = np.linspace(-1, 1, 26)
            plt.hist(train_samples, bins=bins, color='r', alpha=0.5)
            plt.hist(test_samples, color='b', bins=bins, alpha=0.5)
            plt.xlim([-1, 1])
            fig_name = "samples_hist"
            wandb_save_fig(fig_name, results_dir)

            # compute chi-squared distance between the two histograms and log it
            train_counts, bins = np.histogram(train_samples)
            test_counts = np.histogram(test_samples, bins=bins)[0]
            # convert counts to frequencies
            train_counts = train_counts / np.sum(train_counts)
            test_counts = test_counts / np.sum(test_counts)
            stat, p_value = chisquare(f_obs=test_counts, f_exp=train_counts)
            wandb.log({
                "chi_squared_stat": stat,
                "chi_squared_p_value": 1 - p_value
            })

            # visualize the reverse diffusion process which takes samples from a unit gaussian and
            # moves them back to the original distribution
            plt.figure()
            for sample in all_test_samples.T:
                plt.plot(sample, alpha=0.05, c='k')
            fig_name = "sampling_process"
            wandb_save_fig(fig_name, results_dir)


def wandb_save_fig(fig_name, results_dir):
    fig_path = results_dir / f"{fig_name}.png"
    plt.savefig(fig_path)
    plt.close()
    wandb.log({fig_name: wandb.Image(str(fig_path))})


if __name__ == '__main__':
    train()
