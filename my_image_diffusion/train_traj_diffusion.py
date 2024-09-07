from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.my_unet import UNet
from my_image_diffusion.utils import save_images, find_latest_checkpoint, TrajDataset


def train():
    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    batch_size = 16
    dataset = TrajDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # visualize trajs as images
    # import matplotlib.pyplot as plt
    # batch = next(iter(dataloader))
    # plt.figure()
    # for img in batch:
    #     plt.figure()
    #     plt.imshow(img[0].T)
    #     plt.show()
    #
    #     # plt.plot(img[0, :, 0], color='b')
    #     plt.plot(img[0, :, 1], color='r')
    #     # plt.plot(img[0, :, 0], img[0, :, 1], color='r')
    # plt.show()

    model = UNet(c_in=1, c_out=1, time_dim=256)
    # ckpt = find_latest_checkpoint(models_dir)
    # model_state = torch.load(ckpt, weights_only=True)
    # model.load_state_dict(model_state)

    opt = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    diffusion = Diffusion(shape=(1, 50, 2))
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
            sampled_images = diffusion.sample(model, n=4)
            save_images(sampled_images, results_dir / f"sampled_{epoch}.png")
            tb_writer.add_images("Sampled", sampled_images, global_step=epoch)
            torch.save(model.state_dict(), models_dir / f"model_{epoch}.pt")


if __name__ == '__main__':
    train()
