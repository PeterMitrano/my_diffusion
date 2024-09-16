from math import floor
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_images_dataloader(dataset_path, image_size, batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(1.25 * image_size)),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    n_total = floor(len(dataset) / batch_size) * batch_size
    dataset.samples = dataset.samples[:n_total]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class ToyDataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.npy_file = dataset_path / "trajs" / "1d.npy"
        self.data = np.load(self.npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # to make the diffusion generic enough to handle images and trajectories, we need 3 dimensions
        # but here they're all just 1.
        sample = torch.tensor(sample).reshape([1,]).float()
        return sample


class TrajDataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.npy_file = dataset_path / "trajs" / "trajs.npy"
        self.data = np.load(self.npy_file)
        self.time = self.data.shape[1]
        self.action_dim = self.data.shape[2]
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0, 1),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.from_numpy(sample).float()
        sample = sample[None, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample


def find_latest_checkpoint(path: Path):
    ckpts = path.glob("*.pt")
    ckpts = sorted(ckpts, key=lambda x: int(x.stem.split("_")[1]))
    return ckpts[-1]
