import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class Diffusion:

    def __init__(self, shape, noise_steps=1000, beta_start=5e-5, beta_end=0.01, device='cpu'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.shape = shape

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def viz_fwd_diffusion(self, x_0):
        """
        Visualize the forward diffusion process the iterative way and compare the one-step version
        """
        rng = np.random.RandomState(0)
        n_test_samples = x_0.shape[0]
        x_t = x_0.numpy()
        xs = []
        us = []
        sigmas = []
        xs.append(x_t)
        for t in range(self.noise_steps):
            x_t = x_t + rng.normal(0, np.sqrt(1 - self.alpha[t].item()), size=(n_test_samples))
            u = x_t.mean()
            sigma = x_t.std()
            xs.append(x_t)
            us.append(u)
            sigmas.append(sigma)
        xs = np.array(xs)

        plt.figure()
        # plot each sample in xs as a line with low opacity
        for i in range(n_test_samples):
            plt.plot(xs[:, i], alpha=0.05, c='k')
        plt.show()

        plt.figure()
        plt.title("final gaussian")
        plt.hist(xs[-1], bins=100)
        plt.show()
        print(f"final mean: {us[-1]}, final std: {sigmas[-1]}")

    def noise_images(self, x, t):
        """
        Samples noise and adds it to an image. This uses the closed-form analytical solution for the diffusion process,
        instead of actually running the iterative diffusion process.

        :param x: A clean image [b, h, w, c]. Or for trajs, [b, time, action_dim, 1]
        :param t: time index
        :return: noisy image [b, h, w, c] and the unscaled image noise [b, h, w, c]
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])  # beta hat?
        sqrt_alpha_hat = sqrt_alpha_hat[:, None, None, None]
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat[:, None, None, None]

        epsilon = torch.randn_like(x, device=self.device)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timestamps(self, n):
        """
        used to generate targets when training

        :param n: Number of timesteps to sample
        :return: timestep samples
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def sample(self, model, noise_steps):
        model.eval()
        with torch.no_grad():
            x = torch.randn((noise_steps,) + self.shape).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=self.noise_steps):
                t = (torch.ones(noise_steps) * i).long().to(self.device)
                predicted_noise = model(x, t)

                if i % 100 == 0 and x.shape[1] == 1:
                    plt.figure()
                    plt.title(f"pred noise dist when sampling {i=}")
                    plt.hist(predicted_noise.squeeze().detach().numpy(), color='m', bins=50)
                    plt.xlim([-2, 2])
                    plt.ylim([0, 100])
                    plt.show()

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x
