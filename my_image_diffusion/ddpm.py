import numpy as np
import torch
from tqdm import tqdm

import rerun as rr


def rr_log_float_image_tensor(entity_path, image_tensor, bounds):
    lower, upper = bounds
    float_image_np = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))
    float_image_np_scaled = (float_image_np - lower) / (upper - lower)
    int_image_np = np.clip(float_image_np_scaled * 255, 0, 255).astype(np.uint8)
    rr.log(entity_path, rr.Image(int_image_np))


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

    def fwd_diffusion(self, x_0):
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
        us = np.array(us)
        sigmas = np.array(sigmas)
        return xs, us, sigmas

    def noise_images(self, x, t):
        """
        Samples noise and mixes it with the image. This uses the closed-form analytical solution for the diffusion process,
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

        rr_log_float_image_tensor("train/image", x[0], bounds=[0, 1])
        rr_log_float_image_tensor("train/noise", epsilon[0], bounds=[-3, 3])
        rr_log_float_image_tensor("train/mix_image", (sqrt_alpha_hat * x)[0], bounds=[0, 1])
        rr_log_float_image_tensor("train/mix_noise", (sqrt_one_minus_alpha_hat * epsilon)[0], bounds=[-3, 3])
        rr.log("train/t", rr.Scalar(t[0].item()))

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def noise_traj(self, x, t):
        """ for 2D data of shape time, action  """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])  # beta hat?
        sqrt_alpha_hat = sqrt_alpha_hat[:, None, None]
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat[:, None, None]

        epsilon = torch.randn_like(x, device=self.device)
        x_epislon_mix = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

        rr.log("noise_traj/x", rr.LineStrips2D(x.cpu().numpy()[0], radii=0.005))
        rr.log("noise_traj/epsilon", rr.LineStrips2D(epsilon.cpu().numpy()[0], radii=0.0005))
        rr.log("noise_traj/x_mix", rr.LineStrips2D((sqrt_alpha_hat * x).cpu().numpy()[0], radii=0.0005))
        rr.log("noise_traj/epsilon_mix", rr.LineStrips2D((sqrt_one_minus_alpha_hat * epsilon).cpu().numpy()[0], radii=0.0005))
        rr.log("noise_traj/x_epsilon_mix", rr.LineStrips2D((x_epislon_mix).cpu().numpy()[0], radii=0.0005))
        rr.log("noise_traj/t", rr.Scalar(t[0].item()))

        return x_epislon_mix, epsilon

    def noise_scalar(self, x, t):
        """ For 1d data """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])  # beta hat?
        sqrt_alpha_hat = sqrt_alpha_hat[:, None]
        sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat[:, None]

        epsilon = torch.randn_like(x, device=self.device)

        noise_scalar = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noise_scalar, epsilon

    def sample_timestamps(self, n):
        """
        used to generate targets when training

        :param n: Number of timesteps to sample
        :return: timestep samples
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def yield_images(self, model, n_samples):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n_samples,) + self.shape).to(self.device)
            for i in tqdm(reversed(range(0, self.noise_steps)), total=self.noise_steps):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
                image_np = torch.clamp(x.squeeze() * 255, 0, 255).cpu().numpy().transpose(1, 2, 0)
                yield image_np

    def sample_images(self, model, n_samples):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n_samples,) + self.shape).to(self.device)
            sampling_process_images = []
            for i in tqdm(reversed(range(0, self.noise_steps)), total=self.noise_steps):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
                sampling_process_images.append(x.detach().cpu().numpy())
        model.train()
        x = torch.clamp(x * 255, 0, 255)
        sampling_process_images = torch.stack(sampling_process_images, axis=0)
        sampling_process_images = torch.clamp(sampling_process_images * 255, 0, 255)
        return x, sampling_process_images

    def sample_traj(self, model, n_samples):
        model.eval()
        all_samples = []
        with torch.no_grad():
            x = torch.randn((n_samples,) + self.shape).to(self.device)
            for i in reversed(range(0, self.noise_steps)):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)
                all_samples.append(x.detach().numpy())

                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                alpha_pred_noise = ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                x_sub_noise = x - alpha_pred_noise
                x = 1 / torch.sqrt(alpha) * x_sub_noise + torch.sqrt(beta) * noise
        model.train()
        all_samples = np.squeeze(np.array(all_samples))
        x = np.squeeze(x.numpy())
        return x, all_samples

    def sample_scalar(self, model, n_samples):
        model.eval()
        all_samples = []
        with torch.no_grad():
            x = torch.randn((n_samples,) + self.shape).to(self.device)
            for i in reversed(range(0, self.noise_steps)):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                predicted_noise = model(x, t)
                all_samples.append(x.detach().numpy())

                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        all_samples = np.squeeze(np.array(all_samples))
        x = np.squeeze(x.numpy())
        return x, all_samples
