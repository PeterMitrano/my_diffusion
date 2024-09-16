import math

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tensorboard.summary.v1 import image


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            )
        )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            )
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):

    def __init__(self, shape):
        """

        :param shape: a 3-tuple
        """
        super().__init__()
        self.shape = shape
        channels = shape[0]
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Move the channel dimension to the end, and flatten the spatial dimensions
        x = x.reshape(-1, self.shape[0], self.shape[1] * self.shape[2])
        x = x.swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value * x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_out = attention_value.swapaxes(2, 1).reshape(-1, self.shape[0], self.shape[1], self.shape[2])
        return attention_out


class LinearWithPosEmb(nn.Module):

    def __init__(self, in_dim, out_dim, emb_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
        self.a = nn.GELU()
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_dim
            )
        )

    def forward(self, x, t):
        h = self.linear(x)
        a = self.a(h)
        emb = self.emb_layer(t)[:, None, None, :]
        return a + emb


class MyToyMLP(nn.Module):

    def __init__(self, emb_dim=256, device='cpu'):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device

        self.l1 = LinearWithPosEmb(1, 64, emb_dim)
        self.a1 = nn.Tanh()
        self.l2 = LinearWithPosEmb(64, 256, emb_dim)
        self.a2 = nn.Tanh()
        self.l3 = LinearWithPosEmb(256, 256, emb_dim)
        self.a3 = nn.Tanh()
        self.l4 = LinearWithPosEmb(256, 64, emb_dim)
        self.a4 = nn.Tanh()
        self.l4 = LinearWithPosEmb(64, 1, emb_dim)

    def pos_encoding(self, t, pos_embed_dim=64):
        freq = 10_000 ** (torch.arange(0, pos_embed_dim, 2, device=self.device).float() / pos_embed_dim)
        inv_freq = 1 / freq
        pos_enc_a = torch.sin(t.repeat(1, pos_embed_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, pos_embed_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        pos_emb = self.pos_encoding(t, self.emb_dim)

        h1 = self.l1(x, pos_emb)
        z1 = self.a1(h1)
        h2 = self.l2(z1, pos_emb)
        z2 = self.a2(h2)
        h3 = self.l3(z2, pos_emb)
        z3 = self.a3(h3)
        pred_noise = self.l4(z3, pos_emb)

        if self.training:
            plt.figure()
            plt.title("pred noise in MyToyMLP when training")
            plt.hist(pred_noise.squeeze().detach().numpy(), color='r', bins=50)
            plt.xlim([-2, 2])
            plt.ylim([0, 100])
            plt.show()
        return pred_noise


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.emb_dim = embedding_dim
        self.device = device

    # def forward(self, t):
    #     freq = 10_000 ** (torch.arange(0, self.emb_dim, 2, device=self.device).float() / self.emb_dim)
    #     inv_freq = 1 / freq
    #     t.repeat(1, self.emb_dim // 2) * inv_freq
    #     pos_enc_a = torch.sin(t.repeat(1, self.emb_dim // 2) * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(1, self.emb_dim // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc

    def forward(self, t):
        # t is expected to be a Long tensor of shape [batch_size]
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # Shape: [batch_size, embedding_dim]


class MyToyMLP2(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Embedding for the time step t
        self.time_embed = SinusoidalTimeEmbedding(64, device=device)  # Embedding dim for time steps

        # Main MLP for diffusion
        self.mlp = nn.Sequential(
            nn.Linear(1 + 64, 128),  # Input is scalar x and time embedding
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output is scalar prediction
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t)

        x = x.squeeze(1).squeeze(1)  # remove image-like dims
        x_t = torch.cat([x, t_embed], dim=-1)

        out = self.mlp(x_t)
        out = out[..., None, None]  # Add back image-like dims
        return out


class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3, pos_emb_dim=256, image_size=64, device='cpu'):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.pos_emb_dim = pos_emb_dim
        self.device = device

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention((128, int(image_size / 2), int(image_size / 2)))
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention((256, int(image_size / 4), int(image_size / 4)))
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention((256, int(image_size / 8), int(image_size / 8)))

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention((128, int(image_size / 4), int(image_size / 4)))
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention((64, int(image_size / 2), int(image_size / 2)))
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention((64, image_size, image_size))
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.time_embed = SinusoidalTimeEmbedding(pos_emb_dim, device=device)

    def pos_encoding(self, t, pos_emb_dim):
        freq = 10_000 ** (torch.arange(0, pos_emb_dim, 2, device=self.device).float() / pos_emb_dim)
        inv_freq = 1 / freq
        t.repeat(1, pos_emb_dim // 2) * inv_freq
        pos_enc_a = torch.sin(t.repeat(1, pos_emb_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, pos_emb_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        # pos_emb = self.pos_encoding(t, self.pos_emb_dim)
        pos_emb = self.time_embed(t)

        x1 = self.inc(x)
        x2 = self.down1(x1, pos_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, pos_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, pos_emb)
        x4 = self.sa3(x4)

        x5 = self.bot1(x4)
        x6 = self.bot2(x5)
        x7 = self.bot3(x6)

        x8 = self.up1(x7, x3, pos_emb)
        x8 = self.sa4(x8)
        x9 = self.up2(x8, x2, pos_emb)
        x9 = self.sa5(x9)
        x10 = self.up3(x9, x1, pos_emb)
        x10 = self.sa6(x10)
        out = self.outc(x10)

        # if self.training:
        #     fig, ax = plt.subplots(1, 1)
        #     predicted_noise_0 = out[:, 0, 0, 0]
        #     ax.hist(predicted_noise_0.squeeze().detach().numpy(), color='m')
        #     plt.show()

        return out
