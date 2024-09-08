import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from mpmath import hyp3f2


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

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
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

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
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
        x = x.reshape(-1, self.shape[0], self.shape[1] * self.shape[2]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value * x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(-1, self.shape[0], self.shape[1], self.shape[2])


class MyToyMLP(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        self.l1 = nn.Linear(1, 256)
        self.a1 = nn.GELU()
        self.l2 = nn.Linear(256, 256)
        self.a2 = nn.GELU()
        self.l3 = nn.Linear(256, 256)
        self.a3 = nn.GELU()
        self.l4 = nn.Linear(256, 256)
        self.a4 = nn.GELU()
        self.l4 = nn.Linear(256, 1)
        self.a4 = nn.Sigmoid()

    def pos_encoding(self, t, pos_embed_dim=2):
        freq = 10_000 ** (torch.arange(0, pos_embed_dim, 2, device=self.device).float() / pos_embed_dim)
        inv_freq = 1 / freq
        pos_enc_a = torch.sin(t.repeat(1, pos_embed_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, pos_embed_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        import matplotlib.pyplot as plt
        plt.imshow(pos_enc)
        plt.show()
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        pos_emb = self.pos_encoding(t)

        h1 = self.l1(x + pos_emb)
        z1 = self.a1(h1)
        h2 = self.l2(z1 + pos_emb)
        z2 = self.a2(h2)
        h3 = self.l3(z2 + pos_emb)
        z3 = self.a3(h3)
        h4 = self.l4(z3 + pos_emb)
        pred_noise = self.a4(h4)

        return pred_noise


class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3, time_dim=256, device='cpu'):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.time_dim = time_dim
        self.device = device

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention((128, 32, 32))
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention((256, 16, 16))
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention((256, 8, 8))

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention((128, 16, 16))
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention((64, 32, 32))
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention((64, 64, 64))
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        freq = 10_000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        inv_freq = 1 / freq
        t.repeat(1, channels // 2) * inv_freq
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x5 = self.bot1(x4)
        x6 = self.bot2(x5)
        x7 = self.bot3(x6)

        x8 = self.up1(x7, x3, t)
        x8 = self.sa4(x8)
        x9 = self.up2(x8, x2, t)
        x9 = self.sa5(x9)
        x10 = self.up3(x9, x1, t)
        x10 = self.sa6(x10)
        out = self.outc(x10)

        return out
