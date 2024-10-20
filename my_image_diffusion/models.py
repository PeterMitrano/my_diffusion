import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim, device):
        super().__init__()
        self.emb_dim = embedding_dim
        self.device = device

    def forward(self, t):
        emb = torch.exp(-torch.arange(self.emb_dim, dtype=torch.float32, device=t.device))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.sin(emb)
        return emb


class TimeEmbConverter(nn.Module):
    """ Just convert long to float and add a dimension on the end """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(-1).float()


class LinearTimeEmbedding(nn.Module):
    """ Just a linear layer but it converts long to float """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.cvt = TimeEmbConverter()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.cvt(x)
        return self.linear(x)


class IdentityTimeEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float()[..., None]


class DiffusionModel(nn.Module):
    def __init__(self, flat_data_dim=1, h1=128, h2=128, h3=128, time_emb_dim=64, time_emb_mode='sin', device='cpu'):
        super().__init__()
        self.flat_data_dim = flat_data_dim
        self.time_emb_mode = time_emb_mode

        match time_emb_mode:
            case 'sin':
                self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, device=device)
            case 'linear':
                self.time_embed = LinearTimeEmbedding(1, time_emb_dim)
            case 'learned':
                self.time_embed = nn.Embedding(100, time_emb_dim)
            case 'mlp':
                self.time_embed = nn.Sequential(
                    TimeEmbConverter(),
                    nn.Linear(1, h1),
                    nn.ReLU(),
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                    nn.Linear(h2, time_emb_dim)
                )
            case None:
                time_emb_dim = 1
                self.time_embed = IdentityTimeEmbedding()
            case _:
                raise ValueError(f"Invalid time embedding mode: {time_emb_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(flat_data_dim + time_emb_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, flat_data_dim)
        )

    def forward(self, x, t):
        in_shape = x.shape
        x_flat = x.reshape([-1, self.flat_data_dim])
        t_embed = self.time_embed(t)
        x_t = torch.cat([x_flat, t_embed], dim=-1)
        out = self.mlp(x_t)
        out = out.reshape(in_shape)
        return out
