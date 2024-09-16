import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, device='cpu'):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # Shape: [batch_size, embedding_dim]


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layernorm(x + self.block(x))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Self-attention expects [sequence_length, batch_size, embedding_dim] format
        x = x.unsqueeze(0)  # Add a fake sequence length dimension
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm(x + attn_output)
        x = self.layernorm(x + self.ff(x))
        return x.squeeze(0)  # Remove the fake sequence length dimension


class DiffusionModelWithResNet(nn.Module):
    def __init__(self, device='cpu'):
        super(DiffusionModelWithResNet, self).__init__()

        self.device = device

        # Time embedding using sinusoidal encoding
        self.time_embed = SinusoidalTimeEmbedding(64, device=self.device)

        # MLP with Residual Blocks
        self.input_layer = nn.Linear(1 + 64, 128)
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)

        # Self-attention block
        self.attention = SelfAttentionBlock(dim=128, num_heads=4)

        # Output layer for final scalar prediction
        self.final_layer = nn.Linear(128, 1)

        # Move the model to the device
        self.to(self.device)

    def forward(self, x, t):
        # Move x and t to the specified device
        x = x.to(self.device)
        t = t.to(self.device)

        # Create time embedding
        t_embed = self.time_embed(t)

        # Concatenate input scalar x (reshaped to batch_size x 1) with time embedding
        x_t = torch.cat([x, t_embed], dim=-1)

        # Pass through the input layer and residual MLP blocks
        x = self.input_layer(x_t)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Apply self-attention block
        x = self.attention(x)

        # Final prediction
        return self.final_layer(x)
