import torch
import torch.nn as nn
from matf.models.lstm import LSTMEncoder, LSTMDecoder
from matf.models.attention import SocialAttention

class TransformerForecaster(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_heads=4, dropout=0.0,
                 use_residual=True, layer_norm="no_norm"):
        super().__init__()

        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.encoder = LSTMEncoder(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)
        self.attention = SocialAttention(hidden_size=hidden_size,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         use_residual=use_residual,
                                         layer_norm=layer_norm)
        self.decoder = LSTMDecoder(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

    def forward(self, focal_obs, neighbor_obs, neighbor_mask, target=None,
                mode="last_pred", cell_state="zeros", return_attn=False):
        B, N, T, F = neighbor_obs.shape
        focal_h, focal_c = self.encoder(focal_obs)
        focal_token = focal_h[-1]

        neighbor_obs_flat = neighbor_obs[neighbor_mask]
        # neighbor_obs_flat = neighbor_obs.view(-1, T, F)
        # assert neighbor_obs_flat.shape == (B*N, T, F), \
        #        f"Expected {(B*N, T, F)}, got {neighbor_obs_flat.shape}"

        neighbor_h, _ = self.encoder(neighbor_obs_flat)
        real_tokens = neighbor_h[-1]

        # cast the tokens back into the padding
        neighbor_tokens = torch.zeros(B, N, self.hidden_size,
                                      device=focal_obs.device)
        neighbor_tokens[neighbor_mask] = real_tokens
        neighbor_tokens = neighbor_tokens.view(B, N, -1)

        focal_token_3d = focal_token.unsqueeze(1)
        scene_tokens = torch.cat([focal_token_3d, neighbor_tokens], dim=1)

        focal_mask = torch.zeros(B, 1, dtype=torch.bool,
                                  device=focal_obs.device)
        key_padding_mask = torch.cat([focal_mask, ~neighbor_mask], dim=1)

        attended, attn_weights= self.attention(focal_token_3d, scene_tokens, key_padding_mask)

        h_0 = attended.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros(self.num_layers, B, self.hidden_size,
                          device=focal_obs.device)
        if cell_state == "focal":
            c_0 = focal_c
        elif cell_state != "zeros":
            raise ValueError(f"cell_state received invalid input: {cell_state}")

        pred = self.decoder(h_0, c_0, target=target, mode=mode)

        if return_attn:
            return pred, attn_weights
        return pred

# model = TransformerForecaster(hidden_size=256, num_layers=2, num_heads=4)
# focal_obs    = torch.randn(4, 50, 4)
# neighbor_obs = torch.randn(4, 5, 50, 4)
# neighbor_mask = torch.ones(4, 5, dtype=torch.bool)
# neighbor_mask[:, -2:] = False   # last 2 are padding
# 
# pred = model(focal_obs, neighbor_obs, neighbor_mask)
# print(pred.shape)   # expect (4, 1, 60, 2)
