import torch
import torch.nn as nn

class SocialAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_residual=False,
            layer_norm="no_norm"):
        super().__init__()
        self.use_residual = use_residual
        self.layer_norm = layer_norm

        self.attention = nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True
                    )

        if layer_norm != "no_norm":
            self.layernorm = nn.LayerNorm(hidden_size)


    def forward(self, focal_token, scene_tokens, key_padding_mask):

        Q = focal_token
        K = scene_tokens
        V = scene_tokens

        if self.layer_norm == "no_norm":
            pass
        if self.layer_norm == "pre_norm":
            Q = self.layernorm(focal_token)
            K = self.layernorm(scene_tokens)
            V = self.layernorm(scene_tokens)

        attn_output, attn_weights = self.attention(Q, K, V, key_padding_mask)

        attn_output = attn_output.squeeze(dim=1)

        if self.use_residual:
            attn_output = focal_token.squeeze(dim=1) + attn_output

        if self.layer_norm == "post_norm":
            attn_output = self.layernorm(attn_output)
        elif self.layer_norm in ("no_norm", "pre_norm"):
            pass
        else:
            raise ValueError(f"layer_norm received invalid parameter: "
                             f"{self.layer_norm}")

        return attn_output, attn_weights

# attn = SocialAttention(hidden_size=256, num_heads=4,
#                        use_residual=True, layer_norm="no_norm")
# 
# B, N, hidden = 4, 5, 256
# focal_token  = torch.randn(B, 1, hidden)
# scene_tokens = torch.randn(B, N+1, hidden)
# 
# # mask: last 2 neighbors are padding
# mask = torch.zeros(B, N+1, dtype=torch.bool)
# mask[:, -2:] = True   # True = ignore
# 
# out = attn(focal_token, scene_tokens, mask)
# print(out.shape)   # expect (4, 256)
