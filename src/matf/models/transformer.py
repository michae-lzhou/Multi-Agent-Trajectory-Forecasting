import torch
import torch.nn as nn

# The module is normally used as part of the ``matf`` package.  When the
# file is executed directly (e.g. ``python src/matf/models/transformer.py``)
# the relative import below will fail with ``ImportError: attempted relative
# import with no known parent package``.  We provide a fallback path that
# makes the script runnable from either the repository root or the
# ``src/matf/models`` directory by manipulating ``sys.path``.
try:
    from .lstm import LSTMEncoder, LSTMDecoder
except ImportError:  # pragma: no cover - only for direct execution
    import os, sys  # noqa: F401

    # ensure the containing directory is on the path so ``import lstm`` can
    # locate the sibling module.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from lstm import LSTMEncoder, LSTMDecoder


class SocialAttention(nn.Module):
    """Single-head or multi‑head "social" attention module.

    Implements the attention mechanism described in the design notes: the
    focal agent attends over a scene consisting of itself and its neighbours.
    The implementation wraps :class:`nn.MultiheadAttention` with a couple of
    small conveniences (batch‑first tensors, optional residual connection and
    layer norm) so that the calling code stays clean.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_residual: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"

        # we want batch‑first behaviour; MultiheadAttention defaults to
        # (seq, batch, embed) so ``batch_first=True`` saves a lot of transposes.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.use_residual = use_residual
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention.

        Args:
            query: ``(B, 1, H)`` tensor containing the focal token.
            key, value: ``(B, S, H)`` scene tensors. ``S`` is typically ``N+1``.
            mask: boolean or 0/1 mask of shape ``(B, S)``; ``1`` for valid
                positions. ``False``/``0`` entries will be ignored by the
                attention.

        Returns:
            ``(B, 1, H)`` tensor containing the attended focal vector.
        """
        # ``key_padding_mask`` expects True in locations that should be
        # *ignored*, so invert the incoming mask.
        if mask is not None:
            key_padding_mask = mask == 0
        else:
            key_padding_mask = None

        attended, _ = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
        )

        if self.use_residual:
            attended = attended + query
        if self.layer_norm is not None:
            attended = self.layer_norm(attended)
        return attended


class SocialLSTMTrajectoryForecaster(nn.Module):
    """Encoder/decoder LSTM with a social attention bottleneck.

    The class implements the sequence of steps described in the user's notes.
    A common ``LSTMEncoder`` is used for focal and neighbour agents; the
    hidden states are concatenated into a scene tensor and the focal agent
    attends over the scene.  The resulting vector initialises the decoder
    hidden state.  Several ablation options (number of heads, residual
    connection, layer norm, cell state selection) are provided to facilitate
    experiments.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_heads: int = 1,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        cell_state: str = "zeros",
    ):
        super().__init__()
        assert cell_state in ("zeros", "focal", "attend"), (
            "cell_state must be one of 'zeros', 'focal' or 'attend'"
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_state = cell_state

        # shared encoder for all agents
        self.encoder = LSTMEncoder(hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout)

        self.social_attn = SocialAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self,
        focal_obs: torch.Tensor,
        neighbors_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        target: torch.Tensor | None = None,
        mode: str = "zero",
    ) -> torch.Tensor:
        """Predict future trajectory for the focal agent.

        Args:
            focal_obs: ``(B, 50, 4)`` past observation of focal agent.
            neighbors_obs: ``(B, N, 50, 4)`` past observations of neighbours.
            neighbor_mask: ``(B, N)`` binary mask indicating which neighbour
                entries are valid (1) and which are padding (0).
            target: optional ground‑truth future used when ``mode`` is
                ``"teacher_forcing"``.
            mode: decoding strategy passed through to ``LSTMDecoder``.

        Returns:
            ``(B, 1, 60, 2)`` predicted trajectory for the focal agent.
        """
        B = focal_obs.shape[0]
        N = neighbors_obs.shape[1]

        # encode focal agent and extract last layer token
        h_f, c_f = self.encoder(focal_obs)  # each: (L, B, H)
        focal_token = h_f[-1]  # (B, H)

        # encode neighbours in a batched fashion
        # flatten batch and neighbour dims so the encoder can operate once
        neighbors_flat = neighbors_obs.view(B * N, -1, focal_obs.shape[-1])
        h_n, c_n = self.encoder(neighbors_flat)  # (L, B*N, H)
        # reshape back to separate neighbours
        neighbor_tokens = h_n[-1].view(B, N, self.hidden_size)
        neighbor_cells = c_n[-1].view(B, N, self.hidden_size)

        # build scene and mask
        scene_tokens = torch.cat(
            [focal_token.unsqueeze(1), neighbor_tokens], dim=1
        )  # (B, N+1, H)
        mask_ext = torch.cat(
            [torch.ones((B, 1), dtype=neighbor_mask.dtype, device=neighbor_mask.device),
             neighbor_mask],
            dim=1,
        )  # (B, N+1)

        # social attention produces new focal representation
        attended = self.social_attn(
            focal_token.unsqueeze(1), scene_tokens, scene_tokens, mask_ext
        )  # (B, 1, H)
        attended = attended.squeeze(1)  # (B, H)

        # initialise decoder hidden state
        h_0 = attended.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H)

        # cell state initialisation strategy
        if self.cell_state == "zeros":
            c_0 = torch.zeros_like(h_0)
        elif self.cell_state == "focal":
            c_0 = c_f
        else:  # "attend"
            scene_cells = torch.cat(
                [c_f[-1].unsqueeze(1), neighbor_cells], dim=1
            )  # (B, N+1, H)
            cell_attended = self.social_attn(
                c_f[-1].unsqueeze(1), scene_cells, scene_cells, mask_ext
            )  # (B, 1, H)
            c_0 = cell_attended.squeeze(1).unsqueeze(0).repeat(
                self.num_layers, 1, 1
            )

        preds = self.decoder(h_0, c_0, target=target, mode=mode)
        return preds


# quick sanity checks (similar to the LSTM file)
if __name__ == "__main__":
    # make running the script standalone friendly by adding its parent
    # directory to ``sys.path`` (handles being invoked from repo root).
    import os, sys
    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)

    model = SocialLSTMTrajectoryForecaster(hidden_size=128, num_layers=2)
    focal = torch.randn(8, 50, 4) # B, S, F
    neigh = torch.randn(8, 3, 50, 4) # B, N, S, F
    mask = torch.tensor([[1, 1, 0]] * 8, dtype=torch.float32)

    out = model(focal, neigh, mask, mode="zero")
    print(out.shape)  # expect (8, 1, 60, 2)

    out = model(focal, neigh, mask, mode="last_pred")
    print(out.shape)

    gt = torch.randn(8, 60, 2)
    out = model(focal, neigh, mask, target=gt, mode="teacher_forcing")
    print(out.shape)
