"""
Bi-LSTM for LabVIEW inference (Ps spectrum only).

Input: ``[Ps]`` per frequency bin (z-scored).
Output: scalar ``(P_total, Q_total)`` heads (still z-scored; denormalize outside).
"""
import torch
import torch.nn as nn

from utils.constants import DEVICE, DROPOUT, NUM_LAYERS

HIDDEN_SIZE = 128
PROJ_DIM = 32


class LstmModel(nn.Module):
    """Bi-LSTM on projected Ps tokens; mean-pool + linear P/Q heads."""

    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, proj_dim=PROJ_DIM):
        super().__init__()
        if int(input_size) != 1:
            raise ValueError(f'LstmModel expects input_size=1 ([Ps]), got {input_size}')
        if int(proj_dim) < 1:
            raise ValueError(f'proj_dim must be positive, got {proj_dim}')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.proj_dim = int(proj_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
        )
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.proj_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=enc_dropout,
            device=DEVICE,
        )
        trunk_dim = 2 * self.hidden_size
        self.head_p = nn.Linear(trunk_dim, 1)
        self.head_q = nn.Linear(trunk_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f'LstmModel expects input [B, L, 1], got shape {tuple(x.shape)}')
        (enc_out, _) = self.encoder(self.input_proj(x))
        ctx = enc_out.mean(dim=1)
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LstmModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        hidden_size=ckpt['hidden_size'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt.get('dropout', DROPOUT),
        proj_dim=ckpt.get('proj_dim', PROJ_DIM),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


if __name__ == '__main__':
    model = LstmModel(hidden_size=8, num_layers=1, proj_dim=4)
    pred_p, pred_q = model(torch.zeros(2, 10, 1, device=DEVICE))
    assert pred_p.shape == (2,) and pred_q.shape == (2,)
