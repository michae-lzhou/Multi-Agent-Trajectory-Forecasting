import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM (
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, h_n, c_n, steps=60, target=None, mode="zero"):
        B = h_n.shape[1]

        curr_input = torch.zeros(B, 1, 2, device=h_n.device)
        preds = []

        for t in range(steps):
            output, (h_n, c_n) = self.lstm(curr_input, (h_n, c_n))  # (B, 1, 2)
            output = output.squeeze(dim=1)                          # (B, 2)
            pred = self.output_layer(output)                        # (B, 2)
            preds.append(pred)
            if mode == "zero":
                next_input = torch.zeros(B, 1, 2, device=curr_input.device)
            elif mode == "teacher_forcing":
                if target is None:
                    raise ValueError("teacher_forcing mode requires target to "
                                     "be passed in")
                next_input = target[:, t, :].unsqueeze(1)
            elif mode == "last_pred":
                next_input = pred.unsqueeze(1)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            curr_input = next_input

        preds = torch.stack(preds, dim=1)                   # (B, 60, 2)
        return preds.unsqueeze(1)                           # (B, 1, 60, 2)

class LSTMTrajectoryForecaster(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.encoder = LSTMEncoder(hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout)

    def forward(self, focal_obs, target=None, mode="zero"):
        h_n, c_n = self.encoder(focal_obs)
        preds = self.decoder(h_n, c_n, target=target, mode=mode)
        return preds

# model = LSTMTrajectoryForecaster(hidden_size=128, num_layers=2)
# x = torch.randn(8, 50, 4)
# 
# # test all three modes
# pred = model(x, mode="zero")
# print(pred.shape)   # expect (8, 1, 60, 2)
# 
# pred = model(x, mode="last_pred")
# print(pred.shape)   # expect (8, 1, 60, 2)
# 
# gt = torch.randn(8, 60, 2)
# pred = model(x, target=gt, mode="teacher_forcing")
# print(pred.shape)   # expect (8, 1, 60, 2)
