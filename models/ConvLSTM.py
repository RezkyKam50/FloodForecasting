import torch
import torch.nn as nn
import math


def make_norm(kind, ch):
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, ch) or 1, ch)
    return nn.Identity()


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
        nn.init.zeros_(self.conv.bias)
        self.conv.bias.data[hid:2 * hid] = 1.0   # forget-gate bias = 1

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], 1)), self.hid, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

    def init_state(self, b, hw, device, dtype):
        z = torch.zeros(b, self.hid, *hw, device=device, dtype=dtype)
        return z, z.clone()


# model by kelvin
class PlainConvLSTM(nn.Module):
    """ConvLSTM encoder-forecaster; statics are concatenated to the recurrent input."""
    DYN_CH = 2   # depth + rain

    def __init__(self, n_static, hidden=32, kernel=3, norm="none", head_width=32, use_future_rain=True,
                 residual=True, rain_zero_z=0.0, residual_base="persistence", velocity_clip=3.0):
        super().__init__()
        self.n_static, self.use_future_rain, self.residual = n_static, use_future_rain, residual
        self.residual_base, self.velocity_clip = residual_base, float(velocity_clip)
        self.rain_zero_z = float(rain_zero_z)
        self.cell = ConvLSTMCell(self.DYN_CH + n_static, hidden, kernel)
        self.mid = make_norm(norm, hidden)
        self.head = OutputHead(hidden, head_width, kernel)

    def _rnn_input(self, depth_z, rain_z, static):
        return torch.cat([depth_z, rain_z] + ([static] if self.n_static else []), 1)

    def forward(self, depth_hist, rain_hist, rain_fut, static, out_steps=None, teacher=None, tf_prob=0.0):
        B, Tin, _, h, w = depth_hist.shape
        K = out_steps or rain_fut.shape[1]
        state = self.cell.init_state(B, (h, w), depth_hist.device, depth_hist.dtype)

        # encode the history
        for t in range(Tin):
            state = self.cell(self._rnn_input(depth_hist[:, t], rain_hist[:, t], static), state)

        # roll the forecast out autoregressively (with optional teacher forcing)
        cur, outs = depth_hist[:, -1], []
        prev = depth_hist[:, -2] if Tin > 1 else cur
        for k in range(K):
            r = rain_fut[:, k] if self.use_future_rain else torch.full_like(cur, self.rain_zero_z)
            state = self.cell(self._rnn_input(cur, r, static), state)
            d = self.head(self.mid(state[0]))
            if self.residual and self.residual_base == "velocity":
                nxt = cur + (cur - prev).clamp(-self.velocity_clip, self.velocity_clip) + d
            else:
                nxt = cur + d if self.residual else d
            outs.append(nxt)
            if teacher is not None and k < K - 1 and tf_prob > 0:
                m = (torch.rand(B, 1, 1, 1, device=cur.device) < tf_prob).to(cur.dtype)
                prev, cur = cur, m * teacher[:, k] + (1 - m) * nxt
            else:
                prev, cur = cur, nxt
        return torch.stack(outs, 1)


class OutputHead(nn.Module):
    def __init__(self, in_ch, width=32, k=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, width, k, padding=k // 2),
                                 nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)
