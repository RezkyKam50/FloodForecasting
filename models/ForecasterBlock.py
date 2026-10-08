import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class _AutoregressiveForecaster(nn.Module):
    """Handles windowing, residual bases and teacher forcing. Subclasses define `_predict`."""
    DYN_CH = 2  # depth + rain

    def __init__(self, n_static, t_in=4, use_future_rain=True, residual=True, rain_zero_z=0.0,
                 residual_base="persistence", velocity_clip=3.0):
        super().__init__()
        self.n_static, self.t_in = n_static, int(t_in)
        self.use_future_rain, self.residual = use_future_rain, residual
        self.residual_base, self.velocity_clip = residual_base, float(velocity_clip)
        self.rain_zero_z = float(rain_zero_z)
        self.in_ch = self.DYN_CH * self.t_in + n_static

    def _predict(self, x):  # (B, in_ch, H, W) -> (B, 1, H, W)
        raise NotImplementedError

    def _fit(self, seq):
        """(B, T, H, W) -> last t_in frames, left-padded by repeating the first frame."""
        T = seq.shape[1]
        if T >= self.t_in:
            return seq[:, -self.t_in:]
        return torch.cat([seq[:, :1].expand(-1, self.t_in - T, -1, -1), seq], 1)

    def forward(self, depth_hist, rain_hist, rain_fut, static, out_steps=None, teacher=None, tf_prob=0.0):
        B, Tin, _, h, w = depth_hist.shape
        K = out_steps or rain_fut.shape[1]

        d_win = self._fit(depth_hist[:, :, 0])
        r_hist = self._fit(rain_hist[:, :, 0])
        if self.use_future_rain:
            r_fut = rain_fut[:, :K, 0]
        else:
            r_fut = torch.full((B, K, h, w), self.rain_zero_z, device=d_win.device, dtype=d_win.dtype)
        rain_all = torch.cat([r_hist, r_fut], 1)  # window for step k = rain_all[:, k+1 : k+1+t_in]

        cur, outs = depth_hist[:, -1], []
        prev = depth_hist[:, -2] if Tin > 1 else cur
        for k in range(K):
            parts = [d_win, rain_all[:, k + 1:k + 1 + self.t_in]]
            if self.n_static:
                parts.append(static)
            d = self._predict(torch.cat(parts, 1))
            if self.residual and self.residual_base == "velocity":
                nxt = cur + (cur - prev).clamp(-self.velocity_clip, self.velocity_clip) + d
            else:
                nxt = cur + d if self.residual else d
            outs.append(nxt)
            if teacher is not None and k < K - 1 and tf_prob > 0:
                m = (torch.rand(B, 1, 1, 1, device=cur.device) < tf_prob).to(cur.dtype)
                new = m * teacher[:, k] + (1 - m) * nxt
            else:
                new = nxt
            prev, cur = cur, new
            d_win = torch.cat([d_win[:, 1:], cur], 1)
        return torch.stack(outs, 1)