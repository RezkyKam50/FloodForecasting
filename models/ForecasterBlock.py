import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveForecaster(nn.Module):
    DYN_CH = 2  # depth + rain
    def __init__(self, n_static, t_in=4, use_future_rain=True, residual=True,
                 rain_zero_z=0.0, residual_base="persistence", velocity_clip=3.0,
                 n_prior=0):
        super().__init__()
        self.n_static, self.t_in = n_static, int(t_in)
        self.use_future_rain, self.residual = use_future_rain, residual
        self.residual_base, self.velocity_clip = residual_base, float(velocity_clip)
        self.rain_zero_z = float(rain_zero_z)
        self.n_prior = int(n_prior)
        self.in_ch = self.DYN_CH * self.t_in + n_static + self.n_prior

    def _fit(self, seq):
        T = seq.shape[1]
        if T >= self.t_in:
            return seq[:, -self.t_in:]
        return torch.cat([seq[:, :1].expand(-1, self.t_in - T, -1, -1), seq], 1)

    def rollout(self, predict_fn, depth_hist, rain_hist, rain_fut, static,
                out_steps=None, teacher=None, tf_prob=0.0, prior=None):
        B, Tin, _, h, w = depth_hist.shape
        K = out_steps or rain_fut.shape[1]

        d_win = self._fit(depth_hist[:, :, 0])
        r_hist = self._fit(rain_hist[:, :, 0])
        if self.use_future_rain:
            r_fut = rain_fut[:, :K, 0]
        else:
            r_fut = torch.full((B, K, h, w), self.rain_zero_z,
                               device=d_win.device, dtype=d_win.dtype)
        rain_all = torch.cat([r_hist, r_fut], 1)

        cur, outs = depth_hist[:, -1], []
        prev = depth_hist[:, -2] if Tin > 1 else cur
        for k in range(K):
            parts = [d_win, rain_all[:, k + 1:k + 1 + self.t_in]]
            if self.n_static:
                parts.append(static)
            if self.n_prior and prior is not None:
                parts.append(prior)
            d = predict_fn(torch.cat(parts, 1))
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