import math

import torch
import torch.nn as nn
import torch.nn.functional as F

 
def make_norm(kind, ch):
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, ch) or 1, ch)
    return nn.Identity()


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


def _coord_grid(b, h, w, device, dtype):
    gy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    gx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([gy, gx], 1)

 
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


# --------------------------------------------------------------------------- #
# UNet
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, norm):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2), make_norm(norm, out_ch), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=k // 2), make_norm(norm, out_ch), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class PlainUNet(_AutoregressiveForecaster):
    """Classic UNet encoder-decoder; widths = hidden * 2**level."""

    def __init__(self, n_static, hidden=32, kernel=3, norm="none", head_width=32, use_future_rain=True,
                 residual=True, rain_zero_z=0.0, residual_base="persistence", velocity_clip=3.0,
                 t_in=4, levels=3):
        super().__init__(n_static, t_in, use_future_rain, residual, rain_zero_z, residual_base, velocity_clip)
        self.levels = levels
        chs = [hidden * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList()
        c = self.in_ch
        for i in range(levels):
            self.enc.append(ConvBlock(c, chs[i], kernel, norm))
            c = chs[i]
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[levels - 1], chs[levels], kernel, norm)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in range(levels)])
        self.dec = nn.ModuleList([ConvBlock(2 * chs[i], chs[i], kernel, norm) for i in range(levels)])
        self.head = OutputHead(chs[0], head_width, kernel)

    def _predict(self, x):
        H, W = x.shape[-2:]
        s = 2 ** self.levels
        ph, pw = (-H) % s, (-W) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in reversed(range(self.levels)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)[..., :H, :W]


# --------------------------------------------------------------------------- #
# FNO
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.out_ch, self.modes1, self.modes2 = out_ch, modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        # store as real float with a trailing 2 dim = (re, im)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    def _c(self, w):
        return torch.view_as_complex(w.contiguous())

    def forward(self, x):
        B, _, H, W = x.shape
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
            w1 = self._c(self.w1)[:, :, :m1, :m2]
            w2 = self._c(self.w2)[:, :, :m1, :m2]
            out = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
            out[:, :, :m1, :m2]  = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2],  w1)
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], w2)
            y = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return y.to(dtype)


class PlainFNO(_AutoregressiveForecaster):
    """Vanilla FNO: lift -> n_layers x (spectral + 1x1 bypass, GELU) -> OutputHead."""

    def __init__(self, n_static, hidden=32, kernel=3, norm="none", head_width=32, use_future_rain=True,
                 residual=True, rain_zero_z=0.0, residual_base="persistence", velocity_clip=3.0,
                 t_in=4, modes=12, n_layers=4, pad=8, use_grid=True):
        super().__init__(n_static, t_in, use_future_rain, residual, rain_zero_z, residual_base, velocity_clip)
        self.pad, self.use_grid = pad, use_grid
        self.lift = nn.Conv2d(self.in_ch + (2 if use_grid else 0), hidden, 1)
        self.spec = nn.ModuleList([SpectralConv2d(hidden, hidden, modes, modes) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([make_norm(norm, hidden) for _ in range(n_layers)])
        self.head = OutputHead(hidden, head_width, kernel)

    def _predict(self, x):
        B, _, H, W = x.shape
        if self.use_grid:
            x = torch.cat([x, _coord_grid(B, H, W, x.device, x.dtype)], 1)
        x = self.lift(x)
        if self.pad:
            x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        n = len(self.spec)
        for i in range(n):
            x = self.norms[i](self.spec[i](x) + self.skip[i](x))
            if i < n - 1:
                x = F.gelu(x)
        if self.pad:
            x = x[..., :H, :W]
        return self.head(x)


# --------------------------------------------------------------------------- #
# FNO+ (assumed definition, see module docstring)
# --------------------------------------------------------------------------- #
class FNOPlusBlock(nn.Module):
    def __init__(self, ch, modes, kernel, norm, mlp_ratio):
        super().__init__()
        self.n1, self.n2 = make_norm(norm, ch), make_norm(norm, ch)
        self.spec = SpectralConv2d(ch, ch, modes, modes)
        self.local = nn.Conv2d(ch, ch, kernel, padding=kernel // 2)   # local bypass for sharp fronts
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch * mlp_ratio, 1), nn.GELU(), nn.Conv2d(ch * mlp_ratio, ch, 1))

    def forward(self, x):
        y = self.n1(x)
        x = x + F.gelu(self.spec(y) + self.local(y))
        return x + self.mlp(self.n2(x))

 
class PlainFNOPlus(_AutoregressiveForecaster):
    """FNO+: pre-norm residual Fourier blocks with a local conv bypass and channel MLP."""

    def __init__(self, n_static, hidden=32, kernel=3, norm="group", head_width=32, use_future_rain=True,
                 residual=True, rain_zero_z=0.0, residual_base="persistence", velocity_clip=3.0,
                 t_in=4, modes=12, n_layers=4, pad=8, use_grid=True, mlp_ratio=2):
        super().__init__(n_static, t_in, use_future_rain, residual, rain_zero_z, residual_base, velocity_clip)
        self.pad, self.use_grid = pad, use_grid
        self.lift = nn.Sequential(nn.Conv2d(self.in_ch + (2 if use_grid else 0), hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 1))
        self.blocks = nn.ModuleList([FNOPlusBlock(hidden, modes, kernel, norm, mlp_ratio) for _ in range(n_layers)])
        self.out_norm = make_norm(norm, hidden)
        self.head = OutputHead(hidden, head_width, kernel)

    def _predict(self, x):
        B, _, H, W = x.shape
        if self.use_grid:
            x = torch.cat([x, _coord_grid(B, H, W, x.device, x.dtype)], 1)
        x = self.lift(x)
        if self.pad:
            x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        for blk in self.blocks:
            x = blk(x)
        if self.pad:
            x = x[..., :H, :W]
        return self.head(self.out_norm(x))


MODELS = {"unet": PlainUNet, "fno": PlainFNO, "fno_plus": PlainFNOPlus}


def build_model(name, n_static, **kw):
    return MODELS[name.lower()](n_static, **kw)


if __name__ == "__main__":
    B, Tin, K, H, W, S = 2, 4, 3, 50, 70, 3   # deliberately not divisible by 2**levels
    args = (torch.randn(B, Tin, 1, H, W), torch.randn(B, Tin, 1, H, W),
            torch.randn(B, K, 1, H, W), torch.randn(B, S, H, W))
    for name in MODELS:
        m = build_model(name, S)
        y = m(*args, teacher=torch.randn(B, K, 1, H, W), tf_prob=0.5)
        assert y.shape == (B, K, 1, H, W), y.shape
        assert torch.allclose(y[:, 0], args[0][:, -1]), "zero-init head => persistence at start"
        y.mean().backward()
        print(f"{name:9s} ok  params={sum(p.numel() for p in m.parameters()):,}")