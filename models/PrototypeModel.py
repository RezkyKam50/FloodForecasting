import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from .ForecasterBlock import AutoregressiveForecaster
 
class CellularAutomata:
    """
    Physics-prior Cellular Automata module for flood expansion.
    Faithful implementation of Zhou et al., CA-DSUNet, IEEE JSTARS,
    Section III.B.1, Eqs. (3)-(8), Fig. 2.
    """

    def __init__(
        self,
        dem: np.ndarray,
        slope: np.ndarray,
        n_iter: int = 10,
        d_vir: float = 1.0,
        kappa: float = 0.5,
        delta_h: float = 1.0,
        cell_size: float = 30.0,
        slope_eps: float = 1e-3,
    ):
        self.dem = dem.astype(np.float64)
        self.slope = slope.astype(np.float64)
        self.n_iter = n_iter
        self.d_vir = d_vir
        self.kappa = kappa
        self.delta_h = delta_h
        self.cell_size = cell_size
        self.slope_eps = slope_eps

        assert self.dem.shape == self.slope.shape, \
            "DEM and Slope must share the same grid shape."

        self.neighbors = [(-1, -1), (-1, 0), (-1, 1),
                          ( 0, -1),          ( 0, 1),
                          ( 1, -1), ( 1, 0), ( 1, 1)]

    def _init_water_depth(self, water_mask: np.ndarray) -> np.ndarray:
        h = np.zeros_like(self.dem, dtype=np.float64)
        h[water_mask.astype(bool)] = self.d_vir
        return h

    def _hydraulic_head(self, h: np.ndarray) -> np.ndarray:
        return self.dem + h

    def _directional_slope_weight(self, di: int, dj: int) -> np.ndarray:
        return np.roll(np.roll(self.slope, -di, axis=0), -dj, axis=1)

    def _step(self, h: np.ndarray) -> np.ndarray:
        H = self._hydraulic_head(h)
        H_padded = np.pad(H, 1, mode="edge")

        grads = []
        weights = []
        total_weighted_grad = np.zeros_like(h)

        for di, dj in self.neighbors:
            H_nb = H_padded[1 + di: 1 + di + H.shape[0],
                            1 + dj: 1 + dj + H.shape[1]]

            g = np.maximum(H - H_nb, 0.0)
            if self.delta_h > 0:
                g = np.where(g > self.delta_h, g, 0.0)

            w = self._directional_slope_weight(di, dj)
            w = np.maximum(w, self.slope_eps)

            grads.append(g)
            weights.append(w)
            total_weighted_grad += g * w

        safe_total = np.where(total_weighted_grad > 1e-12,
                              total_weighted_grad, 1.0)

        Q_out_list = [
            self.kappa * (g * w / safe_total) * h
            for g, w in zip(grads, weights)
        ]

        dh_out = np.zeros_like(h)
        for Q in Q_out_list:
            dh_out += Q
        scale = np.where(dh_out > h, h / np.maximum(dh_out, 1e-12), 1.0)
        Q_out_list = [Q * scale for Q in Q_out_list]
        dh_out *= scale

        influx = np.zeros_like(h)
        for (di, dj), Q in zip(self.neighbors, Q_out_list):
            influx += np.roll(np.roll(Q, di, axis=0), dj, axis=1)

        h_new = h + influx - dh_out
        h_new = np.clip(h_new, 0.0, None)
        return h_new

    def run(self, water_mask: np.ndarray) -> np.ndarray:
        prior_mask, _ = self.run_with_depth(water_mask)
        return prior_mask

    def run_with_depth(self, water_mask: np.ndarray):
        h = self._init_water_depth(water_mask)
        for _ in range(self.n_iter):
            h = self._step(h)
        prior_mask = (h > 1e-6) | water_mask.astype(bool)
        return prior_mask, h

 
def make_norm(kind, ch):
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, ch) or 1, ch)
    return nn.Identity()


def _coord_grid(b, h, w, device, dtype):
    gy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    gx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([gy, gx], 1)

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.out_ch, self.modes1, self.modes2 = out_ch, modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
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


class FNOPlusBlock(nn.Module):
    def __init__(self, ch, modes, kernel, norm, mlp_ratio):
        super().__init__()
        self.n1, self.n2 = make_norm(norm, ch), make_norm(norm, ch)
        self.spec = SpectralConv2d(ch, ch, modes, modes)
        self.local = nn.Conv2d(ch, ch, kernel, padding=kernel // 2)
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch * mlp_ratio, 1), nn.GELU(),
                                 nn.Conv2d(ch * mlp_ratio, ch, 1))

    def forward(self, x):
        y = self.n1(x)
        x = x + F.gelu(self.spec(y) + self.local(y))
        return x + self.mlp(self.n2(x))
 
class CA_FNOPlus(nn.Module):
    def __init__(self, n_static, hidden=32, kernel=3, norm="group", head_width=32,
                 use_future_rain=True, residual=True, rain_zero_z=0.0,
                 residual_base="persistence", velocity_clip=3.0,
                 t_in=4, modes=12, n_layers=4, pad=8, use_grid=True, mlp_ratio=2,
                 # ---- CA prior knobs ---- #
                 use_ca_prior=True,
                 n_prior=1,                     # 1 = CA depth field `h`
                 dem_mean=0.0, dem_std=1.0,     # to un-normalise `dem` if needed
                 ca_n_iter=30,
                 ca_d_vir=1.0,
                 ca_kappa=0.5,
                 ca_delta_h=1.0,
                 ca_cell_size=30.0,
                 ca_mask_thresh=0.0,            # on normalised depth_hist
                 ca_prior_norm=True,            # z-normalise prior with running stats
                 ca_prior_ema=0.99,
                 ca_slope_eps=1e-6):
        super().__init__()

        self.use_ca_prior = bool(use_ca_prior)
        self.n_prior = int(n_prior) if self.use_ca_prior else 0

        # CA hyperparameters
        self.ca_n_iter = int(ca_n_iter)
        self.ca_d_vir = float(ca_d_vir)
        self.ca_kappa = float(ca_kappa)
        self.ca_delta_h = float(ca_delta_h)
        self.ca_cell_size = float(ca_cell_size)
        self.ca_mask_thresh = float(ca_mask_thresh)
        self.ca_prior_norm = bool(ca_prior_norm)
        self.ca_prior_ema = float(ca_prior_ema)
        self.ca_slope_eps = float(ca_slope_eps)

        # DEM un-normalisation (identity if not provided)
        self.register_buffer("dem_mean", torch.tensor(float(dem_mean)))
        self.register_buffer("dem_std",  torch.tensor(float(dem_std)))

        # Running stats for normalising the CA prior
        if self.ca_prior_norm and self.n_prior > 0:
            self.register_buffer("ca_prior_mean", torch.zeros(self.n_prior))
            self.register_buffer("ca_prior_var",  torch.ones(self.n_prior))
            self.register_buffer("ca_prior_seen", torch.tensor(0.0))
        else:
            self.register_buffer("ca_prior_mean", torch.zeros(0))
            self.register_buffer("ca_prior_var",  torch.ones(0))
            self.register_buffer("ca_prior_seen", torch.tensor(0.0))

        # Rollout mechanics
        self.forecaster = AutoregressiveForecaster(
            n_static=n_static, t_in=t_in, use_future_rain=use_future_rain,
            residual=residual, rain_zero_z=rain_zero_z,
            residual_base=residual_base, velocity_clip=velocity_clip,
            n_prior=self.n_prior,
        )

        self.pad, self.use_grid = pad, use_grid
        in_ch = self.forecaster.in_ch + (2 if use_grid else 0)

        self.lift = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
        )
        self.blocks = nn.ModuleList([
            FNOPlusBlock(hidden, modes, kernel, norm, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.out_norm = make_norm(norm, hidden)
        self.head = OutputHead(hidden, head_width, kernel)
 
    @torch.no_grad()
    def _build_ca_prior(self, depth_hist, dem):
        """
        depth_hist : (B, T_in, 1, H, W)  normalised depth history
        dem        : (B, 1, H, W)        normalised DEM
        returns    : (B, n_prior, H, W)  normalised CA prior (depth field `h`)
        """
        B, _, _, H, W = depth_hist.shape
        device, dtype = dem.device, dem.dtype

        # 1) Water mask from the LAST observed frame (no leakage).
        last_depth_n = depth_hist[:, -1, 0]                      # (B,H,W)
        mask_np = (last_depth_n > self.ca_mask_thresh).detach().cpu().numpy()

        # 2) Un-normalise DEM to metres.
        dem_m = dem.detach().to(torch.float32) * self.dem_std + self.dem_mean
        dem_np = dem_m[:, 0].cpu().numpy()                       # (B,H,W)

        # 3) Slope from DEM (finite differences, in metres / cell).
        gy, gx = np.gradient(dem_np, axis=(-2, -1))
        slope_np = np.sqrt(gy ** 2 + gx ** 2)
        prior_np = np.zeros((B, self.n_prior, H, W), dtype=np.float32)
        for b in range(B):
            ca = CellularAutomata(
                dem=dem_np[b], slope=slope_np[b],
                n_iter=self.ca_n_iter, d_vir=self.ca_d_vir,
                kappa=self.ca_kappa, delta_h=self.ca_delta_h,
                cell_size=self.ca_cell_size, slope_eps=self.ca_slope_eps,
            )
            _, h_final = ca.run_with_depth(mask_np[b])
            if self.n_prior == 1:
                prior_np[b, 0] = h_final
            else:
                # Fallback for n_prior>1: [binary envelope, depth field, ...]
                prior_np[b, 0] = h_final
                prior_np[b, 1] = (h_final > 1e-6).astype(np.float32)
                # remaining channels (if any) left at 0

        prior = torch.from_numpy(prior_np).to(device=device, dtype=dtype)

        # 5) Normalise with running stats (train mode updates them).
        if self.ca_prior_norm and self.n_prior > 0:
            flat = prior.reshape(B, self.n_prior, -1)
            batch_mean = flat.mean(dim=(0, 2))
            batch_var  = flat.var(dim=(0, 2), unbiased=False)

            if self.training:
                with torch.no_grad():
                    ema = self.ca_prior_ema
                    if self.ca_prior_seen.item() < 1.0:
                        self.ca_prior_mean.copy_(batch_mean)
                        self.ca_prior_var.copy_(batch_var)
                        self.ca_prior_seen.fill_(1.0)
                    else:
                        self.ca_prior_mean.mul_(ema).add_(batch_mean, alpha=1 - ema)
                        self.ca_prior_var.mul_(ema).add_(batch_var, alpha=1 - ema)

            mean = self.ca_prior_mean.view(1, -1, 1, 1)
            std  = (self.ca_prior_var.clamp_min(1e-8)).sqrt().view(1, -1, 1, 1)
            prior = (prior - mean) / std

        return prior
 
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
 
    def forward(self, depth_hist, rain_hist, rain_fut, static, dem,
                out_steps=None, teacher=None, tf_prob=0.0):
        prior = None
        if self.use_ca_prior and self.n_prior > 0:
            prior = self._build_ca_prior(depth_hist, dem)

        return self.forecaster.rollout(
            predict_fn=self._predict,
            depth_hist=depth_hist,
            rain_hist=rain_hist,
            rain_fut=rain_fut,
            static=static,
            out_steps=out_steps,
            teacher=teacher,
            tf_prob=tf_prob,
            prior=prior,
        )