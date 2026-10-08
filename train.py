#!/usr/bin/env python3
"""
FloodCastBench - train and compare ALL flood-depth forecasting models in one execution (local script).

Models (names accepted by --models): UNet, FNO, FNO+, ConvLSTM, Hybrid

Run modes
---------
  check  : build caches, run data sanity checks, evaluate reference baselines, smoke-test every model; trains nothing
  single : train every selected model on one (FOLD, SEED) and write the comparison tables + plots
  sweep  : train every model on every (fold, seed) combination and report bootstrap CIs + comparison plots

Examples
--------
  python floodcast_compare.py --mode check
  python floodcast_compare.py --mode single --fold 0 --seed 0
  python floodcast_compare.py --mode single --models UNet,ConvLSTM,Hybrid
  python floodcast_compare.py --mode sweep --folds 0,1 --seeds 0,1,2 --max-hours 6

All models share the same data, split, normalisation, loss, optimiser schedule, curriculum and seed, so the
only thing that differs between rows of the comparison is the architecture.

Every tunable value lives in the CONFIGURATION block below. A few of them can also be
overridden on the command line (see --help).

Requires: numpy, scipy, torch, matplotlib, rasterio   (pip install rasterio)
"""

import argparse
import gc
import hashlib
import inspect
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, asdict, replace

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import Dataset, DataLoader

from models.DynamicConvLSTM import HybridFloodNet
from models.ConvLSTM import PlainConvLSTM
from models.Baselines import *   # PlainUNet, PlainFNO, PlainFNOPlus

# GDAL tuning must be set before rasterio is imported.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:
    sys.exit("rasterio is required to read the GeoTIFFs:  pip install rasterio")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# =====================================================================================
#                                    CONFIGURATION
# =====================================================================================

# ---- run settings (CLI flags override these) -----------------------------------------
RUN_MODE    = "single"            # "check" | "single" | "sweep"
REGION      = "Mozambique"        # Mozambique | Pakistan | Australia30 | Australia60 | UK30 | UK60
FOLD        = 0                   # fold used by RUN_MODE="single"
SEED        = 0                   # seed used by RUN_MODE="single"
SWEEP_FOLDS = (0, 1, 2, 3)        # folds used by RUN_MODE="sweep" (out-of-range folds are dropped)
SWEEP_SEEDS = (0, 1, 2, 3, 4)
MAX_HOURS   = 0.0                 # wall-clock budget for the whole session; 0 = unlimited
SHOW_PLOTS  = False               # True: open matplotlib windows; False: only save PNGs
SAVE_PLOTS  = True

# ---- models --------------------------------------------------------------------------
# display name -> class name (resolved from the imports above; a missing class is skipped with a message)
MODEL_CLASSES = {"UNet": "PlainUNet", "FNO": "PlainFNO", "FNO+": "PlainFNOPlus",
                 "ConvLSTM": "PlainConvLSTM", "Hybrid": "HybridFloodNet"}
MODELS = tuple(MODEL_CLASSES)     # models trained by single / sweep (CLI: --models UNet,FNO,...)

# ---- experiment identity -------------------------------------------------------------
EXPERIMENT_TAG         = "ALL"
EXPERIMENT_DESCRIPTION = "all architectures under one identical protocol"
EXPERIMENT_OVERRIDES   = {}       # any Config field, e.g. {"hidden": 64, "static_set": "minimal"}

# ---- paths ---------------------------------------------------------------------------
DATA_ROOT   = "./dataset/FloodCastBench/"                # path to FloodCastBench/; None = search upward from the cwd
CACHE_DIR   = None                # None = <folder containing FloodCastBench>/cache
RESULTS_DIR = None                # None = <folder containing FloodCastBench>/results_final
DATASET_DIRNAME = "FloodCastBench"
DATASET_SUBDIRS = ("Relevant data", "Low-fidelity flood forecasting", "High-fidelity flood forecasting")

# ---- model / training hyper-parameters (defaults of Config) --------------------------
@dataclass
class Config:
    # identity / paths (filled in at start-up)
    tag: str = EXPERIMENT_TAG         # per-model runs use the model name as tag
    model: str = "ConvLSTM"           # key of MODEL_CLASSES
    data_root: str = ""
    cache_dir: str = ""
    save_dir: str = ""
    region: str = REGION

    # forecasting task
    step_frames: int = 6              # frames per model step (one frame = DT seconds)
    seq_in: int = 12                  # history steps
    seq_out: int = 12                 # forecast steps (lead time)
    use_future_rain: bool = True
    target_transform: str = "log1p"   # "log1p" | anything else = raw depth

    # train / validation / test split
    split_mode: str = "rolling"       # "rolling" (rolling-origin CV) | "chronological"
    fold: int = 0
    n_folds: int = 4
    test_windows: int = 60
    val_windows: int = 30
    test_stride: int = 0              # thin val/test windows to every n-th start
    split_frac: tuple = (0.70, 0.85)  # chronological mode only

    # static channels
    static_set: str = "full"          # "full" | "minimal" | "none"
    misregister_static: bool = False  # ablation: deliberately use the buggy mis-registered statics

    # architecture
    hidden: int = 32
    kernel: int = 3
    norm: str = "none"                # "none" | "batch" | "group"
    head_width: int = 32
    residual_head: bool = True
    residual_base: str = "persistence"  # "persistence" | "velocity"
    velocity_clip: float = 3.0

    # optimisation
    patch: int = 64                   # random crop size during training
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 30
    warmup_frac: float = 0.05         # LR warm-up fraction of total steps
    grad_clip: float = 1.0
    wet_weight: float = 4.0           # extra loss weight on wet cells
    amp: bool = True
    seed: int = 0
    num_workers: int = 0

    # rollout curriculum / teacher forcing
    curriculum: tuple = (2, 4, 8, 12)
    curriculum_every: int = 8
    tf_anneal_epochs: int = 25

    # model selection
    selection_metric: str = "SS_vs_persistence"


PATH_FIELDS = ("data_root", "cache_dir", "save_dir")
PIPELINE_VERSION = 3              # bump to invalidate cached sweep results (3: multi-model, `model` in the key)

# ---- split presets: region -> (independent test windows, independent val windows, n_folds)
USE_SPLIT_PRESETS = True
SPLIT_PRESETS = {
    "Pakistan":    (8, 2, 4),
    "Australia30": (5, 2, 4),
    "Australia60": (5, 2, 4),
    "Mozambique":  (2, 2, 4),
    "UK30":        (1, 1, 2),
    "UK60":        (1, 1, 2),
}
ROLLING_TRAIN_START_FRAC = 0.35   # earliest fraction of the series a rolling fold may start validating

# ---- dataset geometry per region -----------------------------------------------------
GRID_SPEC = {
    "Mozambique":  dict(crop=(150, 50, 50, 50), factor=16, shape=(151, 138),
                        flood=("Low-fidelity flood forecasting", "480m"), dem="Mozambique_DEM.tif",
                        lulc="Mozambique.tif", rain="Mozambique flood", cell_m=485.0),
    "Australia60": dict(crop=(0, 0, 0, 0), factor=2, shape=(536, 536),
                        flood=("High-fidelity flood forecasting", "60m"), dem="Australia_DEM.tif",
                        lulc="Australia.tif", rain="Australia flood", region_dir="Australia", cell_m=60.0),
    "Australia30": dict(crop=(0, 0, 0, 0), factor=1, shape=(1073, 1073),
                        flood=("High-fidelity flood forecasting", "30m"), dem="Australia_DEM.tif",
                        lulc="Australia.tif", rain="Australia flood", region_dir="Australia", cell_m=30.0),
    "UK60":        dict(crop=(5, 5, 5, 5), factor=2, shape=(85, 137),
                        flood=("High-fidelity flood forecasting", "60m"), dem="UK_DEM.tif",
                        lulc="UK.tif", rain="UK flood", region_dir="UK", cell_m=60.0),
    "UK30":        dict(crop=(5, 5, 5, 5), factor=1, shape=(170, 275),
                        flood=("High-fidelity flood forecasting", "30m"), dem="UK_DEM.tif",
                        lulc="UK.tif", rain="UK flood", region_dir="UK", cell_m=30.0),
    "Pakistan":    dict(crop=(40, 35, 182, 60), factor=16, shape=(810, 441),
                        flood=("Low-fidelity flood forecasting", "480m"), dem="Pakistan_DEM.tif",
                        lulc="Pakistan.tif", rain="Pakistan flood", provisional=True, cell_m=485.0),
}

# ---- static terrain / land-cover features --------------------------------------------
STATIC_FULL = ("elevation", "slope", "aspect_sin", "aspect_cos", "curvature",
               "relief_1km", "relief_5km", "sink_depth", "manning", "water_frac", "built_frac")
STATIC_MINIMAL = ("elevation", "manning")
SOURCE_PIXEL_M = 30.0             # DEM / land-cover resolution
LULC_WATER_CLASS = 1
LULC_BUILT_CLASS = 7
DEFAULT_MANNING = 0.0375          # for land-cover classes missing from the table
MANNING_LUT = {0: 0.0375, 1: 0.0350, 2: 0.1200, 4: 0.0800, 5: 0.0350,
               7: 0.3750, 8: 0.0265, 9: 0.0375, 10: 0.0375, 11: 0.0375}
NODATA = -2000.0                  # DEM values below this are treated as missing
NODATA_FILL_OFFSET = 30.0         # missing DEM cells = max valid elevation + this
RELIEF_WINDOWS_M = (1000, 5000)   # relief_1km / relief_5km / sink_depth window sizes

# ---- physical thresholds and units ---------------------------------------------------
RAIN_INTERVAL_S = 1800            # rainfall rasters are half-hourly
WET_THRESHOLD = 0.01              # [m] a cell is "wet" above this depth
CHANGE_THRESHOLD = 0.01           # [m] |last obs - target| above this counts as "changing"
GAMMAS = (0.001, 0.01, 0.05)      # depth thresholds for CSI / IoU / F1
WET_LOSS_SHARPNESS = 4.0          # slope of the sigmoid wet mask inside the loss

# ---- evaluation / reporting ----------------------------------------------------------
EVAL_BATCH = 2                    # batch size when evaluating the model
REF_BATCH = 4                     # batch size when evaluating baselines
BOOTSTRAP_RESAMPLES = 4000        # report() hierarchical bootstrap
BOOTSTRAP_SEED = 0
FOLD_OVERLAP_WARN = 0.25          # warn when fold test blocks overlap more than this
PLOT_DPI = 130
STREAM_CHUNK = 64                 # frames per chunk when streaming statistics from the memmap
CASE_PLOT_WINDOWS = ("first", "middle")   # test windows plotted in the side-by-side case figures

# ---- sanity checks (run on every start) ----------------------------------------------
MIN_MASS_BALANCE_RATIO = 0.5
MAX_MANNING = 0.4

# ---- FloodCastBench paper-protocol reference ----------------------------------------
PAPER_BLOCK_FRAMES = 20           # blocks of 20 frames, t=1 -> t=2..20
PAPER_CHUNK_VALUES = 20_000_000   # memory cap for the paper-protocol computation
PAPER_ROWS = {   # (relL2, NSE, r, CSI@0.001, CSI@0.01)
    "Pakistan":   ("Table 4, in-domain",
                   [("FNO+ ", (0.002107, 0.999994, 0.999997, 0.976716, 0.993993)),
                    ("FNO  ", (0.002232, 0.999993, 0.999997, 0.966631, 0.992669)),
                    ("U-Net", (0.079399, 0.991705, 0.995934, 0.815218, 0.809986))]),
    "Mozambique": ("Table 5, zero-shot from Pakistan",
                   [("FNO+ ", (0.078633, 0.955450, 0.985521, 0.934028, 0.912712)),
                    ("FNO  ", (0.163300, 0.453167, 0.980383, 0.892473, 0.915509)),
                    ("U-Net", (1.602850, -72.350318, 0.903450, 0.724442, 0.667776))]),
}

# =====================================================================================
#                                  END OF CONFIGURATION
# =====================================================================================


RUN_MODES = ("check", "single", "sweep")
SESSION_T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(cfg):
    return torch.autocast(device_type=DEVICE.type, enabled=cfg.amp and DEVICE.type == "cuda")


def time_budget_spent() -> bool:
    return MAX_HOURS > 0 and (time.time() - SESSION_T0) / 3600.0 > MAX_HOURS


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_tag(tag: str) -> str:
    """File-name-safe version of a model tag ('FNO+' -> 'FNOplus')."""
    return tag.replace("+", "plus").replace("/", "_").replace(" ", "_")


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def finish_figure(path=None):
    """Save (if enabled) and show (if enabled) the current figure, then close it."""
    if path and SAVE_PLOTS:
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def find_data_root() -> str:
    def valid(path):
        return all(os.path.isdir(os.path.join(path, d)) for d in DATASET_SUBDIRS)

    if DATA_ROOT:
        if valid(DATA_ROOT):
            return os.path.abspath(DATA_ROOT)
        raise FileNotFoundError(f"DATA_ROOT={DATA_ROOT!r} must contain {DATASET_SUBDIRS}")

    candidates, here = [], os.path.abspath(os.getcwd())
    while True:
        candidates.append(os.path.join(here, DATASET_DIRNAME))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    for candidate in candidates:
        if valid(candidate):
            return candidate
    raise FileNotFoundError(
        f"{DATASET_DIRNAME}/ not found (it must contain {DATASET_SUBDIRS}). Searched:\n  "
        + "\n  ".join(candidates[:8]) + "\nSet DATA_ROOT at the top of the script or pass --data-root.")


# -------------------------------------------------------------------------------------
# Config construction
# -------------------------------------------------------------------------------------
def split_preset(region: str, cfg: Config) -> dict:
    if not USE_SPLIT_PRESETS or region not in SPLIT_PRESETS:
        return {}
    n_test, n_val, n_folds = SPLIT_PRESETS[region]
    tgt = cfg.seq_out * cfg.step_frames
    return dict(test_windows=(n_test - 1) * tgt + 1, val_windows=(n_val - 1) * tgt + 1,
                n_folds=n_folds, test_stride=tgt)


def make_base_config(region, data_root, cache_dir, save_dir) -> Config:
    base = Config(region=region, data_root=data_root, cache_dir=cache_dir, save_dir=save_dir)
    return replace(base, **split_preset(region, base))


def config_for(base: Config, **overrides) -> Config:
    return replace(base, **{**EXPERIMENT_OVERRIDES, **overrides})


def model_config(cfg: Config, name: str) -> Config:
    """Same config, different architecture (the tag follows the model so outputs never collide)."""
    return replace(cfg, model=name, tag=name)


def experiment_key(cfg: Config) -> dict:
    key = json.loads(json.dumps(asdict(cfg), default=str))
    for field in PATH_FIELDS:
        key.pop(field, None)
    key["pipeline"] = PIPELINE_VERSION
    return key


def resolve_models(names) -> tuple:
    """Case-insensitive lookup of model names; raises on an unknown one."""
    lookup = {k.lower(): k for k in MODEL_CLASSES}
    out = []
    for n in names:
        key = lookup.get(str(n).strip().lower())
        if key is None:
            raise ValueError(f"unknown model {n!r}; choose from {tuple(MODEL_CLASSES)}")
        if key not in out:
            out.append(key)
    return tuple(out)


# =====================================================================================
# 1. DATA: terrain features, rainfall, depth cache
# =====================================================================================
def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1)


def resample_grid(a, size):
    """Linearly resample a 2-D array to (h // size, w // size)."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    if size == 1:
        return a.copy()
    nh, nw = h // size, w // size
    f = RegularGridInterpolator((np.arange(h, dtype=float), np.arange(w, dtype=float)), a,
                                method="linear", bounds_error=False, fill_value=None)
    Y, X = np.meshgrid(np.linspace(0, h - 1, nh), np.linspace(0, w - 1, nw), indexing="ij")
    return f(np.stack([Y, X], axis=-1))


def crop_native(a, spec):
    top, bottom, left, right = spec["crop"]
    return a[top: a.shape[0] - bottom if bottom else None, left: a.shape[1] - right if right else None]


def fill_missing_dem(a):
    a = np.asarray(a, dtype=np.float64).copy()
    missing = ~np.isfinite(a) | (a < NODATA)
    if missing.any():
        a[missing] = NODATA_FILL_OFFSET + a[~missing].max()
    return a


def bilinear_taps(n, size):
    x = np.linspace(0, n - 1, n // size)
    i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
    return i0, x - i0


def resample_rain(a, size):
    """Bilinear resampling of a rainfall raster (negative values clipped to 0)."""
    if size == 1:
        g = np.asarray(a, dtype=np.float64).copy()
        g[g < 0] = 0.0
        return g
    (r0, fy), (c0, fx) = bilinear_taps(a.shape[0], size), bilinear_taps(a.shape[1], size)
    nh, nw = len(r0), len(c0)
    g = a[np.ix_(np.concatenate([r0, r0 + 1]), np.concatenate([c0, c0 + 1]))].astype(np.float64)
    g[g < 0] = 0.0
    fy, fx = fy[:, None], fx[None, :]
    return (g[:nh, :nw] * (1 - fy) * (1 - fx) + g[:nh, nw:] * (1 - fy) * fx
            + g[nh:, :nw] * fy * (1 - fx) + g[nh:, nw:] * fy * fx)


class FloodData:
    """Depth (memory-mapped), rainfall and static rasters for one region, cached on disk."""

    def __init__(self, cfg: Config):
        self.region = cfg.region
        self.spec = GRID_SPEC[cfg.region]
        self.H, self.W = self.spec["shape"]
        spec = self.spec

        self.rel_dir = os.path.join(cfg.data_root, "Relevant data")
        self.flood_dir = os.path.join(cfg.data_root, spec["flood"][0], spec["flood"][1],
                                      spec.get("region_dir", cfg.region))

        tag = f"{cfg.region}_" + hashlib.md5(json.dumps(
            [cfg.region, list(spec["crop"]), spec["factor"], list(STATIC_FULL)]).encode()).hexdigest()[:10]
        self.p_depth = os.path.join(cfg.cache_dir, f"depth_{tag}.npy")
        self.p_rain = os.path.join(cfg.cache_dir, f"rain_{tag}.npy")
        self.p_static = os.path.join(cfg.cache_dir, f"static_{tag}.npz")
        self.p_manifest = os.path.join(cfg.cache_dir, f"manifest_{tag}.json")

        if spec.get("provisional"):
            print("NOTE: this region's crop was fitted here, not read off FloodCastBench's code. "
                  "Pakistan's crop is pinned to well inside one 485 m cell.")
        print("target grid:", spec["shape"], "| effective cell size ~", spec["cell_m"], "m")

        self._load_or_build()
        self.depth = np.load(self.p_depth, mmap_mode="r")
        self.nt = self.depth.shape[0]
        n_static = len([k for k in self.static if not k.startswith("bug_")])
        print(f"depth {self.depth.shape} dt={self.dt}s span={self.nt * self.dt / 86400:.2f} d | "
              f"rain {self.rain.shape} | static {n_static} channels")

    # ---- cache handling ----
    def _load_or_build(self):
        paths = (self.p_depth, self.p_rain, self.p_static, self.p_manifest)
        if all(os.path.exists(p) for p in paths):
            self.manifest = load_json(self.p_manifest)
            self.dt = self.manifest["dt_seconds"]
            self.rain = np.load(self.p_rain)
            self.static = dict(np.load(self.p_static))
            print("loaded caches from", os.path.dirname(self.p_depth))
            return

        t0 = time.time()
        n_frames, self.dt = self._build_depth()
        print(f"  depth  ({n_frames}, {self.H}, {self.W})  {time.time() - t0:.0f}s")
        self.rain, rain_files = self._build_rain()
        print(f"  rain   {self.rain.shape}  {time.time() - t0:.0f}s")
        self.static = self._build_static()
        print(f"  static {len(self.static)} arrays  {time.time() - t0:.0f}s")
        np.save(self.p_rain, self.rain)
        np.savez(self.p_static, **self.static)
        self.manifest = dict(region=self.region, crop=list(self.spec["crop"]), factor=self.spec["factor"],
                             shape=[self.H, self.W], n_frames=int(n_frames), dt_seconds=int(self.dt),
                             n_rain=int(self.rain.shape[0]), rain_first=rain_files[0], rain_last=rain_files[-1],
                             static_channels=[k for k in self.static if not k.startswith("bug_")],
                             built=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_json(self.p_manifest, self.manifest)

    def _build_depth(self):
        times = sorted(int(f[:-4]) for f in os.listdir(self.flood_dir) if f.endswith(".tif"))
        dt = times[1] - times[0]
        assert all(t == i * dt for i, t in enumerate(times)), "non-uniform time axis"
        out = np.lib.format.open_memmap(self.p_depth, mode="w+", dtype=np.float32,
                                        shape=(len(times), self.H, self.W))
        every, t0 = max(1, len(times) // 10), time.time()
        for i, t in enumerate(times):
            a = read_tif(os.path.join(self.flood_dir, f"{t}.tif"))
            assert a.shape == (self.H, self.W), f"frame {t}: {a.shape} != {(self.H, self.W)}"
            out[i] = np.maximum(np.nan_to_num(a, nan=0.0), 0.0)
            if (i + 1) % every == 0:
                print(f"    depth {i + 1}/{len(times)}  {time.time() - t0:.0f}s", flush=True)
        out.flush()
        n_frames = int(out.shape[0])
        del out
        return n_frames, dt

    def _build_rain(self):
        rdir = os.path.join(self.rel_dir, "Rainfall", self.spec["rain"])
        files = sorted(f for f in os.listdir(rdir) if f.endswith(".tif"))
        out = np.empty((len(files), self.H, self.W), np.float32)
        every, t0 = max(1, len(files) // 10), time.time()
        for i, f in enumerate(files):
            raw = crop_native(read_tif(os.path.join(rdir, f)), self.spec)
            out[i] = resample_rain(raw, self.spec["factor"]).astype(np.float32)
            if (i + 1) % every == 0:
                print(f"    rain {i + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
        return out, files

    def _build_static(self):
        spec, H, W = self.spec, self.H, self.W
        dem_raw = read_tif(os.path.join(self.rel_dir, "DEM", spec["dem"]))
        lulc_raw = read_tif(os.path.join(self.rel_dir, "Land use and land cover", spec["lulc"]))
        dem = crop_native(fill_missing_dem(dem_raw), spec)
        lulc = crop_native(lulc_raw, spec).astype(np.uint8)

        px = SOURCE_PIXEL_M
        win_1km = int(round(RELIEF_WINDOWS_M[0] / px)) | 1
        win_5km = int(round(RELIEF_WINDOWS_M[1] / px)) | 1
        manning_lut = np.full(256, DEFAULT_MANNING)
        for k, v in MANNING_LUT.items():
            manning_lut[k] = v
        down = lambda a: resample_grid(a, spec["factor"])

        gy, gx = np.gradient(dem, px, px)
        ch = {"elevation": down(dem), "slope": down(np.sqrt(gx ** 2 + gy ** 2))}
        aspect = np.arctan2(-gx, gy)
        del gx, gy
        ch["aspect_sin"], ch["aspect_cos"] = down(np.sin(aspect)), down(np.cos(aspect))
        del aspect
        ch["curvature"] = down(ndimage.laplace(dem) / (px ** 2))
        ch["relief_1km"] = down(dem - ndimage.uniform_filter(dem, win_1km))
        ch["relief_5km"] = down(dem - ndimage.uniform_filter(dem, win_5km))
        ch["sink_depth"] = down(dem - ndimage.minimum_filter(dem, win_1km))
        ch["manning"] = down(manning_lut[lulc])
        ch["water_frac"] = down((lulc == LULC_WATER_CLASS).astype(np.float64))
        ch["built_frac"] = down((lulc == LULC_BUILT_CLASS).astype(np.float64))
        for k, v in ch.items():
            assert v.shape == (H, W), f"static '{k}' is {v.shape}, expected {(H, W)}"
        out = {k: v.astype(np.float32) for k, v in ch.items()}

        # Deliberately mis-registered channels, only used by the `misregister_static` ablation.
        out["bug_elevation"] = np.maximum(dem_raw[:H, :W].astype(np.float32), 0.0)
        out["bug_manning"] = lulc_raw[:H, :W].astype(np.float32)
        return out

    # ---- access helpers ----
    def rain_index(self, frame_idx: int) -> int:
        return min(int(frame_idx * self.dt // RAIN_INTERVAL_S), self.rain.shape[0] - 1)

    def rain_for_step(self, frame_idx: int, step: int, rows=slice(None), cols=slice(None)) -> np.ndarray:
        """Mean rainfall over the `step` frames that end at `frame_idx`."""
        g = np.arange(frame_idx - step, frame_idx)
        g = g[g >= 0]
        if len(g) == 0:
            return np.zeros(self.rain[0, rows, cols].shape, np.float32)
        u, cnt = np.unique([self.rain_index(int(x)) for x in g], return_counts=True)
        return (np.tensordot(cnt.astype(np.float32), self.rain[u, rows, cols], axes=1) / step).astype(np.float32)

    def frame_stats(self, lo: int = 0, hi: int = None, chunk: int = STREAM_CHUNK):
        """Per-frame domain-mean depth and wet fraction."""
        hi = self.nt if hi is None else hi
        mean, wet = np.empty(hi - lo), np.empty(hi - lo)
        for a in range(lo, hi, chunk):
            x = np.asarray(self.depth[a:min(a + chunk, hi)], dtype=np.float32)
            mean[a - lo:a - lo + len(x)] = x.mean(axis=(1, 2))
            wet[a - lo:a - lo + len(x)] = (x > WET_THRESHOLD).mean(axis=(1, 2))
        return mean, wet


def run_data_sanity_checks(cfg: Config, data: FloodData, plot_path_prefix=None):
    """Verify grid registration, units and rainfall timing; plot the inputs."""
    rain_mm = float(data.rain.mean(axis=(1, 2)).sum() * RAIN_INTERVAL_S / 3600.0)
    depth_mm = float(np.asarray(data.depth[-1]).mean() * 1000)
    ratio = depth_mm / max(rain_mm, 1e-9)
    print(f"[1] mass balance : rained {rain_mm:7.1f} mm | stored {depth_mm:7.1f} mm | ratio {ratio:.2f}"
          f"   (~1.0 where the only water source is rain and the only sink is free outflow: Mozambique, UK,\n"
          f"                    Australia. Pakistan has a prescribed Indus INFLOW boundary, so >1 is expected there.)")

    d_last = np.log1p(np.asarray(data.depth[-1]).ravel())
    r_ok = float(np.corrcoef(data.static["elevation"].ravel(), d_last)[0, 1])
    r_bug = float(np.corrcoef(data.static["bug_elevation"].ravel(), d_last)[0, 1])
    print(f"[2] registration : corr(elevation, log1p depth) = {r_ok:+.4f}   (must be NEGATIVE)")
    print(f"    the v1 misregistered crop gives             = {r_bug:+.4f}   (the bug ablation measures this)")

    manning = data.static["manning"]
    print(f"[3] manning      : min {manning.min():.4f} max {manning.max():.4f} mean {manning.mean():.4f}"
          f"  (expect 0.0265..0.3750)")

    domain_mean, wet_frac = data.frame_stats()
    s = cfg.step_frames
    frames = np.arange(s, data.nt, s)
    storage = (domain_mean[frames] - domain_mean[frames - s]) * 1000.0
    hours = s * data.dt / 3600.0
    rain_step = np.array([data.rain_for_step(int(f), s).mean() for f in frames]) * hours
    rain_old = np.array([data.rain[data.rain_index(int(f))].mean() for f in frames]) * hours
    r_rain = float(np.corrcoef(storage, rain_step)[0, 1])
    r_rain_old = float(np.corrcoef(storage, rain_old)[0, 1])
    print(f"[4] rain timing  : corr(storage change, rain during the step) = {r_rain:+.4f}"
          f"   (old pairing, interval starting at the target frame: {r_rain_old:+.4f})")

    assert ratio > MIN_MASS_BALANCE_RATIO, "mass balance failed - grid mapping or rainfall units are wrong"
    assert r_ok < 0, "registration failed - static channels are not on the flood grid"
    assert manning.max() <= MAX_MANNING, "manning channel still holds land-cover class IDs"
    assert r_rain >= r_rain_old, "rain timing: the step-aligned rainfall fits the storage change worse"
    print("\nall checks passed")

    # plots
    fig, ax = plt.subplots(3, 4, figsize=(17, 11))
    for a, k in zip(ax.ravel(), STATIC_FULL):
        im = a.imshow(data.static[k], cmap="terrain")
        a.set_title(k, fontsize=9)
        a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.046)
    for a in ax.ravel()[len(STATIC_FULL):]:
        a.axis("off")
    plt.suptitle(f"{cfg.region}: {len(STATIC_FULL)} static channels on the {data.H}x{data.W} flood grid")
    plt.tight_layout()
    finish_figure(plot_path_prefix and plot_path_prefix + "_static.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    days = np.arange(data.nt) * data.dt / 86400
    ax[0].plot(days, domain_mean)
    ax[0].set(xlabel="days", ylabel="mean depth [m]", title="domain-mean water depth")
    ax[1].plot(days, wet_frac * 100)
    ax[1].set(xlabel="days", ylabel=f"% cells > {WET_THRESHOLD * 100:g} cm", title="wet fraction")
    ax[2].plot(np.arange(data.rain.shape[0]) * RAIN_INTERVAL_S / 86400, data.rain.mean(axis=(1, 2)))
    ax[2].set(xlabel="days", ylabel="mm/h", title="domain-mean rainfall")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    finish_figure(plot_path_prefix and plot_path_prefix + "_timeseries.png")


# =====================================================================================
# 2. SPLITS, NORMALISATION AND DATASET
# =====================================================================================
def span_of(cfg: Config) -> int:
    return (cfg.seq_in + cfg.seq_out) * cfg.step_frames


def window_starts(lo, hi, span):
    return np.arange(lo, max(lo, hi - span + 1), dtype=np.int64)


def thin(starts, cfg: Config):
    return starts[::cfg.test_stride] if cfg.test_stride and cfg.test_stride > 1 else starts


def make_splits(cfg: Config, nt: int):
    span = span_of(cfg)
    if cfg.split_mode == "chronological":
        b1, b2 = int(nt * cfg.split_frac[0]), int(nt * cfg.split_frac[1])
        return {"train": window_starts(0, b1, span),
                "val": thin(window_starts(b1, b2, span), cfg),
                "test": thin(window_starts(b2, nt, span), cfg)}
    block = span + cfg.test_windows - 1
    vlen = span + cfg.val_windows - 1
    lo, hi = int(ROLLING_TRAIN_START_FRAC * nt) + vlen + span, nt - block
    assert hi > lo, "series too short for rolling-origin CV - lower test_windows / val_windows or seq_out"
    cut = int(np.linspace(lo, hi, cfg.n_folds)[cfg.fold])
    return {"train": window_starts(0, cut - vlen, span),
            "val": thin(window_starts(cut - vlen, cut, span), cfg),
            "test": thin(window_starts(cut, cut + block, span), cfg)}


def static_names(cfg: Config):
    if cfg.misregister_static:
        return ("bug_elevation", "bug_manning")
    return {"full": STATIC_FULL, "minimal": STATIC_MINIMAL, "none": ()}[cfg.static_set]


def depth_transform(x, cfg: Config):
    return np.log1p(np.maximum(x, 0.0)) if cfg.target_transform == "log1p" else np.maximum(x, 0.0)


def stream_mean_std(chunk_fn, n: int, chunk: int = STREAM_CHUNK):
    cnt = s = s2 = 0.0
    for a in range(0, n, chunk):
        x = np.asarray(chunk_fn(a, min(a + chunk, n)), dtype=np.float64)
        cnt += x.size
        s += x.sum()
        s2 += (x * x).sum()
    mean = s / cnt
    return float(mean), float(math.sqrt(max(s2 / cnt - mean * mean, 0.0)) + 1e-6)


@dataclass
class Context:
    """Everything derived from (config, data) that training and evaluation need.
    It does not depend on the architecture, so one Context serves every model on a fold."""
    cfg: Config
    data: FloodData
    splits: dict
    norm: dict
    static: np.ndarray
    names: tuple
    train_hi: int


def build_context(cfg: Config, data: FloodData) -> Context:
    splits = make_splits(cfg, data.nt)
    if len(splits["train"]) == 0:
        raise RuntimeError("the train split is empty, so there are no frames to estimate the normalisation from. "
                           "Lower seq_out / step_frames / test_windows / val_windows, or use a longer region.")
    train_hi = int(splits["train"][-1] + span_of(cfg))
    rain_hi = data.rain_index(max(train_hi - cfg.step_frames - 1, 0)) + 1
    norm = {"depth": stream_mean_std(
                lambda a, b: depth_transform(np.asarray(data.depth[a:b], dtype=np.float32), cfg), train_hi),
            "rain": stream_mean_std(lambda a, b: np.log1p(data.rain[a:b]), rain_hi)}
    names = static_names(cfg)
    if names:
        static = np.stack([(data.static[k] - data.static[k].mean()) / (data.static[k].std() + 1e-6)
                           for k in names]).astype(np.float32)
    else:
        static = np.zeros((0, data.H, data.W), np.float32)
    return Context(cfg, data, splits, norm, static, names, train_hi)


def describe_splits(ctx: Context):
    cfg, data, span = ctx.cfg, ctx.data, span_of(ctx.cfg)
    lag = cfg.seq_out * cfg.step_frames
    for name, st in ctx.splits.items():
        if len(st) == 0:
            print(f"  !! {name} split is EMPTY - lower seq_out / step_frames / test_windows, or use a longer region")
            continue
        lo, hi = int(st[0]), int(st[-1] + span)
        mean, wet = data.frame_stats(lo, hi)
        if hi - lo > lag:
            se, n, num, den = 0.0, 0.0, [], []
            for a in range(lo + lag, hi, STREAM_CHUNK):
                b = min(a + STREAM_CHUNK, hi)
                y = np.asarray(data.depth[a:b], dtype=np.float32)
                p = np.asarray(data.depth[a - lag:b - lag], dtype=np.float32)
                e2 = (y - p) ** 2
                num += np.sqrt(e2.sum(axis=(1, 2))).tolist()
                den += np.sqrt((y ** 2).sum(axis=(1, 2))).tolist()
                se += float(e2.sum(dtype=np.float64))
                n += e2.size
            rel = float(np.mean(np.array(num) / (np.array(den) + 1e-12)))
            rmse = float(np.sqrt(se / n))
        else:
            rel = rmse = float("nan")
        print(f"  {name:5s}: {len(st):5d} windows | frames [{lo}, {hi}) = days "
              f"[{lo * data.dt / 86400:.2f}, {hi * data.dt / 86400:.2f}) | mean h {mean.mean():.4f} m "
              f"| wet {100 * wet.mean():.1f}% | persistence relL2 {rel:.4f} RMSE {rmse:.4f} m")
        if name != "train":
            tgt = cfg.seq_out * cfg.step_frames
            n_indep = max(1, (hi - lo - span) // tgt + 1) if hi - lo >= span else 0
            print(f"         block {(hi - lo) * data.dt / 3600:5.1f} h | window span {span * data.dt / 3600:.1f} h |"
                  f" independent (target-disjoint) windows: {n_indep}")
            if len(st) > 2 * n_indep:
                print(f"         !! {len(st)} {name} windows carry at most {n_indep} independent forecast(s); "
                      f"set test_stride={tgt} in the config and widen the block before quoting a CI")


def describe_folds(cfg: Config, data: FloodData):
    if cfg.split_mode != "rolling":
        return
    span, blocks = span_of(cfg), []
    for f in range(cfg.n_folds):
        st = make_splits(replace(cfg, fold=f), data.nt)["test"]
        blocks.append((int(st[0]), int(st[-1]) + span) if len(st) else (0, 0))
    print("  fold test blocks (days): " + "  ".join(
        "%d:[%.2f,%.2f)" % (f, a * data.dt / 86400, b * data.dt / 86400) for f, (a, b) in enumerate(blocks)))
    worst = 0.0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            (a1, b1), (a2, b2) = blocks[i], blocks[j]
            overlap = max(0, min(b1, b2) - max(a1, a2))
            worst = max(worst, overlap / max(min(b1 - a1, b2 - a2), 1))
    print(f"  worst pairwise overlap between fold test blocks: {100 * worst:.0f}%")
    if worst > FOLD_OVERLAP_WARN:
        print(f"  !! the folds are NOT independent replicates - they retest the same frames, so n_folds="
              f"{cfg.n_folds} buys far less than {cfg.n_folds}x the evidence and the hierarchical CI in report() "
              f"will be too narrow. This event is too short for test_windows={cfg.test_windows} at "
              f"seq_out={cfg.seq_out}. Lower test_windows / val_windows / seq_out or n_folds, or hold this "
              f"region out as a transfer target.")


def z_depth(h, ctx: Context):
    mean, std = ctx.norm["depth"]
    return ((depth_transform(h, ctx.cfg) - mean) / std).astype(np.float32)


def z_rain(r, ctx: Context):
    mean, std = ctx.norm["rain"]
    return ((np.log1p(np.maximum(r, 0.0)) - mean) / std).astype(np.float32)


def z_depth_t(h: torch.Tensor, ctx: Context) -> torch.Tensor:
    mean, std = ctx.norm["depth"]
    x = torch.log1p(h.clamp(min=0)) if ctx.cfg.target_transform == "log1p" else h.clamp(min=0)
    return (x - mean) / std


def inv_depth_t(z: torch.Tensor, ctx: Context) -> torch.Tensor:
    mean, std = ctx.norm["depth"]
    x = z * std + mean
    return torch.expm1(x.clamp(min=0.0)) if ctx.cfg.target_transform == "log1p" else x.clamp(min=0.0)


class FloodWindows(Dataset):
    """Sliding windows: (depth history, rain history, future rain, statics, target depth)."""

    def __init__(self, split: str, ctx: Context, patch=None):
        self.idx, self.ctx, self.patch = ctx.splits[split], ctx, patch
        self.data = ctx.data
        cfg = ctx.cfg
        self.step, self.tin, self.tout = cfg.step_frames, cfg.seq_in, cfg.seq_out

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        data = self.data
        frames = int(self.idx[i]) + np.arange(self.tin + self.tout) * self.step
        if self.patch:
            p = min(self.patch, data.H, data.W)
            top, left = np.random.randint(0, data.H - p + 1), np.random.randint(0, data.W - p + 1)
            rows, cols = slice(top, top + p), slice(left, left + p)
        else:
            rows = cols = slice(None)
        d = np.asarray(data.depth[frames, rows, cols], dtype=np.float32)
        r = np.stack([data.rain_for_step(int(f), self.step, rows, cols) for f in frames])
        st = self.ctx.static[:, rows, cols]
        return (torch.from_numpy(z_depth(d[:self.tin], self.ctx))[:, None],
                torch.from_numpy(z_rain(r[:self.tin], self.ctx))[:, None],
                torch.from_numpy(z_rain(r[self.tin:], self.ctx))[:, None],
                torch.from_numpy(st.copy()),
                torch.from_numpy(d[self.tin:])[:, None])


# =====================================================================================
# 3. METRICS, BASELINES AND TABLES
# =====================================================================================
class MetricAccumulator:
    """Streams batch predictions into global error / skill / categorical statistics."""

    def __init__(self, n_lead, gammas=GAMMAS):
        self.g, self.K = tuple(gammas), n_lead
        self.n = self.sy = self.sy2 = self.sp = self.sp2 = self.syp = self.se = self.sae = 0.0
        self.se_k = np.zeros(n_lead)
        self.n_k = np.zeros(n_lead)
        self.se_ref_k = np.zeros(n_lead)
        self.se_lin_k = np.zeros(n_lead)
        self.has_lin = False
        self.se_chg = self.n_chg = 0.0
        self.s_dp = self.s_dy = 0.0
        self.tp = {g: 0.0 for g in self.g}
        self.fp = dict(self.tp)
        self.fn = dict(self.tp)
        self.on_tp = self.on_fp = self.on_fn = 0.0
        self.rel_num, self.rel_den = [], []

    @torch.no_grad()
    def update(self, pred, targ, last_obs, prev_obs=None):
        p, y = pred.double(), targ.double()
        ref = last_obs.double().expand_as(y)
        e = p - y
        self.n += y.numel()
        self.sy += y.sum().item()
        self.sy2 += (y * y).sum().item()
        self.sp += p.sum().item()
        self.sp2 += (p * p).sum().item()
        self.syp += (p * y).sum().item()
        self.se += (e * e).sum().item()
        self.sae += e.abs().sum().item()
        self.se_k += (e * e).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        self.n_k += y[:, 0].numel()
        self.se_ref_k += ((ref - y) ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        if prev_obs is not None:
            lead = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype).view(1, -1, 1, 1, 1)
            lin = (last_obs.double() + (last_obs.double() - prev_obs.double()) * lead).clamp(min=0)
            self.se_lin_k += ((lin - y) ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
            self.has_lin = True
        self.s_dp += (p - ref).abs().sum().item()
        self.s_dy += (y - ref).abs().sum().item()
        chg = (ref - y).abs() > CHANGE_THRESHOLD
        if chg.any():
            self.se_chg += (e[chg] ** 2).sum().item()
            self.n_chg += chg.sum().item()
        for g in self.g:
            pm, ym = p >= g, y >= g
            self.tp[g] += (pm & ym).sum().item()
            self.fp[g] += (pm & ~ym).sum().item()
            self.fn[g] += (~pm & ym).sum().item()
        dry = ref < WET_THRESHOLD   # onset = cells that were dry at the last observation
        pm, ym = (p >= WET_THRESHOLD) & dry, (y >= WET_THRESHOLD) & dry
        self.on_tp += (pm & ym).sum().item()
        self.on_fp += (pm & ~ym).sum().item()
        self.on_fn += (~pm & ym).sum().item()
        self.rel_num += torch.sqrt((e ** 2).sum(dim=(1, 2, 3, 4))).cpu().tolist()
        self.rel_den += torch.sqrt((y ** 2).sum(dim=(1, 2, 3, 4))).cpu().tolist()

    def result(self):
        n = max(self.n, 1.0)
        ybar, pbar = self.sy / n, self.sp / n
        sdy = math.sqrt(max(self.sy2 / n - ybar ** 2, 0.0))
        sdp = math.sqrt(max(self.sp2 / n - pbar ** 2, 0.0))
        out = {"MSE_m2": self.se / n,
               "RMSE_m": math.sqrt(self.se / n),
               "NSE": 1.0 - self.se / max(self.sy2 - n * ybar ** 2, 1e-12),
               "r": (self.syp / n - ybar * pbar) / (sdy * sdp + 1e-12),
               "relL2": float(np.mean(np.array(self.rel_num) / (np.array(self.rel_den) + 1e-12))),
               "MAE_m": self.sae / n,
               "mass_err": abs(self.sp - self.sy) / (abs(self.sy) + 1e-12),
               "SS_vs_persistence": 1.0 - self.se / max(self.se_ref_k.sum(), 1e-12),
               "SS_vs_linear": (1.0 - self.se / max(self.se_lin_k.sum(), 1e-12)) if self.has_lin else float("nan"),
               "RMSE_changing_m": math.sqrt(self.se_chg / max(self.n_chg, 1.0)),
               "CSI_onset_1cm": self.on_tp / max(self.on_tp + self.on_fp + self.on_fn, 1e-12),
               "change_ratio": self.s_dp / max(self.s_dy, 1e-12)}
        for g in self.g:
            tp, fp, fn = self.tp[g], self.fp[g], self.fn[g]
            out[f"CSI@{g}"] = tp / max(tp + fp + fn, 1e-12)
            out[f"IoU@{g}"] = out[f"CSI@{g}"]
            out[f"F1@{g}"] = 2 * tp / max(2 * tp + fp + fn, 1e-12)
        nan_k = [float("nan")] * self.K
        out["_per_lead_RMSE_m"] = np.sqrt(self.se_k / np.maximum(self.n_k, 1)).tolist()
        out["_per_lead_SS"] = (1.0 - self.se_k / np.maximum(self.se_ref_k, 1e-12)).tolist()
        out["_per_lead_SS_linear"] = ((1.0 - self.se_k / np.maximum(self.se_lin_k, 1e-12)).tolist()
                                      if self.has_lin else nan_k)
        out["_per_lead_RMSE_persistence_m"] = np.sqrt(self.se_ref_k / np.maximum(self.n_k, 1)).tolist()
        out["_per_lead_RMSE_linear_m"] = (np.sqrt(self.se_lin_k / np.maximum(self.n_k, 1)).tolist()
                                          if self.has_lin else nan_k)
        return out


MAIN_COLS = ("MSE_m2", "RMSE_m", "NSE", "r", "IoU@0.01", "IoU@0.05")
EXTRA_COLS = ("relL2", "MAE_m", "CSI@0.001", "SS_vs_persistence", "SS_vs_linear", "RMSE_changing_m",
              "CSI_onset_1cm", "mass_err", "change_ratio")
LABELS = {"MSE_m2": "MSE [m2]", "RMSE_m": "RMSE [m]", "NSE": "NSE", "r": "Pearson r",
          "IoU@0.01": "IoU 1cm", "IoU@0.05": "IoU 5cm", "relL2": "rel-L2", "MAE_m": "MAE [m]",
          "SS_vs_linear": "SS vs linear", "CSI@0.001": "CSI 1mm", "SS_vs_persistence": "SS vs pers",
          "RMSE_changing_m": "RMSE chg [m]", "CSI_onset_1cm": "onset CSI", "mass_err": "mass err",
          "change_ratio": "change ratio"}
BETTER = {"MSE_m2": "min", "RMSE_m": "min", "MAE_m": "min", "relL2": "min", "RMSE_changing_m": "min",
          "mass_err": "min", "change_ratio": "one"}   # everything else: higher is better


def with_aliases(m):
    m = dict(m)
    if "MSE_m2" not in m and "RMSE_m" in m:
        m["MSE_m2"] = m["RMSE_m"] ** 2
    for g in GAMMAS:
        if f"IoU@{g}" not in m and f"CSI@{g}" in m:
            m[f"IoU@{g}"] = m[f"CSI@{g}"]
    return m


def _best(rows, col):
    vals = {k: m[col] for k, m in rows.items() if col in m and np.isfinite(m[col])}
    if not vals:
        return None
    score = {"max": lambda v: v, "min": lambda v: -v, "one": lambda v: -abs(v - 1.0)}[BETTER.get(col, "max")]
    return max(vals, key=lambda k: score(vals[k]))


def _text_table(rows, cols, title):
    best = {c: _best(rows, c) for c in cols}
    print(title)
    print(f"  {'model':22s}" + "".join(f"{LABELS.get(c, c):>15s}" for c in cols))
    for name, m in rows.items():
        cells = "".join(f"{m.get(c, float('nan')):14.6f}" + ("*" if best[c] == name else " ") for c in cols)
        print(f"  {name:22s}" + cells)


def results_table(rows, title=""):
    rows = {k: with_aliases(v) for k, v in rows.items()}
    if title:
        print(title)
    print("=" * 110)
    _text_table(rows, MAIN_COLS, "  MAIN METRICS   (* = best per column; IoU = CSI of the wet mask)")
    print("=" * 110)
    _text_table(rows, EXTRA_COLS, "secondary metrics (* = best per column; change ratio best = closest to 1)")


def check_result(test, ref, val=None, val_ref=None, name=""):
    t_ss = test["SS_vs_persistence"]
    pre = f"[{name}] " if name else ""
    if val is not None and val_ref is not None:
        v_ss = val["SS_vs_persistence"]
        v_pers, t_pers = val_ref["persistence"]["RMSE_m"], ref["persistence"]["RMSE_m"]
        print(f"\n{pre}val SS={v_ss:+.4f}  vs  test SS={t_ss:+.4f}"
              f"   | persistence RMSE {v_pers:.4f} m (val) vs {t_pers:.4f} m (test)"
              f"   -> test block is {v_pers / max(t_pers, 1e-12):.1f}x easier")
        if (v_ss > 0) != (t_ss > 0):
            print(f"  {pre}WARNING: validation and test disagree on the SIGN of the skill score. The two blocks are "
                  f"not exchangeable - typically the test block sits in the recession where persistence is "
                  f"near-perfect. This fold's headline number is not evidence either way; run all folds and "
                  f"report the spread.")
    lin_ss = ref["linear"]["SS_vs_persistence"]
    if lin_ss > max(t_ss, 0.0):
        print(f"  {pre}WARNING: linear extrapolation (SS={lin_ss:+.4f}) beats the model (SS={t_ss:+.4f}). On this "
              f"block the field is close to locally linear in time, so 'beats persistence' is too weak a bar - "
              f"report the model against linear extrapolation as well.")


@torch.no_grad()
def eval_reference(split: str, ctx: Context, batch: int = REF_BATCH):
    """Training-free baselines: persistence, linear extrapolation and a rain-bucket model."""
    cfg = ctx.cfg
    loader = DataLoader(FloodWindows(split, ctx), batch_size=batch, shuffle=False)
    accs = {k: MetricAccumulator(cfg.seq_out) for k in ("persistence", "linear", "rain_bucket")}
    step_h = cfg.step_frames * ctx.data.dt / 3600.0
    lead = torch.arange(1, cfg.seq_out + 1, device=DEVICE, dtype=torch.float32).view(1, -1, 1, 1, 1)
    rain_mean, rain_std = ctx.norm["rain"][0], ctx.norm["rain"][1]
    for dh, rh, rf, st, y in loader:
        dh, rf, y = dh.to(DEVICE), rf.to(DEVICE), y.to(DEVICE)
        last = inv_depth_t(dh[:, -1:], ctx)
        prev = inv_depth_t(dh[:, -2:-1], ctx)
        accs["persistence"].update(last.expand_as(y), y, last, prev)
        accs["linear"].update((last + (last - prev) * lead).clamp(min=0), y, last, prev)
        rain_mm_h = torch.expm1(rf * rain_std + rain_mean).clamp(min=0)
        accs["rain_bucket"].update(last + torch.cumsum(rain_mm_h * step_h / 1000.0, dim=1), y, last, prev)
    return {k: a.result() for k, a in accs.items()}


def paper_protocol_reference(cfg: Config, data: FloodData):
    """Persistence evaluated exactly like the FloodCastBench paper (blocks of N frames)."""
    block = PAPER_BLOCK_FRAMES
    H, W = data.H, data.W
    n_blocks = data.nt // block
    per_chunk = max(1, PAPER_CHUNK_VALUES // ((block - 1) * H * W))
    gammas = (0.001, 0.01)
    n = sy = sp = sy2 = sp2 = syp = se = 0.0
    tp = {g: 0 for g in gammas}
    fp, fn = dict(tp), dict(tp)
    rel = []
    for b0 in range(0, n_blocks, per_chunk):
        b1 = min(b0 + per_chunk, n_blocks)
        a = np.asarray(data.depth[b0 * block: b1 * block], dtype=np.float64).reshape(-1, block, H, W)
        y = a[:, 1:]
        p = np.broadcast_to(a[:, 0:1], y.shape)
        e2 = (y - p) ** 2
        rel += (np.sqrt(e2.sum(axis=(1, 2, 3))) / (np.sqrt((y ** 2).sum(axis=(1, 2, 3))) + 1e-12)).tolist()
        n += y.size
        se += e2.sum()
        del e2
        sy += y.sum()
        sp += p.sum()
        sy2 += (y * y).sum()
        sp2 += (p * p).sum()
        syp += (y * p).sum()
        for g in gammas:
            pm, ym = p >= g, y >= g
            tp[g] += int((pm & ym).sum())
            fp[g] += int((pm & ~ym).sum())
            fn[g] += int((~pm & ym).sum())
    ybar, pbar = sy / n, sp / n
    sdy, sdp = math.sqrt(max(sy2 / n - ybar ** 2, 0.0)), math.sqrt(max(sp2 / n - pbar ** 2, 0.0))
    out = {"blocks": int(n_blocks), "relL2": float(np.mean(rel)),
           "RMSE_m": math.sqrt(se / n),
           "NSE": float(1 - se / (sy2 - n * ybar ** 2)),
           "r": float((syp / n - ybar * pbar) / (sdy * sdp))}
    for g in gammas:
        out[f"CSI@{g}"] = float(tp[g] / (tp[g] + fp[g] + fn[g]))
    return out


def print_paper_comparison(cfg: Config, data: FloodData):
    paper = paper_protocol_reference(cfg, data)
    print(f"FloodCastBench protocol ({PAPER_BLOCK_FRAMES}-frame blocks, t=1 -> t=2..{PAPER_BLOCK_FRAMES}), "
          f"{cfg.region} {data.spec['flood'][1]}\n")
    print(f"{'model':32s}{'relL2':>10s}{'NSE':>12s}{'r':>10s}{'CSI@0.001':>12s}{'CSI@0.01':>11s}")
    print(f"{'persistence  (measured here)':32s}{paper['relL2']:10.6f}{paper['NSE']:12.6f}{paper['r']:10.6f}"
          f"{paper['CSI@0.001']:12.6f}{paper['CSI@0.01']:11.6f}")
    if cfg.region in PAPER_ROWS:
        where, rows = PAPER_ROWS[cfg.region]
        for name, v in rows:
            print(f"{name + '  (paper ' + where.split(',')[0] + ')':32s}"
                  f"{v[0]:10.6f}{v[1]:12.6f}{v[2]:10.6f}{v[3]:12.6f}{v[4]:11.6f}")
        print(f"\nContext, not a leaderboard. The paper rows are {where}, on the paper's own test split; the\n"
              f"persistence row above covers all {paper['blocks']} blocks of the event and needs no training. "
              f"Persistence\nmatching or beating them says the short-lead protocol is too easy to separate models "
              f"- not that it beats\na model trained on this region. The long-lead task used throughout is the "
              f"one that separates.")
    else:
        print(f"\n(no published rows for {cfg.region})")


# =====================================================================================
# 4. MODEL REGISTRY
# =====================================================================================
def n_params(model):
    return sum(p.numel() for p in model.parameters())


def zero_rain_z(ctx):
    return 0.0 if ctx is None else float(z_rain(np.float32(0.0), ctx))


def accepted_kwargs(cls, kw: dict) -> dict:
    """Drop keyword arguments the model class does not take (unless it has **kwargs)."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def build_model(cfg: Config, device=None, verbose=True, ctx=None):
    if cfg.model not in MODEL_CLASSES:
        raise ValueError(f"unknown model {cfg.model!r}; choose from {tuple(MODEL_CLASSES)}")
    cls = globals().get(MODEL_CLASSES[cfg.model])
    if cls is None:
        raise ImportError(f"class {MODEL_CLASSES[cfg.model]} for model {cfg.model!r} was not found - "
                          f"check the imports from models/ at the top of the script")
    n_static = len(static_names(cfg))
    kw = dict(hidden=cfg.hidden, kernel=cfg.kernel, norm=cfg.norm, head_width=cfg.head_width,
              use_future_rain=cfg.use_future_rain, residual=cfg.residual_head,
              rain_zero_z=zero_rain_z(ctx), residual_base=cfg.residual_base,
              velocity_clip=cfg.velocity_clip, t_in=cfg.seq_in)
    model = cls(n_static, **accepted_kwargs(cls, kw))
    if verbose:
        print(f"  {cfg.model}: {n_params(model):,} parameters | hidden {cfg.hidden} | {n_static} static channels")
    return model.to(DEVICE if device is None else device)


def smoke_test_models(cfg: Config, ctx: Context, names):
    """Push one batch through every model (training regime on a patch, evaluation regime on the full grid)
    so a broken architecture is dropped now instead of after hours of training the others."""
    print("\nmodel smoke test (one forward/backward on a patch, one forward on the full grid):")
    usable, K = [], min(2, cfg.seq_out)
    train_ds = FloodWindows("train", ctx, patch=cfg.patch)
    full_ds = FloodWindows("val" if len(ctx.splits["val"]) else "train", ctx)
    for name in names:
        mcfg = model_config(cfg, name)
        model = None
        try:
            model = build_model(mcfg, verbose=False, ctx=ctx)
            dh, rh, rf, st, y = [t[None].to(DEVICE) for t in train_ds[0]]
            model.train()
            with autocast(mcfg):
                pz = model(dh, rh, rf, st, K, teacher=z_depth_t(y, ctx)[:, :K], tf_prob=0.5)
            expect = (1, K, 1, dh.shape[-2], dh.shape[-1])
            assert tuple(pz.shape) == expect, f"output {tuple(pz.shape)}, expected {expect}"
            pz.float().mean().backward()
            fdh, frh, frf, fst, _ = [t[None].to(DEVICE) for t in full_ds[0]]
            model.eval()
            with torch.no_grad(), autocast(mcfg):
                pe = model(fdh, frh, frf, fst, 1)
            assert tuple(pe.shape[-2:]) == (ctx.data.H, ctx.data.W), \
                f"full-grid output {tuple(pe.shape[-2:])}, expected {(ctx.data.H, ctx.data.W)}"
            print(f"  ok    {name:10s} {n_params(model):>12,} parameters")
            usable.append(name)
        except Exception as e:   # noqa: BLE001 - any failure means this model cannot be trained
            print(f"  SKIP  {name:10s} {type(e).__name__}: {e}")
        finally:
            del model
            free_gpu()
    if not usable:
        sys.exit("no model passed the smoke test - nothing to train")
    return tuple(usable)


# =====================================================================================
# 5. TRAINING AND EVALUATION
# =====================================================================================
class WetWeightedLoss(nn.Module):
    """MSE in normalised space, up-weighted where the target is wet."""

    def __init__(self, w=4.0):
        super().__init__()
        self.w = w

    def forward(self, pred_z, targ_z, thresh_z):
        weights = 1.0 + self.w * torch.sigmoid((targ_z - thresh_z) * WET_LOSS_SHARPNESS)
        return ((pred_z - targ_z) ** 2 * weights).mean()


@torch.no_grad()
def evaluate(model, split: str, ctx: Context, batch: int = EVAL_BATCH):
    cfg = ctx.cfg
    model.eval()
    loader = DataLoader(FloodWindows(split, ctx), batch_size=batch, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    acc = MetricAccumulator(cfg.seq_out)
    for dh, rh, rf, st, y in loader:
        dh, rh, rf, st, y = [t.to(DEVICE, non_blocking=True) for t in (dh, rh, rf, st, y)]
        with autocast(cfg):
            pz = model(dh, rh, rf, st, cfg.seq_out)
        acc.update(inv_depth_t(pz.float(), ctx), y, inv_depth_t(dh[:, -1:].float(), ctx),
                   inv_depth_t(dh[:, -2:-1].float(), ctx))
    return acc.result()


@torch.no_grad()
def predict_window(model, ctx: Context, i: int, split: str = "test"):
    """(prediction, truth, last observation) in metres for one window, as numpy arrays."""
    cfg = ctx.cfg
    dh, rh, rf, st, y = [t[None].to(DEVICE) for t in FloodWindows(split, ctx)[i]]
    model.eval()
    with autocast(cfg):
        pz = model(dh, rh, rf, st, cfg.seq_out)
    pred = inv_depth_t(pz.float(), ctx)[0, :, 0].cpu().numpy()
    last = inv_depth_t(dh[:, -1:].float(), ctx)[0, 0, 0].cpu().numpy()
    return pred, y[0, :, 0].cpu().numpy(), last


def schedule_warmup(cfg: Config) -> int:
    """Epochs until the rollout curriculum and teacher forcing are both finished."""
    full_rollout = cfg.curriculum_every * (len(cfg.curriculum) - 1)
    return int(max(full_rollout, cfg.tf_anneal_epochs))


def load_state(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def train(cfg: Config, ctx: Context, verbose=True):
    if cfg.epochs < 1:
        raise ValueError(f"epochs={cfg.epochs}; at least one epoch is needed to produce a checkpoint")
    set_seed(cfg.seed)
    model = build_model(cfg, verbose=verbose, ctx=ctx)
    set_seed(cfg.seed)
    loader = DataLoader(FloodWindows("train", ctx, patch=cfg.patch), batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    if len(loader) == 0:
        raise RuntimeError(f"{len(loader.dataset)} training windows is fewer than batch_size={cfg.batch_size}")

    # optimiser + warm-up/cosine LR schedule
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    warm_steps = int(cfg.warmup_frac * total_steps)

    def lr_factor(step):
        if step < warm_steps:
            return step / max(warm_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(total_steps - warm_steps, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = make_scaler(cfg.amp and DEVICE.type == "cuda")
    loss_fn = WetWeightedLoss(cfg.wet_weight).to(DEVICE)
    thresh_z = torch.tensor(float(z_depth(np.float32(WET_THRESHOLD), ctx)), device=DEVICE)

    lower_is_better = ("RMSE_m", "RMSE_changing_m", "MSE_m2", "relL2", "MAE_m", "mass_err")
    sel_metric = cfg.selection_metric
    run_warnings = []

    run_id = f"{safe_tag(cfg.tag)}_{cfg.region}_fold{cfg.fold}_seed{cfg.seed}"
    ckpt = os.path.join(cfg.save_dir, run_id + ".pt")
    sel_split = "val" if len(ctx.splits["val"]) else "test"
    if sel_split == "test":
        run_warnings.append("empty validation split - selected on TEST")
        print(f"  WARNING [{run_id}]: empty validation split - selecting on TEST; fix the split before reporting")
    es_start = schedule_warmup(cfg)
    if cfg.epochs <= es_start:
        run_warnings.append(f"epochs={cfg.epochs} ends before the schedule finishes at epoch {es_start + 1}")
        print(f"  WARNING [{run_id}]: epochs={cfg.epochs} but the curriculum / teacher-forcing schedule only "
              f"finishes at epoch {es_start + 1} - the model is never trained in the regime it is evaluated in")
    if verbose:
        print(f"  schedule: K={cfg.seq_out} and tf=0 from epoch {es_start + 1}; "
              f"early-stopping patience ({cfg.patience}) counts from there | selecting on {sel_metric}")

    hist, best, best_epoch, bad, stopped = [], -math.inf, 0, 0, "completed"
    for ep in range(cfg.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        K = min(cfg.curriculum[min(ep // cfg.curriculum_every, len(cfg.curriculum) - 1)], cfg.seq_out)
        tf_prob = max(0.0, 1.0 - ep / max(cfg.tf_anneal_epochs, 1))
        for dh, rh, rf, st, y in loader:
            dh, rh, rf, st, y = [t.to(DEVICE, non_blocking=True) for t in (dh, rh, rf, st, y)]
            tz = z_depth_t(y, ctx)
            opt.zero_grad(set_to_none=True)
            with autocast(cfg):
                pz = model(dh, rh, rf, st, K, teacher=tz[:, :K], tf_prob=tf_prob)
                loss = loss_fn(pz, tz[:, :K], thresh_z)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()

        vm = evaluate(model, sel_split, ctx)
        score = vm.get(sel_metric, float("nan"))
        score = (-score if sel_metric in lower_is_better else score) if math.isfinite(score) else -math.inf
        hist.append(dict(epoch=ep + 1, train_loss=loss_sum / len(loader), K=K, tf=tf_prob,
                         **{k: v for k, v in vm.items() if not k.startswith("_")}))

        flag = ""
        if ep == 0 or score > best + 1e-5:
            best, best_epoch, bad, flag = score, ep + 1, 0, "  *"
            torch.save(model.state_dict(), ckpt)
        elif ep >= es_start:
            bad += 1
        else:
            flag = "  (warm-up)"
        if verbose:
            print(f"  ep {ep + 1:3d}/{cfg.epochs} K={K:2d} tf={tf_prob:.2f} loss={loss_sum / len(loader):.5f}"
                  f" | {sel_split} SS={vm['SS_vs_persistence']:+.4f} RMSE={vm['RMSE_m']:.4f} m"
                  f" CSI1cm={vm['CSI@0.01']:.4f} | {time.time() - t0:.0f}s{flag}")
        if bad >= cfg.patience:
            stopped = "early_stop"
            if verbose:
                print(f"  early stop at epoch {ep + 1} ({cfg.patience} epochs without improvement since the "
                      f"schedule finished)")
            break
        if time_budget_spent():
            stopped = "time_budget"
            run_warnings.append(f"stopped by MAX_HOURS at epoch {ep + 1} - incomplete")
            print(f"  [{run_id}] MAX_HOURS={MAX_HOURS} reached - stopped at epoch {ep + 1}; this run is incomplete")
            break

    model.load_state_dict(load_state(ckpt))
    if verbose:
        print(f"  restored best checkpoint: epoch {best_epoch} ({sel_split} SS={best:+.4f})")
    if best_epoch <= es_start:
        row = hist[best_epoch - 1]
        run_warnings.append(f"best epoch {best_epoch} is inside the schedule warm-up "
                            f"(K={row['K']}, tf={row['tf']:.2f}) - the run never converged in the evaluated regime")
        print(f"  NOTE [{run_id}]: best epoch {best_epoch} is inside the schedule warm-up (K={row['K']}, "
              f"tf={row['tf']:.2f}), i.e. from a shorter-rollout, teacher-forced regime. Validation free-runs "
              f"seq_out every epoch, so the comparison is fair, but check this run before trusting it.")
    info = dict(run_id=run_id, model=cfg.model, params=n_params(model), best_epoch=best_epoch,
                best_selection_SS=best, selection_split=sel_split, selection_metric=sel_metric,
                es_start_epoch=es_start + 1, epochs_run=len(hist), stopped=stopped, warnings=run_warnings,
                ok=not run_warnings)
    return model, hist, info


# =====================================================================================
# 6. REPORTING AND COMPARISON PLOTS
# =====================================================================================
BASELINE_ORDER = (("persistence", "persistence"), ("linear extrap", "linear"), ("rain bucket", "rain_bucket"))
BASELINE_COLORS = {"persistence": "0.30", "linear extrap": "0.50", "rain bucket": "0.70"}
BASELINE_STYLE = {"persistence": "--", "linear extrap": ":", "rain bucket": "-."}


def palette(names):
    """Fixed colour per model so the same architecture has the same colour in every figure and run."""
    cmap = plt.get_cmap("tab10")
    pal = {n: cmap(i % 10) for i, n in enumerate(MODEL_CLASSES)}
    for n in names:
        pal.setdefault(n, cmap(len(pal) % 10))
    pal.update(BASELINE_COLORS)
    return pal


def hier_bootstrap(by_fold, n_boot=10000, seed=BOOTSTRAP_SEED):
    """95% CI of the grand mean, resampling folds first and then seeds within each fold."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(v, dtype=float) for v in by_fold.values() if len(v)]
    n_folds = len(arrs)
    fold_boot = np.stack([a[rng.integers(0, len(a), (n_boot, n_folds, len(a)))].mean(axis=-1) for a in arrs])
    pick = rng.integers(0, n_folds, (n_boot, n_folds))
    boot = fold_boot[pick, np.arange(n_boot)[:, None], np.arange(n_folds)[None, :]].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def report(rows, keys=MAIN_COLS + ("SS_vs_persistence", "change_ratio")):
    rows = [with_aliases(r) for r in rows]
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    folds = sorted({r["fold"] for r in rows})
    print(f"\n{len(rows)} runs | folds {folds} | mean [95% CI: folds, then seeds within a fold, resampled]")
    print(f"{'metric':20s}" + "".join(f"{t:>34s}" for t in tags))
    for k in keys:
        line = f"{k:20s}"
        for t in tags:
            by_fold = {f: [r[k] for r in rows if r["tag"] == t and r["fold"] == f] for f in folds}
            by_fold = {f: v for f, v in by_fold.items() if v}
            if not by_fold:
                line += f"{'-':>34s}"
                continue
            lo, hi = hier_bootstrap(by_fold, n_boot=BOOTSTRAP_RESAMPLES)
            line += f"{np.mean([x for v in by_fold.values() for x in v]):>12.5f} [{lo:9.5f},{hi:9.5f}]"
        print(line)
    print("\nper-fold means of SS_vs_persistence:")
    for f in folds:
        line = f"  fold {f}: "
        for t in tags:
            v = [r["SS_vs_persistence"] for r in rows if r["tag"] == t and r["fold"] == f]
            line += f"{t}={np.mean(v):+.4f}  " if v else f"{t}=   -     "
        print(line)


def comparison_rows(runs, ref):
    """Ordered {name: test metrics}: training-free baselines first, then every trained model."""
    rows = {name: ref[key] for name, key in BASELINE_ORDER}
    rows.update({name: r["test"] for name, r in runs.items()})
    return rows


def plot_compare_summary(runs, ref, ctx: Context, path=None):
    """One dashboard: learning curves, error / skill versus lead time and final RMSE for every model."""
    cfg, pal = ctx.cfg, palette(runs)
    lead_h = (np.arange(cfg.seq_out) + 1) * cfg.step_frames * ctx.data.dt / 3600
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    for name, r in runs.items():
        hist, c = r["hist"], pal[name]
        ep = [h["epoch"] for h in hist]
        ax[0, 0].plot(ep, [h["train_loss"] for h in hist], color=c, label=name)
        ax[0, 1].plot(ep, [h["SS_vs_persistence"] for h in hist], color=c, label=name)
        ax[0, 2].plot(ep, [h["RMSE_m"] for h in hist], color=c, label=name)
        be = r["info"]["best_epoch"]
        if 1 <= be <= len(hist):
            ax[0, 1].plot(be, hist[be - 1]["SS_vs_persistence"], "o", ms=8, mfc=c, mec="k", zorder=5)
            ax[0, 2].plot(be, hist[be - 1]["RMSE_m"], "o", ms=8, mfc=c, mec="k", zorder=5)
    es = max(r["info"]["es_start_epoch"] for r in runs.values())
    for a in (ax[0, 0], ax[0, 1], ax[0, 2]):
        a.axvline(es, color="k", lw=0.8, ls=":")
        a.grid(alpha=0.3)
        a.set_xlabel("epoch")
    ax[0, 0].set(yscale="log", title="training loss (normalised, wet-weighted)")
    ax[0, 1].axhline(0, color="k", lw=1, ls="--")
    ax[0, 1].set(title="validation skill vs persistence  (o = selected epoch, dotted = schedule finished)")
    ax[0, 2].set(yscale="log", ylabel="m", title="validation RMSE (full free-run rollout)")
    ax[0, 0].legend(fontsize=9)

    # error / skill versus lead time, test split
    for name, key in BASELINE_ORDER[:2]:
        ax[1, 0].plot(lead_h, ref[key]["_per_lead_RMSE_m"], BASELINE_STYLE[name], color=BASELINE_COLORS[name],
                      lw=1.8, label=name)
    ax[1, 1].axhline(0, color=BASELINE_COLORS["persistence"], lw=1.8, ls="--", label="persistence")
    ax[1, 1].plot(lead_h, ref["linear"]["_per_lead_SS"], ":", color=BASELINE_COLORS["linear extrap"], lw=1.8,
                  label="linear extrap")
    for name, r in runs.items():
        ax[1, 0].plot(lead_h, r["test"]["_per_lead_RMSE_m"], "o-", ms=4, color=pal[name], label=name)
        ax[1, 1].plot(lead_h, r["test"]["_per_lead_SS"], "o-", ms=4, color=pal[name], label=name)
    ax[1, 0].set(xlabel="lead time [h]", ylabel="RMSE [m]", title="test RMSE vs lead time (lower is better)")
    ax[1, 1].set(xlabel="lead time [h]", ylabel="skill score", title="test skill vs persistence per lead (>0 beats it)")
    ax[1, 0].legend(fontsize=9)
    for a in (ax[1, 0], ax[1, 1]):
        a.grid(alpha=0.3)

    # final test RMSE ranking
    rows = comparison_rows(runs, ref)
    order = sorted(rows, key=lambda n: rows[n]["RMSE_m"])
    vals = [rows[n]["RMSE_m"] for n in order]
    bars = ax[1, 2].barh(range(len(order)), vals, color=[pal.get(n, "0.5") for n in order])
    ax[1, 2].set_yticks(range(len(order)))
    ax[1, 2].set_yticklabels(order)
    ax[1, 2].invert_yaxis()
    for b, n, v in zip(bars, order, vals):
        extra = f"  ({runs[n]['info']['params']:,} par.)" if n in runs else ""
        ax[1, 2].text(v, b.get_y() + b.get_height() / 2, f" {v:.4f}{extra}", va="center", fontsize=8)
    ax[1, 2].set(xlabel="RMSE [m]", title="test RMSE ranking (lower is better)")
    ax[1, 2].set_xlim(0, max(vals) * 1.35)
    ax[1, 2].grid(alpha=0.3, axis="x")

    plt.suptitle(f"{cfg.region} | fold {cfg.fold} | seed {cfg.seed} | lead "
                 f"{cfg.seq_out * cfg.step_frames * ctx.data.dt / 3600:.1f} h | {len(runs)} models, identical protocol",
                 fontsize=13)
    plt.tight_layout()
    finish_figure(path)


def plot_metric_bars(runs, ref, ctx: Context, path=None):
    """One bar chart per headline metric, all models and baselines side by side (best bar outlined)."""
    rows = {k: with_aliases(v) for k, v in comparison_rows(runs, ref).items()}
    pal = palette(runs)
    metrics = ("RMSE_m", "NSE", "SS_vs_persistence", "IoU@0.01", "IoU@0.05", "change_ratio")
    names = list(rows)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for a, k in zip(ax.ravel(), metrics):
        vals = np.array([rows[n].get(k, np.nan) for n in names], dtype=float)
        bars = a.bar(range(len(names)), np.nan_to_num(vals), color=[pal.get(n, "0.5") for n in names])
        best = _best(rows, k)
        for b, n, v in zip(bars, names, vals):
            if n == best:
                b.set_edgecolor("k")
                b.set_linewidth(2.5)
            if np.isfinite(v):
                a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center",
                       va="bottom" if b.get_height() >= 0 else "top", fontsize=7)
        if k == "change_ratio":
            a.axhline(1.0, color="k", lw=1, ls="--")
        if k == "SS_vs_persistence":
            a.axhline(0.0, color="k", lw=1, ls="--")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        a.set_title(LABELS.get(k, k) + ("  (lower is better)" if BETTER.get(k) == "min" else
                                        "  (closest to 1 is best)" if BETTER.get(k) == "one" else
                                        "  (higher is better)"), fontsize=10)
        a.grid(alpha=0.3, axis="y")
    plt.suptitle(f"{ctx.cfg.region} | test | fold {ctx.cfg.fold} | seed {ctx.cfg.seed} (outlined bar = best)",
                 fontsize=13)
    plt.tight_layout()
    finish_figure(path)


def plot_case_comparison(case, i, ctx: Context, path_prefix=None, split="test"):
    """Two figures for one window: predictions (truth on top) and errors, all models on shared colour scales."""
    cfg = ctx.cfg
    truth, last, preds = case["truth"], case["last"], case["preds"]
    names = list(preds)
    leads = sorted({0, cfg.seq_out // 2, cfg.seq_out - 1})
    lead_h = [(k + 1) * cfg.step_frames * ctx.data.dt / 3600 for k in leads]
    vmax = float(np.percentile(truth, 99.5)) or 1.0
    cell = 3.3

    # predictions
    rows = ["truth"] + names
    fig, ax = plt.subplots(len(rows), len(leads), figsize=(cell * len(leads) + 1.5, 2.8 * len(rows)),
                           squeeze=False, constrained_layout=True)
    im = None
    for r, name in enumerate(rows):
        field = truth if name == "truth" else preds[name]
        for j, k in enumerate(leads):
            im = ax[r, j].imshow(field[k], cmap="Blues", vmin=0, vmax=vmax)
            ax[r, j].set_xticks([])
            ax[r, j].set_yticks([])
            if r == 0:
                ax[r, j].set_title(f"t+{lead_h[j]:.1f} h")
        ax[r, 0].set_ylabel(name, fontsize=11)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, label="depth [m]")
    fig.suptitle(f"{split} window {i} | predicted depth")
    finish_figure(path_prefix and f"{path_prefix}_pred.png")

    # errors
    err_max = max(float(np.percentile(np.abs(preds[n] - truth), 99.5)) for n in names) or 1.0
    fig, ax = plt.subplots(len(names), len(leads), figsize=(cell * len(leads) + 1.5, 2.8 * len(names)),
                           squeeze=False, constrained_layout=True)
    for r, name in enumerate(names):
        for j, k in enumerate(leads):
            err = preds[name][k] - truth[k]
            im = ax[r, j].imshow(err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
            ax[r, j].set_xticks([])
            ax[r, j].set_yticks([])
            ax[r, j].set_title(f"RMSE {np.sqrt((err ** 2).mean()):.4f} m", fontsize=9)
            if r == 0:
                ax[r, j].annotate(f"t+{lead_h[j]:.1f} h", xy=(0.5, 1.22), xycoords="axes fraction",
                                  ha="center", fontsize=11)
        ax[r, 0].set_ylabel(name, fontsize=11)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, label="prediction - truth [m]")
    pers = float(np.sqrt(((last - truth[leads[-1]]) ** 2).mean()))
    fig.suptitle(f"{split} window {i} | error (persistence RMSE at the last lead shown = {pers:.4f} m)")
    finish_figure(path_prefix and f"{path_prefix}_err.png")


def plot_sweep(rows, path=None):
    """Per-metric view of a sweep: bar = mean over runs, dots = individual (fold, seed) runs coloured by fold."""
    rows = [with_aliases(r) for r in rows]
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    folds = sorted({r["fold"] for r in rows})
    pal = palette(tags)
    fold_color = {f: plt.get_cmap("viridis")(f / max(len(folds) - 1, 1) if len(folds) > 1 else 0.5) for f in folds}
    metrics = ("RMSE_m", "SS_vs_persistence", "NSE", "IoU@0.01", "IoU@0.05", "CSI_onset_1cm")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for a, k in zip(ax.ravel(), metrics):
        for x, t in enumerate(tags):
            pts = [(r["fold"], r[k]) for r in rows if r["tag"] == t and k in r and np.isfinite(r[k])]
            if not pts:
                continue
            ys = np.array([v for _, v in pts])
            a.bar(x, ys.mean(), color=pal.get(t, "0.5"), alpha=0.4, width=0.7)
            a.scatter(x + rng.uniform(-0.18, 0.18, len(ys)), ys, c=[fold_color[f] for f, _ in pts], s=26,
                      edgecolor="k", linewidth=0.4, zorder=3)
        if k == "SS_vs_persistence":
            a.axhline(0.0, color="k", lw=1, ls="--")
        a.set_xticks(range(len(tags)))
        a.set_xticklabels(tags, rotation=35, ha="right", fontsize=9)
        a.set_title(LABELS.get(k, k) + ("  (lower is better)" if BETTER.get(k) == "min" else "  (higher is better)"),
                    fontsize=10)
        a.grid(alpha=0.3, axis="y")
    handles = [Line2D([], [], marker="o", ls="", mfc=fold_color[f], mec="k", label=f"fold {f}") for f in folds]
    ax[0, 0].legend(handles=handles, fontsize=8, title="dots: one run each")
    plt.suptitle(f"sweep | {len(rows)} runs | bar = mean, dots = individual (fold, seed) results", fontsize=13)
    plt.tight_layout()
    finish_figure(path)


# =====================================================================================
# 7. RUN MODES
# =====================================================================================
def run_compare(cfg: Config, ctx: Context, ref, models):
    """Train every model on one (fold, seed), then tabulate and plot them together."""
    lead_h = cfg.seq_out * cfg.step_frames * ctx.data.dt / 3600
    print(f"\ntraining {len(models)} models on {cfg.region} | fold {cfg.fold} | seed {cfg.seed}: {', '.join(models)}")
    n_test = len(ctx.splits["test"])
    case_ids = {"first": 0, "middle": n_test // 2, "last": n_test - 1}
    case_idx = sorted({case_ids[c] for c in CASE_PLOT_WINDOWS})
    cases = {i: dict(preds={}) for i in case_idx}
    val_ref = eval_reference("val", ctx) if len(ctx.splits["val"]) else None
    runs, failed, t_all = {}, {}, time.time()

    for n, name in enumerate(models, 1):
        if time_budget_spent():
            print(f"\nMAX_HOURS reached - {name} and the models after it were not trained")
            break
        mcfg = model_config(cfg, name)
        print(f"\n{'=' * 100}\n[{n}/{len(models)}] {name}\n{'=' * 100}")
        t0, model = time.time(), None
        try:
            model, hist, info = train(mcfg, ctx)
            test = evaluate(model, "test", ctx)
            val = evaluate(model, "val", ctx) if val_ref is not None else None
            for i in case_idx:
                pred, truth, last = predict_window(model, ctx, i, "test")
                cases[i]["truth"], cases[i]["last"] = truth, last
                cases[i]["preds"][name] = pred
        except Exception as e:   # noqa: BLE001 - one broken model must not cost the other runs
            traceback.print_exc()
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"  !! {name} failed and is left out of the comparison")
            continue
        finally:
            del model
            free_gpu()
        if info["stopped"] == "time_budget":
            print(f"  WARNING: {name} was cut short by MAX_HOURS - its numbers are from an incomplete run")
        runs[name] = dict(hist=hist, test=test, val=val, info=info, seconds=time.time() - t0)
        save_json(os.path.join(cfg.save_dir, info["run_id"] + ".json"),
                  dict(key=experiment_key(mcfg), config=asdict(mcfg), info=info, history=hist, test=test, val=val,
                       reference=ref, val_reference=val_ref, manifest=ctx.data.manifest,
                       torch=torch.__version__, device=str(DEVICE)))
        print(f"  {name} done in {time.time() - t0:.0f}s | test SS={test['SS_vs_persistence']:+.4f} "
              f"RMSE={test['RMSE_m']:.5f} m")

    if not runs:
        print("\nno model finished - nothing to compare")
        return runs

    rows = comparison_rows(runs, ref)
    results_table(rows, title=f"\nTEST  |  {cfg.region} {ctx.data.H}x{ctx.data.W}  |  fold {cfg.fold}  |  "
                              f"seed {cfg.seed}  |  lead {lead_h:.1f} h  |  {n_test} windows\n")
    print("\nSS_vs_persistence <= 0 means a model has no skill over copying the last frame.")
    for name, r in runs.items():
        check_result(r["test"], ref, r["val"], val_ref, name=name)

    ranked = sorted(runs, key=lambda m: runs[m]["test"]["RMSE_m"])
    print("\nranking by test RMSE:  " + "  <  ".join(f"{m} ({runs[m]['test']['RMSE_m']:.5f} m)" for m in ranked))
    print(f"{'model':12s}{'params':>14s}{'best ep':>9s}{'epochs':>8s}{'stopped':>13s}{'time [s]':>10s}")
    for name, r in runs.items():
        i = r["info"]
        print(f"{name:12s}{i['params']:>14,}{i['best_epoch']:>9d}{i['epochs_run']:>8d}{i['stopped']:>13s}"
              f"{r['seconds']:>10.0f}")
    for name, why in failed.items():
        print(f"FAILED {name}: {why}")

    stem = os.path.join(cfg.save_dir, f"compare_{EXPERIMENT_TAG}_{cfg.region}_fold{cfg.fold}_seed{cfg.seed}")
    save_json(stem + ".json", dict(config=asdict(cfg), models=list(runs), failed=failed, reference=ref,
                                   val_reference=val_ref, runs=runs, torch=torch.__version__,
                                   total_seconds=time.time() - t_all))
    plot_compare_summary(runs, ref, ctx, stem + "_summary.png")
    plot_metric_bars(runs, ref, ctx, stem + "_metrics.png")
    for i in case_idx:
        plot_case_comparison(cases[i], i, ctx, f"{stem}_case{i}")
    print("\nsaved to", cfg.save_dir, f"(files start with {os.path.basename(stem)})")
    return runs


def fold_context(base: Config, data: FloodData, fold: int, shared: dict):
    """Context and reference baselines for one fold, built once and reused by every model and seed."""
    if fold not in shared:
        ctx = build_context(config_for(base, fold=fold), data)
        shared[fold] = (ctx, eval_reference("test", ctx))
    return shared[fold]


def run_one(base: Config, data: FloodData, fold: int, seed: int, model_name: str, shared: dict):
    """One sweep cell; loads the saved result when the config and pipeline version match."""
    cfg = config_for(base, fold=fold, seed=seed, model=model_name, tag=model_name)
    path = os.path.join(cfg.save_dir, f"sweep_{safe_tag(cfg.tag)}_{cfg.region}_fold{fold}_seed{seed}.json")
    cached = load_json(path)
    if isinstance(cached, dict):
        if cached.get("key") == experiment_key(cfg):
            cached["cached"] = True
            return cached
        print(f"  {os.path.basename(path)} was produced with a different config or pipeline version - re-running")
    if time_budget_spent():
        return None
    ctx, ref = fold_context(base, data, fold, shared)
    model = None
    try:
        model, hist, info = train(cfg, ctx, verbose=True)
        if info["stopped"] == "time_budget":
            print(f"  {info['run_id']} not saved: MAX_HOURS cut it short")
            return None
        result = dict(key=experiment_key(cfg), config=asdict(cfg), tag=cfg.tag, fold=fold, seed=seed, info=info,
                      history=hist, test=evaluate(model, "test", ctx), reference=ref)
    finally:
        del model
        free_gpu()
    save_json(path, result)
    return result


def result_row(result):
    return dict(tag=result["tag"], fold=result["fold"], seed=result["seed"], params=result["info"]["params"],
                **{k: v for k, v in result["test"].items() if not k.startswith("_")})


def reference_rows(result):
    return [dict(tag=name, fold=result["fold"], seed=result["seed"], params=0,
                 **{k: v for k, v in result["reference"][key].items() if not k.startswith("_")})
            for name, key in (("persistence", "persistence"), ("linear extrap", "linear"))]


def run_sweep(base: Config, data: FloodData, folds, seeds, models):
    """Every model x fold x seed. Loop order is (fold, seed, model) so a MAX_HOURS stop leaves complete
    model-vs-model comparisons for the cells that did finish."""
    cells = list(itertools.product(folds, seeds))
    total = len(cells) * len(models)
    print(f"{total} runs: {len(models)} models ({', '.join(models)}) x folds {folds} x seeds {seeds} "
          f"(runs on disk with the same config are loaded)")
    rows, references, ref_seen, skipped, done, shared = [], [], set(), 0, 0, {}
    for fold, seed in cells:
        for name in models:
            done += 1
            t0 = time.time()
            try:
                result = run_one(base, data, fold, seed, name, shared)
            except Exception as e:   # noqa: BLE001 - keep sweeping the other cells
                traceback.print_exc()
                print(f"[{done}/{total}] {name} fold {fold} seed {seed} FAILED: {type(e).__name__}: {e}")
                skipped += 1
                continue
            if result is None:
                skipped += 1
                continue
            rows.append(result_row(result))
            if (fold, seed) not in ref_seen:       # baselines are identical for every model: count them once
                ref_seen.add((fold, seed))
                references += reference_rows(result)
            took = "cached" if result.get("cached") else f"{time.time() - t0:.0f}s"
            print(f"[{done}/{total}] {name:9s} fold {fold} seed {seed} | SS={rows[-1]['SS_vs_persistence']:+.4f} "
                  f"RMSE={rows[-1]['RMSE_m']:.5f} m | {took}")
    if skipped:
        print(f"\n{skipped} run(s) not done (MAX_HOURS or an error). Run the script again to continue - "
              f"finished runs load from disk.")
    if not rows:
        print("no finished runs to report yet")
        return rows

    save_json(os.path.join(base.save_dir, f"sweep_summary_{EXPERIMENT_TAG}_{base.region}.json"), references + rows)
    report(references + rows)
    plot_sweep(references + rows, os.path.join(base.save_dir, f"sweep_compare_{EXPERIMENT_TAG}_{base.region}.png"))
    print(f"\nPer-run results: {base.save_dir}/sweep_<model>_{base.region}_fold*_seed*.json")
    print(f"Comparison plot: {base.save_dir}/sweep_compare_{EXPERIMENT_TAG}_{base.region}.png")
    return rows


# =====================================================================================
# 8. ENTRY POINT
# =====================================================================================
def _int_tuple(text):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _str_tuple(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def parse_args():
    p = argparse.ArgumentParser(description="FloodCastBench: train and compare all models in one run. Defaults "
                                            "come from the CONFIGURATION block at the top of this file.")
    p.add_argument("--mode", choices=RUN_MODES, default=RUN_MODE)
    p.add_argument("--region", choices=sorted(GRID_SPEC), default=REGION)
    p.add_argument("--models", type=_str_tuple, default=MODELS,
                   help=f"comma-separated subset of {','.join(MODEL_CLASSES)} (default: all)")
    p.add_argument("--fold", type=int, default=FOLD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--folds", type=_int_tuple, default=SWEEP_FOLDS, help="comma-separated, sweep mode")
    p.add_argument("--seeds", type=_int_tuple, default=SWEEP_SEEDS, help="comma-separated, sweep mode")
    p.add_argument("--max-hours", type=float, default=MAX_HOURS)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)
    return p.parse_args()


def main():
    # cuda determinism
    random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("WARNING: no CUDA GPU found - may differ.")


    global MAX_HOURS, SHOW_PLOTS, DATA_ROOT
    args = parse_args()
    MAX_HOURS, SHOW_PLOTS, DATA_ROOT = args.max_hours, args.show_plots, args.data_root
    if not SHOW_PLOTS:
        plt.switch_backend("Agg")
    requested = resolve_models(args.models)

    print("python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", np.__version__)
    if DEVICE.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"gpu: {gpu.name}  {gpu.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA GPU found - training runs on the CPU and is 10-50x slower.\n"
              "         --mode check (caches, pipeline checks, reference baselines) is still quick.")

    # paths
    data_root = find_data_root()
    local_base = os.path.dirname(data_root)
    cache_dir = args.cache_dir or os.path.join(local_base, "cache")
    save_dir = args.results_dir or os.path.join(local_base, "results_final")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # config
    base = make_base_config(args.region, data_root, cache_dir, save_dir)
    folds = tuple(f for f in args.folds if 0 <= f < base.n_folds) or (0,)
    if folds != tuple(args.folds):
        print(f"NOTE: {base.region} has only {base.n_folds} folds, so the sweep folds become {folds}")
    if not 0 <= args.fold < base.n_folds:
        raise ValueError(f"fold {args.fold} is outside 0..{base.n_folds - 1}; this region's split preset "
                         f"gives n_folds={base.n_folds}")
    cfg = config_for(base, fold=args.fold, seed=args.seed)
    set_seed(cfg.seed)

    print(f"RUN_MODE={args.mode} | {EXPERIMENT_TAG} ({EXPERIMENT_DESCRIPTION}) | models {', '.join(requested)} "
          f"| fold {cfg.fold} | seed {cfg.seed} | device {DEVICE}")
    print("data_root:", cfg.data_root)
    print("cache_dir:", cfg.cache_dir, "| save_dir:", cfg.save_dir)
    print(f"one model step = {cfg.step_frames * 300} s | history {cfg.seq_in * cfg.step_frames * 300 / 3600:.1f} h"
          f" | lead {cfg.seq_out * cfg.step_frames * 300 / 3600:.1f} h")
    preset = split_preset(cfg.region, base)
    if preset:
        n_test, n_val, _ = SPLIT_PRESETS[cfg.region]
        print(f"split preset for {cfg.region}: test_windows={cfg.test_windows} val_windows={cfg.val_windows} "
              f"n_folds={cfg.n_folds} test_stride={cfg.test_stride}"
              f"  ->  {n_test} independent test and {n_val} independent val window(s) per fold")
    elif USE_SPLIT_PRESETS:
        print(f"NOTE: no split preset for region {cfg.region!r} - using the Config defaults. "
              f"Check the fold-overlap line below before reporting anything from this region.")

    # data + checks
    data = FloodData(cfg)
    run_data_sanity_checks(cfg, data, os.path.join(save_dir, f"data_{cfg.region}") if SAVE_PLOTS else None)

    ctx = build_context(cfg, data)
    print(f"split_mode={cfg.split_mode}"
          + (f" fold {cfg.fold + 1}/{cfg.n_folds}" if cfg.split_mode == "rolling" else "")
          + f" | {len(ctx.names)} static channels {ctx.names}")
    describe_splits(ctx)
    describe_folds(cfg, data)
    print("  (if the test block's persistence error is far below the train block's, the event is already over "
          "there - run all folds and report the spread, never one fold)")
    print(f"normalisation (train frames [0, {ctx.train_hi}) only):",
          {k: (round(a, 3), round(b, 3)) for k, (a, b) in ctx.norm.items()})

    # reference baselines
    ref = eval_reference("test", ctx)
    results_table(ref, title=f"reference baselines | TEST fold {cfg.fold} | {len(ctx.splits['test'])} windows"
                             f" | lead {cfg.seq_out * cfg.step_frames * data.dt / 3600:.1f} h\n")
    save_json(os.path.join(save_dir, f"baselines_{cfg.region}_test_fold{cfg.fold}.json"), ref)
    print_paper_comparison(cfg, data)

    # every requested model must build and run before any training time is spent
    models = smoke_test_models(cfg, ctx, requested)
    if models != requested:
        print(f"NOTE: continuing with {models}; dropped {tuple(m for m in requested if m not in models)}")

    # run
    if args.mode == "check":
        n = len(folds) * len(args.seeds) * len(models)
        print(f"\nRUN_MODE = 'check': caches, pipeline checks, reference baselines and the model smoke test are "
              f"done; nothing was trained.")
        print(f"Next: --mode single trains all {len(models)} models on fold {args.fold}, seed {args.seed}. "
              f"The sweep is {n} runs ({len(models)} models x folds {folds} x seeds {tuple(args.seeds)}) - "
              f"time one single run and multiply.")
    elif args.mode == "single":
        run_compare(cfg, ctx, ref, models)
    else:
        del ctx
        free_gpu()
        run_sweep(base, data, folds, tuple(args.seeds), models)


if __name__ == "__main__":
    main()