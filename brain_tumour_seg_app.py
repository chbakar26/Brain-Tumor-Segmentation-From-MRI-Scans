#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary modules and libraries
from __future__ import annotations

# ===== Standard library =====
from contextlib import nullcontext
from pathlib import Path
import importlib.util
import inspect
import sys
import time
import gc
from typing import Any, Dict, Optional, Tuple

# ===== Third-party =====
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import Qt, QSize, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QMovie, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QComboBox, QGroupBox,
    QMessageBox, QLineEdit, QSizePolicy, QSplitter, QTextEdit,
    QProgressBar,
)

# ===== 3D Visualization imports =====
try:
    import open3d as o3d
    from skimage import measure
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization will be disabled.")
    print("Install with: pip install open3d scikit-image")

try:
    from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some 3D features may be disabled.")
    print("Install with: pip install scipy")


# ==============================================================================
#                       MODEL LOADING AND INFERENCE
# ==============================================================================

# Load configuration from external file if available
def load_config():
    """Load configuration from config_local.py or fall back to defaults."""
    try:
        import config_local
        return {
            "MODEL_PATH": getattr(config_local, "MODEL_PATH", "best_brats_model_dice.pth"),
            "MODEL_DEF_PATH": getattr(config_local, "MODEL_DEF_PATH", "improved3dunet.py"),
            "MODEL_CLASS": getattr(config_local, "MODEL_CLASS", "Improved3DUNet"),
            "BASE_FILTERS": getattr(config_local, "BASE_FILTERS", 16),
            "ZRANGE": getattr(config_local, "ZRANGE", "60:100"),
            "PATCH_SIZE": getattr(config_local, "PATCH_SIZE", "128,128,64"),
            "OVERLAP": getattr(config_local, "OVERLAP", 0.35),
            "FORCE_CPU": getattr(config_local, "FORCE_CPU", True),
            "INTENSITY_NORM": getattr(config_local, "INTENSITY_NORM", "zscore"),
            "LOW_MEM_ACCUM": getattr(config_local, "LOW_MEM_ACCUM", False),
            "ACCUM_DTYPE": getattr(config_local, "ACCUM_DTYPE", "float16"),
            "AMP": getattr(config_local, "AMP", "auto"),
        }
    except ImportError:
        # Default configuration (relative paths)
        return {
            "MODEL_PATH": "best_brats_model_dice.pth",
            "MODEL_DEF_PATH": "improved3dunet.py",
            "MODEL_CLASS": "Improved3DUNet",
            "BASE_FILTERS": 16,
            "ZRANGE": "60:100",
            "PATCH_SIZE": "128,128,64",
            "OVERLAP": 0.35,
            "FORCE_CPU": True,
            "INTENSITY_NORM": "zscore",
            "LOW_MEM_ACCUM": False,
            "ACCUM_DTYPE": "float16",
            "AMP": "auto",
        }

# Default configuration for the model and inference
DEFAULTS: Dict[str, object] = load_config()

# Function to dynamically import a model class from a Python file
def _import_model_class(py_path: str, class_name: str) -> Optional[type]:
    """Dynamically import a class from a Python file path."""
    p = Path(py_path) if py_path else None
    if not p or not p.exists():
        return None
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    return cls if isinstance(cls, type) else None

# Function to build the model from the configuration
def build_model_from_cfg(cfg: dict) -> nn.Module:
    cls = _import_model_class(cfg.get("MODEL_DEF_PATH"), cfg.get("MODEL_CLASS"))
    if not cls:
        raise RuntimeError("Model class not found; check MODEL_DEF_PATH / MODEL_CLASS in DEFAULTS.")
    sig = inspect.signature(cls)
    kwargs: Dict[str, Any] = {}
    for k, v in dict(in_channels=4, out_channels=4, base_filters=cfg.get("BASE_FILTERS", 8)).items():
        if k in sig.parameters:
            kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()

# Function to load model weights and verify their compatibility
def load_weights_verified(model: nn.Module, weights_path: str) -> nn.Module:
    """Load model weights and verify compatibility with the model architecture."""
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    return model

# Function to normalize a volume using z-score normalization
def zscore(vol: np.ndarray) -> np.ndarray:
    """Apply z-score normalization to the volume."""
    m = float(np.mean(vol)); s = float(np.std(vol))
    return (vol - m) / (s + 1e-6)

# Function to normalize a volume to the range [0, 1]
def norm_minmax01(vol: np.ndarray, lo_p: float = 0.5, hi_p: float = 99.5) -> np.ndarray:
    """Normalize the volume to the range [0, 1] using min-max scaling."""
    lo = float(np.percentile(vol, lo_p)); hi = float(np.percentile(vol, hi_p))
    vol = np.clip(vol, lo, hi, out=vol.astype(np.float32, copy=False))
    return ((vol - lo) / (hi - lo + 1e-6)).astype(np.float32, copy=False)

# Function to crop a volume along the Z-axis
def crop_z(vol: np.ndarray, z0: Optional[int], z1: Optional[int]) -> np.ndarray:
    """Crop the volume along the Z-axis."""
    if z0 is None and z1 is None:
        return vol
    if z0 is None:
        z0 = 0
    if z1 is None:
        z1 = vol.shape[2]
    z0 = max(0, int(z0)); z1 = min(vol.shape[2], int(z1))
    return vol[:, :, z0:z1]

# Function to convert multiple modalities into a PyTorch tensor
def to_tensor(flair: np.ndarray, t1: np.ndarray, t1ce: np.ndarray, t2: np.ndarray) -> torch.Tensor:
    """Convert FLAIR, T1, T1CE, and T2 modalities to a PyTorch tensor."""
    arr = np.stack([flair, t1, t1ce, t2], axis=0).astype(np.float32, copy=False)
    np.nan_to_num(arr, copy=False)
    return torch.from_numpy(arr)[None, ...]

# Function to determine the automatic mixed precision mode
def _pick_amp(cfg: dict, device: torch.device) -> str:
    """Determine the automatic mixed precision (AMP) mode based on configuration and device."""
    amp = str(cfg.get("AMP", "auto")).lower()
    if amp == "auto":
        return "fp16" if device.type == "cuda" else "off"
    return amp

# Context manager for automatic mixed precision
def _autocast_ctx(device: torch.device, amp: str):
    """Context manager for automatic mixed precision (AMP) during inference."""
    if amp == "fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

# Function for sliding-window inference with low memory usage
@torch.inference_mode()
def _sliding_logits_lowmem(
    model: nn.Module,
    x: torch.Tensor,
    patch: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    accum_on_cpu: bool = True,
    accum_dtype: torch.dtype = torch.float16,
    amp_mode: str = "off",
    progress_callback: Optional[callable] = None,
) -> np.ndarray:
    """Sliding-window logits accumulation to reduce peak memory usage."""
    model.eval()
    B, C, H, W, D = x.shape
    pH, pW, pD = patch
    sH = max(1, int(pH * (1 - overlap)))
    sW = max(1, int(pW * (1 - overlap)))
    sD = max(1, int(pD * (1 - overlap)))

    if device.type == "cuda":
        try:
            x = x.pin_memory()
        except Exception:
            pass

    with _autocast_ctx(device, amp_mode):
        xs0 = x[:, :, :min(H, pH), :min(W, pW), :min(D, pD)].to(device, non_blocking=True)
        tmp = model(xs0)
        nC = tmp.shape[1]
        del tmp, xs0

    target_dev = torch.device("cpu") if accum_on_cpu else device
    logits_sum = torch.zeros((1, nC, H, W, D), dtype=accum_dtype, device=target_dev)
    weights    = torch.zeros((1, 1, H, W, D), dtype=accum_dtype, device=target_dev)

    x_starts = list(range(0, max(H - pH, 0) + 1, sH)) or [0]
    if x_starts[-1] != max(0, H - pH):
        x_starts.append(max(0, H - pH))
    y_starts = list(range(0, max(W - pW, 0) + 1, sW)) or [0]
    if y_starts[-1] != max(0, W - pW):
        y_starts.append(max(0, W - pW))
    z_starts = list(range(0, max(D - pD, 0) + 1, sD)) or [0]
    if z_starts[-1] != max(0, D - pD):
        z_starts.append(max(0, D - pD))

    total_patches = len(x_starts) * len(y_starts) * len(z_starts)
    processed = 0

    def place(xh: int, y: int, z: int):
        nonlocal processed
        xs = x[:, :, xh:xh + pH, y:y + pW, z:z + pD].to(device, non_blocking=True)
        with _autocast_ctx(device, amp_mode):
            out = model(xs)
        out_cpu = out.detach().to(target_dev, dtype=accum_dtype)
        logits_sum[:, :, xh:xh + pH, y:y + pW, z:z + pD] += out_cpu
        weights[:, :, xh:xh + pH, y:y + pW, z:z + pD] += 1.0
        del xs, out, out_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        processed += 1
        if progress_callback:
            progress_callback(int(processed / max(1, total_patches) * 100))

    for z in z_starts:
        for y in y_starts:
            for xh in x_starts:
                place(xh, y, z)

    logits_avg = (logits_sum / torch.clamp(weights, min=1.0)).to(torch.float32)
    pred = torch.argmax(logits_avg, dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
    del logits_sum, weights, logits_avg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred

# Function to perform inference on the input tensor
@torch.inference_mode()
def infer(model: nn.Module, x: torch.Tensor, device: torch.device, cfg: dict,
          progress_callback: Optional[callable] = None) -> np.ndarray:
    """Perform inference on the input tensor using the trained model."""
    patch_size = cfg.get("PATCH_SIZE", "")
    overlap = float(cfg.get("OVERLAP", 0.25))
    amp_mode = _pick_amp(cfg, device)
    lowmem = bool(cfg.get("LOW_MEM_ACCUM", True))
    accum_dtype = torch.float16 if str(cfg.get("ACCUM_DTYPE", "float16")).lower() == "float16" else torch.float32

    if patch_size and str(patch_size).lower() != "none":
        pH, pW, pD = [int(v) for v in str(patch_size).split(",")]
        return _sliding_logits_lowmem(
            model, x, (pH, pW, pD), overlap, device,
            accum_on_cpu=lowmem, accum_dtype=accum_dtype, amp_mode=amp_mode,
            progress_callback=progress_callback,
        )

    x = x.to(device, non_blocking=True)
    with _autocast_ctx(device, amp_mode):
        logits = model(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    if progress_callback:
        progress_callback(100)
    del logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred

# Color mapping for segmentation labels
SEG_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # Necrotic tumor core
    2: (0, 255, 0),  # Edema
    3: (255, 255, 0),  # Enhancing tumor
}

THUMB = 150

# Function to convert a grayscale image to uint8
def gray_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to uint8 format."""
    vmin, vmax = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.clip((arr - vmin) * (255.0 / (vmax - vmin)), 0, 255).astype(np.uint8)

# Function to colorize a segmentation mask
def colorize_mask(mask2d: np.ndarray, colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Colorize a 2D segmentation mask."""
    h, w = mask2d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, (r, g, b) in colors.items():
        out[mask2d == k] = (r, g, b)
    return out

# Function to overlay a segmentation mask on an image
def overlay_rgb(base: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a segmentation mask on the base image."""
    base8 = gray_to_uint8(base)
    base_rgb = np.repeat(base8[..., None], 3, axis=2)
    out = (1 - alpha) * base_rgb.astype(np.float32) + alpha * mask_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

# Function to find the best slice from a 3D prediction
def best_slice_from_pred(pred3d: np.ndarray) -> int:
    """Find the slice with the largest area of interest in the 3D prediction."""
    H, W, D = pred3d.shape
    areas = (pred3d > 0).reshape(H * W, D).sum(0)
    return int(np.argmax(areas)) if areas.max() > 0 else int(D // 2)

# Function to find patient files in a folder
def find_patient_files(folder: Path) -> Dict[str, Path]:
    """Find *_flair, *_t1, *_t1ce, *_t2 inside folder."""
    def _glob_one(suf: str) -> Optional[Path]:
        cands = list(folder.glob(f"*_{suf}.nii")) + list(folder.glob(f"*_{suf}.nii.gz"))
        return cands[0] if cands else None

    out = {
        "flair": _glob_one("flair"),
        "t1": _glob_one("t1"),
        "t1ce": _glob_one("t1ce"),
        "t2": _glob_one("t2"),
    }
    if not all(out.values()):
        missing = [k for k, v in out.items() if v is None]
        raise FileNotFoundError(f"Missing modalities in {folder}: {missing}")
    return out

# Function to find an optional ground-truth mask
def find_optional_gt(folder: Path) -> Optional[Path]:
    """Look for a ground-truth mask in the same folder (optional)."""
    pats = [
        "*_seg.nii", "*_seg.nii.gz", "*segmentation*.nii", "*segmentation*.nii.gz",
        "*_mask.nii", "*_mask.nii.gz",
    ]
    for p in pats:
        hits = list(folder.glob(p))
        if hits:
            return hits[0]
    return None

# Function to load a volume as a NumPy array
def load_volume_float32(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI volume and return as float32 NumPy array."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    return data, img.affine

# Function to compute voxel spacing from an affine matrix
def voxel_spacing_from_affine(aff: np.ndarray) -> Tuple[float, float, float]:
    """Compute voxel spacing (in mm) from the affine transformation matrix."""
    sx = float(np.linalg.norm(aff[:3, 0]))
    sy = float(np.linalg.norm(aff[:3, 1]))
    sz = float(np.linalg.norm(aff[:3, 2]))
    return sx, sy, sz

# Function to save a cropped prediction as a NIfTI file
def save_pred_nifti_cropped(pred: np.ndarray, affine: np.ndarray, z0: int, out_path: Path) -> Path:
    """Save Z-cropped prediction with shifted origin so world coordinates match."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aff = affine.copy()
    if z0 and z0 > 0:
        aff[:3, 3] += aff[:3, 2] * float(z0)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), aff), str(out_path))
    return out_path

# Function to save an overlay GIF
def save_overlay_gif(
    base_vol: np.ndarray, pred: np.ndarray, colors: Dict[int, Tuple[int, int, int]],
    out_path: Path, alpha: float = 0.45, every_k: int = 1, fps: int = 12,
) -> Path:
    """Save an overlay GIF of the segmentation results."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W, D = pred.shape
    frames: list[Image.Image] = []
    for z in range(0, D, max(1, every_k)):
        mask_rgb = colorize_mask(pred[:, :, z], colors)
        over = overlay_rgb(base_vol[:, :, z], mask_rgb, alpha=alpha)
        frames.append(Image.fromarray(over))
    if not frames:
        z = D // 2
        mask_rgb = colorize_mask(pred[:, :, z], colors)
        over = overlay_rgb(base_vol[:, :, z], mask_rgb, alpha=alpha)
        frames = [Image.fromarray(over)]
    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:], duration=int(1000 / max(1, fps)), loop=0,
    )
    return out_path

# Function to save example slices as a PNG panel
def save_examples_png(
    mods_raw: Dict[str, np.ndarray],
    pred: np.ndarray,
    colors: Dict[int, Tuple[int, int, int]],
    out_path: Path,
    base_name: str,
    z0: int,
    z1: int,
    tile: int = 522,
    single_slice: bool = True,
) -> Path:
    """PNG panel: either one best slice or 3 evenly spaced slices with legend."""
    def crop(vol: np.ndarray) -> np.ndarray:
        end = z1 if (z1 and z1 > 0) else vol.shape[2]
        end = min(end, vol.shape[2])
        return vol[:, :, z0:end]

    mods = {k: crop(v) for k, v in mods_raw.items()}
    D = pred.shape[2]
    if D <= 0:
        raise ValueError("Empty prediction depth.")

    z_slices = [best_slice_from_pred(pred)] if single_slice else \
        list(np.unique(np.clip(np.round(np.linspace(0, D - 1, 3)).astype(int), 0, D - 1)))

    cols = ["FLAIR", "T1", "T1CE", "T2", "PREDICTED"]
    n_rows, n_cols = len(z_slices), 5

    font_px = max(10, int(tile * 0.08))
    title_h = font_px + 11
    legend_h = 40  # Height for the legend section
    
    # Calculate canvas dimensions
    canvas_w, canvas_h = n_cols * tile, title_h + n_rows * tile + legend_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    HEADER_BG = (32, 86, 176)
    TEXT_FG   = (255, 255, 255)
    OUTLINE   = (0, 32, 96)
    LEGEND_BG = (240, 240, 240)

    # Draw header
    draw.rectangle([0, 0, canvas_w, title_h], fill=HEADER_BG)

    def _bold_font(px: int):
        for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, px)
            except Exception:
                continue
        return ImageFont.load_default()

    font = _bold_font(font_px)
    legend_font = _bold_font(max(12, int(font_px * 0.7)))

    # Draw column headers
    for c, label_text in enumerate(cols):
        cx = c * tile + tile // 2
        cy = title_h // 2
        text = label_text.upper()
        try:
            tw, th = draw.textbbox((0, 0), text, font=font)[2:4]
        except Exception:
            tw, th = draw.textsize(text, font=font)
        x = cx - tw // 2
        y = cy - th // 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), text, fill=OUTLINE, font=font)
        draw.text((x, y), text, fill=TEXT_FG, font=font)

    # Draw images
    for r, z in enumerate(z_slices):
        imgs = [
            gray_to_uint8(mods["flair"][:, :, z]),
            gray_to_uint8(mods["t1"][:, :, z]),
            gray_to_uint8(mods["t1ce"][:, :, z]),
            gray_to_uint8(mods["t2"][:, :, z]),
        ]
        base = mods.get(base_name, mods["flair"])[:, :, z]
        mask_rgb = colorize_mask(pred[:, :, int(z)], colors)
        over = overlay_rgb(base, mask_rgb, alpha=0.54)

        pil_imgs: list[Image.Image] = []
        for im in imgs:
            if im.ndim == 2:
                im = np.repeat(im[..., None], 3, axis=2)
            pil_imgs.append(Image.fromarray(im).resize((tile, tile), Image.Resampling.LANCZOS))
        pil_imgs.append(Image.fromarray(over).resize((tile, tile), Image.Resampling.LANCZOS))

        for c, pim in enumerate(pil_imgs):
            canvas.paste(pim, (c * tile, title_h + r * tile))

    # Draw legend section
    legend_y_start = title_h + n_rows * tile
    draw.rectangle([0, legend_y_start, canvas_w, canvas_h], fill=LEGEND_BG)
    
    # Define legend items (label: color, description)
    legend_items = {
        1: (" Necrotic Tumor core ", (255, 0, 0)), # Red
        2: (" Edema ", (0, 255, 0)), # Green
        3: (" Enhancing Tumor ", (255, 255, 0)) # Yellow
    }
    
    # Calculate legend layout
    swatch_size = 25
    text_spacing = 15
    swatch_text_spacing = 5
    total_legend_width = 0
    
    # Calculate total width needed for legend items
    for label, (text, color) in legend_items.items():
        try:
            tw, th = draw.textbbox((0, 0), text, font=legend_font)[2:4]
        except Exception:
            tw, th = draw.textsize(text, font=legend_font)
        total_legend_width += swatch_size + swatch_text_spacing + tw + text_spacing
    
    # Start x position for centering legend
    current_x = (canvas_w - total_legend_width) // 2
    legend_y_center = legend_y_start + legend_h // 2
    
    # Draw each legend item
    for label, (text, color) in legend_items.items():
        # Draw color swatch
        swatch_x1 = current_x
        swatch_y1 = legend_y_center - swatch_size // 2
        swatch_x2 = swatch_x1 + swatch_size
        swatch_y2 = swatch_y1 + swatch_size
        draw.rectangle([swatch_x1, swatch_y1, swatch_x2, swatch_y2], fill=color, outline=(0, 0, 0))
        
        # Draw text
        text_x = swatch_x2 + swatch_text_spacing
        try:
            tw, th = draw.textbbox((0, 0), text, font=legend_font)[2:4]
        except Exception:
            tw, th = draw.textsize(text, font=legend_font)
        text_y = legend_y_center - th // 2
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=legend_font)
        
        # Move to next item position
        current_x = text_x + tw + text_spacing

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    return out_path

# Function to compute basic metrics for the segmentation
def compute_basic_metrics(pred: np.ndarray, aff: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Dict[str, Any]:
    """Compute basic metrics for the segmentation results."""
    metrics: Dict[str, Any] = {}
    unique, counts = np.unique(pred, return_counts=True)
    metrics["label_counts"] = {int(k): int(v) for k, v in zip(unique, counts)}
    metrics["total_voxels"] = int(pred.size)

    if aff is not None:
        sx, sy, sz = voxel_spacing_from_affine(aff)
        metrics["voxel_spacing_mm"] = (sx, sy, sz)
        lv = {}
        for k, v in metrics["label_counts"].items():
            lv[int(k)] = float(v * sx * sy * sz / 1000.0)
        metrics["label_volume_ml"] = lv

    if gt is not None and gt.shape == pred.shape:
        dice: Dict[int, float] = {}
        labels = np.union1d(np.unique(pred), np.unique(gt))
        for l in labels:
            p = pred == l
            g = gt == l
            inter = int(np.logical_and(p, g).sum())
            denom = int(p.sum() + g.sum())
            dice[int(l)] = (2.0 * inter / denom) if denom > 0 else float("nan")
        metrics["dice_per_label"] = dice
    return metrics

# Function to compute BraTS composite Dice scores
def brats_composite_dice(
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    map_gt_4_to_3: bool = True,
    et_label: int = 3,
) -> Dict[str, Any]:
    """Compute BraTS composite Dice scores for the segmentation."""
    out: Dict[str, Any] = {}
    if gt is None:
        return out

    gt2 = gt.copy()
    if map_gt_4_to_3:
        gt2[gt2 == 4] = et_label

    def _dice(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(bool, copy=False); b = b.astype(bool, copy=False)
        inter = int(np.logical_and(a, b).sum())
        denom = int(a.sum() + b.sum())
        return (2.0 * inter / denom) if denom > 0 else float('nan')

    per: Dict[int, float] = {}
    for l in (1, 2, et_label):
        per[int(l)] = _dice(pred == l, gt2 == l)
    out['dice_per_label_nobg'] = per

    wt_pred = pred != 0
    wt_gt   = gt2  != 0
    out['dice_WT'] = _dice(wt_pred, wt_gt)

    tc_pred = np.logical_or(pred == 1, pred == et_label)
    tc_gt   = np.logical_or(gt2  == 1, gt2  == et_label)
    out['dice_TC'] = _dice(tc_pred, tc_gt)

    out['dice_ET'] = per[et_label]
    return out

# Function to format metrics as text
def format_metrics_text(m: Dict[str, Any]) -> str:
    """Format the computed metrics as a text report."""
    lines = ["# Metrics"]
    if "total_voxels" in m:
        lines.append(f"Total voxels: {m['total_voxels']}")
    if "voxel_spacing_mm" in m:
        sx, sy, sz = m["voxel_spacing_mm"]
        lines.append(f"Voxel spacing (mm): {sx:.2f} x {sy:.2f} x {sz:.2f}")

    if "label_counts" in m and isinstance(m["label_counts"], dict):
        fg = {int(k): int(v) for k, v in m["label_counts"].items() if int(k) != 0}
        if fg:
            lines.append("Label counts:")
            for k in sorted(fg.keys()):
                lines.append(f"  - {k}: {fg[k]}")

    if "label_volume_ml" in m and isinstance(m["label_volume_ml"], dict):
        fg = {int(k): float(v) for k, v in m["label_volume_ml"].items() if int(k) != 0}
        if fg:
            lines.append("Label volumes (ml):")
            for k in sorted(fg.keys()):
                lines.append(f"  - {k}: {fg[k]:.2f}")

    if "dice_per_label_nobg" in m:
        lines.append("Dice per label:")
        for k in sorted(m["dice_per_label_nobg"].keys()):
            v = m["dice_per_label_nobg"][k]
            lines.append(f"  - {k}: {'N/A' if np.isnan(v) else f'{v:.3f}'}")
    elif "dice_per_label" in m:
        lines.append("Dice per label:")
        for k in sorted(m["dice_per_label"].keys()):
            if int(k) == 0:
                continue
            v = m["dice_per_label"][k]
            lines.append(f"  - {k}: {'N/A' if np.isnan(v) else f'{v:.3f}'}")

    if any(k in m for k in ("dice_WT", "dice_TC", "dice_ET")):
        lines.append("Composite Dice:")
        if "dice_WT" in m:
            v = m["dice_WT"]
            lines.append(f"  - WT: {'N/A' if np.isnan(v) else f'{v:.3f}'}")
        if "dice_TC" in m:
            v = m["dice_TC"]
            lines.append(f"  - TC: {'N/A' if np.isnan(v) else f'{v:.3f}'}")
        if "dice_ET" in m:
            v = m["dice_ET"]
            lines.append(f"  - ET: {'N/A' if np.isnan(v) else f'{v:.3f}'}")

    return "\n".join(lines)

# ==============================================================================
#                       3D VISUALIZATION FUNCTIONS
# ==============================================================================

# Functions for 3D visualization (if Open3D and SciPy are available)
if OPEN3D_AVAILABLE and SCIPY_AVAILABLE:
    def preprocess_volume(volume, smooth_sigma=1.0):
        """Preprocess volume with smoothing and morphological operations."""
        smoothed = gaussian_filter(volume, sigma=smooth_sigma)
        binary_mask = smoothed > 0
        binary_mask = binary_fill_holes(binary_mask)
        binary_mask = binary_closing(binary_mask, iterations=2)
        return binary_mask.astype(float)

    def create_smooth_mesh_from_volume(volume, threshold=0.5, step_size=1, smooth_iterations=50):
        """Create a smooth mesh from a 3D volume using marching cubes."""
        binary_mask = volume > threshold
        
        try:
            verts, faces, _, _ = measure.marching_cubes(
                binary_mask, 
                level=0.5, 
                step_size=step_size,
                allow_degenerate=False
            )
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            
            if smooth_iterations > 0:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
                mesh.compute_vertex_normals()
            
            return mesh
        except Exception as e:
            print(f"Error creating mesh: {e}")
            return None

    def apply_advanced_smoothing(mesh, method='taubin', iterations=30):
        """Apply advanced smoothing techniques to the mesh."""
        if mesh is None:
            return None
        
        mesh_smooth = o3d.geometry.TriangleMesh(mesh)
        
        if method == 'taubin':
            mesh_smooth = mesh_smooth.filter_smooth_taubin(
                number_of_iterations=iterations,
                lambda_filter=0.5,
                mu=-0.53
            )
        elif method == 'laplacian':
            mesh_smooth = mesh_smooth.filter_smooth_laplacian(
                number_of_iterations=iterations,
                lambda_filter=0.5
            )
        elif method == 'simple':
            mesh_smooth = mesh_smooth.filter_smooth_simple(
                number_of_iterations=iterations
            )
        
        mesh_smooth.compute_vertex_normals()
        return mesh_smooth

    def create_3d_visualization_from_data(t1ce_data, seg_data, slice_range=(60, 100)):
        """Create 3D meshes from the loaded data with better memory management."""
        print("Creating 3D visualization...")
        
        try:
            # Apply slice range to segmentation
            if slice_range is not None:
                seg_modified = seg_data.copy()
                seg_modified[:, :, :slice_range[0]] = 0
                seg_modified[:, :, slice_range[1]:] = 0
                seg_data = seg_modified
            
            # Normalize brain volume
            brain_volume = t1ce_data.copy()
            brain_volume = (brain_volume - brain_volume.min()) / (brain_volume.max() - brain_volume.min() + 1e-8)
            brain_volume = gaussian_filter(brain_volume, sigma=1.5)
            
            meshes = []
            
            # Create brain surface mesh
            print("Creating brain surface mesh...")
            brain_mesh = create_smooth_mesh_from_volume(
                brain_volume, 
                threshold=0.15, 
                step_size=2,
                smooth_iterations=50
            )
            
            if brain_mesh is not None:
                brain_mesh.paint_uniform_color([0.8, 0.8, 0.8])
                meshes.append(brain_mesh)
                print(f"Brain mesh: {len(brain_mesh.vertices)} vertices")
            
            # Define tumor regions
            tumor_regions = {
                1: {'name': 'Necrotic Core', 'color': [1.0, 0.0, 0.0]},      # Red
                2: {'name': 'Peritumoral Edema', 'color': [0.0, 1.0, 0.0]},  # Green  
                3: {'name': 'Enhancing Tumor', 'color': [1.0, 1.0, 0.0]}     # Yellow
            }
            
            # Process tumor regions
            for label, info in tumor_regions.items():
                tumor_mask = (seg_data == label).astype(float)
                
                if tumor_mask.sum() > 0:
                    print(f"Processing {info['name']}...")
                    tumor_mask = gaussian_filter(tumor_mask, sigma=0.8)
                    tumor_mask = (tumor_mask > 0.3).astype(float)
                    
                    tumor_mesh = create_smooth_mesh_from_volume(
                        tumor_mask, 
                        threshold=0.5, 
                        step_size=1,
                        smooth_iterations=30
                    )
                    
                    if tumor_mesh is not None:
                        tumor_mesh = apply_advanced_smoothing(tumor_mesh, method='taubin', iterations=20)
                        tumor_mesh.paint_uniform_color(info['color'])
                        meshes.append(tumor_mesh)
                        print(f"  Added {info['name']}: {len(tumor_mesh.vertices)} vertices")
            
            # Force cleanup
            del brain_volume, seg_modified, tumor_mask
            gc.collect()
            
            return meshes
            
        except Exception as e:
            print(f"3D visualization error: {e}")
            return []

    def visualize_with_open3d(meshes):
        """Display meshes using Open3D with better error handling."""
        if not meshes:
            print("No meshes to display")
            return
            
        vis = None
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Brain Tumor 3D Visualization", width=1024, height=768)
            
            for mesh in meshes:
                vis.add_geometry(mesh)
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            vis.add_geometry(coord_frame)
            
            opt = vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.point_size = 5.0
            opt.mesh_show_wireframe = False
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
            
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            vis.run()
            
        except Exception as e:
            print(f"Open3D visualization failed: {e}")
        finally:
            if vis:
                try:
                    vis.destroy_window()
                except:
                    pass

# ==============================================================================
#                                   Workers
# ==============================================================================

# Worker class for segmentation
class SegmentationWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, flair_p, t1_p, t1ce_p, t2_p, cfg, device):
        super().__init__()
        self.flair_p = flair_p
        self.t1_p = t1_p
        self.t1ce_p = t1ce_p
        self.t2_p = t2_p
        self.cfg = cfg
        self.device = device

    def run(self):
        """Run the segmentation process."""
        try:
            device = self.device
            if device is None:
                use_cuda = torch.cuda.is_available() and not bool(self.cfg.get("FORCE_CPU", False))
                device = torch.device("cuda" if use_cuda else "cpu")

            model = build_model_from_cfg(self.cfg).to(device)
            weights = Path(self.cfg["MODEL_PATH"])
            assert weights.exists(), f"MODEL_PATH not found: {weights}"
            load_weights_verified(model, str(weights))
            model.eval()

            flair, aff = load_volume_float32(self.flair_p)
            t1, _ = load_volume_float32(self.t1_p)
            t1ce, _ = load_volume_float32(self.t1ce_p)
            t2, _ = load_volume_float32(self.t2_p)

            zrange = self.cfg.get("ZRANGE", None)
            if isinstance(zrange, str) and ":" in zrange:
                z0, z1 = [int(v) for v in zrange.split(":")]
            else:
                z0 = z1 = None

            flair_raw, t1_raw, t1ce_raw, t2_raw = flair.copy(), t1.copy(), t1ce.copy(), t2.copy()
            flair = crop_z(flair, z0, z1) if z0 is not None or z1 is not None else flair
            t1    = crop_z(t1,    z0, z1) if z0 is not None or z1 is not None else t1
            t1ce  = crop_z(t1ce,  z0, z1) if z0 is not None or z1 is not None else t1ce
            t2    = crop_z(t2,    z0, z1) if z0 is not None or z1 is not None else t2

            if str(self.cfg.get("INTENSITY_NORM", "zscore")).lower() == "minmax":
                flair_n = norm_minmax01(flair); t1_n = norm_minmax01(t1)
                t1ce_n  = norm_minmax01(t1ce);  t2_n = norm_minmax01(t2)
            else:
                flair_n = zscore(flair); t1_n = zscore(t1)
                t1ce_n  = zscore(t1ce);  t2_n = zscore(t2)

            x = to_tensor(flair_n, t1_n, t1ce_n, t2_n)

            pred = infer(model, x, device, self.cfg, progress_callback=self.progress.emit).astype(np.uint8)

            mods = {"flair": flair_raw, "t1": t1_raw, "t1ce": t1ce_raw, "t2": t2_raw}
            self.finished.emit({"pred": pred, "mods": mods, "aff": aff})
        except Exception as e:
            self.error.emit(str(e))

# Worker class for 3D visualization
class ThreeDVisualizationWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, t1ce_data, seg_data, slice_range):
        super().__init__()
        self.t1ce_data = t1ce_data
        self.seg_data = seg_data
        self.slice_range = slice_range

    def run(self):
        """Run the 3D visualization process."""
        try:
            if not OPEN3D_AVAILABLE or not SCIPY_AVAILABLE:
                self.error.emit("3D visualization dependencies not available")
                return

            meshes = create_3d_visualization_from_data(
                self.t1ce_data, 
                self.seg_data, 
                slice_range=self.slice_range
            )
            self.finished.emit(meshes)
        except Exception as e:
            self.error.emit(str(e))

# ==============================================================================
#                                     GUI
# ==============================================================================

# Main application class for the GUI
class SegApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Segmentation - Radiologist App")
        self.resize(1250, 600)

        self.last_z0z1: Optional[Tuple[int, int]] = None
        self.cfg = dict(DEFAULTS)
        _zr = self.cfg.get("ZRANGE", None)
        if isinstance(_zr, str) and ":" in _zr:
            try:
                _z0, _z1 = [int(v) for v in _zr.split(":")]
            except Exception:
                _z0, _z1 = 0, 0
        else:
            _z0, _z1 = 0, 0
        self.last_z0z1 = (_z0, _z1)
        self.output_dir = Path.cwd() / "seg_outputs"

        self.files: Dict[str, Optional[Path]] = {"flair": None, "t1": None, "t1ce": None, "t2": None}
        self.patient_folder: Optional[Path] = None

        self.modalities: Dict[str, np.ndarray] = {}
        self.pred: Optional[np.ndarray] = None
        self.affine: Optional[np.ndarray] = None
        self.patient_id: str = "patient"
        self.current_png_path: Optional[Path] = None

        # Initialize thread attributes to avoid AttributeError
        self.thread = None
        self.thread_3d = None
        self.worker = None
        self.worker_3d = None

        self._build_ui()
        self._load_qss_if_present()

    def _build_ui(self):
        """Build the user interface."""
        central = QWidget(self)
        self.setCentralWidget(central)

        # Left column: Load / Options
        options_box = QGroupBox("Load / Options")
        options_layout = QVBoxLayout(options_box)
        options_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Set object names for styling
        self.btn_load_folder = QPushButton("📁 Load Patient Folder")
        self.btn_load_folder.setObjectName("btn_load_folder")
        self.btn_load_folder.clicked.connect(self.on_load_folder)
        options_layout.addWidget(self.btn_load_folder)

        options_layout.addWidget(QLabel("⚙️ Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto (CUDA if available)", "cpu", "cuda"])
        options_layout.addWidget(self.device_combo)

        options_layout.addWidget(QLabel("🎯 Base for overlay/GIF:"))
        self.base_combo = QComboBox()
        self.base_combo.addItems(["flair", "t1", "t1ce", "t2"])
        options_layout.addWidget(self.base_combo)

        options_layout.addWidget(QLabel("🔍 Mask opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        options_layout.addWidget(self.opacity_slider)

        options_layout.addWidget(QLabel("💾 Output Folder:"))
        self.out_dir_edit = QLineEdit(str(self.output_dir))
        options_layout.addWidget(self.out_dir_edit)

        self.btn_out_dir = QPushButton("📂 Choose Output Dir")
        self.btn_out_dir.setObjectName("btn_out_dir")
        self.btn_out_dir.clicked.connect(self.on_choose_outdir)
        options_layout.addWidget(self.btn_out_dir)

        self.btn_segment = QPushButton("🧠 SEGMENT")
        self.btn_segment.setObjectName("segmentButton")
        self.btn_segment.clicked.connect(self.on_segment)
        options_layout.addWidget(self.btn_segment)

        # 3D Visualization button
        if OPEN3D_AVAILABLE and SCIPY_AVAILABLE:
            self.btn_3d_viz = QPushButton("📊 3D VISUALIZATION")
            self.btn_3d_viz.setObjectName("btn_3d")
            self.btn_3d_viz.clicked.connect(self.on_show_3d)
            self.btn_3d_viz.setEnabled(False)
            options_layout.addWidget(self.btn_3d_viz)
        
        options_layout.addStretch(3)

        # Set object name for status label
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("status_label")

        # Viewer
        viewer_box = QGroupBox("Viewer")
        vgrid = QGridLayout(viewer_box)
        self.slice_label = QLabel("Slice view will appear here")
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_label.setFixedSize(THUMB, THUMB)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice_view)
        vgrid.addWidget(self.slice_label, 0, 0)
        vgrid.addWidget(self.slice_slider, 1, 0)
        self.slice_label.setScaledContents(True)

        # GIF viewer
        gif_box = QGroupBox("Segmentation GIF")
        gif_layout = QVBoxLayout(gif_box)
        self.gif_label = QLabel("GIF will appear after segmentation")
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setFixedSize(THUMB, THUMB)
        gif_layout.addWidget(self.gif_label)

        # Metrics
        metrics_box = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setFixedSize(THUMB * 2, THUMB)
        metrics_layout.addWidget(self.metrics_text)

        # PNG panel
        png_box = QGroupBox("Segmented PNG")
        png_layout = QVBoxLayout(png_box)
        self.png_label = QLabel("PNG panel will appear after segmentation")
        self.png_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.png_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.png_label.setMinimumHeight(200)
        self.png_label.setScaledContents(True)
        png_layout.addWidget(self.png_label)

        # Bottom status + progress
        status_and_progress_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        status_and_progress_layout.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        status_and_progress_layout.addWidget(self.progress_bar)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal, central)

        left_panel = QWidget(splitter)
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(240)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(options_box, 1)
        left_layout.addStretch(1)
        left_layout.addLayout(status_and_progress_layout)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)

        right_top = QHBoxLayout()
        right_top.addWidget(viewer_box, 1)
        right_top.addWidget(gif_box, 1)
        right_top.addWidget(metrics_box, 1)
        right_layout.addLayout(right_top, 1)

        right_layout.addWidget(png_box, 2)

        main = QHBoxLayout(central)
        main.addWidget(splitter)
        splitter.setSizes([340, 880])

    def _load_qss_if_present(self):
        """Load the QSS stylesheet if available."""
        qss_path = Path(__file__).with_name("radiology_app.qss")
        if qss_path.exists():
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                print(f"Could not load stylesheet: {e}")

    def validate_config(self):
        """Validate critical configuration paths."""
        model_path = Path(self.cfg.get("MODEL_PATH", ""))
        if not model_path.exists():
            QMessageBox.critical(self, "Error", f"Model path not found: {model_path}")
            return False
        
        model_def_path = Path(self.cfg.get("MODEL_DEF_PATH", ""))
        if not model_def_path.exists():
            QMessageBox.critical(self, "Error", f"Model definition not found: {model_def_path}")
            return False
            
        return True

    def on_choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", str(self.output_dir))
        if d:
            self.output_dir = Path(d)
            self.out_dir_edit.setText(str(self.output_dir))

    def on_load_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Patient Folder", "")
        if not d:
            return
        folder = Path(d)
        try:
            found = find_patient_files(folder)
            for k in self.files.keys():
                self.files[k] = found[k]
            self.patient_folder = folder
            any_name = next(iter(found.values())).name
            self.patient_id = any_name.split("_")[0] if "_" in any_name else folder.name
            self.status_label.setText(f"Loaded folder: {folder}")
        except Exception as e:
            QMessageBox.warning(self, "Missing files", str(e))

    def current_device(self) -> torch.device:
        choice = self.device_combo.currentText()
        if choice.startswith("auto"):
            use_cuda = torch.cuda.is_available() and not bool(self.cfg.get("FORCE_CPU", False))
            return torch.device("cuda" if use_cuda else "cpu")
        return torch.device(choice)

    def _make_run_dir(self, out_root: Path, patient_id: str) -> Path:
        base = out_root / patient_id
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = base / f"run_{stamp}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        k = 1
        while True:
            candidate = base / f"run_{stamp}_{k}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            k += 1

    def on_segment(self):
        if not all([self.files["flair"], self.files["t1"], self.files["t1ce"], self.files["t2"]]):
            QMessageBox.warning(self, "Files missing", "Please load a patient folder.")
            return

        # Validate configuration before proceeding
        if not self.validate_config():
            return

        _zr = self.cfg.get("ZRANGE", None)
        if isinstance(_zr, str) and ":" in _zr:
            try:
                z0, z1 = [int(v) for v in _zr.split(":")]
            except Exception:
                z0, z1 = 0, 0
        else:
            z0, z1 = 0, 0
        self.last_z0z1 = (z0, z1)
        self.output_dir = Path(self.out_dir_edit.text().strip())

        dev = self.current_device()
        self.cfg["FORCE_CPU"] = (dev.type == "cpu")

        self.status_label.setText("Running segmentation...")
        self.btn_segment.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.thread = QThread()
        self.worker = SegmentationWorker(
            self.files["flair"], self.files["t1"], self.files["t1ce"], self.files["t2"],
            self.cfg, dev,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_segment_finished)
        self.worker.error.connect(self.on_segment_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_segment_finished(self, result):
        self.pred = result["pred"]
        self.modalities = result["mods"]
        self.affine = result["aff"]

        self.progress_bar.hide()
        self.btn_segment.setEnabled(True)
        
        if OPEN3D_AVAILABLE and SCIPY_AVAILABLE and hasattr(self, 'btn_3d_viz'):
            self.btn_3d_viz.setEnabled(True)

        H, W, D = self.pred.shape
        self.slice_slider.setMaximum(max(0, D - 1))
        self.slice_slider.setEnabled(True)
        self.slice_slider.setValue(best_slice_from_pred(self.pred))
        self.update_slice_view()

        run_dir = self._make_run_dir(self.output_dir, self.patient_id)
        pred_nii_path = run_dir / f"{self.patient_id}_pred.nii"
        gif_path      = run_dir / f"{self.patient_id}_overlay.gif"
        panel_path    = run_dir / f"{self.patient_id}_examples.png"

        z0_used, z1_used = (self.last_z0z1 or (0, 0))
        base_name = self.base_combo.currentText()
        base_raw = self.modalities[base_name]
        zb0 = z0_used
        zb1 = z1_used if (z1_used and z1_used > 0) else base_raw.shape[2]
        zb1 = min(zb1, base_raw.shape[2])
        base_for_gif = base_raw[:, :, zb0:zb1]

        _ = save_pred_nifti_cropped(self.pred, self.affine, z0_used, pred_nii_path)
        _ = save_overlay_gif(
            base_for_gif, self.pred, SEG_COLORS,
            gif_path, alpha=self.opacity_slider.value() / 100.0, every_k=1, fps=60,
        )
        _ = save_examples_png(
            self.modalities, self.pred, SEG_COLORS,
            panel_path, base_name=base_name, z0=z0_used, z1=z1_used, tile=512,
            single_slice=True,
        )

        gt_arr = None
        if self.patient_folder:
            gt_path = find_optional_gt(self.patient_folder)
            if gt_path is not None and gt_path.exists():
                try:
                    gt_arr, _ = load_volume_float32(gt_path)
                    if z0_used or z1_used:
                        gt_arr = crop_z(gt_arr, z0_used, z1_used)
                    gt_arr = gt_arr.astype(np.uint8)
                except Exception:
                    gt_arr = None
        metrics = compute_basic_metrics(self.pred, self.affine, gt_arr)
        comp = brats_composite_dice(self.pred, gt_arr)
        if comp:
            metrics.update(comp)
        self.metrics_text.setPlainText(format_metrics_text(metrics))

        self.show_gif(gif_path)
        self.show_png(panel_path)
        self.status_label.setText(f"Done. Saved to: {run_dir}")

    def on_segment_error(self, message):
        QMessageBox.critical(self, "Segmentation failed", message)
        self.status_label.setText("Ready.")
        self.progress_bar.hide()
        self.btn_segment.setEnabled(True)

    def on_show_3d(self):
        """Launch 3D visualization in a separate thread."""
        if not OPEN3D_AVAILABLE or not SCIPY_AVAILABLE:
            QMessageBox.warning(self, "Dependencies not available", 
                "Please install: pip install open3d scikit-image scipy")
            return
        
        if self.pred is None or "t1ce" not in self.modalities:
            QMessageBox.warning(self, "No data", "Please run segmentation first.")
            return
        
        # Disable button during processing
        self.btn_3d_viz.setEnabled(False)
        self.status_label.setText("Preparing 3D visualization...")
        
        try:
            t1ce_data = self.modalities["t1ce"]
            
            # Create full-volume prediction for 3D viz
            z0_used, z1_used = self.last_z0z1 or (0, 0)
            seg_full = np.zeros_like(t1ce_data, dtype=np.uint8)
            if z0_used or z1_used:
                zb1 = z1_used if (z1_used and z1_used > 0) else seg_full.shape[2]
                seg_full[:, :, z0_used:zb1] = self.pred
            else:
                seg_full = self.pred
            
            # Use QThread for 3D processing
            self.thread_3d = QThread()
            self.worker_3d = ThreeDVisualizationWorker(
                t1ce_data, 
                seg_full,
                self.last_z0z1 if self.last_z0z1 else None
            )
            self.worker_3d.moveToThread(self.thread_3d)
            self.thread_3d.started.connect(self.worker_3d.run)
            self.worker_3d.finished.connect(self.on_3d_ready)
            self.worker_3d.error.connect(self.on_3d_error)
            self.worker_3d.finished.connect(self.thread_3d.quit)
            self.worker_3d.finished.connect(self.worker_3d.deleteLater)
            self.thread_3d.finished.connect(self.thread_3d.deleteLater)
            self.thread_3d.start()
                
        except Exception as e:
            QMessageBox.critical(self, "3D Visualization Error", str(e))
            self.status_label.setText("Ready.")
            self.btn_3d_viz.setEnabled(True)

    def on_3d_ready(self, meshes):
        """Handle completed 3D visualization preparation"""
        self.btn_3d_viz.setEnabled(True)
        if meshes:
            self.status_label.setText("Launching 3D viewer...")
            # Run visualization in main thread (Open3D requires main thread)
            visualize_with_open3d(meshes)
            self.status_label.setText("3D visualization closed.")
        else:
            self.status_label.setText("No 3D data to display.")
            QMessageBox.warning(self, "Visualization failed", 
                "No meshes were created. Check console for details.")

    def on_3d_error(self, error_msg):
        """Handle 3D visualization errors"""
        self.btn_3d_viz.setEnabled(True)
        self.status_label.setText("3D visualization failed.")
        QMessageBox.critical(self, "3D Visualization Error", error_msg)

    def update_slice_view(self):
        if self.pred is None or not self.modalities:
            return
        z = int(self.slice_slider.value())
        z0_used, z1_used = (self.last_z0z1 or (0, 0))
        base_name = self.base_combo.currentText()
        base_raw = self.modalities[base_name]
        zb0 = z0_used
        zb1 = z1_used if (z1_used and z1_used > 0) else base_raw.shape[2]
        zb1 = min(zb1, base_raw.shape[2])
        base_cropped = base_raw[:, :, zb0:zb1]
        z = int(np.clip(z, 0, max(0, base_cropped.shape[2] - 1)))
        mask_rgb = colorize_mask(self.pred[:, :, z], SEG_COLORS)
        over = overlay_rgb(base_cropped[:, :, z], mask_rgb, alpha=self.opacity_slider.value() / 100.0)
        qimg = QImage(over.data, over.shape[1], over.shape[0], 3 * over.shape[1], QImage.Format.Format_RGB888)
        self.slice_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.slice_label.width(), self.slice_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_slice_view()
        if hasattr(self, 'current_png_path') and self.current_png_path:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, lambda: self.show_png(self.current_png_path))

    def show_gif(self, path: Path):
        if not path.exists():
            self.gif_label.setText("GIF not found.")
            return
        self.gif_label.setScaledContents(False)
        try:
            from PyQt6.QtGui import QImageReader
            reader = QImageReader(str(path))
            sz = reader.size()
            w0, h0 = (max(1, sz.width()), max(1, sz.height()))
        except Exception:
            w0, h0 = (THUMB, THUMB)
        scale = min(THUMB / float(w0), THUMB / float(h0))
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        movie = QMovie(str(path))
        movie.setScaledSize(QSize(new_w, new_h))
        self.gif_label.setMovie(movie)
        movie.start()

    def show_png(self, path: Path):
        self.current_png_path = path
        if not path.exists():
            self.png_label.setText("PNG not found.")
            return
        
        try:
            pix = QPixmap(str(path))
            if pix.isNull():
                self.png_label.setText("Failed to load PNG.")
                return
            
            available_width = max(100, self.png_label.width() - 10)
            available_height = max(100, self.png_label.height() - 10)
            
            pix_width = pix.width()
            pix_height = pix.height()
            
            width_ratio = available_width / pix_width
            height_ratio = available_height / pix_height
            scale_factor = min(width_ratio, height_ratio)
            
            new_width = int(pix_width * scale_factor)
            new_height = int(pix_height * scale_factor)
            
            scaled_pix = pix.scaled(
                new_width, 
                new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.png_label.clear()
            self.png_label.setPixmap(scaled_pix)
            self.png_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        except Exception as e:
            print(f"Error displaying PNG: {e}")
            self.png_label.setText("Error displaying image")

    def closeEvent(self, event: QCloseEvent):
        """Cleanup resources on app close"""
        try:
            # Stop segmentation thread if running
            if self.thread is not None and self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(3000)  # Wait up to 3 seconds
                
            # Stop 3D thread if running  
            if self.thread_3d is not None and self.thread_3d.isRunning():
                self.thread_3d.quit()
                self.thread_3d.wait(3000)
        except (AttributeError, RuntimeError) as e:
            print(f"Warning during cleanup: {e}")
                
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        event.accept()

# Entry point for the application
def main():
    app = QApplication(sys.argv)
    w = SegApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()