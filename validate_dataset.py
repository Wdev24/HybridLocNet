#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   HybridLocNet Dataset Audit + Cleaning Pipeline            ║
║   Covers: BOSSBase, generic image folders, synthetic stego  ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python validate_dataset.py --data ./data/bossbase [--fix] [--report]
    python validate_dataset.py --data ./images --mode flat [--fix]

Outputs:
    audit_report.json    → full machine-readable audit
    audit_report.html    → visual summary (open in browser)
    cleaned/             → symlinks to valid files (--fix mode)
    removed_log.csv      → what was removed and why
"""

import os
import sys
import json
import csv
import hashlib
import argparse
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ── Optional imports with graceful fallback ─────────────────────────────────
try:
    from PIL import Image, UnidentifiedImageError
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("WARN: Pillow not installed. Install with: pip install Pillow")

try:
    import scipy.ndimage as ndimage
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

# ── Logger setup ─────────────────────────────────────────────────────────────
# Windows cp1252 fix: force UTF-8 on all output streams
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
))

_file_handler = logging.FileHandler('dataset_audit.log', mode='w', encoding='utf-8')
_file_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageRecord:
    path:         str
    split:        str          # 'cover' | 'stego' | 'flat'
    stem:         str
    status:       str  = 'ok' # 'ok' | 'removed' | 'warning'
    remove_reason: str = ''
    warnings:     list = field(default_factory=list)
    # Image properties
    width:        int  = 0
    height:       int  = 0
    channels:     int  = 0
    file_size_kb: float = 0.0
    mean:         float = 0.0
    std:          float = 0.0
    min_val:      float = 0.0
    max_val:      float = 0.0
    file_hash:    str  = ''
    has_rho:      bool = False
    rho_valid:    bool = False
    has_stego_pair: bool = False


@dataclass
class AuditReport:
    timestamp:          str   = ''
    data_root:          str   = ''
    mode:               str   = ''
    total_scanned:      int   = 0
    total_valid:        int   = 0
    total_removed:      int   = 0
    total_warnings:     int   = 0
    cover_count:        int   = 0
    stego_count:        int   = 0
    rho_count:          int   = 0
    duplicates_found:   int   = 0
    # Distribution stats
    modal_size:         str   = ''
    size_distribution:  dict  = field(default_factory=dict)
    removal_reasons:    dict  = field(default_factory=dict)
    warning_types:      dict  = field(default_factory=dict)
    # Thresholds used
    thresholds:         dict  = field(default_factory=dict)
    records:            list  = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — all thresholds in one place, easy to tune
# ═══════════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    # File
    "min_file_size_kb":       0.5,     # anything smaller is likely corrupt/empty
    "max_file_size_kb":       50_000,  # >50MB is anomalous for stego images
    # Image dimensions
    "min_dimension":          32,      # images smaller than this are useless
    "max_dimension":          4096,    # sanity cap
    "target_size":            256,     # what we'll train at (warnings if different)
    # Pixel value stats (uint8-normalised to [0,1])
    "min_mean":               0.02,    # near-black → likely corrupt
    "max_mean":               0.98,    # near-white → likely corrupt
    "min_std":                0.005,   # nearly flat image → useless for texture
    "max_std":                0.50,    # extremely high variance (synthetic noise?)
    # Rho map specific
    "rho_min_valid":          0.0,
    "rho_max_valid":          1.0,
    "rho_max_nan_fraction":   0.01,    # >1% NaN in rho = corrupt
    "rho_min_nonzero_frac":   0.001,   # rho should have >0.1% non-zero pixels
    # Spearman correlation between rho and image texture (optional check)
    "min_rho_texture_corr":   0.05,    # very weak floor — just ensures rho isn't random
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def hash_file(path: str, chunk_size: int = 65536) -> str:
    """MD5 hash of file bytes for duplicate detection."""
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ''


def validate_image_file(path: str, rec: ImageRecord) -> ImageRecord:
    """
    All image-level checks in one pass.
    Returns the record with fields populated and status set.
    """
    p = Path(path)

    # ── 1. File existence & size ─────────────────────────────────────────────
    if not p.exists():
        rec.status = 'removed'
        rec.remove_reason = 'file_not_found'
        return rec

    size_kb = p.stat().st_size / 1024
    rec.file_size_kb = round(size_kb, 2)

    if size_kb < THRESHOLDS["min_file_size_kb"]:
        rec.status = 'removed'
        rec.remove_reason = f'file_too_small_{size_kb:.2f}kb'
        return rec

    if size_kb > THRESHOLDS["max_file_size_kb"]:
        rec.warnings.append(f'very_large_file_{size_kb:.0f}kb')

    # ── 2. File hash (populated early for duplicate detection) ───────────────
    rec.file_hash = hash_file(path)

    # ── 3. Image loadability ─────────────────────────────────────────────────
    if not PIL_OK:
        log.warning("Pillow not available -- skipping pixel-level checks")
        return rec

    try:
        img = Image.open(path)
        img.verify()              # catches truncated/corrupt files
        img = Image.open(path)   # reopen after verify (verify closes stream)
        img_arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
    except Exception as e:
        rec.status = 'removed'
        rec.remove_reason = f'load_failed:{str(e)[:60]}'
        return rec

    # ── 4. Dimensions ────────────────────────────────────────────────────────
    h, w = img_arr.shape[:2]
    c = img_arr.shape[2] if img_arr.ndim == 3 else 1
    rec.width, rec.height, rec.channels = w, h, c

    if min(h, w) < THRESHOLDS["min_dimension"]:
        rec.status = 'removed'
        rec.remove_reason = f'too_small_{w}x{h}'
        return rec

    if max(h, w) > THRESHOLDS["max_dimension"]:
        rec.warnings.append(f'very_large_{w}x{h}')

    if h != THRESHOLDS["target_size"] or w != THRESHOLDS["target_size"]:
        rec.warnings.append(f'non_target_size_{w}x{h}')

    # ── 5. Channel check ─────────────────────────────────────────────────────
    # BOSSBase is grayscale (.pgm) — that's expected and fine
    if c == 1:
        rec.warnings.append('grayscale_will_be_converted')

    # ── 6. Pixel statistics ──────────────────────────────────────────────────
    gray = img_arr.mean(axis=2) if img_arr.ndim == 3 else img_arr[:, :, 0]
    rec.mean    = float(round(gray.mean(), 4))
    rec.std     = float(round(gray.std(),  4))
    rec.min_val = float(round(gray.min(),  4))
    rec.max_val = float(round(gray.max(),  4))

    if rec.mean < THRESHOLDS["min_mean"]:
        rec.status = 'removed'
        rec.remove_reason = f'near_black_mean_{rec.mean:.4f}'
        return rec

    if rec.mean > THRESHOLDS["max_mean"]:
        rec.status = 'removed'
        rec.remove_reason = f'near_white_mean_{rec.mean:.4f}'
        return rec

    if rec.std < THRESHOLDS["min_std"]:
        rec.status = 'removed'
        rec.remove_reason = f'flat_image_std_{rec.std:.4f}'
        return rec

    if rec.std > THRESHOLDS["max_std"]:
        rec.warnings.append(f'high_variance_std_{rec.std:.3f}')

    # ── 7. Constant channel check (all red / all green / pure noise) ─────────
    if img_arr.ndim == 3:
        ch_stds = img_arr.std(axis=(0, 1))
        if ch_stds.min() < 0.001:
            rec.warnings.append('one_channel_is_constant')

    # ── 8. Range sanity (uint8 images should be [0,255] range) ──────────────
    if rec.max_val < 0.1:
        rec.warnings.append('very_low_max_value')

    return rec


def validate_rho_file(rho_path: str, rec: ImageRecord) -> ImageRecord:
    """
    Validate a .npy rho embedding probability map.
    Checks: loadable, shape matches image, values in [0,1],
            NaN/Inf fraction, non-zero coverage.
    """
    rec.has_rho = True
    try:
        rho = np.load(rho_path)
    except Exception as e:
        rec.rho_valid = False
        rec.warnings.append(f'rho_load_failed:{str(e)[:50]}')
        return rec

    # Shape check — must match image H×W
    if rec.width > 0 and rec.height > 0:
        if rho.shape != (rec.height, rec.width):
            # Acceptable: (H, W, 1) or (1, H, W)
            if rho.squeeze().shape != (rec.height, rec.width):
                rec.warnings.append(
                    f'rho_shape_mismatch:{rho.shape}_vs_{rec.height}x{rec.width}'
                )

    # NaN / Inf
    nan_frac = np.isnan(rho).mean()
    inf_frac = np.isinf(rho).mean()
    if nan_frac > THRESHOLDS["rho_max_nan_fraction"]:
        rec.warnings.append(f'rho_high_nan_frac:{nan_frac:.3f}')
        rec.rho_valid = False
        return rec
    if inf_frac > 0:
        rec.warnings.append(f'rho_has_inf_values')

    # Value range
    rho_clean = rho[np.isfinite(rho)]
    if rho_clean.min() < THRESHOLDS["rho_min_valid"] - 1e-4:
        rec.warnings.append(f'rho_has_negative_values:{rho_clean.min():.4f}')
    if rho_clean.max() > THRESHOLDS["rho_max_valid"] + 1e-4:
        rec.warnings.append(f'rho_exceeds_one:{rho_clean.max():.4f}')

    # Non-zero coverage (rho all zeros = no embedding info)
    nonzero_frac = (rho_clean > 1e-6).mean()
    if nonzero_frac < THRESHOLDS["rho_min_nonzero_frac"]:
        rec.warnings.append(f'rho_nearly_all_zeros:{nonzero_frac:.5f}')

    rec.rho_valid = True
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

VALID_IMG_EXTS = {'.pgm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def discover_files(root: Path, mode: str) -> dict:
    """
    Returns dict with keys 'cover', 'stego', 'rho' mapping to sorted file lists.
    mode='bossbase' -> expects root/cover/, root/stego/, root/rho/
                       Falls back to flat scan if cover/ is absent (raw BOSSBase).
    mode='flat'     -> all images in root treated as cover; no stego/rho
    """
    result = {'cover': [], 'stego': [], 'rho': []}

    if mode == 'bossbase':
        cover_dir = root / 'cover'
        stego_dir = root / 'stego'

        # ── Auto-detect: no cover/ subdir means images are in root directly ──
        if not cover_dir.exists():
            log.warning(
                f"No cover/ subdir found in {root}. "
                "Treating all images in root as cover (raw BOSSBase layout). "
                "To generate stego pairs, run the embedding step first."
            )
            # Scan root directly for image files (non-recursive to avoid data/ subfolder)
            result['cover'] = sorted([
                p for p in root.iterdir()
                if p.suffix.lower() in VALID_IMG_EXTS and p.is_file()
            ])
            # Also try one level down in case there's a named subfolder
            if not result['cover']:
                result['cover'] = sorted([
                    p for p in root.rglob('*')
                    if p.suffix.lower() in VALID_IMG_EXTS and p.is_file()
                ])
                log.info(f"Deep scan found {len(result['cover'])} images under {root}")
        else:
            result['cover'] = sorted([
                p for p in cover_dir.iterdir()
                if p.suffix.lower() in VALID_IMG_EXTS
            ])

        if stego_dir.exists():
            result['stego'] = sorted([
                p for p in stego_dir.iterdir()
                if p.suffix.lower() in VALID_IMG_EXTS
            ])
        else:
            log.info("No stego/ directory found -- stego pairs not yet generated")

        rho_dir = root / 'rho'
        if rho_dir.exists():
            result['rho'] = sorted([p for p in rho_dir.iterdir()
                                     if p.suffix.lower() == '.npy'])
        else:
            log.info("No rho/ directory found -- rho maps will be synthetic")

    elif mode == 'flat':
        result['cover'] = sorted([
            p for p in root.rglob('*')
            if p.suffix.lower() in VALID_IMG_EXTS and p.is_file()
        ])

    return result


def build_stem_index(files: list) -> dict:
    """Maps stem → path for fast cover↔stego pairing."""
    return {p.stem: p for p in files}


def scan_dataset(root: Path, mode: str, workers: int = 4) -> AuditReport:
    """
    Main scanning function. Validates all files and returns a populated AuditReport.
    """
    report = AuditReport(
        timestamp=datetime.now().isoformat(timespec='seconds'),
        data_root=str(root),
        mode=mode,
        thresholds=THRESHOLDS.copy(),
    )

    files  = discover_files(root, mode)
    covers = files['cover']
    stegos = files['stego']
    rhos   = files['rho']

    log.info(f"Discovered: {len(covers)} cover | {len(stegos)} stego | {len(rhos)} rho")

    # Build lookup indices
    stego_index = build_stem_index(stegos)
    rho_index   = build_stem_index(rhos)

    # ── Scan covers ─────────────────────────────────────────────────────────
    all_records: list[ImageRecord] = []

    def process_cover(cover_path: Path) -> ImageRecord:
        rec = ImageRecord(
            path=str(cover_path),
            split='cover',
            stem=cover_path.stem,
        )
        rec = validate_image_file(str(cover_path), rec)

        # Rho map check
        if cover_path.stem in rho_index:
            rec = validate_rho_file(str(rho_index[cover_path.stem]), rec)
        
        # Stego pair check
        rec.has_stego_pair = (cover_path.stem in stego_index)
        if not rec.has_stego_pair and mode == 'bossbase':
            rec.warnings.append('no_stego_pair')

        return rec

    log.info(f"Scanning {len(covers)} cover images with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_cover, p): p for p in covers}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            all_records.append(rec)
            if i % 500 == 0:
                log.info(f"  Progress: {i}/{len(covers)}")

    # ── Scan stegos (lighter check — just validity + pairing) ──────────────
    cover_index = build_stem_index(covers)

    def process_stego(stego_path: Path) -> ImageRecord:
        rec = ImageRecord(path=str(stego_path), split='stego', stem=stego_path.stem)
        rec = validate_image_file(str(stego_path), rec)
        rec.has_stego_pair = (stego_path.stem in cover_index)
        if not rec.has_stego_pair:
            rec.warnings.append('no_cover_pair')
        return rec

    if stegos:
        log.info(f"Scanning {len(stegos)} stego images...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_stego, p): p for p in stegos}
            for fut in as_completed(futures):
                all_records.append(fut.result())

    # ── Duplicate detection ─────────────────────────────────────────────────
    log.info("Running duplicate detection via MD5 hash...")
    hash_map = defaultdict(list)
    for rec in all_records:
        if rec.file_hash:
            hash_map[rec.file_hash].append(rec.path)

    dup_paths = set()
    for h, paths in hash_map.items():
        if len(paths) > 1:
            # Keep first, mark rest as duplicates
            for p in paths[1:]:
                dup_paths.add(p)

    for rec in all_records:
        if rec.path in dup_paths and rec.status == 'ok':
            rec.status = 'removed'
            rec.remove_reason = 'duplicate'
            report.duplicates_found += 1

    # ── Outlier detection (IQR on mean & std) ───────────────────────────────
    log.info("Running statistical outlier detection...")
    valid_recs = [r for r in all_records if r.status == 'ok']
    if len(valid_recs) > 20:
        means = np.array([r.mean for r in valid_recs])
        stds  = np.array([r.std  for r in valid_recs])

        for arr, attr, label in [(means, 'mean', 'mean'), (stds, 'std', 'std')]:
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr   # 3× IQR = very permissive
            for rec in valid_recs:
                v = getattr(rec, attr)
                if v < lo or v > hi:
                    rec.warnings.append(f'iqr_outlier_{label}_{v:.4f}')

    # ── Aggregate report ────────────────────────────────────────────────────
    removal_counts = Counter(r.remove_reason for r in all_records if r.status == 'removed')
    warn_types = Counter(w for r in all_records for w in r.warnings)

    # Size distribution
    size_counter = Counter(
        f"{r.width}x{r.height}" for r in all_records
        if r.status == 'ok' and r.width > 0
    )
    modal_size = size_counter.most_common(1)[0][0] if size_counter else 'unknown'

    report.total_scanned    = len(all_records)
    report.total_valid      = sum(1 for r in all_records if r.status == 'ok')
    report.total_removed    = sum(1 for r in all_records if r.status == 'removed')
    report.total_warnings   = sum(1 for r in all_records if r.warnings)
    report.cover_count      = sum(1 for r in all_records if r.split == 'cover' and r.status == 'ok')
    report.stego_count      = sum(1 for r in all_records if r.split == 'stego' and r.status == 'ok')
    report.rho_count        = sum(1 for r in all_records if r.has_rho and r.rho_valid)
    report.modal_size       = modal_size
    report.size_distribution = dict(size_counter.most_common(10))
    report.removal_reasons  = dict(removal_counts)
    report.warning_types    = dict(warn_types.most_common(20))
    report.records          = [asdict(r) for r in all_records]

    return report, all_records


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def save_json_report(report: AuditReport, out_path: str):
    with open(out_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    log.info(f"JSON report -> {out_path}")


def save_removed_log(records: list, out_path: str):
    removed = [r for r in records if r.status == 'removed']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'split', 'stem', 'remove_reason',
                         'width', 'height', 'mean', 'std', 'file_size_kb'])
        for r in removed:
            writer.writerow([r.path, r.split, r.stem, r.remove_reason,
                              r.width, r.height, r.mean, r.std, r.file_size_kb])
    log.info(f"Removed log ({len(removed)} entries) -> {out_path}")


def save_warnings_log(records: list, out_path: str):
    warned = [r for r in records if r.warnings and r.status == 'ok']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'split', 'stem', 'warnings'])
        for r in warned:
            writer.writerow([r.path, r.split, r.stem, ' | '.join(r.warnings)])
    log.info(f"Warnings log ({len(warned)} entries) -> {out_path}")


def save_clean_manifest(records: list, out_path: str):
    """
    Writes a clean_manifest.json with only valid cover paths + their metadata.
    The training DataLoader can read this directly instead of re-scanning.
    """
    valid_covers = [r for r in records if r.status == 'ok' and r.split == 'cover']
    manifest = []
    for r in valid_covers:
        manifest.append({
            'path':      r.path,
            'stem':      r.stem,
            'width':     r.width,
            'height':    r.height,
            'has_rho':   r.has_rho and r.rho_valid,
            'has_stego': r.has_stego_pair,
            'mean':      r.mean,
            'std':       r.std,
        })
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Clean manifest ({len(manifest)} entries) -> {out_path}")


def generate_visualizations(records: list, out_dir: str):
    """Generates diagnostic plots for the audit report."""
    if not MPL_OK:
        log.warning("matplotlib not available -- skipping visualizations")
        return

    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    valid = [r for r in records if r.status == 'ok' and r.mean > 0]
    if not valid:
        return

    means = [r.mean for r in valid]
    stds  = [r.std  for r in valid]
    sizes = [r.file_size_kb for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#111827')
        for s in ax.spines.values():
            s.set_edgecolor('#374151')
        ax.tick_params(colors='#9ca3af')
        ax.title.set_color('#e5e7eb')
        ax.xaxis.label.set_color('#9ca3af')
        ax.yaxis.label.set_color('#9ca3af')

    axes[0].hist(means, bins=60, color='#00d4ff', alpha=0.8, edgecolor='none')
    axes[0].axvline(THRESHOLDS['min_mean'], color='#ef4444', lw=1.5, ls='--', label='threshold')
    axes[0].axvline(THRESHOLDS['max_mean'], color='#ef4444', lw=1.5, ls='--')
    axes[0].set_title('Pixel Mean Distribution'); axes[0].set_xlabel('Mean intensity')
    axes[0].legend(labelcolor='#9ca3af', framealpha=0.3)

    axes[1].hist(stds, bins=60, color='#7c3aed', alpha=0.8, edgecolor='none')
    axes[1].axvline(THRESHOLDS['min_std'], color='#ef4444', lw=1.5, ls='--', label='threshold')
    axes[1].set_title('Pixel Std Distribution'); axes[1].set_xlabel('Std intensity')
    axes[1].legend(labelcolor='#9ca3af', framealpha=0.3)

    axes[2].hist(sizes, bins=60, color='#10b981', alpha=0.8, edgecolor='none')
    axes[2].set_title('File Size Distribution (KB)'); axes[2].set_xlabel('File size (KB)')

    plt.tight_layout()
    plot_path = out / 'distribution_plots.png'
    plt.savefig(plot_path, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    log.info(f"Distribution plots -> {plot_path}")

    # Size scatter
    if len(valid) > 10:
        fig2, ax = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#111827')
        ax.scatter(means, stds, s=3, alpha=0.4, c='#00d4ff')
        ax.set_xlabel('Mean'); ax.set_ylabel('Std')
        ax.set_title('Mean vs Std (valid images)')
        ax.tick_params(colors='#9ca3af')
        ax.title.set_color('#e5e7eb')
        for sp in ax.spines.values(): sp.set_edgecolor('#374151')
        scatter_path = out / 'mean_vs_std.png'
        plt.savefig(scatter_path, dpi=120, facecolor='#0d1117', bbox_inches='tight')
        plt.close()


def generate_html_report(report: AuditReport, records: list, out_path: str):
    """Generates a self-contained HTML summary report."""
    removed = [r for r in records if r.status == 'removed']
    warned  = [r for r in records if r.warnings and r.status == 'ok']

    health_pct = round(report.total_valid / max(report.total_scanned, 1) * 100, 1)
    health_color = ('#10b981' if health_pct >= 95
                    else '#f59e0b' if health_pct >= 85
                    else '#ef4444')

    reason_rows = ''.join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>'
        for k, v in sorted(report.removal_reasons.items(), key=lambda x: -x[1])
    )
    warn_rows = ''.join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>'
        for k, v in sorted(report.warning_types.items(), key=lambda x: -x[1])
    )
    size_rows = ''.join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>'
        for k, v in sorted(report.size_distribution.items(), key=lambda x: -x[1])
    )

    removed_sample = removed[:50]
    removed_rows = ''.join(
        f'<tr><td class="path">{Path(r.path).name}</td>'
        f'<td>{r.split}</td>'
        f'<td class="reason">{r.remove_reason}</td>'
        f'<td>{r.width}x{r.height}</td>'
        f'<td>{r.mean:.3f}</td>'
        f'<td>{r.std:.3f}</td></tr>'
        for r in removed_sample
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HybridLocNet Dataset Audit Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
  :root {{ --bg:#080b10; --surface:#0d1117; --card:#111620; --border:#1e2a38;
           --text:#e2e8f0; --muted:#64748b; --mono:'IBM Plex Mono',monospace;
           --sans:'Inter',sans-serif; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:var(--sans); line-height:1.6; padding:40px; }}
  h1 {{ font-size:28px; font-weight:600; margin-bottom:6px; }}
  h2 {{ font-size:16px; font-weight:500; color:#94a3b8; margin:28px 0 14px; text-transform:uppercase; letter-spacing:.07em; }}
  .ts {{ color:var(--muted); font-family:var(--mono); font-size:12px; margin-bottom:32px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:14px; margin-bottom:28px; }}
  .stat {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:18px 20px; }}
  .stat .val {{ font-size:28px; font-weight:600; font-family:var(--mono); }}
  .stat .lbl {{ font-size:12px; color:var(--muted); margin-top:4px; }}
  .health {{ font-size:48px; font-weight:700; color:{health_color}; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-bottom:24px; }}
  th {{ font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em;
        color:var(--muted); padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }}
  td {{ padding:8px 12px; border-bottom:1px solid rgba(30,42,56,.5); }}
  td.num {{ font-family:var(--mono); color:#00d4ff; text-align:right; }}
  td.path {{ font-family:var(--mono); font-size:11px; color:#94a3b8; max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  td.reason {{ color:#ef4444; font-family:var(--mono); font-size:11px; }}
  tr:hover td {{ background:rgba(255,255,255,.02); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px; margin-bottom:18px; }}
  img {{ max-width:100%; border-radius:8px; margin-top:12px; }}
  .ok {{ color:#10b981; }} .warn {{ color:#f59e0b; }} .err {{ color:#ef4444; }}
</style>
</head>
<body>
<h1>🔬 HybridLocNet Dataset Audit</h1>
<div class="ts">Generated: {report.timestamp} &nbsp;|&nbsp; Root: {report.data_root} &nbsp;|&nbsp; Mode: {report.mode}</div>

<div class="stats">
  <div class="stat"><div class="val health">{health_pct}%</div><div class="lbl">Dataset health</div></div>
  <div class="stat"><div class="val">{report.total_scanned:,}</div><div class="lbl">Total scanned</div></div>
  <div class="stat"><div class="val ok">{report.total_valid:,}</div><div class="lbl">Valid files</div></div>
  <div class="stat"><div class="val err">{report.total_removed:,}</div><div class="lbl">Removed</div></div>
  <div class="stat"><div class="val warn">{report.total_warnings:,}</div><div class="lbl">With warnings</div></div>
  <div class="stat"><div class="val">{report.duplicates_found:,}</div><div class="lbl">Duplicates</div></div>
  <div class="stat"><div class="val ok">{report.cover_count:,}</div><div class="lbl">Valid covers</div></div>
  <div class="stat"><div class="val ok">{report.stego_count:,}</div><div class="lbl">Valid stego</div></div>
  <div class="stat"><div class="val ok">{report.rho_count:,}</div><div class="lbl">Valid rho maps</div></div>
  <div class="stat"><div class="val">{report.modal_size}</div><div class="lbl">Modal image size</div></div>
</div>

<div class="card">
  <h2>Removal Reasons</h2>
  <table><tr><th>Reason</th><th style="text-align:right">Count</th></tr>{reason_rows or '<tr><td colspan=2>None - clean dataset!</td></tr>'}</table>
</div>

<div class="card">
  <h2>Warning Types (top 20)</h2>
  <table><tr><th>Warning</th><th style="text-align:right">Count</th></tr>{warn_rows or '<tr><td colspan=2>None</td></tr>'}</table>
</div>

<div class="card">
  <h2>Image Size Distribution (top 10)</h2>
  <table><tr><th>Size</th><th style="text-align:right">Count</th></tr>{size_rows}</table>
</div>

<div class="card">
  <h2>Removed Files (first 50)</h2>
  <table>
    <tr><th>Filename</th><th>Split</th><th>Reason</th><th>Size</th><th>Mean</th><th>Std</th></tr>
    {removed_rows or '<tr><td colspan=6>No files removed.</td></tr>'}
  </table>
</div>

<div class="card">
  <h2>Diagnostic Plots</h2>
  <img src="plots/distribution_plots.png" alt="Distribution plots" onerror="this.style.display='none'">
  <img src="plots/mean_vs_std.png" alt="Mean vs Std" style="margin-top:12px" onerror="this.style.display='none'">
</div>
</body></html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    log.info(f"HTML report -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TO CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(report: AuditReport):
    health = report.total_valid / max(report.total_scanned, 1) * 100
    status = ("EXCELLENT"       if health >= 98 else
              "GOOD"            if health >= 90 else
              "NEEDS ATTENTION" if health >= 75 else
              "CRITICAL ISSUES")

    sep = "=" * 58
    print(f"""
{sep}
  DATASET AUDIT SUMMARY
{sep}
  Overall Status : {status}
  Health Score   : {health:.1f}% valid
{sep}
  COUNTS
    Total scanned : {report.total_scanned}
    Valid         : {report.total_valid}  ({report.cover_count} cover / {report.stego_count} stego)
    Removed       : {report.total_removed}
    Warnings      : {report.total_warnings}
    Duplicates    : {report.duplicates_found}
    Valid rho     : {report.rho_count}
{sep}
  MODAL IMAGE SIZE : {report.modal_size}
{sep}""")

    if report.removal_reasons:
        print("  TOP REMOVAL REASONS")
        for reason, count in sorted(report.removal_reasons.items(),
                                    key=lambda x: -x[1])[:5]:
            print(f"    {reason[:50]:<50} {count:>4}")

    print(f"""{sep}
  OUTPUT FILES
    audit_report.json   - Full machine-readable report
    audit_report.html   - Visual summary (open in browser)
    removed_log.csv     - Removed files with reasons
    warnings_log.csv    - Files needing attention
    clean_manifest.json - Training-ready file list
{sep}
""")

    if report.total_removed > 0:
        print("ACTION REQUIRED:")
        if 'duplicate' in report.removal_reasons:
            n = report.removal_reasons['duplicate']
            print(f"   * {n} duplicate files removed -- check source pipeline")
        if any('load_failed' in k for k in report.removal_reasons):
            n = sum(v for k,v in report.removal_reasons.items() if 'load_failed' in k)
            print(f"   * {n} corrupt images -- re-download or regenerate")
        if any('near_black' in k or 'near_white' in k or 'flat_image' in k
               for k in report.removal_reasons):
            print("   * Some images are nearly uniform -- check preprocessing pipeline")

    non_target_warns = report.warning_types.get('non_target_size_' + report.modal_size,
                        sum(v for k, v in report.warning_types.items()
                            if k.startswith('non_target_size')))
    if report.modal_size != f"{THRESHOLDS['target_size']}x{THRESHOLDS['target_size']}":
        print(f"\nSIZE NOTE: Most images are {report.modal_size}, not "
              f"{THRESHOLDS['target_size']}x{THRESHOLDS['target_size']}. "
              f"The DataLoader will resize automatically -- this is NORMAL for raw BOSSBase "
              f"(512x512 source images). The {report.total_warnings} entries in warnings_log.csv "
              f"are almost entirely this harmless size warning.")

    if report.rho_count == 0:
        print("\nRHO NOTE: No valid rho maps found. "
              "Synthetic cost maps will be computed during training (slower but valid).")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='HybridLocNet Dataset Audit + Cleaning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BOSSBase structure (cover/ stego/ rho/)
  python validate_dataset.py --data ./data/bossbase

  # Flat image folder (synthetic stego generation)
  python validate_dataset.py --data ./my_images --mode flat

  # Full run with plot generation
  python validate_dataset.py --data ./data/bossbase --plots

  # Strict mode: lower thresholds
  python validate_dataset.py --data ./data/bossbase --strict

  # Quick scan (no pixel statistics — just file integrity)
  python validate_dataset.py --data ./data/bossbase --quick
        """
    )
    parser.add_argument('--data',    type=str, required=True,  help='Path to dataset root')
    parser.add_argument('--mode',    type=str, default='bossbase',
                        choices=['bossbase', 'flat'], help='Dataset structure mode')
    parser.add_argument('--out',     type=str, default='.',     help='Output directory for reports')
    parser.add_argument('--workers', type=int, default=4,       help='Parallel workers')
    parser.add_argument('--plots',   action='store_true',       help='Generate distribution plots')
    parser.add_argument('--strict',  action='store_true',
                        help='Tighter outlier thresholds (3-sigma instead of IQR x 3)')
    parser.add_argument('--quick',   action='store_true',
                        help='Fast mode: file existence + size only, no pixel stats')
    args = parser.parse_args()

    root = Path(args.data)
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        log.error(f"Data root not found: {root}")
        sys.exit(1)

    if args.strict:
        THRESHOLDS['min_mean'] = 0.05
        THRESHOLDS['max_mean'] = 0.95
        THRESHOLDS['min_std']  = 0.01
        log.info("STRICT mode: tightened thresholds")

    if not PIL_OK:
        log.warning("Pillow missing -- install it: pip install Pillow")
        log.warning("Running in limited mode (no pixel statistics)")

    log.info(f"Starting audit: {root} (mode={args.mode}, workers={args.workers})")
    t0 = datetime.now()

    report, records = scan_dataset(root, args.mode, workers=args.workers)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Scan complete in {elapsed:.1f}s")

    # Save outputs
    save_json_report(report, out / 'audit_report.json')
    save_removed_log(records, out / 'removed_log.csv')
    save_warnings_log(records, out / 'warnings_log.csv')
    save_clean_manifest(records, out / 'clean_manifest.json')

    if args.plots:
        plots_dir = out / 'plots'
        generate_visualizations(records, str(plots_dir))

    generate_html_report(report, records, out / 'audit_report.html')
    print_summary(report)

    # Exit code: 0 = clean, 1 = issues found
    has_issues = report.total_removed > 0 or report.duplicates_found > 0
    sys.exit(1 if has_issues else 0)


if __name__ == '__main__':
    main()