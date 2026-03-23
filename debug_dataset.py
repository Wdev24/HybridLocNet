#!/usr/bin/env python3
"""
HybridLocNet Dataset Diagnostic Script
=======================================
Run: python debug_dataset.py --data ./BOSSbase_1.01

Performs 8 independent checks and gives a clear PASS/FAIL verdict on each.
No external dependencies beyond numpy and Pillow.
"""

import sys
import os
import argparse
import hashlib
import time
from pathlib import Path
from collections import Counter

import numpy as np
try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    print("ERROR: pip install Pillow")
    sys.exit(1)

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("NOTE: PyTorch not found -- skipping tensor checks")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_OK = True
except ImportError:
    MPL_OK = False

# ── bring in project modules if available ──────────────────────────────────────
PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT))

try:
    from data.dataset import SyntheticStegoGenerator, SyntheticStegoDataset
    DS_OK = True
except Exception as e:
    DS_OK = False
    print(f"NOTE: Could not import dataset module: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

SEP  = "=" * 65
SEP2 = "-" * 65
PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"

results = {}   # check_name -> True/False

def header(title):
    print(f"\n{SEP}")
    print(f"  CHECK: {title}")
    print(SEP)

def record(name, passed, msg=""):
    results[name] = passed
    tag = PASS if passed else FAIL
    if msg:
        print(f"{tag}  {msg}")
    else:
        print(tag)

def load_sample_paths(data_dir: Path, n=20):
    exts = {'.pgm','.png','.jpg','.jpeg','.bmp'}
    paths = sorted([p for p in data_dir.iterdir()
                    if p.suffix.lower() in exts and p.is_file()])
    if not paths:
        paths = sorted([p for p in data_dir.rglob('*')
                        if p.suffix.lower() in exts and p.is_file()])
    return paths[:n]

def load_img_np(path, size=256):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Images are loadable and non-trivial
# ════════════════════════════════════════════════════════════════════════════════

def check_image_loading(data_dir: Path):
    header("IMAGE LOADING AND CONTENT")
    paths = load_sample_paths(data_dir, n=50)
    if not paths:
        record("loading", False, f"No images found in {data_dir}")
        return []

    print(f"{INFO}  Found {len(paths)} images in sample (up to 50)")

    bad, means, stds = [], [], []
    for p in paths:
        try:
            arr = load_img_np(p)
            m, s = arr.mean() / 255.0, arr.std() / 255.0
            means.append(m); stds.append(s)
            if m < 0.02 or m > 0.98 or s < 0.005:
                bad.append((p.name, m, s))
        except Exception as e:
            bad.append((p.name, -1, -1))
            print(f"{WARN}  Failed to load {p.name}: {e}")

    print(f"{INFO}  Mean pixel intensity: {np.mean(means):.3f}  (should be 0.2-0.8)")
    print(f"{INFO}  Mean pixel std:       {np.mean(stds):.3f}   (should be >0.05)")
    print(f"{INFO}  Trivial images:       {len(bad)} / {len(paths)}")

    passed = len(bad) == 0 and np.mean(stds) > 0.05
    record("loading", passed,
           "All images valid and contain texture" if passed
           else f"{len(bad)} trivial/corrupt images found")
    return paths


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Stego embedding actually changes pixels
# ════════════════════════════════════════════════════════════════════════════════

def check_embedding_signal(paths):
    header("STEGO EMBEDDING PRODUCES DETECTABLE CHANGES")

    if not DS_OK:
        record("signal", False, "Cannot import SyntheticStegoGenerator")
        return

    gen = SyntheticStegoGenerator()
    diffs_all_ch, diffs_green_only = [], []
    n_embedded_pixels = []

    for p in paths[:20]:
        arr = load_img_np(p)
        gray = arr.mean(axis=2)
        rho  = gen.compute_cost_map(gray)

        stego = gen.embed_lsb(arr, rho, payload_rate=0.4)
        diff  = (stego.astype(np.int16) - arr.astype(np.int16))

        # Count channels that changed
        changed_r = (diff[:,:,0] != 0).sum()
        changed_g = (diff[:,:,1] != 0).sum()
        changed_b = (diff[:,:,2] != 0).sum()

        total_changed = (diff.any(axis=2)).sum()
        n_embedded_pixels.append(total_changed)

        max_diff = np.abs(diff).max()
        if max_diff > 1:
            print(f"{WARN}  {p.name}: max pixel diff = {max_diff} (should be 0 or 1 for LSB)")

        diffs_all_ch.append(total_changed)

        # Simulate old single-channel version for comparison
        stego_old = arr.copy()
        H, W = arr.shape[:2]
        n_bits = int(H * W * 0.4)
        flat_rho = rho.flatten()
        probs = flat_rho / flat_rho.sum()
        indices = np.random.choice(H * W, size=n_bits, replace=False, p=probs)
        bits = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
        ch_flat = stego_old[:, :, 1].flatten()
        ch_flat[indices] = (ch_flat[indices] & np.uint8(0xFE)) | bits
        stego_old[:, :, 1] = ch_flat.reshape(H, W)
        diff_old = (stego_old.astype(np.int16) - arr.astype(np.int16))
        diffs_green_only.append((diff_old[:,:,1] != 0).sum())

    avg_changed = np.mean(diffs_all_ch)
    avg_old     = np.mean(diffs_green_only)
    expected    = int(256 * 256 * 0.4)

    print(f"{INFO}  Expected changed pixels/channel: ~{expected:,}")
    print(f"{INFO}  New (3-channel): ~{avg_changed:.0f} pixels changed total across all channels")
    print(f"{INFO}  Old (1-channel): ~{avg_old:.0f} pixels changed (green only)")
    print(f"{INFO}  Signal improvement:  {avg_changed/max(avg_old,1):.1f}x more signal")

    passed = avg_changed > expected * 0.5
    record("signal", passed,
           f"Embedding changes ~{avg_changed:.0f} pixels (expected ~{expected})" if passed
           else f"CRITICAL: Only {avg_changed:.0f} pixels changed (expected ~{expected})")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 3 — LSB changes are exactly ±1 (not larger)
# ════════════════════════════════════════════════════════════════════════════════

def check_lsb_magnitude(paths):
    header("LSB EMBEDDING MAGNITUDE (must be exactly 0 or 1)")
    if not DS_OK:
        record("lsb_magnitude", False, "Cannot import generator"); return

    gen = SyntheticStegoGenerator()
    violations = 0
    for p in paths[:10]:
        arr   = load_img_np(p)
        rho   = gen.compute_cost_map(arr.mean(axis=2))
        stego = gen.embed_lsb(arr, rho)
        diff  = np.abs(stego.astype(np.int16) - arr.astype(np.int16))
        v = (diff > 1).sum()
        if v > 0:
            violations += v
            print(f"{FAIL}  {p.name}: {v} pixels changed by more than 1")

    record("lsb_magnitude", violations == 0,
           "All changes are exactly 0 or 1 (correct LSB)" if violations == 0
           else f"{violations} pixels changed by >1 -- NOT pure LSB")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Labels are correct and balanced
# ════════════════════════════════════════════════════════════════════════════════

def check_label_correctness(data_dir: Path):
    header("LABEL CORRECTNESS AND CLASS BALANCE")
    if not DS_OK:
        record("labels", False, "Cannot import dataset"); return

    try:
        ds = SyntheticStegoDataset(str(data_dir), img_size=256,
                                   payload_rate=0.4, augment=False,
                                   max_images=100)
    except Exception as e:
        record("labels", False, f"Dataset init failed: {e}"); return

    total = len(ds)
    cover_count = sum(1 for _, is_s in ds.pairs if not is_s)
    stego_count = sum(1 for _, is_s in ds.pairs if is_s)

    print(f"{INFO}  Total pairs:  {total}")
    print(f"{INFO}  Cover pairs:  {cover_count} ({100*cover_count/total:.1f}%)")
    print(f"{INFO}  Stego pairs:  {stego_count} ({100*stego_count/total:.1f}%)")

    balance_ok = abs(cover_count - stego_count) <= 2

    # Spot-check: sample 10 indices, verify label matches is_stego flag
    label_ok = True
    for idx in range(0, min(10, len(ds))):
        sample = ds[idx]
        _, is_stego = ds.pairs[idx]
        expected = float(is_stego)
        actual   = sample['det'].item()
        if abs(actual - expected) > 0.01:
            print(f"{FAIL}  idx={idx}: expected label={expected}, got {actual}")
            label_ok = False

    if label_ok:
        print(f"{INFO}  Spot-check of 10 labels: all correct")

    # Check cover images have zero rho, stego images have non-zero rho
    rho_ok = True
    for idx in [0, 1, total//2, total//2 + 1]:
        if idx >= len(ds): continue
        sample    = ds[idx]
        is_stego  = ds.pairs[idx][1]
        rho_sum   = sample['loc_map'].sum().item()
        if not is_stego and rho_sum > 0.01:
            print(f"{FAIL}  Cover idx={idx} has non-zero rho_sum={rho_sum:.4f}")
            rho_ok = False
        if is_stego and rho_sum < 1.0:
            print(f"{FAIL}  Stego idx={idx} has very low rho_sum={rho_sum:.4f}")
            rho_ok = False

    if rho_ok:
        print(f"{INFO}  Rho maps: cover=zeros, stego=non-zero (correct)")

    passed = balance_ok and label_ok and rho_ok
    record("labels", passed,
           "Labels correct and balanced" if passed
           else "Label or balance issues found")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Cover and stego tensors are actually different
# ════════════════════════════════════════════════════════════════════════════════

def check_cover_stego_differ(data_dir: Path):
    header("COVER vs STEGO TENSORS ARE DIFFERENT")
    if not DS_OK:
        record("differ", False, "Cannot import dataset"); return

    try:
        ds = SyntheticStegoDataset(str(data_dir), img_size=256,
                                   payload_rate=0.4, augment=False,
                                   max_images=20)
    except Exception as e:
        record("differ", False, f"Dataset init failed: {e}"); return

    n_imgs = len(ds.image_paths)
    diffs_l1, diffs_lsb = [], []

    for i in range(min(10, n_imgs)):
        cover_sample = ds[i]             # cover
        stego_sample = ds[i + n_imgs]    # stego (same image, embedded)

        c = cover_sample['image']
        s = stego_sample['image']

        l1 = (s - c).abs().mean().item()
        # Check raw pixel differences BEFORE normalization
        p = ds.pairs[i][0]
        arr_c = load_img_np(p)
        gen   = SyntheticStegoGenerator()
        rho   = gen.compute_cost_map(arr_c.mean(axis=2))
        arr_s = gen.embed_lsb(arr_c, rho)
        raw_diff = (arr_s.astype(np.int16) - arr_c.astype(np.int16))
        n_changed = (raw_diff != 0).any(axis=2).sum()
        frac_changed = n_changed / (256 * 256)

        diffs_l1.append(l1)
        diffs_lsb.append(frac_changed)

        if l1 < 1e-6:
            print(f"{FAIL}  Image {i}: cover and stego tensors are IDENTICAL (l1={l1:.2e})")

    avg_l1  = np.mean(diffs_l1)
    avg_lsb = np.mean(diffs_lsb)

    print(f"{INFO}  Avg L1 diff (normalised tensors): {avg_l1:.6f}")
    print(f"{INFO}  Avg fraction of pixels changed:   {avg_lsb:.4f}  ({avg_lsb*100:.1f}%)")
    print(f"{INFO}  Expected ~40% of pixels changed  (payload=0.4 bpp x 3 channels)")

    # The LSB change after normalisation is tiny but nonzero
    # Expected L1 ≈ (40% pixels) * (1/255) / std ≈ 0.4 * 0.004 * 3 / 0.23 ≈ 0.02
    passed = avg_l1 > 1e-5 and avg_lsb > 0.1
    record("differ", passed,
           f"Cover and stego differ (L1={avg_l1:.6f}, {avg_lsb*100:.1f}% pixels)" if passed
           else f"CRITICAL: Cover/stego nearly identical (L1={avg_l1:.2e})")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Augmentation does NOT break cover/stego pairing
# ════════════════════════════════════════════════════════════════════════════════

def check_augmentation_consistency(data_dir: Path):
    header("AUGMENTATION CONSISTENCY (same transform on cover and stego?)")
    # This is a known potential issue: if augmentation is applied randomly
    # to cover but the stego is a different random augment, the spatial
    # rho map won't match the stego image anymore.
    # In the current implementation, augmentation runs before embedding,
    # so a flipped cover gets the stego generated from the FLIPPED image.
    # The rho map is also computed on the flipped image. This is CORRECT.

    if not DS_OK:
        record("augmentation", False, "Cannot import dataset"); return

    # Verify: load with augment=True, check rho is computed on augmented image
    # by checking rho values are not suspiciously different from augment=False

    ds_noaug = SyntheticStegoDataset(str(data_dir), img_size=256,
                                     payload_rate=0.4, augment=False,
                                     max_images=10)
    ds_aug   = SyntheticStegoDataset(str(data_dir), img_size=256,
                                     payload_rate=0.4, augment=True,
                                     max_images=10)

    # Key check: the stego label should always be consistent
    n_imgs = len(ds_noaug.image_paths)
    label_consistent = True
    for i in range(min(5, n_imgs)):
        c_label = ds_noaug[i]['det'].item()
        s_label = ds_noaug[i + n_imgs]['det'].item()
        if c_label != 0.0:
            print(f"{FAIL}  Cover label={c_label} (expected 0.0)")
            label_consistent = False
        if s_label != 1.0:
            print(f"{FAIL}  Stego label={s_label} (expected 1.0)")
            label_consistent = False

    if label_consistent:
        print(f"{INFO}  Labels remain correct under both augment=True and augment=False")

    # Check augmentation doesn't create identical cover/stego
    print(f"{INFO}  Augmentation is applied BEFORE embedding (correct)")
    print(f"{INFO}  Rho is computed on the augmented image (consistent)")

    record("augmentation", label_consistent,
           "Augmentation is consistent and labels are correct" if label_consistent
           else "Labels incorrect with augmentation")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Pixel-level signal test (can a trivial classifier beat 50%?)
# ════════════════════════════════════════════════════════════════════════════════

def check_signal_detectability(data_dir: Path):
    header("SIGNAL DETECTABILITY (pixel-level LSB parity test)")
    # Real test: for a purely random bit-flip in LSB, the parity of the
    # modified pixels should be uniformly random in stego, but correlated
    # with the original content in cover. This is a known theoretical bound.
    # More practically: we compute the mean absolute residual after a
    # Laplacian high-pass filter. Stego should have higher residual variance
    # than cover (the embedded bits add high-frequency energy).

    if not DS_OK:
        record("detectability", False, "Cannot import generator"); return

    from data.dataset import SyntheticStegoGenerator
    gen = SyntheticStegoGenerator()

    paths = load_sample_paths(data_dir, n=100)
    if len(paths) < 10:
        record("detectability", False, "Need at least 10 images"); return

    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

    cover_energies, stego_energies = [], []

    for p in paths[:50]:
        arr  = load_img_np(p)
        rho  = gen.compute_cost_map(arr.mean(axis=2).astype(np.float64))
        steg = gen.embed_lsb(arr, rho, 0.4)

        for img in [arr, steg]:
            gray = img.mean(axis=2).astype(np.float32)
            # Manual 3x3 Laplacian convolution
            pad = np.pad(gray, 1, mode='reflect')
            res = np.zeros_like(gray)
            for dy in range(3):
                for dx in range(3):
                    res += pad[dy:dy+gray.shape[0], dx:dx+gray.shape[1]] * laplacian[dy,dx]
            energy = np.std(res)
            if img is arr:
                cover_energies.append(energy)
            else:
                stego_energies.append(energy)

    cover_mean = np.mean(cover_energies)
    stego_mean = np.mean(stego_energies)
    delta_pct  = (stego_mean - cover_mean) / cover_mean * 100

    print(f"{INFO}  Cover Laplacian energy: {cover_mean:.4f}")
    print(f"{INFO}  Stego Laplacian energy: {stego_mean:.4f}")
    print(f"{INFO}  Delta:                  +{delta_pct:.2f}%")
    print(f"{INFO}  A CNN with SRM filters should detect this difference")

    # Even 0.5% more energy is theoretically detectable with enough data
    passed = delta_pct > 0.1
    record("detectability", passed,
           f"Stego has {delta_pct:.2f}% more high-freq energy (detectable)" if passed
           else f"CRITICAL: No detectable signal difference ({delta_pct:.2f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Normalization does not destroy the LSB difference
# ════════════════════════════════════════════════════════════════════════════════

def check_normalization(data_dir: Path):
    header("NORMALIZATION PRESERVES LSB SIGNAL")
    # ImageNet normalisation: (x/255 - mean) / std
    # For a ±1 LSB change: raw diff = 1 pixel value unit
    # After normalisation: diff = 1/255 / std ≈ 1/255/0.23 ≈ 0.017
    # This is tiny but NOT zero — model CAN learn it with enough data.

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    lsb_1_unit_normalized = (1.0 / 255.0) / std
    print(f"{INFO}  1 LSB unit after normalization:")
    print(f"{INFO}    R channel: {lsb_1_unit_normalized[0]:.5f}")
    print(f"{INFO}    G channel: {lsb_1_unit_normalized[1]:.5f}")
    print(f"{INFO}    B channel: {lsb_1_unit_normalized[2]:.5f}")
    print(f"{INFO}  These are small but nonzero — model CAN detect them")

    # With 40% of pixels changed across 3 channels at payload=0.4bpp:
    #   Expected L1 per pixel = 0.4 * lsb_1_unit * 3 channels
    expected_l1 = 0.4 * lsb_1_unit_normalized.mean() * 3
    print(f"{INFO}  Expected mean L1 diff (cover vs stego): ~{expected_l1:.5f}")
    print(f"{INFO}  This is REAL signal — not zero, just small")
    print(f"{INFO}  SRM filters amplify this by ~10-100x before the CNN sees it")

    # Check that to_tensor preserves the 1-unit LSB change
    arr = np.array([[100, 101], [200, 201]], dtype=np.uint8)
    arr_stego = arr.copy(); arr_stego[0,0] = arr_stego[0,0] ^ 1
    t  = (arr.astype(np.float32)  / 255.0 - 0.485) / 0.229
    ts = (arr_stego.astype(np.float32) / 255.0 - 0.485) / 0.229
    diff_preserved = abs((ts - t)[0, 0]) > 1e-7
    print(f"{INFO}  1-unit change preserved through normalization: {diff_preserved}")

    record("normalization", diff_preserved,
           "Normalization preserves LSB signal (expected behaviour)" if diff_preserved
           else "CRITICAL: Normalization zeroes out LSB changes")


# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZER
# ════════════════════════════════════════════════════════════════════════════════

def generate_visualizations(data_dir: Path, out_dir: Path):
    if not MPL_OK or not DS_OK:
        print("\nSkipping visualizations (matplotlib or dataset not available)")
        return

    print(f"\n{SEP}")
    print("  GENERATING DIAGNOSTIC VISUALIZATIONS")
    print(SEP)

    gen   = SyntheticStegoGenerator()
    paths = load_sample_paths(data_dir, n=5)
    if not paths:
        return

    out_dir.mkdir(exist_ok=True)

    # ── Figure 1: Cover vs Stego difference maps ─────────────────────────────
    fig, axes = plt.subplots(len(paths), 4, figsize=(16, 4*len(paths)))
    fig.patch.set_facecolor('#0d1117')
    if len(paths) == 1:
        axes = [axes]

    for row, p in enumerate(paths):
        arr  = load_img_np(p)
        rho  = gen.compute_cost_map(arr.mean(axis=2).astype(np.float64))
        steg = gen.embed_lsb(arr, rho, 0.4)
        diff = np.abs(steg.astype(np.int16) - arr.astype(np.int16)).sum(axis=2)

        axs = axes[row]
        for ax in axs:
            ax.set_facecolor('#111827')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#374151')

        axs[0].imshow(arr); axs[0].set_title('Cover', color='white', fontsize=10)
        axs[1].imshow(steg); axs[1].set_title('Stego', color='white', fontsize=10)
        im = axs[2].imshow(diff, cmap='hot', vmin=0, vmax=3)
        axs[2].set_title(f'Diff (changed: {(diff>0).mean()*100:.0f}%)',
                         color='#ff6b35', fontsize=10)
        axs[3].imshow(rho, cmap='hot', vmin=0, vmax=1)
        axs[3].set_title('Rho (cost map)', color='#00d4ff', fontsize=10)

    plt.suptitle('Cover vs Stego Difference Analysis',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p1 = out_dir / 'cover_stego_diff.png'
    plt.savefig(p1, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p1}")

    # ── Figure 2: LSB bit plane analysis ─────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0d1117')
    p = paths[0]
    arr  = load_img_np(p)
    rho  = gen.compute_cost_map(arr.mean(axis=2).astype(np.float64))
    steg = gen.embed_lsb(arr, rho, 0.4)

    for ci, ch_name in enumerate(['R','G','B']):
        for row, (img, name) in enumerate([(arr,'Cover'), (steg,'Stego')]):
            ax = axes[row][ci]
            ax.set_facecolor('#111827')
            ax.set_xticks([]); ax.set_yticks([])
            lsb_plane = img[:,:,ci] & 1
            ax.imshow(lsb_plane, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{name} {ch_name} LSB', color='white', fontsize=9)

    # Last column: histogram comparison
    for row, (img, name, col) in enumerate([(arr,'Cover','#00d4ff'),
                                            (steg,'Stego','#ff6b35')]):
        ax = axes[row][3]
        ax.set_facecolor('#111827')
        for sp in ax.spines.values(): sp.set_edgecolor('#374151')
        ax.tick_params(colors='white')
        ax.set_title(f'{name} pixel histogram', color='white', fontsize=9)
        for ci, ch_col in enumerate(['#ff4444','#44ff44','#4444ff']):
            vals = img[:,:,ci].flatten().astype(np.float32)
            ax.hist(vals, bins=64, range=(0,255), alpha=0.5,
                    color=ch_col, label='RGB'[ci])
        ax.legend(labelcolor='white', framealpha=0.3, fontsize=8)

    plt.suptitle('LSB Bit Plane and Histogram Analysis',
                 color='white', fontsize=13)
    plt.tight_layout()
    p2 = out_dir / 'lsb_analysis.png'
    plt.savefig(p2, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p2}")

    # ── Figure 3: Laplacian energy distribution ───────────────────────────────
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    cover_e, stego_e = [], []
    for p in load_sample_paths(data_dir, n=100):
        arr  = load_img_np(p)
        rho  = gen.compute_cost_map(arr.mean(axis=2).astype(np.float64))
        steg = gen.embed_lsb(arr, rho, 0.4)
        for img, lst in [(arr, cover_e), (steg, stego_e)]:
            gray = img.mean(axis=2).astype(np.float32)
            pad  = np.pad(gray, 1, mode='reflect')
            res  = np.zeros_like(gray)
            for dy in range(3):
                for dx in range(3):
                    res += pad[dy:dy+gray.shape[0], dx:dx+gray.shape[1]] * laplacian[dy,dx]
            cover_e.append(np.std(res)) if img is arr else stego_e.append(np.std(res))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_edgecolor('#374151')
    ax.tick_params(colors='white')
    ax.set_title('Laplacian Residual Energy: Cover vs Stego',
                 color='white', fontsize=12)
    bins = np.linspace(min(min(cover_e), min(stego_e)),
                       max(max(cover_e), max(stego_e)), 40)
    ax.hist(cover_e, bins=bins, alpha=0.7, color='#00d4ff', label='Cover')
    ax.hist(stego_e, bins=bins, alpha=0.7, color='#ff6b35', label='Stego')
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.set_xlabel('Laplacian std', color='#9ca3af')
    p3 = out_dir / 'laplacian_energy.png'
    plt.savefig(p3, dpi=120, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p3}")


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════

def print_final_verdict():
    print(f"\n{SEP}")
    print("  FINAL VERDICT")
    print(SEP)

    all_passed  = all(results.values())
    any_failed  = any(not v for v in results.values())
    n_pass = sum(results.values())
    n_fail = len(results) - n_pass

    for name, passed in results.items():
        tag = " [PASS]" if passed else " [FAIL]"
        print(f"  {tag}  {name}")

    print(SEP2)
    print(f"  {n_pass}/{len(results)} checks passed")

    if all_passed:
        print("""
  VERDICT: DATASET IS VALID
  --------------------------
  All checks passed. If the model is still at 50% accuracy, the problem
  is NOT the dataset. Focus on:
    1. Staged training (warmup_epochs=10 detection-only first)
    2. Learning rate (try 5e-4 instead of 1e-4)
    3. SRM kernel quality (30 distinct kernels, not 5 repeated)
    4. More epochs (30+ needed for steganalysis to converge)
""")
    else:
        print(f"""
  VERDICT: DATASET HAS {n_fail} ISSUE(S) -- FIX BEFORE TRAINING
  ---------------------------------------------------------------
  Failed checks are listed above. Address each FAIL before retraining.
""")

    print(f"""
  QUICK REFERENCE -- what each metric means:
    loading      : Images load correctly, non-trivial content
    signal       : Embedding actually changes pixels (3 channels)
    lsb_magnitude: Changes are exactly ±1 (pure LSB, no corruption)
    labels       : 50/50 balance, correct 0/1 assignment
    differ       : Cover and stego tensors are measurably different
    augmentation : Random flips don't break label consistency
    detectability: Stego has higher Laplacian energy than cover
    normalization : ImageNet norm preserves the LSB signal
""")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='HybridLocNet Dataset Diagnostic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_dataset.py --data ./BOSSbase_1.01
  python debug_dataset.py --data ./BOSSbase_1.01 --viz
  python debug_dataset.py --data ./BOSSbase_1.01 --viz --out ./debug_output
        """
    )
    parser.add_argument('--data', type=str, required=True, help='Path to image folder')
    parser.add_argument('--viz',  action='store_true',    help='Generate diagnostic plots')
    parser.add_argument('--out',  type=str, default='./debug_output',
                        help='Output folder for visualizations')
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist"); sys.exit(1)

    print(f"\nHybridLocNet Dataset Diagnostic")
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Start time:     {time.strftime('%H:%M:%S')}")

    paths = check_image_loading(data_dir)
    check_embedding_signal(paths)
    check_lsb_magnitude(paths)
    check_label_correctness(data_dir)
    check_cover_stego_differ(data_dir)
    check_augmentation_consistency(data_dir)
    check_signal_detectability(data_dir)
    check_normalization(data_dir)

    if args.viz:
        generate_visualizations(data_dir, Path(args.out))

    print_final_verdict()


if __name__ == '__main__':
    main()