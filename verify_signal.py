#!/usr/bin/env python3
"""
verify_signal.py — Quick pre-training signal sanity check.

Runs in ~10 seconds. Confirms:
  1. Embedding actually modifies pixels (3 channels, correct count)
  2. Signal is strong enough for the model to learn
  3. Labels are correct

Run: python verify_signal.py --data ./BOSSbase_1.01
Expected output: ALL CHECKS PASS — READY TO TRAIN
"""

import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from PIL import Image
except ImportError:
    print("pip install Pillow"); sys.exit(1)

try:
    from data.dataset import SyntheticStegoGenerator, SyntheticStegoDataset
except Exception as e:
    print(f"ERROR importing dataset: {e}"); sys.exit(1)


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"        {detail}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--n-bit-planes', type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data)
    gen = SyntheticStegoGenerator()

    exts = {'.pgm','.png','.jpg','.jpeg','.bmp'}
    paths = sorted([p for p in data_dir.iterdir()
                    if p.suffix.lower() in exts])[:20]
    if not paths:
        paths = sorted([p for p in data_dir.rglob('*')
                        if p.suffix.lower() in exts])[:20]

    if not paths:
        print("ERROR: No images found"); sys.exit(1)

    print(f"\nverify_signal.py — n_bit_planes={args.n_bit_planes}")
    print(f"Images: {len(paths)} samples from {data_dir}\n")

    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    all_pass = True

    # ── Test 1: Pixel change count ─────────────────────────────────────────────
    changed_counts, expected_counts = [], []
    max_diffs = []

    for p in paths[:10]:
        img = Image.open(p).convert('RGB')
        img = img.resize((256,256), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        rho = gen.compute_cost_map(arr.mean(axis=2))
        stego = gen.embed(arr, rho, 0.4, args.n_bit_planes)

        diff = np.abs(stego.astype(np.int16) - arr.astype(np.int16))
        changed = (diff > 0).any(axis=2).sum()
        changed_counts.append(changed)
        expected = int(256*256*0.4 * (1 - (0.5)**args.n_bit_planes))
        expected_counts.append(expected)
        max_diffs.append(diff.max())

    avg_changed  = np.mean(changed_counts)
    avg_expected = np.mean(expected_counts)
    max_diff_val = max(max_diffs)
    expected_max = 2**args.n_bit_planes - 1

    ok1 = avg_changed > avg_expected * 0.6
    all_pass &= check(
        "Pixel change count",
        ok1,
        f"Changed {avg_changed:.0f} pixels (expected ~{avg_expected:.0f}, "
        f"min acceptable {avg_expected*0.6:.0f})"
    )

    ok2 = max_diff_val <= expected_max
    all_pass &= check(
        "Max pixel change magnitude",
        ok2,
        f"Max diff = {max_diff_val}  (expected <= {expected_max} for {args.n_bit_planes}-bit planes)"
    )

    ok3 = max_diff_val >= 1
    all_pass &= check(
        "Embedding is non-trivial",
        ok3,
        "At least 1 pixel value changed"
    )

    # ── Test 2: Laplacian energy difference ────────────────────────────────────
    cover_e, stego_e = [], []
    for p in paths[:20]:
        img = Image.open(p).convert('RGB')
        img = img.resize((256,256), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        rho = gen.compute_cost_map(arr.mean(axis=2))
        stego = gen.embed(arr, rho, 0.4, args.n_bit_planes)

        for src, lst in [(arr, cover_e), (stego, stego_e)]:
            gray = src.mean(axis=2).astype(np.float32)
            pad  = np.pad(gray, 1, mode='reflect')
            res  = np.zeros_like(gray)
            for dy in range(3):
                for dx in range(3):
                    res += pad[dy:dy+256, dx:dx+256] * laplacian[dy, dx]
            lst.append(np.std(res))

    delta_pct = (np.mean(stego_e) - np.mean(cover_e)) / np.mean(cover_e) * 100

    # Thresholds: 2-bit should give ~0.4-1.0%, 3-bit ~1-3%
    min_delta = {1: 0.03, 2: 0.3, 3: 1.0}.get(args.n_bit_planes, 0.1)
    ok4 = delta_pct >= min_delta
    all_pass &= check(
        "Laplacian energy delta",
        ok4,
        f"Delta = {delta_pct:.3f}%  (min for {args.n_bit_planes}-bit: {min_delta}%)"
    )

    # ── Test 3: Dataset label correctness ──────────────────────────────────────
    try:
        ds = SyntheticStegoDataset(str(data_dir), img_size=256,
                                   payload_rate=0.4,
                                   n_bit_planes=args.n_bit_planes,
                                   augment=False, max_images=20)
        n_imgs = len(ds.image_paths)
        n_cover = sum(1 for _,s in ds.pairs if not s)
        n_stego = sum(1 for _,s in ds.pairs if s)

        ok5 = n_cover == n_stego
        all_pass &= check(
            "Class balance",
            ok5,
            f"Cover={n_cover}, Stego={n_stego}"
        )

        # Check a cover sample has det=0 and a stego sample has det=1
        c_sample = ds[0]
        s_sample = ds[n_imgs]  # stego half
        ok6 = c_sample['det'].item() == 0.0 and s_sample['det'].item() == 1.0
        all_pass &= check(
            "Label assignment",
            ok6,
            f"Cover label={c_sample['det'].item()}, Stego label={s_sample['det'].item()}"
        )

        # Check cover rho is zero, stego rho is nonzero
        ok7 = (c_sample['loc_map'].sum().item() == 0 and
               s_sample['loc_map'].sum().item() > 1.0)
        all_pass &= check(
            "Rho map assignment",
            ok7,
            f"Cover rho_sum={c_sample['loc_map'].sum():.2f}, "
            f"Stego rho_sum={s_sample['loc_map'].sum():.1f}"
        )

        # Check cover and stego tensors are different
        c_img = c_sample['image']
        s_img = s_sample['image']
        l1 = (s_img - c_img).abs().mean().item()
        ok8 = l1 > 1e-4
        all_pass &= check(
            "Cover != Stego tensors",
            ok8,
            f"L1 difference = {l1:.6f}  (expected > 1e-4)"
        )

    except Exception as e:
        print(f"  [FAIL] Dataset init: {e}")
        all_pass = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    if all_pass:
        print("  ALL CHECKS PASS — READY TO TRAIN")
        print()
        print("  Run:")
        print(f"  python train.py --data {args.data} --mode flat \\")
        print(f"    --epochs 30 --batch-size 8 \\")
        print(f"    --n-bit-planes {args.n_bit_planes} \\")
        print(f"    --warmup-epochs 10 --lr 5e-4")
        print()
        print("  Expected: acc > 70% by epoch 15, > 80% by epoch 25")
    else:
        print("  SOME CHECKS FAILED — see above")
        print("  Do NOT start training until all checks pass")
    print("=" * 50)

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()