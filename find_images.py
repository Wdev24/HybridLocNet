#!/usr/bin/env python3
"""Quick script to find actual image paths for demo.py"""
import sys
from pathlib import Path

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./BOSSbase_1.01')
exts = {'.pgm', '.png', '.jpg', '.jpeg', '.bmp'}

paths = sorted([p for p in data_dir.iterdir()
                if p.suffix.lower() in exts and p.is_file()])

if not paths:
    paths = sorted([p for p in data_dir.rglob('*')
                    if p.suffix.lower() in exts and p.is_file()])

if not paths:
    print(f"No images found in {data_dir}")
    sys.exit(1)

print(f"Found {len(paths)} images. First 5 paths:")
for p in paths[:5]:
    print(f"  {p}")

print()
print("Demo commands to copy-paste:")
print(f'  python demo.py --image "{paths[0]}" --checkpoint checkpoints/best.pt --overlay')
print()
print("Batch demo (cover vs stego comparison):")
if len(paths) >= 3:
    imgs = " ".join(f'"{p}"' for p in paths[:3])
    print(f'  python demo.py --batch {imgs} --checkpoint checkpoints/best.pt')