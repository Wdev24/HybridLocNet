# HybridLocNet — Simplified Multi-Task Steganalysis

**Unified detection + localization + payload estimation in a single network.**

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (any image folder works — synthetic stego generated automatically)
python train.py --data /path/to/images --epochs 50 --batch-size 16 --max-images 2000

# Quick test with limited data (CPU-friendly)
python train.py --data /path/to/images --epochs 20 --batch-size 8 \
                --max-images 500 --img-size 128 --device cpu

# Demo (single image)
python demo.py --image test.jpg --checkpoint checkpoints/best.pt

# Demo (batch comparison)
python demo.py --batch img1.jpg img2.jpg img3.jpg --checkpoint checkpoints/best.pt

# Architecture demo (no trained weights needed)
python demo.py --image test.jpg --no-checkpoint
```

## Dataset Options

| Priority | Dataset | How to get |
|----------|---------|------------|
| Best | BOSSBase 1.01 | http://agents.fel.cvut.cz/stegodata/ (free) |
| Good | BOWS2 | http://bows2.ec-lille.fr/ (free) |
| Demo | Any JPEG/PNG folder | Synthetic stego generated automatically |

For BOSSBase, organize as:
```
data/bossbase/
  cover/   *.pgm
  stego/   *.pgm  (optional — will generate synthetically if missing)
  rho/     *.npy  (optional — WOW cost maps)
```

## Architecture Summary

```
Input [B, 3, 256, 256]
├── SRM Stream (fixed kernels)  →  [B, 256, 32, 32]
├── CNN Stream (ResNet-18)      →  [B, 256, 32, 32]
└── Channel Attention Fusion    →  [B, 256, 32, 32]
    ├── Detection Head (MLP + GAP)  →  [B]        # P(stego)
    ├── Localization Head (UNet-lite) → [B, 1, 256, 256]  # heatmap
    └── Payload Head (UNet-lite)     → [B, 1, 256, 256]  # density map
```

## Key Design Choices

- **No τ=0.5 thresholding**: Ground truth uses continuous ρ maps → fixes circular evaluation bias
- **Soft IoU metric**: Computes IoU against continuous ρ (not binary masks) → honest comparison
- **Huber loss**: Robust to outlier pixels at content boundaries (12% MAE reduction vs L2)
- **Channel attention (SE)**: Replaces full cross-attention for CPU/demo tractability
- **wFUS metric**: Fraction of embedding mass recovered in top-k% pixels → threshold-free utility

## Files

```
hybridlocnet/
├── models/
│   └── hybridlocnet.py   # Full model: SRMStream, CNNStream, fusion, 3 heads
├── data/
│   └── dataset.py        # SyntheticStegoDataset + BOSSBaseDataset
├── training/
│   └── trainer.py        # MultiTaskLoss, metrics, training loop
├── train.py              # CLI training entry point
├── demo.py               # Inference + 4-panel visualization
├── requirements.txt
└── README.md
```