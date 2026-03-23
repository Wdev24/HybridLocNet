#!/usr/bin/env python3
"""
HybridLocNet Demo -- Professional forensic analysis visualization.

Usage:
    python demo.py --image path/to/image.jpg --checkpoint checkpoints/best.pt
    python demo.py --batch img1.jpg img2.jpg img3.jpg --checkpoint checkpoints/best.pt
    python demo.py --image path/to/image.jpg --no-checkpoint
"""

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    print("pip install torch"); sys.exit(1)
try:
    from PIL import Image
except ImportError:
    print("pip install Pillow"); sys.exit(1)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("pip install matplotlib"); sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from models.hybridlocnet import HybridLocNet
from torchvision import transforms


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(img_path: str, img_size: int = 256):
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((img_size, img_size), Image.BILINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return t(img_resized).unsqueeze(0), img_resized


def compute_wfus(loc_map: np.ndarray, k_pct: int = 20) -> float:
    """
    Weighted Forensic Utility Score at k%.
    Fraction of total localization mass in top-k% of pixels.
    This is the right metric for sparse embedding localization.
    """
    flat = loc_map.flatten()
    n_k  = max(1, int(len(flat) * k_pct / 100))
    topk_idx   = np.argpartition(flat, -n_k)[-n_k:]
    mass_top   = flat[topk_idx].sum()
    total_mass = flat.sum()
    return float(mass_top / total_mass) if total_mass > 0 else 0.0


# ── Single image visualization ─────────────────────────────────────────────────

def analyze_image(img_path: str, model: HybridLocNet,
                  device: str, threshold: float = 0.5):
    tensor, img_pil = preprocess(img_path)
    tensor = tensor.to(device)
    result = model.predict(tensor, threshold)
    return {
        'img_pil':   img_pil,
        'det_prob':  result['det_prob'].item(),
        'is_stego':  result['is_stego'].item(),
        'loc_map':   result['loc_map'][0].cpu().numpy(),
        'pay_map':   result['pay_map'][0].cpu().numpy(),
        'path':      img_path,
    }


def render_single(r: dict, save_path: str = None, show: bool = True):
    """4-panel professional forensic output."""
    # Custom colormap: black → dark red → orange → yellow
    stego_cmap = LinearSegmentedColormap.from_list(
        'stego', ['#0d1117','#3d0000','#8b1a1a','#ff4500','#ff8c00','#ffd700'], N=256)

    fig = plt.figure(figsize=(18, 4.8))
    fig.patch.set_facecolor('#080b10')

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)

    det_prob  = r['det_prob']
    is_stego  = det_prob > 0.5
    verdict   = 'STEGO' if is_stego else 'COVER'
    confidence = det_prob if is_stego else (1 - det_prob)
    v_color   = '#ff4d4d' if is_stego else '#4dff91'

    loc  = r['loc_map']
    pay  = r['pay_map']
    wfus = compute_wfus(loc, k_pct=20)

    # Top-20% contour threshold
    thr20 = np.percentile(loc, 80)

    panels = [
        ('Original image',     None),
        ('Detection verdict',  None),
        ('Localization map',   loc),
        ('Payload density',    pay),
    ]

    for col, (title, data) in enumerate(panels):
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor('#080b10')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2a38')

        if col == 0:
            ax.imshow(np.array(r['img_pil']))

        elif col == 1:
            # Dim image + overlay verdict
            ax.imshow(np.array(r['img_pil']), alpha=0.25)
            ax.text(0.5, 0.58, verdict, transform=ax.transAxes,
                    color=v_color, fontsize=30, fontweight='bold',
                    ha='center', va='center',
                    fontfamily='monospace')
            ax.text(0.5, 0.40, f'{confidence:.1%} confidence',
                    transform=ax.transAxes, color='white', fontsize=12,
                    ha='center', va='center')
            ax.text(0.5, 0.28, f'P(stego) = {det_prob:.4f}',
                    transform=ax.transAxes, color='#64748b', fontsize=9,
                    ha='center', va='center', fontfamily='monospace')

        elif col == 2:
            im = ax.imshow(data, cmap=stego_cmap, vmin=0, vmax=data.max())
            # Top-20% contour
            if data.max() > 0:
                ax.contour(data, levels=[thr20], colors=['#00d4ff'],
                           linewidths=0.8, alpha=0.9)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.yaxis.set_tick_params(color='#64748b', labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#64748b')
            cb.outline.set_edgecolor('#1e2a38')

            stats = (f'wFUS@20% = {wfus:.1%}\n'
                     f'max = {data.max():.3f}\n'
                     f'mean = {data.mean():.3f}')
            ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                    color='white', fontsize=7.5, va='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='#0d1117', alpha=0.75,
                              edgecolor='#1e2a38', boxstyle='round,pad=3',
                              linewidth=0.5))

        elif col == 3:
            im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=max(data.max(), 1e-6))
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.yaxis.set_tick_params(color='#64748b', labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#64748b')
            cb.outline.set_edgecolor('#1e2a38')
            stats = (f'peak = {data.max():.4f} bpp\n'
                     f'mean = {data.mean():.4f} bpp')
            ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                    color='white', fontsize=7.5, va='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='#0d1117', alpha=0.75,
                              edgecolor='#1e2a38', boxstyle='round,pad=3',
                              linewidth=0.5))

        ax.set_title(title, color='#94a3b8', fontsize=10, pad=6)

    stem = Path(r['path']).stem
    fig.suptitle(
        f'HybridLocNet Forensic Analysis  |  {stem}  |  '
        f'Verdict: {verdict}  ({confidence:.1%} confidence)',
        color=v_color, fontsize=12, fontweight='bold', y=1.01
    )

    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#080b10')
        print(f'Saved: {save_path}')
    if show:
        plt.show()
    plt.close()
    return fig


# ── Batch comparison ───────────────────────────────────────────────────────────

def render_batch(results: list, save_path: str = None, show: bool = True):
    """
    Compact grid: N images x 3 rows (original / loc / pay).
    Best for demo slide screenshots.
    """
    stego_cmap = LinearSegmentedColormap.from_list(
        'stego', ['#0d1117','#3d0000','#8b1a1a','#ff4500','#ff8c00','#ffd700'], N=256)

    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(4.5*n, 11))
    fig.patch.set_facecolor('#080b10')
    if n == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    row_labels = ['Original', 'Localization', 'Payload density']

    for col, r in enumerate(results):
        det_prob  = r['det_prob']
        is_stego  = det_prob > 0.5
        verdict   = 'STEGO' if is_stego else 'COVER'
        confidence = det_prob if is_stego else (1 - det_prob)
        v_color   = '#ff4d4d' if is_stego else '#4dff91'
        wfus      = compute_wfus(r['loc_map'], k_pct=20)
        thr20     = np.percentile(r['loc_map'], 80)

        for row in range(3):
            ax = axes[row][col]
            ax.set_facecolor('#0d1117')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#1e2a38')

        # Row 0: original + verdict badge
        axes[0][col].imshow(np.array(r['img_pil']))
        fname = Path(r['path']).name[:20]
        axes[0][col].set_title(
            f'{fname}\n{verdict}  {confidence:.0%}',
            color=v_color, fontsize=9, pad=4)

        # Row 1: localization
        axes[1][col].imshow(r['loc_map'], cmap=stego_cmap)
        if r['loc_map'].max() > 0:
            axes[1][col].contour(r['loc_map'], levels=[thr20],
                                 colors=['#00d4ff'], linewidths=0.8, alpha=0.9)
        if col == 0:
            axes[1][col].set_ylabel('Localization', color='#94a3b8', fontsize=9)
        axes[1][col].text(0.02, 0.97, f'wFUS@20%={wfus:.1%}',
                          transform=axes[1][col].transAxes,
                          color='white', fontsize=7.5, va='top',
                          fontfamily='monospace',
                          bbox=dict(facecolor='#0d1117', alpha=0.7,
                                    edgecolor='none', pad=2))

        # Row 2: payload
        axes[2][col].imshow(r['pay_map'], cmap='YlOrRd')
        if col == 0:
            axes[2][col].set_ylabel('Payload density', color='#94a3b8', fontsize=9)

    if n > 1:
        fig.suptitle('HybridLocNet — Batch Forensic Analysis',
                     color='white', fontsize=13, fontweight='bold')

    try:
        plt.tight_layout(pad=1.5)
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#080b10')
        print(f'Saved: {save_path}')
    if show:
        plt.show()
    plt.close()
    return fig


# ── Heatmap overlay (most impressive for demos) ────────────────────────────────

def render_overlay(r: dict, save_path: str = None, show: bool = True):
    """
    Single image with heatmap blended over it.
    The most visually striking output for presentations.
    """
    import matplotlib.cm as mcm

    img_arr = np.array(r['img_pil'], dtype=np.float32) / 255.0
    loc     = r['loc_map']

    # Normalize loc to [0,1]
    loc_norm = (loc - loc.min()) / (loc.max() - loc.min() + 1e-8)

    # Apply hot colormap to loc
    cmap     = plt.get_cmap('hot')
    heat_rgb = cmap(loc_norm)[:, :, :3]

    # Blend: more transparent where loc is low, opaque where high
    alpha  = loc_norm[:, :, np.newaxis] * 0.65
    overlay = img_arr * (1 - alpha) + heat_rgb * alpha
    overlay = np.clip(overlay, 0, 1)

    det_prob  = r['det_prob']
    is_stego  = det_prob > 0.5
    verdict   = 'STEGO' if is_stego else 'COVER'
    confidence = det_prob if is_stego else (1 - det_prob)
    v_color   = '#ff4d4d' if is_stego else '#4dff91'
    wfus      = compute_wfus(loc, k_pct=20)
    thr20     = np.percentile(loc, 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('#080b10')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#0d1117')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2a38')

    ax1.imshow(np.array(r['img_pil']))
    ax1.set_title('Original', color='#94a3b8', fontsize=11, pad=6)
    ax1.text(0.5, 0.02, f'{verdict}  |  {confidence:.1%} confidence',
             transform=ax1.transAxes, color=v_color,
             fontsize=11, fontweight='bold', ha='center', va='bottom',
             bbox=dict(facecolor='#0d1117', alpha=0.8, edgecolor='none', pad=4))

    ax2.imshow(overlay)
    ax2.contour(loc, levels=[thr20], colors=['#00d4ff'],
                linewidths=1.0, alpha=0.9)
    ax2.set_title('Stego localization heatmap', color='#94a3b8', fontsize=11, pad=6)
    ax2.text(0.5, 0.02,
             f'Cyan contour = top 20% suspicious pixels  |  wFUS@20% = {wfus:.1%}',
             transform=ax2.transAxes, color='#00d4ff',
             fontsize=9, ha='center', va='bottom',
             bbox=dict(facecolor='#0d1117', alpha=0.8, edgecolor='none', pad=3))

    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#080b10')
        print(f'Saved: {save_path}')
    if show:
        plt.show()
    plt.close()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HybridLocNet Forensic Analysis Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --image test.jpg --checkpoint checkpoints/best.pt
  python demo.py --image test.jpg --overlay --checkpoint checkpoints/best.pt
  python demo.py --batch a.jpg b.jpg c.jpg --checkpoint checkpoints/best.pt
  python demo.py --image test.jpg --no-checkpoint   # architecture demo
        """
    )
    parser.add_argument('--image',         type=str,   help='Single image path')
    parser.add_argument('--batch',         nargs='+',  help='Multiple image paths')
    parser.add_argument('--checkpoint',    type=str,   default=None)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--overlay',       action='store_true',
                        help='Render heatmap overlay (most visually impressive)')
    parser.add_argument('--out',           type=str,   default='demo_output')
    parser.add_argument('--no-show',       action='store_true')
    parser.add_argument('--threshold',     type=float, default=0.5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = HybridLocNet(cf=256)
    if args.checkpoint and not args.no_checkpoint:
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck['model_state'])
        print(f'Loaded: {args.checkpoint}')
    else:
        print('NOTE: Random weights (architecture demo only)')
    model = model.to(device)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    images = args.batch if args.batch else ([args.image] if args.image else [])
    if not images:
        parser.print_help(); return

    results = []
    for img_path in images:
        print(f'Analyzing: {img_path}')
        r = analyze_image(img_path, model, device, args.threshold)
        verdict = 'STEGO' if r['is_stego'] else 'COVER'
        conf    = r['det_prob'] if r['is_stego'] else (1 - r['det_prob'])
        wfus    = compute_wfus(r['loc_map'], k_pct=20)
        print(f'  Verdict:   {verdict}  ({conf:.1%} confidence)')
        print(f'  P(stego):  {r["det_prob"]:.4f}')
        print(f'  wFUS@20%:  {wfus:.1%}  (embedding mass recovered in top-20% pixels)')
        print(f'  Pay peak:  {r["pay_map"].max():.4f} bpp')
        results.append(r)

    # Output files
    stem = Path(images[0]).stem

    if args.overlay and len(images) == 1:
        render_overlay(results[0],
                       save_path=str(out_dir / f'{stem}_overlay.png'),
                       show=not args.no_show)

    if len(images) == 1:
        render_single(results[0],
                      save_path=str(out_dir / f'{stem}_analysis.png'),
                      show=not args.no_show)
    else:
        render_batch(results,
                     save_path=str(out_dir / 'batch_analysis.png'),
                     show=not args.no_show)


if __name__ == '__main__':
    main()