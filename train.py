"""
HybridLocNet — Training Entry Point (Improved)
===============================================

CHANGES FROM ORIGINAL:

1. n_epochs passed to Trainer:
   CosineAnnealingLR needs T_max = total epochs. Old Trainer hardcoded T_max=50.
   Now train.py passes args.epochs to Trainer.__init__ so T_max is always correct.

2. --start-epoch removed (now handled inside load_checkpoint):
   trainer.load_checkpoint() sets self.start_epoch = saved_epoch + 1 automatically.
   No need for a separate CLI arg.

3. Final test report includes AUC, F1, wFUS@20%:
   These are the metrics reviewers and paper readers care about, not just accuracy.

4. if __name__ == '__main__' guard:
   Required on Windows for DataLoader with num_workers > 0.
   Without this, each worker process re-runs the module on spawn, causing
   recursive process creation and immediate crash.

5. Training time estimate printed before starting:
   Helps decide whether to wait for training to finish or use latest.pt.

6. --lambda1 / --lambda2 CLI args:
   Allows overriding Stage 2 loss weights at launch without editing trainer.py.
   Critical for Phase 2 fine-tuning runs with boosted lambda1.
"""

import argparse
import os
import sys
import json
import time
import torch
from pathlib import Path

# Required for Windows DataLoader multiprocessing
# Must be at top-level before any torch imports in spawned workers
if __name__ == '__main__':
    # Prevents recursive worker spawn on Windows
    torch.multiprocessing.set_start_method('spawn', force=True)

sys.path.insert(0, str(Path(__file__).parent))

from models.hybridlocnet import HybridLocNet
from data.dataset import get_dataloaders
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train HybridLocNet — Multi-task Steganalysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    parser.add_argument('--data',         type=str, required=True,
                        help='Path to image directory or BOSSBase root')
    parser.add_argument('--mode',         type=str, default='auto',
                        choices=['auto', 'bossbase', 'flat'],
                        help='Dataset layout mode')
    parser.add_argument('--max-images',   type=int, default=None,
                        help='Cap number of images (for quick tests)')
    # Model
    parser.add_argument('--img-size',     type=int,   default=256)
    parser.add_argument('--n-bit-planes', type=int,   default=2,
                        help='Embedding strength: 1=LSB, 2=2-bit, 3=3-bit')
    parser.add_argument('--payload',      type=float, default=0.4,
                        help='Embedding payload rate (fraction of pixels)')
    # Training
    parser.add_argument('--epochs',       type=int,   default=40)
    parser.add_argument('--batch-size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--warmup-epochs',type=int,   default=10,
                        help='Epochs of detection-only Stage 1')
    parser.add_argument('--ramp-epochs',  type=int,   default=5,
                        help='Epochs to ramp multi-task lambdas to final values')
    # CHANGE: Lambda overrides — needed for Phase 2 fine-tuning with boosted
    # localization weight. Without these, --lambda1 on the CLI is silently
    # ignored and training defaults to 0.30 regardless of what you pass.
    parser.add_argument('--lambda1',      type=float, default=0.3,
                        help='Final localization loss weight (Stage 2)')
    parser.add_argument('--lambda2',      type=float, default=0.1,
                        help='Final payload loss weight (Stage 2)')
    # Infrastructure
    parser.add_argument('--output-dir',   type=str, default='./checkpoints')
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--device',       type=str, default='auto')
    parser.add_argument('--resume',       type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. checkpoints/latest.pt)')
    parser.add_argument('--save-every',   type=int, default=10,
                        help='Save epoch_NNN.pt every N epochs')
    args = parser.parse_args()

    # ── Environment setup ────────────────────────────────────────────────────────
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nDevice: cuda ({total_gb:.1f} GB)")
        # AMP halves VRAM usage, so batch=16 is safe on 8GB with AMP on
        if total_gb <= 6 and args.batch_size > 8:
            args.batch_size = 8
            print(f"  Capped batch_size=8 for {total_gb:.1f}GB VRAM")
    else:
        print(f"\nDevice: {device}")
        if args.workers > 0:
            print("  Note: num_workers>0 on CPU can be slow. Consider --workers 0")

    args.img_size = 256  # Hard-coded: SRM kernel geometry assumes 256x256

    if args.mode == 'auto':
        args.mode = 'bossbase' if (Path(args.data) / 'cover').exists() else 'flat'

    print(f"Mode: {args.mode}  |  n_bit_planes: {args.n_bit_planes}  "
          f"(max pixel change: ±{2**args.n_bit_planes - 1})")

    # ── Reproducibility ──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ── Data ─────────────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data}")
    loaders = get_dataloaders(
        image_dir    = args.data,
        batch_size   = args.batch_size,
        img_size     = args.img_size,
        payload_rate = args.payload,
        n_bit_planes = args.n_bit_planes,
        max_images   = args.max_images,
        num_workers  = args.workers,
    )
    n_train = len(loaders['train'].dataset)
    n_val   = len(loaders['val'].dataset)
    n_test  = len(loaders['test'].dataset)
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # ── Model ────────────────────────────────────────────────────────────────────
    model    = HybridLocNet(cf=256)
    n_params = model.count_parameters()
    print(f"  Model parameters: {n_params/1e6:.2f}M")

    # Rough time estimate (based on observed ~330s/epoch at batch=8 on RTX)
    batches_per_epoch = n_train // args.batch_size
    est_sec_per_epoch = batches_per_epoch * 0.08   # ~80ms per batch on RTX with AMP
    est_total_min     = (args.epochs * est_sec_per_epoch) / 60
    print(f"  Estimated training time: ~{est_total_min:.0f} min "
          f"({est_sec_per_epoch:.0f}s/epoch × {args.epochs} epochs)")

    # ── Trainer ──────────────────────────────────────────────────────────────────
    # CHANGE: Pass n_epochs to Trainer so CosineAnnealingLR T_max is correct.
    # CHANGE: Pass lambda1/lambda2 from CLI so Phase 2 fine-tuning runs pick
    #         up the boosted weights without editing trainer.py directly.
    trainer = Trainer(
        model,
        loaders,
        device         = device,
        output_dir     = args.output_dir,
        lr             = args.lr,
        warmup_epochs  = args.warmup_epochs,
        ramp_epochs    = args.ramp_epochs,
        n_epochs       = args.epochs,
        lambda1_final  = args.lambda1,   # CHANGE: was hardcoded 0.3 in Trainer default
        lambda2_final  = args.lambda2,   # CHANGE: was hardcoded 0.1 in Trainer default
    )

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        # CHANGE: load_checkpoint now sets trainer.start_epoch automatically.
        # No need to pass --start-epoch separately.
        trainer.load_checkpoint(args.resume)

    # ── Training ─────────────────────────────────────────────────────────────────
    print()
    t_start  = time.time()
    history  = trainer.train(n_epochs=args.epochs, save_every=args.save_every)
    t_elapsed= time.time() - t_start
    print(f"\nTotal training time: {t_elapsed/3600:.2f}h")

    # ── Final test evaluation ─────────────────────────────────────────────────────
    print("\nRunning final test evaluation (best.pt)...")
    # Load best weights for test evaluation
    best_pt = Path(args.output_dir) / 'best.pt'
    if best_pt.exists():
        ck = torch.load(best_pt, map_location=device)
        trainer.model.load_state_dict(ck['model_state'], strict=False)
        print(f"  Loaded best.pt from epoch {ck.get('epoch', '?')}")

    test_m = trainer._run_epoch('test')

    # CHANGE: Extended final report — includes AUC, F1, wFUS.
    sep = '=' * 55
    print(f"\n{sep}")
    print("  FINAL TEST RESULTS (best.pt)")
    print(sep)
    print(f"  Detection Accuracy : {test_m['det_acc']:.4f}  "
          f"({test_m['det_acc']*100:.1f}%)")
    print(f"  AUC-ROC            : {test_m['auc']:.4f}")
    print(f"  F1 Score           : {test_m['f1']:.4f}")
    print(f"  Soft IoU           : {test_m['iou']:.4f}")
    print(f"  wFUS @ 20%         : {test_m['wfus20']:.4f}  "
          f"(baseline: 0.200, {test_m['wfus20']/0.2:.1f}x improvement)")
    print(f"  Payload MAE        : {test_m['mae']:.6f} bpp")
    print(sep)

    # ── Save history ─────────────────────────────────────────────────────────────
    history_path = Path(args.output_dir) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(
            {k: [{kk: float(vv) for kk, vv in m.items()} for m in v]
             for k, v in history.items()},
            f, indent=2
        )
    print(f"\nHistory saved -> {history_path}")
    print(f"Checkpoints  -> {args.output_dir}/best.pt  (and latest.pt)")


if __name__ == '__main__':
    main()