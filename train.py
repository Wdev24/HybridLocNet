"""
HybridLocNet -- Training Entry Point
"""
import argparse
import os
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.hybridlocnet import HybridLocNet
from data.dataset import get_dataloaders
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train HybridLocNet')
    parser.add_argument('--data',          type=str, required=True)
    parser.add_argument('--mode',          type=str, default='auto',
                        choices=['auto','bossbase','flat'])
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch-size',    type=int,   default=16)
    parser.add_argument('--img-size',      type=int,   default=256)
    parser.add_argument('--payload',       type=float, default=0.4)
    parser.add_argument('--n-bit-planes',  type=int,   default=2,
                        help='Embedding strength: 1=pure LSB, 2=2-bit (default), 3=3-bit')
    parser.add_argument('--max-images',    type=int,   default=None)
    parser.add_argument('--output-dir',    type=str,   default='./checkpoints')
    parser.add_argument('--workers',       type=int,   default=4)
    parser.add_argument('--resume',        type=str,   default=None)
    parser.add_argument('--device',        type=str,   default='auto')
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int,   default=10)
    parser.add_argument('--ramp-epochs',   type=int,   default=5)
    args = parser.parse_args()

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nDevice: cuda ({total_gb:.1f} GB)")
        if total_gb <= 10 and args.batch_size > 8:
            args.batch_size = 8
            print(f"  Capped batch_size=8 for {total_gb:.1f}GB VRAM")
    else:
        print(f"\nDevice: {device}")

    args.img_size = 256

    if args.mode == 'auto':
        args.mode = 'bossbase' if (Path(args.data) / 'cover').exists() else 'flat'
    print(f"Mode: {args.mode}  |  n_bit_planes: {args.n_bit_planes}  "
          f"(max pixel change: +-{2**args.n_bit_planes - 1})")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Loading data from: {args.data}")
    loaders = get_dataloaders(
        image_dir=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        payload_rate=args.payload,
        n_bit_planes=args.n_bit_planes,
        max_images=args.max_images,
        num_workers=args.workers,
    )
    print(f"  Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    model    = HybridLocNet(cf=256)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.2f}M")

    trainer = Trainer(
        model, loaders,
        device=device,
        output_dir=args.output_dir,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train(n_epochs=args.epochs, save_every=10)

    print("\nFinal test evaluation...")
    test_m = trainer._run_epoch('test')
    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Detection Acc : {test_m['det_acc']:.4f} ({test_m['det_acc']*100:.1f}%)")
    print(f"Soft IoU      : {test_m['iou']:.4f}")
    print(f"Payload MAE   : {test_m['mae']:.6f} bpp")
    print(f"{'='*50}")

    import json
    with open(Path(args.output_dir) / 'history.json', 'w') as f:
        json.dump({k: [{kk: float(vv) for kk, vv in m.items()} for m in v]
                   for k, v in history.items()}, f, indent=2)
    print(f"History saved -> {args.output_dir}/history.json")


if __name__ == '__main__':
    main()