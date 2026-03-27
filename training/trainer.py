"""
Training pipeline for HybridLocNet — fixed multi-task loss and staged training.

ROOT CAUSE FIXED:
  Old: L = L_cls + 0.5*L_loc(w=5) + 0.3*L_pay at ALL epochs.
       L_loc dominates early, L_cls gets no gradient, acc stuck at 50%.
  New: Stage 1 (warmup_epochs): L = L_cls only, LR=5e-4
       Stage 2 (remaining):     L = L_cls + lambda1*L_loc + lambda2*L_pay
       lambda1 and lambda2 start at 0.05 and ramp up over 5 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
import numpy as np
import time


# ── Loss Functions ──────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """
    Focal loss for detection. Focuses training on hard examples.
    gamma=2: standard value, down-weights easy (confident) predictions.
    Works with raw logits (more stable than BCE on sigmoid output).
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class WeightedBCELoss(nn.Module):
    """Weighted BCE for localization vs soft rho maps."""
    def __init__(self, w_pos=5.0):
        super().__init__()
        self.w_pos = w_pos

    def forward(self, pred, target):
        weight = torch.where(target > 0.5,
                             torch.full_like(target, self.w_pos),
                             torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')


class HuberPayloadLoss(nn.Module):
    def __init__(self, delta=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')

    def forward(self, pred, target):
        return self.huber(pred, target)


class MultiTaskLoss(nn.Module):
    """
    Staged multi-task loss.
    During warmup (set_stage(1)): only L_cls.
    After warmup (set_stage(2)):  L_cls + lambda1*L_loc + lambda2*L_pay
                                  lambdas ramp up over ramp_epochs.

    WHY BCEWithLogitsLoss NOT FocalBCE:
    At random init (logit~0), p_t=0.5, focal_weight=0.25 -> gradients are
    4x weaker. Focal is designed for 1000:1 class imbalance (object detection).
    Our 50/50 balanced binary task needs full gradient from epoch 1.
    """
    def __init__(self, lambda1_final=0.3, lambda2_final=0.1):
        super().__init__()
        self.lambda1_final = lambda1_final
        self.lambda2_final = lambda2_final
        self.lambda1 = 0.0
        self.lambda2 = 0.0
        self._stage  = 1
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.loc_loss = WeightedBCELoss(w_pos=5.0)
        self.pay_loss = HuberPayloadLoss(delta=0.1)

    def set_stage(self, stage: int):
        self._stage = stage

    def ramp_lambdas(self, frac: float):
        """frac in [0,1]: how far through Stage 2 ramp we are."""
        frac = min(frac, 1.0)
        self.lambda1 = self.lambda1_final * frac
        self.lambda2 = self.lambda2_final * frac

    def forward(self, preds, targets):
        l_cls = self.cls_loss(preds['det'], targets['det'])
        l_loc = self.loc_loss(preds['loc'], targets['loc_map'])
        l_pay = self.pay_loss(preds['pay'], targets['pay_map'])

        if self._stage == 1:
            total = l_cls
        else:
            total = l_cls + self.lambda1 * l_loc + self.lambda2 * l_pay

        return {'total': total, 'cls': l_cls, 'loc': l_loc, 'pay': l_pay}


# ── Metrics ─────────────────────────────────────────────────────────────────────

def compute_detection_accuracy(det_logits, labels, threshold=0.5):
    probs = torch.sigmoid(det_logits)
    preds = (probs > threshold).float()
    return (preds == labels).float().mean().item()


def compute_soft_iou(pred_map, target_map, eps=1e-6):
    intersection = (pred_map * target_map).sum(dim=(-2, -1))
    union = (pred_map + target_map - pred_map * target_map).sum(dim=(-2, -1))
    return ((intersection + eps) / (union + eps)).mean().item()


def compute_payload_mae(pred_pay, target_pay):
    return F.l1_loss(pred_pay, target_pay).item()


def compute_wfus(loc_map, rho_map, k_pct=20):
    B = loc_map.shape[0]
    scores = []
    for i in range(B):
        loc = loc_map[i].flatten()
        rho = rho_map[i].flatten()
        n_k = max(1, int(len(loc) * k_pct / 100))
        topk_idx = loc.topk(n_k).indices
        total_mass = rho.sum()
        if total_mass > 0:
            scores.append((rho[topk_idx].sum() / total_mass).item())
    return np.mean(scores) if scores else 0.0


# ── Training Loop ───────────────────────────────────────────────────────────────

class Trainer:
    """
    Staged trainer:
      Stage 1 (epochs 1..warmup_epochs):  detection only, LR=lr
      Stage 2 (epochs warmup+1..n_epochs): multi-task, lambdas ramp over ramp_epochs
    """
    def __init__(self, model, dataloaders, device='cuda',
                 output_dir='./checkpoints',
                 lr=5e-4, warmup_epochs=10, ramp_epochs=5,
                 lambda1_final=0.3, lambda2_final=0.1):
        self.model       = model.to(device)
        self.dataloaders = dataloaders
        self.device      = device
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.warmup_epochs  = warmup_epochs
        self.ramp_epochs    = ramp_epochs
        self.criterion      = MultiTaskLoss(lambda1_final, lambda2_final)
        self.optimizer      = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.history        = {'train': [], 'val': []}
        self.best_val_acc   = 0.0   # track by accuracy, not loss (more meaningful)

        # Cosine annealing over everything
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

    def _run_epoch(self, split='train'):
        is_train = (split == 'train')
        self.model.train() if is_train else self.model.eval()
        loader = self.dataloaders[split]

        total_losses = {'total': 0.0, 'cls': 0.0, 'loc': 0.0, 'pay': 0.0}
        det_acc = iou = mae = n_batches = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                imgs = batch['image'].to(self.device)
                targets = {
                    'det':     batch['det'].to(self.device),
                    'loc_map': batch['loc_map'].to(self.device),
                    'pay_map': batch['pay_map'].to(self.device),
                }

                preds  = self.model(imgs)
                losses = self.criterion(preds, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                for k in total_losses:
                    total_losses[k] += losses[k].item()

                det_acc += compute_detection_accuracy(preds['det'], targets['det'])
                iou     += compute_soft_iou(preds['loc'].squeeze(1),
                                            targets['loc_map'].squeeze(1))
                mae     += compute_payload_mae(preds['pay'].squeeze(1),
                                               targets['pay_map'].squeeze(1))
                n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()} | {
            'det_acc': det_acc / n_batches,
            'iou':     iou     / n_batches,
            'mae':     mae     / n_batches,
        }

    def train(self, n_epochs=30, save_every=5):
        print(f"{'='*60}")
        print(f"Training HybridLocNet for {n_epochs} epochs on {self.device}")
        print(f"  Stage 1 (detection only): epochs 1-{self.warmup_epochs}")
        print(f"  Stage 2 (multi-task):    epochs {self.warmup_epochs+1}-{n_epochs}")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # ── Set stage and lambda schedule ────────────────────────────────
            if epoch <= self.warmup_epochs:
                self.criterion.set_stage(1)
                stage_str = "DET"
            else:
                self.criterion.set_stage(2)
                ramp_frac = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
                self.criterion.ramp_lambdas(ramp_frac)
                stage_str = f"MT(l1={self.criterion.lambda1:.2f})"

            train_m = self._run_epoch('train')
            val_m   = self._run_epoch('val')
            self.scheduler.step()

            self.history['train'].append(train_m)
            self.history['val'].append(val_m)

            elapsed = time.time() - t0
            # Log fusion alpha so we can monitor stream balance
            try:
                alpha_val = torch.sigmoid(self.model.fusion.alpha).item()
                alpha_str = f" alpha={alpha_val:.2f}"
            except Exception:
                alpha_str = ""

            print(
                f"Ep {epoch:3d}/{n_epochs} [{stage_str:10s}] | "
                f"cls={train_m['cls']:.4f} loc={train_m['loc']:.4f} | "
                f"acc={train_m['det_acc']:.3f} IoU={train_m['iou']:.3f} | "
                f"val_acc={val_m['det_acc']:.3f} val_IoU={val_m['iou']:.3f} | "
                f"{elapsed:.0f}s{alpha_str}"
            )

            # Save on best val accuracy (more meaningful than val loss in staged training)
            if val_m['det_acc'] > self.best_val_acc:
                self.best_val_acc = val_m['det_acc']
                self.save_checkpoint('best.pt', epoch, val_m)
                print(f"  >> Best val_acc={self.best_val_acc:.3f} saved")

            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch:03d}.pt', epoch, val_m)

        print(f"\nTraining complete. Best val_acc: {self.best_val_acc:.3f}")
        return self.history

    def save_checkpoint(self, name, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
        }, self.output_dir / name)

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model_state'])
        print(f"Loaded from epoch {ck['epoch']} "
              f"(val_acc={ck['metrics'].get('det_acc', '?'):.3f})")
        return ck