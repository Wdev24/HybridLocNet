"""
HybridLocNet - Simplified Multi-Task Steganalysis
Architecture: SRM + ResNet18 + Lightweight Attention Fusion
Tasks: Detection | Localization | Payload Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_srm_kernels():
    """
    30 DISTINCT SRM high-pass kernels.
    BUG FIXED: old version did repeat(6) = 6 copies of 5 kernels.
    This version has 30 genuinely distinct patterns covering
    1st, 2nd, 3rd order residuals in all 8 directions.
    """
    k = [
        # 1st order: 8 directions
        [[ 0, 0, 0], [ 0,-1, 1], [ 0, 0, 0]],
        [[ 0, 0, 0], [ 1,-1, 0], [ 0, 0, 0]],
        [[ 0, 1, 0], [ 0,-1, 0], [ 0, 0, 0]],
        [[ 0, 0, 0], [ 0,-1, 0], [ 0, 1, 0]],
        [[ 0, 0, 1], [ 0,-1, 0], [ 0, 0, 0]],
        [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0, 0]],
        [[ 0, 0, 0], [ 0,-1, 0], [ 0, 0, 1]],
        [[ 0, 0, 0], [ 0,-1, 0], [ 1, 0, 0]],
        # 2nd order Laplacians
        [[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]],
        [[ 1, 1, 1], [ 1,-8, 1], [ 1, 1, 1]],
        [[ 1,-2, 1], [ 0, 0, 0], [ 0, 0, 0]],
        [[ 0, 0, 0], [ 1,-2, 1], [ 0, 0, 0]],
        [[ 0, 0, 0], [ 0, 0, 0], [ 1,-2, 1]],
        [[ 1, 0, 0], [-2, 0, 0], [ 1, 0, 0]],
        [[ 0, 1, 0], [ 0,-2, 0], [ 0, 1, 0]],
        [[ 0, 0, 1], [ 0, 0,-2], [ 0, 0, 1]],
        # 3rd order / LSB-sensitive
        [[ 1,-2, 1], [-2, 4,-2], [ 1,-2, 1]],
        [[-1, 2,-1], [ 2,-4, 2], [-1, 2,-1]],
        [[ 1, 0,-1], [ 2, 0,-2], [ 1, 0,-1]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]],
        [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]],
        [[ 1, 0,-1], [ 0, 0, 0], [-1, 0, 1]],
        [[-1, 0, 1], [ 0, 0, 0], [ 1, 0,-1]],
        # Edge-sensitive
        [[-1, 2,-1], [ 0, 0, 0], [ 1,-2, 1]],
        [[ 0,-1, 0], [ 0, 2, 0], [ 0,-1, 0]],
        [[ 0, 0, 0], [-1, 2,-1], [ 0, 0, 0]],
        [[ 1, 0,-1], [-2, 0, 2], [ 1, 0,-1]],
        [[ 2,-1, 2], [-1,-4,-1], [ 2,-1, 2]],
        [[-1,-1, 2], [-1, 2,-1], [ 2,-1,-1]],
    ]
    k_tensor = torch.tensor(k, dtype=torch.float32).unsqueeze(1)  # [30, 1, 3, 3]
    return k_tensor


class TLU(nn.Module):
    def __init__(self, T=3.0):
        super().__init__()
        self.T = T
    def forward(self, x):
        return torch.clamp(x, -self.T, self.T)


class SRMStream(nn.Module):
    """
    Fixed SRM extractor. Applied per-channel (groups=3) so each of R,G,B
    gets all 30 kernels -> 90 residual channels total.
    This correctly captures LSB signal in ALL channels.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        srm_k = get_srm_kernels()          # [30, 1, 3, 3]
        srm_3ch = srm_k.repeat(3, 1, 1, 1) # [90, 1, 3, 3]
        self.srm_conv = nn.Conv2d(3, 90, kernel_size=3, padding=1,
                                  groups=3, bias=False)
        self.srm_conv.weight = nn.Parameter(srm_3ch, requires_grad=False)
        self.tlu = TLU(T=3.0)
        self.proj = nn.Sequential(
            nn.Conv2d(90, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

    def forward(self, x):
        res     = self.tlu(self.srm_conv(x))
        feat    = self.proj(res)
        feat_ds = self.pool(feat)
        return feat, feat_ds


class CNNStream(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.proj   = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)
        return self.proj(f)


class LightweightAttentionFusion(nn.Module):
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.se_srm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )
        self.se_cnn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))  # CNN-dominant init: exp 6 showed CNN>SRM
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, srm_feat, cnn_feat):
        w_srm = self.se_srm(srm_feat).view(-1, 256, 1, 1)
        w_cnn = self.se_cnn(cnn_feat).view(-1, 256, 1, 1)
        # Clamp alpha to [0.1, 0.9]: prevents either stream being fully suppressed.
        # Exp 6 showed alpha saturated at 1.02 (SRM dominant), hurting performance.
        alpha = torch.sigmoid(self.alpha).clamp(0.1, 0.9)
        fused = torch.cat([alpha * srm_feat * w_srm,
                           (1 - alpha) * cnn_feat * w_cnn], dim=1)
        return self.fusion_conv(fused)


class UNetDecoderLite(nn.Module):
    """stride-8 -> full resolution (3x 2x upsample)."""
    def __init__(self, in_ch=256, out_ch=1, final_activation='sigmoid'):
        super().__init__()
        self.up1 = self._up_block(in_ch, 128)
        self.up2 = self._up_block(128, 64)
        self.up3 = self._up_block(64, 32)
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)
        self.final_act = nn.Sigmoid() if final_activation == 'sigmoid' else nn.ReLU()

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final_act(self.out_conv(x))


class DetectionHead(nn.Module):
    """Outputs RAW LOGIT — use BCEWithLogitsLoss (more numerically stable)."""
    def __init__(self, in_ch=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            # NO Sigmoid — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        return self.mlp(self.gap(x)).squeeze(1)  # [B] logit


class HybridLocNet(nn.Module):
    def __init__(self, cf=256):
        super().__init__()
        self.srm         = SRMStream(out_channels=cf)
        self.cnn         = CNNStream(out_channels=cf)
        self.fusion      = LightweightAttentionFusion(channels=cf)
        self.det_head    = DetectionHead(in_ch=cf)
        self.loc_decoder = UNetDecoderLite(cf, 1, 'sigmoid')
        self.pay_decoder = UNetDecoderLite(cf, 1, 'relu')

    def forward(self, x):
        _srm_full, srm_ds = self.srm(x)
        cnn_ds  = self.cnn(x)
        fused   = self.fusion(srm_ds, cnn_ds)
        det     = self.det_head(fused)         # [B] raw logit
        loc     = self.loc_decoder(fused)      # [B, 1, H, W]
        pay     = self.pay_decoder(fused)      # [B, 1, H, W]
        return {'det': det, 'loc': loc, 'pay': pay}

    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        det_prob = torch.sigmoid(out['det'])
        return {
            'is_stego': (det_prob > threshold).float(),
            'det_prob':  det_prob,
            'loc_map':   out['loc'].squeeze(1),
            'pay_map':   out['pay'].squeeze(1),
        }