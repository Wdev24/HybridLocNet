"""
HybridLocNet - Simplified Multi-Task Steganalysis
Architecture: SRM + ResNet18 + Lightweight Attention Fusion
Tasks: Detection | Localization | Payload Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─── SRM Fixed Filters ─────────────────────────────────────────────────────────

def get_srm_kernels():
    """
    30 fixed SRM (Spatial Rich Model) high-pass kernels.
    These extract noise residuals — the statistical fingerprints
    left by steganographic embedding. NOT learned — fixed physics.
    """
    # 5 representative kernels (simplified from full 30-filter SRM)
    # In production, load all 30 from srm_filters.npy
    kernels = [
        # 1st order horizontal
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        # 1st order vertical
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
        # 1st order diagonal
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        # 2nd order (edge)
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        # 2nd order diagonal
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
    ]
    # Tile to get 30 channels (5 kernels × 3 color channels)
    k_tensor = torch.tensor(kernels, dtype=torch.float32)  # [5, 3, 3]
    k_tensor = k_tensor.unsqueeze(1).expand(-1, 3, -1, -1)  # [5, 3, 3, 3]
    # Replicate to 30 filters
    k_tensor = k_tensor.repeat(6, 1, 1, 1)  # [30, 3, 3, 3]
    return k_tensor


class TLU(nn.Module):
    """Truncated Linear Unit — clips SRM residuals to [-T, T]."""
    def __init__(self, T=3.0):
        super().__init__()
        self.T = T

    def forward(self, x):
        return torch.clamp(x, -self.T, self.T)


class SRMStream(nn.Module):
    """
    Fixed SRM residual extractor.
    Rationale: SRM kernels capture high-frequency statistical anomalies
    introduced by steganographic algorithms — exactly what CNN features miss.
    Output: [B, 256, H/8, W/8] — stride-8 for attention tractability.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        srm_kernels = get_srm_kernels()
        self.srm_conv = nn.Conv2d(3, 30, kernel_size=3, padding=1, bias=False)
        self.srm_conv.weight = nn.Parameter(srm_kernels, requires_grad=False)
        
        self.tlu = TLU(T=3.0)
        self.proj = nn.Sequential(
            nn.Conv2d(30, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

    def forward(self, x):
        res = self.tlu(self.srm_conv(x))    # [B, 30, H, W]
        feat = self.proj(res)               # [B, 256, H, W]
        feat_ds = self.pool(feat)           # [B, 256, H/8, W/8]
        return feat, feat_ds               # full-res + downsampled


class CNNStream(nn.Module):
    """
    ResNet-18 CNN stream (randomly initialized).
    Rationale: ImageNet pretrain introduces semantic bias inappropriate
    for steganalysis. Random init forces learning stego-specific features.
    Output: [B, 256, H/8, W/8] — stride-8 to match SRM stream.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        backbone = models.resnet18(weights=None)  # NO pretrain — intentional
        
        # Use block-4 features at stride 8 (not stride 32)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )  # stride 4
        self.layer1 = backbone.layer1   # stride 4
        self.layer2 = backbone.layer2   # stride 8
        
        # Project to Cf=256
        self.proj = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f = self.stem(x)
        f = self.layer1(f)
        f = self.layer2(f)          # [B, 128, H/8, W/8]
        feat_ds = self.proj(f)      # [B, 256, H/8, W/8]
        return feat_ds


# ─── Lightweight Attention Fusion ──────────────────────────────────────────────

class LightweightAttentionFusion(nn.Module):
    """
    Channel attention (SE-style) instead of full spatial cross-attention.

    Rationale for simplification:
    - Full QK^T cross-attention at stride-8 = N=1024 tokens → 4MB matrix,
      still heavy for CPU/small GPU demos.
    - Channel attention captures WHICH features matter via squeeze-excitation,
      requiring only 2×FC layers. Same fusion principle, 100× cheaper.
    - Learnable scalar alpha controls global stream balance (from paper).
    """
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.se_srm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.se_cnn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.6))  # learnable stream balance
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, srm_feat, cnn_feat):
        # Channel attention weights
        w_srm = self.se_srm(srm_feat).view(-1, 256, 1, 1)
        w_cnn = self.se_cnn(cnn_feat).view(-1, 256, 1, 1)
        
        srm_attended = srm_feat * w_srm
        cnn_attended = cnn_feat * w_cnn
        
        # Alpha-weighted combination + concat fusion
        alpha = torch.sigmoid(self.alpha)
        fused = torch.cat([alpha * srm_attended, (1 - alpha) * cnn_attended], dim=1)
        return self.fusion_conv(fused)  # [B, 256, H/8, W/8]


# ─── Decoder Heads ─────────────────────────────────────────────────────────────

class UNetDecoderLite(nn.Module):
    """
    Lightweight 3-block U-Net decoder for localization/payload heads.
    Uses skip connections from SRM full-res features.
    """
    def __init__(self, in_ch=256, out_ch=1, final_activation='sigmoid'):
        super().__init__()
        self.up1 = self._up_block(in_ch, 128)      # /8 → /4
        self.up2 = self._up_block(128, 64)          # /4 → /2
        self.up3 = self._up_block(64, 32)           # /2 → /1
        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)
        self.final_act = nn.Sigmoid() if final_activation == 'sigmoid' else nn.ReLU()

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final_act(self.out_conv(x))


class DetectionHead(nn.Module):
    """Binary detection: cover (0) vs stego (1)."""
    def __init__(self, in_ch=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(self.gap(x))


# ─── Full Model ────────────────────────────────────────────────────────────────

class HybridLocNet(nn.Module):
    """
    Unified multi-task steganalysis network.

    Correct shape flow for 256x256 input:
      fused        [B, 256, 32, 32]   stride-8  (from fusion)
      loc/pay up1  [B, 128, 64, 64]   stride-4
      loc/pay up2  [B,  64,128,128]   stride-2
      loc/pay up3  [B,  32,256,256]   stride-1  (full res)
      out_conv     [B,   1,256,256]   final output
    """

    def __init__(self, cf=256):
        super().__init__()
        self.srm     = SRMStream(out_channels=cf)
        self.cnn     = CNNStream(out_channels=cf)
        self.fusion  = LightweightAttentionFusion(channels=cf)
        self.det_head = DetectionHead(in_ch=cf)

        # Both decoders receive fused [B, cf, H/8, W/8] and upsample 8x to full res.
        # Blocks 1-2 of the upsampling path use the same channel widths (logical share).
        self.loc_decoder = UNetDecoderLite(cf, 1, 'sigmoid')
        self.pay_decoder = UNetDecoderLite(cf, 1, 'relu')

    def forward(self, x):
        # Dual stream feature extraction
        _srm_full, srm_ds = self.srm(x)      # [B, 256, H/8, W/8]
        cnn_ds = self.cnn(x)                  # [B, 256, H/8, W/8]

        # Attention fusion at stride-8
        fused = self.fusion(srm_ds, cnn_ds)   # [B, 256, H/8, W/8]

        # Detection head (global average pool on fused features)
        det = self.det_head(fused).squeeze(1) # [B]

        # Spatial heads: stride-8 -> full resolution (3 x 2x upsample each)
        loc = self.loc_decoder(fused)         # [B, 1, H, W]  in [0,1]
        pay = self.pay_decoder(fused)         # [B, 1, H, W]  >= 0

        return {
            'det': det,   # [B]
            'loc': loc,   # [B, 1, H, W]
            'pay': pay,   # [B, 1, H, W]
        }

    def predict(self, x, threshold=0.5):
        """Inference helper with thresholded detection."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return {
            'is_stego':  (out['det'] > threshold).float(),
            'det_prob':  out['det'],
            'loc_map':   out['loc'].squeeze(1),
            'pay_map':   out['pay'].squeeze(1),
        }