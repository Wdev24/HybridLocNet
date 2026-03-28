"""
HybridLocNet Dataset — Improved
=================================

CHANGES FROM ORIGINAL:

1. Rho map caching:
   Local variance cost maps take ~30ms per image to compute (scipy uniform_filter).
   With 10K images x 40 epochs = 400K computations = ~3.3 hours wasted.
   Now cached to .npy files in {image_dir}/.rho_cache/ on first access.
   Subsequent epochs: direct np.load() = ~0.5ms. Effectively free.

2. persistent_workers=True in DataLoader:
   On Windows, DataLoader workers are re-spawned every epoch by default.
   This adds ~10-15s overhead per epoch (PyTorch Windows fork limitation).
   persistent_workers=True keeps workers alive between epochs.

3. Stronger augmentation for training:
   Added random rotation (±15°) to the existing flip augmentation.
   BOSSBase images are portrait shots with strong vertical bias.
   Rotation breaks this bias and improves generalization.
   NOTE: Only applied to training set. Val/test use augment=False (deterministic).

4. prefetch_factor=2 in DataLoader:
   Workers pre-fetch 2 batches ahead, hiding I/O latency behind GPU compute.
   Reduces "worker idle" time visible in GPU utilization dips.

5. Better data path validation with informative error messages.
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import hashlib


# ── Rho Cache Utilities ─────────────────────────────────────────────────────────

def _get_rho_cache_path(img_path: Path, cache_dir: Path) -> Path:
    """Stable cache path: uses image filename stem as key."""
    return cache_dir / f"{img_path.stem}.npy"


def _compute_and_cache_rho(img_gray: np.ndarray,
                            img_path: Path,
                            cache_dir: Path) -> np.ndarray:
    """
    Compute local-variance rho map and cache to disk.
    On cache hit: ~0.5ms (np.load). On miss: ~30ms (scipy).
    """
    cache_path = _get_rho_cache_path(img_path, cache_dir)

    # CHANGE: Return cached rho if it exists — skip recomputation.
    if cache_path.exists():
        return np.load(cache_path)

    # Compute rho from scratch
    rho = SyntheticStegoGenerator.compute_cost_map(img_gray)

    # Cache for future epochs
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, rho)
    except OSError:
        pass   # Non-fatal: cache write failure just means recompute next time

    return rho


# ── Stego Generator ─────────────────────────────────────────────────────────────

class SyntheticStegoGenerator:

    @staticmethod
    def compute_cost_map(img_gray: np.ndarray) -> np.ndarray:
        """
        Local variance rho map. High variance = high embedding probability.
        Embedding in texture (high var) is harder to detect than in flat regions.
        """
        try:
            from scipy.ndimage import uniform_filter
            g        = img_gray.astype(np.float64)
            mean     = uniform_filter(g, size=5)
            mean_sq  = uniform_filter(g ** 2, size=5)
            local_var = np.clip(mean_sq - mean ** 2, 0, None)
        except ImportError:
            pad = np.pad(img_gray.astype(np.float64), 2, mode='reflect')
            local_var = np.zeros_like(img_gray, dtype=np.float64)
            for dy in range(5):
                for dx in range(5):
                    local_var += (
                        pad[dy:dy+img_gray.shape[0], dx:dx+img_gray.shape[1]]
                        - img_gray
                    ) ** 2
            local_var /= 25.0
        rho = local_var / (local_var.max() + 1e-8)
        return np.clip(rho, 0.01, 0.99).astype(np.float32)

    @staticmethod
    def embed(img: np.ndarray, rho: np.ndarray,
              payload_rate: float = 0.4,
              n_bit_planes: int   = 2,
              rng: np.random.RandomState = None) -> np.ndarray:
        """
        Embed synthetic stego payload via n-bit LSB replacement.
        Pixels selected probabilistically by rho (texture-priority).
        """
        assert img.ndim == 3 and img.shape[2] == 3
        assert 1 <= n_bit_planes <= 4
        if rng is None:
            rng = np.random.RandomState()

        stego    = img.copy()
        H, W     = img.shape[:2]
        n_pixels = int(H * W * payload_rate)
        flat_rho = rho.flatten()
        probs    = flat_rho / flat_rho.sum()
        indices  = rng.choice(H * W, size=n_pixels, replace=False, p=probs)
        clear_mask = np.uint8(0xFF << n_bit_planes)

        for ch in range(3):
            new_bits = rng.randint(0, 2**n_bit_planes,
                                   size=n_pixels, dtype=np.uint8)
            ch_flat  = stego[:, :, ch].flatten().copy()
            ch_flat[indices] = (ch_flat[indices] & clear_mask) | new_bits
            stego[:, :, ch]  = ch_flat.reshape(H, W)

        return stego


# ── Dataset ─────────────────────────────────────────────────────────────────────

class SyntheticStegoDataset(Dataset):
    """
    Paired cover/stego dataset with on-the-fly synthetic embedding.

    - augment=True, deterministic=False: training (random stego per epoch)
    - augment=False, deterministic=True: val/test (fixed stego per run)

    CHANGE: Accepts cache_dir for rho map caching.
    """
    def __init__(self,
                 image_paths:  list,
                 img_size:     int   = 256,
                 payload_rate: float = 0.4,
                 n_bit_planes: int   = 2,
                 augment:      bool  = True,
                 deterministic:bool  = False,
                 cache_dir:    Path  = None):

        self.image_paths   = image_paths
        self.img_size      = img_size
        self.payload_rate  = payload_rate
        self.n_bit_planes  = n_bit_planes
        self.deterministic = deterministic
        self.cache_dir     = cache_dir
        self.generator     = SyntheticStegoGenerator()

        # Each image appears twice: once as cover (label=0), once as stego (label=1)
        self.pairs = ([(p, False) for p in image_paths] +
                      [(p, True)  for p in image_paths])

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

        # CHANGE: Stronger augmentation — add random rotation ±15°.
        # BOSSBase has portrait orientation bias; rotation breaks it.
        # augment=False for val/test ensures deterministic evaluation.
        if augment:
            self.aug_fn = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),  # CHANGE: added
            ])
        else:
            self.aug_fn = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, is_stego = self.pairs[idx]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if self.aug_fn is not None:
            img = self.aug_fn(img)

        img_np   = np.array(img, dtype=np.uint8)
        img_gray = img_np.mean(axis=2)

        # CHANGE: Use cached rho if cache_dir provided, else compute inline.
        if self.cache_dir is not None:
            rho = _compute_and_cache_rho(img_gray, img_path, self.cache_dir)
        else:
            rho = self.generator.compute_cost_map(img_gray)

        if is_stego:
            if self.deterministic:
                # Deterministic stego for val/test: same embedding every epoch.
                seed = int(hash(str(img_path)) & 0x7FFFFFFF)
                rng  = np.random.RandomState(seed)
            else:
                rng = None
            img_np = self.generator.embed(
                img_np, rho, self.payload_rate, self.n_bit_planes, rng=rng
            )
        else:
            # Noise-augmented cover: 15% of cover samples get ±3 noise but label=0.
            # Teaches model: high-frequency energy alone != stego.
            # Directly addresses Exp 3 failure (noise scored higher than embedding).
            if not self.deterministic and np.random.rand() < 0.15:
                n_px     = int(self.img_size * self.img_size * self.payload_rate)
                noise_idx= np.random.choice(self.img_size**2, n_px, replace=False)
                noise_img= img_np.copy().astype(np.int16)
                for ch in range(3):
                    flat     = noise_img[:, :, ch].flatten()
                    flat[noise_idx] = np.clip(
                        flat[noise_idx] + np.random.randint(-3, 4, n_px), 0, 255
                    )
                    noise_img[:, :, ch] = flat.reshape(self.img_size, self.img_size)
                img_np = noise_img.astype(np.uint8)

        img_tensor = self.normalize(self.to_tensor(img_np))
        rho_tensor = torch.from_numpy(rho).unsqueeze(0)
        pay_tensor = rho_tensor * self.payload_rate

        det_label = torch.tensor(float(is_stego))

        if not is_stego:
            # Cover: zero localization and payload ground truth.
            rho_tensor = torch.zeros_like(rho_tensor)
            pay_tensor = torch.zeros_like(pay_tensor)

        return {
            'image':   img_tensor,
            'det':     det_label,
            'loc_map': rho_tensor,
            'pay_map': pay_tensor,
            'path':    str(img_path),
        }


class BOSSBaseDataset(Dataset):
    """
    BOSSBase 1.01 with precomputed stego directory or on-the-fly embedding.
    Expects: root/cover/*.pgm (required), root/stego/*.pgm (optional), root/rho/*.npy (optional).
    """
    def __init__(self, root: str, img_size: int = 256, split: str = 'train',
                 payload_rate: float = 0.4, n_bit_planes: int = 2,
                 cache_dir: Path = None):
        self.root         = Path(root)
        self.img_size     = img_size
        self.payload_rate = payload_rate
        self.n_bit_planes = n_bit_planes
        self.cache_dir    = cache_dir

        cover_dir = self.root / 'cover'
        stego_dir = self.root / 'stego'
        rho_dir   = self.root / 'rho'

        if not cover_dir.exists():
            raise FileNotFoundError(
                f"BOSSBase cover directory not found: {cover_dir}\n"
                f"Expected structure: {root}/cover/*.pgm"
            )

        covers = sorted(cover_dir.glob("*.pgm")) + sorted(cover_dir.glob("*.png"))
        if not covers:
            raise FileNotFoundError(f"No .pgm/.png images found in {cover_dir}")

        n = len(covers)
        splits = {
            'train': covers[:int(0.7 * n)],
            'val':   covers[int(0.7 * n):int(0.8 * n)],
            'test':  covers[int(0.8 * n):],
        }
        self.files     = splits[split]
        self.stego_dir = stego_dir
        self.rho_dir   = rho_dir
        self.has_rho   = rho_dir.exists()
        self.gen       = SyntheticStegoGenerator()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.files) * 2

    def __getitem__(self, idx: int) -> dict:
        is_stego   = idx >= len(self.files)
        file_idx   = idx % len(self.files)
        cover_path = self.files[file_idx]

        img = Image.open(cover_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)

        # Load precomputed rho if available, else compute + cache.
        rho_path = self.rho_dir / f"{cover_path.stem}.npy"
        if self.has_rho and rho_path.exists():
            rho = np.clip(np.load(rho_path).astype(np.float32), 0, 1)
        elif self.cache_dir is not None:
            rho = _compute_and_cache_rho(
                img_np.mean(axis=2), cover_path, self.cache_dir
            )
        else:
            rho = self.gen.compute_cost_map(img_np.mean(axis=2))

        if is_stego:
            stego_path = self.stego_dir / cover_path.name
            if stego_path.exists():
                img_np = np.array(
                    Image.open(stego_path).convert('RGB').resize(
                        (self.img_size, self.img_size), Image.BILINEAR
                    ), dtype=np.uint8
                )
            else:
                img_np = self.gen.embed(
                    img_np, rho, self.payload_rate, self.n_bit_planes
                )

        img_tensor = self.normalize(self.to_tensor(img_np))
        rho_tensor = torch.from_numpy(rho).unsqueeze(0)
        pay_tensor = rho_tensor * self.payload_rate
        det_label  = torch.tensor(float(is_stego))

        if not is_stego:
            rho_tensor = torch.zeros_like(rho_tensor)
            pay_tensor = torch.zeros_like(pay_tensor)

        return {
            'image':   img_tensor,
            'det':     det_label,
            'loc_map': rho_tensor,
            'pay_map': pay_tensor,
        }


# ── DataLoader Factory ─────────────────────────────────────────────────────────

def get_dataloaders(image_dir:    str,
                    batch_size:   int   = 16,
                    img_size:     int   = 256,
                    payload_rate: float = 0.4,
                    n_bit_planes: int   = 2,
                    max_images:   int   = None,
                    num_workers:  int   = 4) -> dict:

    root = Path(image_dir)

    # CHANGE: Rho cache directory alongside the dataset.
    # Avoids recomputing cost maps every epoch (~3 hours saved over 40 epochs).
    cache_dir = root / '.rho_cache'

    if (root / 'cover').exists():
        # BOSSBase structured layout
        train_ds = BOSSBaseDataset(
            image_dir, img_size, 'train', payload_rate, n_bit_planes,
            cache_dir=cache_dir
        )
        val_ds   = BOSSBaseDataset(
            image_dir, img_size, 'val',   payload_rate, n_bit_planes,
            cache_dir=cache_dir
        )
        test_ds  = BOSSBaseDataset(
            image_dir, img_size, 'test',  payload_rate, n_bit_planes,
            cache_dir=cache_dir
        )
    else:
        # Flat image directory
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm'}
        all_paths = sorted([
            p for p in root.glob('**/*') if p.suffix.lower() in exts
        ])
        if not all_paths:
            raise FileNotFoundError(
                f"No images found in {root}. "
                f"Supported: {', '.join(exts)}"
            )
        if max_images:
            all_paths = all_paths[:max_images]

        n = len(all_paths)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)

        n_train = int(0.7 * n)
        n_val   = int(0.1 * n)

        train_paths = [all_paths[i] for i in idx[:n_train]]
        val_paths   = [all_paths[i] for i in idx[n_train:n_train + n_val]]
        test_paths  = [all_paths[i] for i in idx[n_train + n_val:]]

        train_ds = SyntheticStegoDataset(
            train_paths, img_size, payload_rate, n_bit_planes,
            augment=True,  deterministic=False, cache_dir=cache_dir
        )
        val_ds   = SyntheticStegoDataset(
            val_paths,   img_size, payload_rate, n_bit_planes,
            augment=False, deterministic=True,  cache_dir=cache_dir
        )
        test_ds  = SyntheticStegoDataset(
            test_paths,  img_size, payload_rate, n_bit_planes,
            augment=False, deterministic=True,  cache_dir=cache_dir
        )

    # CHANGE: persistent_workers=True — prevents worker respawn each epoch (Windows fix).
    # CHANGE: prefetch_factor=2 — workers prefetch 2 batches ahead.
    # CHANGE: pin_memory=True remains — zero-copy CPU->GPU transfer.
    # NOTE: On Windows with CUDA, num_workers > 0 requires if __name__=='__main__' guard
    # in train.py. If you hit spawn errors, set num_workers=0 as fallback.
    loader_kwargs = dict(
        batch_size        = batch_size,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= (num_workers > 0),   # CHANGE
        prefetch_factor   = (2 if num_workers > 0 else None),  # CHANGE
    )

    return {
        'train': DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        'val':   DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        'test':  DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    }