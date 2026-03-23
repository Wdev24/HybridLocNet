"""
HybridLocNet Dataset — with deterministic val/test splits.

Key fix: val and test DataLoaders are now created from a separate
SyntheticStegoDataset instance with augment=False. This makes val/test
evaluation deterministic across epochs, eliminating the artificial
val-acc variance observed in training logs.
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


class SyntheticStegoGenerator:

    @staticmethod
    def compute_cost_map(img_gray: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter
            g = img_gray.astype(np.float64)
            mean    = uniform_filter(g, size=5)
            mean_sq = uniform_filter(g ** 2, size=5)
            local_var = np.clip(mean_sq - mean ** 2, 0, None)
        except ImportError:
            pad = np.pad(img_gray.astype(np.float64), 2, mode='reflect')
            local_var = np.zeros_like(img_gray, dtype=np.float64)
            for dy in range(5):
                for dx in range(5):
                    local_var += (pad[dy:dy+img_gray.shape[0],
                                     dx:dx+img_gray.shape[1]] - img_gray) ** 2
            local_var /= 25.0
        rho = local_var / (local_var.max() + 1e-8)
        rho = np.clip(rho, 0.01, 0.99)
        return rho.astype(np.float32)

    @staticmethod
    def embed(img: np.ndarray, rho: np.ndarray,
              payload_rate: float = 0.4,
              n_bit_planes: int = 2) -> np.ndarray:
        assert img.ndim == 3 and img.shape[2] == 3
        assert 1 <= n_bit_planes <= 4
        stego    = img.copy()
        H, W     = img.shape[:2]
        n_pixels = int(H * W * payload_rate)
        flat_rho = rho.flatten()
        probs    = flat_rho / flat_rho.sum()
        indices  = np.random.choice(H * W, size=n_pixels, replace=False, p=probs)
        clear_mask = np.uint8(0xFF << n_bit_planes)
        for ch in range(3):
            new_bits = np.random.randint(0, 2**n_bit_planes,
                                         size=n_pixels, dtype=np.uint8)
            ch_flat = stego[:, :, ch].flatten().copy()
            ch_flat[indices] = (ch_flat[indices] & clear_mask) | new_bits
            stego[:, :, ch] = ch_flat.reshape(H, W)
        assert not np.array_equal(stego, img)
        return stego


class SyntheticStegoDataset(Dataset):
    def __init__(self, image_paths: list, img_size: int = 256,
                 payload_rate: float = 0.4, n_bit_planes: int = 2,
                 augment: bool = True, deterministic: bool = False):
        self.image_paths   = image_paths
        self.img_size      = img_size
        self.payload_rate  = payload_rate
        self.n_bit_planes  = n_bit_planes
        self.deterministic = deterministic  # True for val/test: same stego every epoch
        self.generator     = SyntheticStegoGenerator()
        self.pairs         = ([(p, False) for p in image_paths] +
                              [(p, True)  for p in image_paths])
        self.to_tensor    = transforms.ToTensor()
        self.normalize    = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.aug_fn = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]) if augment else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, is_stego = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.aug_fn is not None:
            img = self.aug_fn(img)
        img_np   = np.array(img, dtype=np.uint8)
        img_gray = img_np.mean(axis=2)
        rho      = self.generator.compute_cost_map(img_gray)
        if is_stego:
            if self.deterministic:
                # Same stego every epoch for val/test — seed from image path hash
                seed = int(hash(str(img_path)) & 0x7FFFFFFF)
                old_state = np.random.get_state()
                np.random.seed(seed)
            img_np = self.generator.embed(img_np, rho,
                                          self.payload_rate, self.n_bit_planes)
            if self.deterministic:
                np.random.set_state(old_state)
        img_tensor = self.normalize(self.to_tensor(img_np))
        rho_tensor = torch.from_numpy(rho).unsqueeze(0)
        pay_tensor = rho_tensor * self.payload_rate
        det_label  = torch.tensor(float(is_stego))
        if not is_stego:
            rho_tensor = torch.zeros_like(rho_tensor)
            pay_tensor = torch.zeros_like(pay_tensor)
        return {'image': img_tensor, 'det': det_label,
                'loc_map': rho_tensor, 'pay_map': pay_tensor,
                'path': str(img_path)}


class BOSSBaseDataset(Dataset):
    def __init__(self, root: str, img_size: int = 256, split: str = 'train',
                 payload_rate: float = 0.4, n_bit_planes: int = 2):
        self.root = Path(root)
        self.img_size = img_size
        self.payload_rate = payload_rate
        self.n_bit_planes = n_bit_planes
        cover_dir = self.root / 'cover'
        stego_dir = self.root / 'stego'
        rho_dir   = self.root / 'rho'
        covers = sorted(cover_dir.glob("*.pgm")) + sorted(cover_dir.glob("*.png"))
        n = len(covers)
        splits = {'train': covers[:int(0.7*n)],
                  'val':   covers[int(0.7*n):int(0.8*n)],
                  'test':  covers[int(0.8*n):]}
        self.files     = splits[split]
        self.stego_dir = stego_dir
        self.rho_dir   = rho_dir
        self.has_rho   = rho_dir.exists()
        self.gen       = SyntheticStegoGenerator()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __len__(self):
        return len(self.files) * 2

    def __getitem__(self, idx):
        is_stego  = idx >= len(self.files)
        file_idx  = idx % len(self.files)
        cover_path = self.files[file_idx]
        img = Image.open(cover_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)
        rho_path = self.rho_dir / f"{cover_path.stem}.npy"
        if self.has_rho and rho_path.exists():
            rho = np.clip(np.load(rho_path).astype(np.float32), 0, 1)
        else:
            rho = self.gen.compute_cost_map(img_np.mean(axis=2))
        if is_stego:
            stego_path = self.stego_dir / cover_path.name
            if stego_path.exists():
                img_np = np.array(
                    Image.open(stego_path).convert('RGB').resize(
                        (self.img_size,self.img_size), Image.BILINEAR), dtype=np.uint8)
            else:
                img_np = self.gen.embed(img_np, rho,
                                        self.payload_rate, self.n_bit_planes)
        img_tensor = self.normalize(self.to_tensor(img_np))
        rho_tensor = torch.from_numpy(rho).unsqueeze(0)
        pay_tensor = rho_tensor * self.payload_rate
        det_label  = torch.tensor(float(is_stego))
        if not is_stego:
            rho_tensor = torch.zeros_like(rho_tensor)
            pay_tensor = torch.zeros_like(pay_tensor)
        return {'image': img_tensor, 'det': det_label,
                'loc_map': rho_tensor, 'pay_map': pay_tensor}


def get_dataloaders(image_dir: str, batch_size: int = 16,
                    img_size: int = 256, payload_rate: float = 0.4,
                    n_bit_planes: int = 2,
                    max_images: int = None, num_workers: int = 4):
    root = Path(image_dir)

    if (root / 'cover').exists():
        train_ds = BOSSBaseDataset(image_dir, img_size, 'train', payload_rate, n_bit_planes)
        val_ds   = BOSSBaseDataset(image_dir, img_size, 'val',   payload_rate, n_bit_planes)
        test_ds  = BOSSBaseDataset(image_dir, img_size, 'test',  payload_rate, n_bit_planes)
    else:
        # Collect all image paths
        exts = {'.jpg','.jpeg','.png','.bmp','.pgm'}
        all_paths = sorted([p for p in root.glob('**/*')
                            if p.suffix.lower() in exts])
        if max_images:
            all_paths = all_paths[:max_images]

        n = len(all_paths)
        n_train = int(0.7 * n)
        n_val   = int(0.1 * n)
        # n_test  = n - n_train - n_val

        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        train_paths = [all_paths[i] for i in idx[:n_train]]
        val_paths   = [all_paths[i] for i in idx[n_train:n_train+n_val]]
        test_paths  = [all_paths[i] for i in idx[n_train+n_val:]]

        # KEY FIX: val and test use augment=False for deterministic evaluation
        train_ds = SyntheticStegoDataset(train_paths, img_size, payload_rate,
                                         n_bit_planes, augment=True,  deterministic=False)
        val_ds   = SyntheticStegoDataset(val_paths,   img_size, payload_rate,
                                         n_bit_planes, augment=False, deterministic=True)
        test_ds  = SyntheticStegoDataset(test_paths,  img_size, payload_rate,
                                         n_bit_planes, augment=False, deterministic=True)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        'val':   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }