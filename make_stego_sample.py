# make_stego_sample.py
import numpy as np
from PIL import Image
import sys, pathlib
sys.path.insert(0, '.')
from data.dataset import SyntheticStegoGenerator

src = "BOSSbase_1.01/1.pgm"
img = Image.open(src).convert('RGB').resize((256,256))
arr = np.array(img, dtype=np.uint8)
gen = SyntheticStegoGenerator()
rho = gen.compute_cost_map(arr.mean(axis=2))
stego = gen.embed(arr, rho, 0.4, n_bit_planes=2)
Image.fromarray(stego).save("stego_sample.png")
print("Saved: stego_sample.png")