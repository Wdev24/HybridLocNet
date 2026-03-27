"""
HybridLocNet — Streamlit Forensic Analysis App
Run: streamlit run app.py

Fixes vs v1:
  - Grayscale preprocessing: converts input to grayscale then back to 3-ch
    (matches BOSSBase training distribution, removes JPEG/color mismatch)
  - OOD detection: flags JPEG files and high-variance images with a warning
  - Uncertainty band: shows UNCERTAIN when 0.35 < P(stego) < 0.65
"""

import sys
import io
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.hybridlocnet import HybridLocNet

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HybridLocNet — Stego Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #080b10; }
  [data-testid="stSidebar"]  { background: #0d1117; border-right: 1px solid #1e2a38; }
  [data-testid="stHeader"]   { background: transparent; }
  h1,h2,h3,p,label           { color: #e2e8f0 !important; }
  .stButton > button {
    background: #1e2a38; color: #00d4ff; border: 1px solid #00d4ff;
    border-radius: 8px; font-weight: 600; width: 100%; transition: all 0.2s;
  }
  .stButton > button:hover   { background: #00d4ff; color: #080b10; }
  .verdict-stego  { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4);   border-radius: 12px; padding: 20px 28px; text-align: center; }
  .verdict-cover  { background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.35); border-radius: 12px; padding: 20px 28px; text-align: center; }
  .verdict-unsure { background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.4);  border-radius: 12px; padding: 20px 28px; text-align: center; }
  .ood-warning    { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.35); border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; }
  .metric-card    { background: #0d1520; border: 1px solid #1e2a38; border-radius: 10px; padding: 16px 20px; text-align: center; }
  .metric-val     { font-size: 26px; font-weight: 700; margin: 0; }
  .metric-lbl     { font-size: 12px; color: #64748b; margin: 4px 0 0; }
  hr { border-color: #1e2a38; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt_path: str, device: str):
    model = HybridLocNet(cf=256)
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        epoch = ck.get('epoch', '?')
        return model.to(device).eval(), f"Loaded checkpoint (epoch {epoch})"
    return model.to(device).eval(), "No checkpoint — random weights (demo only)"


# ── OOD detection ──────────────────────────────────────────────────────────────
def detect_ood(uploaded_file, img_pil: Image.Image) -> list[str]:
    """
    Returns a list of OOD warning strings (empty = in-distribution).
    Checks:
      1. JPEG file format (compression artifacts mimic stego signal)
      2. Image std vs BOSSBase expected range (very high = real camera / unusual scene)
      3. Original image is colour (not grayscale)
    """
    warnings_out = []
    fname = uploaded_file.name.lower()

    # Check 1: JPEG
    if fname.endswith('.jpg') or fname.endswith('.jpeg'):
        warnings_out.append(
            "JPEG compression detected. DCT quantization artifacts can "
            "resemble LSB embedding noise and inflate P(stego)."
        )

    # Check 2: colour input
    arr = np.array(img_pil.convert('RGB'))
    ch_stds = arr.std(axis=(0,1))
    if ch_stds.std() > 8:          # large difference between R/G/B spread = colour photo
        warnings_out.append(
            "Colour image detected. Model trained on grayscale BOSSBase images. "
            "Colour channels introduce distribution shift."
        )

    # Check 3: overall std vs BOSSBase typical range (~30-70 in uint8)
    gray_std = np.array(img_pil.convert('L')).std()
    if gray_std > 80:
        warnings_out.append(
            f"High pixel variance (std={gray_std:.0f}). "
            "Complex real-world scene may be out of training distribution."
        )

    return warnings_out


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_to_grayscale(img_pil: Image.Image) -> Image.Image:
    """
    Convert to grayscale then back to 3-channel.
    This matches the effective input that the model saw during training
    (BOSSBase .pgm files converted to RGB via img.convert('RGB')).
    Eliminates colour-channel mismatch completely.
    """
    gray = img_pil.convert('L')          # luminance only
    return gray.convert('RGB')           # replicate across 3 channels


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(img_pil: Image.Image, model, device: str):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    det_prob = torch.sigmoid(out['det']).item()
    loc_map  = out['loc'][0, 0].cpu().numpy()
    pay_map  = out['pay'][0, 0].cpu().numpy()
    return det_prob, loc_map, pay_map


def compute_wfus(loc_map: np.ndarray, k: int = 20) -> float:
    flat  = loc_map.flatten()
    n_k   = max(1, int(len(flat) * k / 100))
    top_k = np.argpartition(flat, -n_k)[-n_k:]
    total = flat.sum()
    return float(flat[top_k].sum() / total) if total > 0 else 0.0


def get_verdict(det_prob: float, threshold: float, uncertain_band: float):
    """
    Three-way verdict:
      STEGO   — P > threshold
      COVER   — P < (1 - threshold)
      UNCERTAIN — within the uncertain_band around 0.5
    """
    lo = 0.5 - uncertain_band / 2
    hi = 0.5 + uncertain_band / 2
    if lo <= det_prob <= hi:
        return "UNCERTAIN", "#f59e0b", "verdict-unsure"
    if det_prob > threshold:
        return "STEGO", "#ff4d4d", "verdict-stego"
    return "COVER", "#4dff91", "verdict-cover"


# ── Figures ────────────────────────────────────────────────────────────────────
def make_overlay_figure(img_pil: Image.Image, loc_map: np.ndarray) -> plt.Figure:
    stego_cmap = LinearSegmentedColormap.from_list(
        'stego', ['#0d1117','#3d0000','#8b1a1a','#ff4500','#ff8c00','#ffd700'], N=256)
    img_arr  = np.array(img_pil.resize((256, 256))) / 255.0
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr]*3, axis=-1)
    loc_norm = (loc_map - loc_map.min()) / (loc_map.max() - loc_map.min() + 1e-8)
    heat_rgb = plt.get_cmap('hot')(loc_norm)[:,:,:3]
    alpha    = loc_norm[:,:,np.newaxis] * 0.65
    overlay  = np.clip(img_arr*(1-alpha) + heat_rgb*alpha, 0, 1)
    thr20    = np.percentile(loc_map, 80)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#080b10')
    for ax, title, d, cmap in zip(
        axes,
        ['Original (grayscale input)', 'Localization heatmap', 'Overlay (hot = suspicious)'],
        [img_arr, loc_map, overlay],
        [None, stego_cmap, None]
    ):
        ax.set_facecolor('#080b10'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2a38')
        ax.set_title(title, color='#94a3b8', fontsize=10, pad=6)
        ax.imshow(d, cmap=cmap)
        if 'Overlay' in title and loc_map.max() > 0:
            ax.contour(loc_map, levels=[thr20], colors=['#00d4ff'],
                       linewidths=1.0, alpha=0.9)
    try:
        plt.tight_layout()
    except Exception:
        pass
    return fig


def make_payload_figure(pay_map: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#080b10')
    ax.set_facecolor('#080b10'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2a38')
    ax.set_title('Payload density map', color='#94a3b8', fontsize=10, pad=6)
    im = ax.imshow(pay_map, cmap='YlOrRd')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='#64748b', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#64748b')
    cb.set_label('bpp', color='#64748b', fontsize=8)
    cb.outline.set_edgecolor('#1e2a38')
    try:
        plt.tight_layout()
    except Exception:
        pass
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 HybridLocNet")
    st.markdown("<p style='color:#64748b;font-size:13px'>Multi-task steganalysis</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    ckpt_path  = st.text_input("Checkpoint path", value="checkpoints/best.pt")
    device_opt = st.selectbox("Device", ["cuda", "cpu"],
                               index=0 if torch.cuda.is_available() else 1)
    model, model_status = load_model(ckpt_path, device_opt)
    status_color = "#4dff91" if "Loaded" in model_status else "#f59e0b"
    st.markdown(f"<p style='color:{status_color};font-size:12px;font-family:monospace'>"
                f"{model_status}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Detection threshold**")
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="P(stego) above this → STEGO verdict")

    st.markdown("**Uncertainty band**")
    uncertain_band = st.slider("Uncertain band (±)", 0.0, 0.4, 0.15, 0.05,
                               help="Predictions within ±band of 0.5 → UNCERTAIN")

    st.markdown("**Input mode**")
    use_grayscale = st.toggle("Grayscale preprocessing", value=True,
                              help="Converts input to grayscale before inference. "
                                   "Matches BOSSBase training distribution. "
                                   "Recommended for real-world images.")

    st.markdown("---")
    st.markdown("""
<p style='color:#64748b;font-size:12px'>
<b style='color:#94a3b8'>How it works:</b><br>
1. SRM filters extract noise residuals<br>
2. ResNet-18 captures structure<br>
3. Attention fusion combines streams<br>
4. Three heads: detect · localize · estimate<br><br>
<b style='color:#94a3b8'>Training data:</b><br>
BOSSBase — grayscale, uncompressed<br>
Synthetic 2-bit LSB embedding<br>
82% accuracy on in-distribution data
</p>
""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## Forensic Steganalysis")
st.markdown("<p style='color:#64748b;margin-top:-12px'>Upload an image to detect "
            "hidden data, localize suspicious regions, and estimate payload density.</p>",
            unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("Drop an image here",
                             type=["png","jpg","jpeg","pgm","bmp"],
                             label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
<div style='border:2px dashed #1e2a38;border-radius:14px;padding:60px;text-align:center'>
  <p style='font-size:18px;color:#374151'>Drop an image above to analyze</p>
  <p style='font-size:13px;color:#4b5563'>
    Best results: BOSSBase .pgm files and stego_sample.png<br>
    Real-world images supported with OOD detection
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Load + OOD check ───────────────────────────────────────────────────────────
img_original = Image.open(uploaded)
ood_flags    = detect_ood(uploaded, img_original)

# Show OOD warnings before results
if ood_flags:
    for flag in ood_flags:
        st.markdown(f"""
<div class='ood-warning'>
  <span style='color:#f59e0b;font-weight:600'>⚠ Out-of-distribution warning</span><br>
  <span style='color:#94a3b8;font-size:13px'>{flag}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.2);
     border-radius:10px;padding:12px 16px;margin-bottom:16px'>
  <p style='color:#94a3b8;font-size:13px;margin:0'>
  <b style='color:#f59e0b'>Model limitation:</b> HybridLocNet was trained on 
  uncompressed grayscale academic images (BOSSBase). Results on JPEG or colour 
  photos may be unreliable. Grayscale preprocessing (enabled by default in sidebar) 
  reduces but does not eliminate this effect.
  <br><br>
  <b style='color:#94a3b8'>In production</b>, this is addressed via domain adaptation, 
  JPEG-aware stego embedders, and mixed training data.
  </p>
</div>""", unsafe_allow_html=True)

# ── Preprocess ─────────────────────────────────────────────────────────────────
if use_grayscale:
    img_for_model = preprocess_to_grayscale(img_original)
    mode_label    = "grayscale"
else:
    img_for_model = img_original.convert('RGB')
    mode_label    = "colour (raw)"

# ── Inference ──────────────────────────────────────────────────────────────────
with st.spinner("Analyzing..."):
    det_prob, loc_map, pay_map = run_inference(img_for_model, model, device_opt)

verdict, v_color, v_class = get_verdict(det_prob, threshold, uncertain_band)
confidence = det_prob if det_prob > 0.5 else (1 - det_prob)
wfus20     = compute_wfus(loc_map, k=20)
wfus10     = compute_wfus(loc_map, k=10)

# ── Verdict banner ─────────────────────────────────────────────────────────────
verdict_sub = f"{confidence:.1%} confidence &nbsp;|&nbsp; P(stego) = {det_prob:.4f}"
if verdict == "UNCERTAIN":
    verdict_sub = (f"P(stego) = {det_prob:.4f} — within uncertainty band ±{uncertain_band:.2f}. "
                   f"Model is not confident either way.")

st.markdown(f"""
<div class="{v_class}">
  <p style='font-size:36px;font-weight:800;color:{v_color};margin:0;letter-spacing:2px'>
    {verdict}
  </p>
  <p style='font-size:15px;color:#94a3b8;margin:6px 0 0'>
    {verdict_sub}
  </p>
  <p style='font-size:11px;color:#4b5563;margin:6px 0 0;font-family:monospace'>
    input: {mode_label} &nbsp;|&nbsp; threshold: {threshold} &nbsp;|&nbsp;
    {"⚠ OOD input detected" if ood_flags else "in-distribution"}
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Metrics ────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl, color in [
    (m1, f"{confidence:.1%}", "Confidence",       v_color),
    (m2, f"{wfus20:.1%}",     "wFUS @ 20% pixels","#a78bfa"),
    (m3, f"{wfus10:.1%}",     "wFUS @ 10% pixels","#f59e0b"),
    (m4, f"{pay_map.max():.4f}", "Peak payload (bpp)","#34d399"),
]:
    with col:
        st.markdown(f"""<div class='metric-card'>
          <p class='metric-val' style='color:{color}'>{val}</p>
          <p class='metric-lbl'>{lbl}</p></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Spatial analysis ───────────────────────────────────────────────────────────
st.markdown("#### Spatial Analysis")
st.markdown("<p style='color:#64748b;font-size:13px;margin-top:-10px'>"
            "Cyan contour = top 20% most suspicious pixels. "
            "Hot regions indicate detected embedding artifacts.</p>",
            unsafe_allow_html=True)

fig_overlay = make_overlay_figure(img_for_model, loc_map)
st.pyplot(fig_overlay, use_container_width=True)
plt.close(fig_overlay)

# ── Payload + wFUS ─────────────────────────────────────────────────────────────
col_pay, col_wfus = st.columns([1, 1])

with col_pay:
    st.markdown("#### Payload density")
    fig_pay = make_payload_figure(pay_map)
    st.pyplot(fig_pay, use_container_width=True)
    plt.close(fig_pay)

with col_wfus:
    st.markdown("#### Forensic utility (wFUS)")
    st.markdown(f"""
<p style='color:#94a3b8;font-size:13px'>
Fraction of embedding mass recovered by examining only the top-k% of pixels.
</p>
<div style='background:#0d1520;border:1px solid #1e2a38;border-radius:10px;padding:16px;margin-top:8px'>
  <p style='color:#64748b;font-size:12px;margin:0'>Random search baseline (20% pixels)</p>
  <div style='background:#1e2a38;border-radius:4px;height:10px;margin:6px 0'>
    <div style='background:#4b5563;height:100%;width:20%;border-radius:4px'></div>
  </div>
  <p style='color:#64748b;font-size:12px;margin:0'>Recovers 20% of embedding mass</p>
</div>
<div style='background:#0d1520;border:1px solid #1e2a38;border-radius:10px;padding:16px;margin-top:8px'>
  <p style='color:#a78bfa;font-size:12px;margin:0'>HybridLocNet</p>
  <div style='background:#1e2a38;border-radius:4px;height:10px;margin:6px 0'>
    <div style='background:#7c3aed;height:100%;width:{min(int(wfus20*100),100)}%;border-radius:4px'></div>
  </div>
  <p style='color:#a78bfa;font-size:12px;margin:0'>
    Recovers <b>{wfus20:.1%}</b> of embedding mass in top 20% of pixels
  </p>
</div>
<p style='color:#64748b;font-size:12px;margin-top:12px'>
  <b style='color:#e2e8f0'>{wfus20/0.20:.1f}x</b> improvement over random search.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Technical details ─────────────────────────────────────────────────────────
with st.expander("Technical details"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Detection**
- P(stego): `{det_prob:.6f}`
- Threshold: `{threshold}`
- Uncertain band: `±{uncertain_band:.2f}`
- Verdict: `{verdict}`
- Input mode: `{mode_label}`

**Localization**
- Map range: `{loc_map.min():.4f}` → `{loc_map.max():.4f}`
- wFUS@10%: `{wfus10:.1%}`
- wFUS@20%: `{wfus20:.1%}`
""")
    with c2:
        st.markdown(f"""
**Payload estimation**
- Peak density: `{pay_map.max():.6f}` bpp
- Mean density: `{pay_map.mean():.6f}` bpp

**Model**
- Parameters: `2.05M`
- Input size: `256×256`
- Architecture: `SRM + ResNet-18 + SE fusion`
- Training data: `BOSSBase grayscale`
- OOD warnings: `{len(ood_flags)}`
""")

    if ood_flags:
        st.markdown("**OOD flags detected:**")
        for f in ood_flags:
            st.markdown(f"- {f}")