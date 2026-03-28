"""
HybridLocNet — Professional Forensic Steganalysis Tool
Run: streamlit run app.py
"""

import sys, io, warnings, os
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
    page_title="HybridLocNet — Forensic Steganalysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
  --bg:        #030508;
  --surface:   #080c12;
  --surface2:  #0d1520;
  --border:    #1a2535;
  --border2:   #243040;
  --text:      #c8d8e8;
  --muted:     #4a6080;
  --accent:    #00c8ff;
  --green:     #00ff88;
  --red:       #ff3355;
  --amber:     #ffaa00;
  --purple:    #9966ff;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* Typography */
h1,h2,h3,h4,h5,h6 { font-family: 'Rajdhani', sans-serif !important; color: var(--text) !important; letter-spacing: 1px; }
p, label, span, div { color: var(--text); }
.mono { font-family: 'Share Tech Mono', monospace !important; }

/* Scanline overlay effect */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,200,255,0.012) 2px, rgba(0,200,255,0.012) 4px);
  pointer-events: none; z-index: 9999;
}

/* Buttons */
.stButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 4px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 12px !important;
  letter-spacing: 1px !important;
  transition: all 0.2s !important;
  text-transform: uppercase !important;
  padding: 8px 16px !important;
}
.stButton > button:hover {
  background: rgba(0,200,255,0.1) !important;
  box-shadow: 0 0 12px rgba(0,200,255,0.3) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 12px !important; letter-spacing: 1px !important; border: none !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; }
.streamlit-expanderContent { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-top: none !important; }

/* Selectbox + text inputs */
.stSelectbox > div > div, .stTextInput > div > div > input {
  background: var(--surface) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
}

hr { border-color: var(--border) !important; opacity: 0.6; }

/* Verdict cards */
.verdict-stego  { background: rgba(255,51,85,0.08); border: 1px solid rgba(255,51,85,0.5); border-radius: 8px; padding: 28px 32px; text-align: center; position:relative; overflow:hidden; }
.verdict-cover  { background: rgba(0,255,136,0.07); border: 1px solid rgba(0,255,136,0.4); border-radius: 8px; padding: 28px 32px; text-align: center; position:relative; overflow:hidden; }
.verdict-unsure { background: rgba(255,170,0,0.07);  border: 1px solid rgba(255,170,0,0.4);  border-radius: 8px; padding: 28px 32px; text-align: center; position:relative; overflow:hidden; }

.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px 16px; text-align: center; }
.metric-val  { font-size: 28px; font-weight: 700; font-family: 'Rajdhani', sans-serif; margin: 0; }
.metric-lbl  { font-size: 11px; color: var(--muted); margin: 4px 0 0; font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; text-transform: uppercase; }

.report-box  { background: var(--surface); border: 1px solid var(--border2); border-left: 3px solid var(--accent); border-radius: 8px; padding: 24px 28px; }
.ood-box     { background: rgba(255,170,0,0.05); border: 1px solid rgba(255,170,0,0.3); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; }
.info-row    { display:flex; justify-content:space-between; padding: 8px 0; border-bottom: 1px solid var(--border); }
.progress-bg { background: var(--border); border-radius: 4px; height: 8px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

.section-header { font-family: 'Share Tech Mono', monospace; font-size: 11px; letter-spacing: 3px; color: var(--muted); text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(ckpt_path: str, device: str):
    model = HybridLocNet(cf=256)
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model_state'])
        epoch = ck.get('epoch', '?')
        metrics = ck.get('metrics', {})
        return model.to(device).eval(), f"checkpoint · epoch {epoch}", True, metrics
    return model.to(device).eval(), "no checkpoint · random weights", False, {}


def preprocess(img_pil: Image.Image, grayscale: bool = True) -> torch.Tensor:
    if grayscale:
        img_pil = img_pil.convert('L').convert('RGB')
    else:
        img_pil = img_pil.convert('RGB')
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(img_pil).unsqueeze(0)


@st.cache_data
def run_inference_cached(img_bytes: bytes, ckpt_path: str, device: str, grayscale: bool):
    """Cached inference — re-runs only if image/checkpoint/settings change."""
    img_pil = Image.open(io.BytesIO(img_bytes))
    model, _, _, _ = load_model(ckpt_path, device)
    tensor = preprocess(img_pil, grayscale).to(device)
    with torch.no_grad():
        out = model(tensor)
    det_prob = torch.sigmoid(out['det']).item()
    loc_map  = torch.sigmoid(out['loc'])[0, 0].cpu().numpy()
    pay_map  = out['pay'][0, 0].cpu().numpy()
    return det_prob, loc_map, pay_map


def get_stego_tensor_direct():
    """
    Load a stego sample directly from dataset pipeline — bypasses PNG quantization.
    Stego embedding lives in normalized float tensor space (sub-0.01 value changes).
    Saving to PNG quantizes those changes away, so any PNG-based demo will always
    read as COVER. This function pulls the tensor before it ever touches uint8.
    """
    try:
        from data.dataset import get_dataloaders
        loaders = get_dataloaders('./BOSSbase_1.01', batch_size=1,
                                  img_size=256, n_bit_planes=2, num_workers=0)
        for batch in loaders['val']:
            if batch['det'].item() == 1.0:
                return batch['image'], batch['loc_map']
    except Exception:
        return None, None
    return None, None


def wfus(loc_map: np.ndarray, k: int) -> float:
    flat = loc_map.flatten()
    n_k  = max(1, int(len(flat) * k / 100))
    idx  = np.argpartition(flat, -n_k)[-n_k:]
    tot  = flat.sum()
    return float(flat[idx].sum() / tot) if tot > 0 else 0.0


def get_verdict(p: float, threshold: float, band: float):
    lo, hi = 0.5 - band / 2, 0.5 + band / 2
    if lo <= p <= hi:
        return "UNCERTAIN", "var(--amber)", "verdict-unsure", "⚠"
    if p > threshold:
        return "STEGO", "var(--red)", "verdict-stego", "⛔"
    return "COVER", "var(--green)", "verdict-cover", "✅"


def detect_ood(fname: str, img_pil: Image.Image):
    flags = []
    if fname.lower().endswith(('.jpg', '.jpeg')):
        flags.append("JPEG compression — DCT artifacts may inflate P(stego)")
    arr = np.array(img_pil.convert('RGB'))
    if arr.std(axis=(0, 1)).std() > 8:
        flags.append("Colour image — trained on grayscale BOSSBase (enable grayscale mode)")
    if np.array(img_pil.convert('L')).std() > 80:
        flags.append(f"High pixel variance (σ={np.array(img_pil.convert('L')).std():.0f}) — complex real-world scene")
    return flags


STEGO_CMAP = LinearSegmentedColormap.from_list(
    'stego', ['#030508','#3d0000','#8b1a1a','#ff4500','#ff8c00','#ffd700'], N=256)


def fig_spatial(img_pil, loc_map):
    img_arr  = np.array(img_pil.convert('RGB').resize((256, 256))) / 255.0
    loc_norm = (loc_map - loc_map.min()) / (loc_map.max() - loc_map.min() + 1e-8)
    heat     = plt.get_cmap('hot')(loc_norm)[:, :, :3]
    alpha    = loc_norm[:, :, np.newaxis] * 0.70
    overlay  = np.clip(img_arr * (1 - alpha) + heat * alpha, 0, 1)
    thr20    = np.percentile(loc_map, 80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#030508')
    titles = ['Original Image', 'Localization Heatmap', 'Heatmap Overlay']
    data   = [img_arr, loc_map, overlay]
    cmaps  = [None, STEGO_CMAP, None]

    for ax, title, d, cm in zip(axes, titles, data, cmaps):
        ax.set_facecolor('#030508')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('#1a2535')
        ax.set_title(title, color='#6a8aaa', fontsize=10, pad=8, fontfamily='monospace')
        im = ax.imshow(d, cmap=cm)
        if title == 'Localization Heatmap':
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.yaxis.set_tick_params(color='#4a6080', labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='#4a6080')
            cb.outline.set_edgecolor('#1a2535')
        if 'Overlay' in title and loc_map.max() > 0:
            ax.contour(loc_map, levels=[thr20], colors=['#00c8ff'], linewidths=1.2, alpha=0.9)
            ax.text(0.03, 0.03, '— cyan = top 20% suspicious pixels',
                    transform=ax.transAxes, color='#00c8ff', fontsize=7.5,
                    fontfamily='monospace',
                    bbox=dict(facecolor='#030508', alpha=0.75, edgecolor='none', pad=3))

    plt.tight_layout(pad=0.5)
    return fig


def fig_payload(pay_map):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#030508')
    ax.set_facecolor('#030508'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor('#1a2535')
    ax.set_title('Payload Density Map', color='#6a8aaa', fontsize=10, pad=8, fontfamily='monospace')
    im = ax.imshow(pay_map, cmap='YlOrRd', vmin=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='#4a6080', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#4a6080')
    cb.set_label('bits per pixel', color='#4a6080', fontsize=8)
    cb.outline.set_edgecolor('#1a2535')
    plt.tight_layout()
    return fig


def fig_compare(results):
    """Side-by-side comparison figure for 2 images."""
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(6*n, 12))
    fig.patch.set_facecolor('#030508')
    if n == 1: axes = [[axes[0]], [axes[1]], [axes[2]]]

    row_titles = ['Original Image', 'Localization Heatmap', 'Payload Density']
    for col, r in enumerate(results):
        img_arr = np.array(r['img'].convert('RGB').resize((256, 256))) / 255.0
        for row in range(3):
            ax = axes[row][col]
            ax.set_facecolor('#030508'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#1a2535')
            if col == 0:
                ax.set_ylabel(row_titles[row], color='#6a8aaa', fontsize=9, fontfamily='monospace')

        v, vc, _, icon = get_verdict(r['det'], 0.5, 0.15)
        axes[0][col].imshow(img_arr)
        axes[0][col].set_title(f"{icon} {v}  |  P(stego)={r['det']:.4f}\n{r['label']}",
                               color=vc, fontsize=10, pad=6)
        axes[1][col].imshow(r['loc'], cmap=STEGO_CMAP)
        w = wfus(r['loc'], 20)
        axes[1][col].text(0.03, 0.96, f'wFUS@20%={w:.1%}',
                          transform=axes[1][col].transAxes, color='white',
                          fontsize=8, va='top', fontfamily='monospace',
                          bbox=dict(facecolor='#030508', alpha=0.8, edgecolor='none', pad=3))
        axes[2][col].imshow(r['pay'], cmap='YlOrRd')
        axes[2][col].text(0.03, 0.96, f'peak={r["pay"].max():.4f} bpp',
                          transform=axes[2][col].transAxes, color='white',
                          fontsize=8, va='top', fontfamily='monospace',
                          bbox=dict(facecolor='#030508', alpha=0.8, edgecolor='none', pad=3))

    fig.suptitle('HybridLocNet — Side-by-Side Comparison', color='#6a8aaa',
                 fontsize=12, fontfamily='monospace', y=1.01)
    plt.tight_layout(pad=1.0)
    return fig


def generate_explanation(verdict, det_prob, w20, w10, peak_pay, ood_flags):
    lines = []
    if verdict == "STEGO":
        lines.append(f"🔴 Model detected steganographic artifacts with {det_prob:.1%} probability.")
        lines.append(f"📍 Examining only the top 20% of suspicious pixels recovers {w20:.1%} of estimated embedding mass — a {w20/0.20:.1f}× improvement over random search.")
        lines.append(f"📍 Even in the top 10% of pixels, {w10:.1%} of embedding mass is captured.")
        lines.append(f"📦 Estimated embedding density: {peak_pay:.4f} bits per pixel at peak.")
        if peak_pay < 0.01:
            lines.append("💡 Low payload density suggests minimal embedding (1–2 bit planes, low rate).")
        elif peak_pay < 0.05:
            lines.append("💡 Moderate payload density — consistent with 2-bit LSB adaptive embedding.")
        else:
            lines.append("💡 High payload density detected — significant data may be hidden.")
        lines.append("⚠ Conclusion: Hidden data detected and spatially localized.")
    elif verdict == "COVER":
        lines.append(f"🟢 No steganographic embedding detected. P(stego) = {det_prob:.4f}.")
        lines.append(f"📍 Localization map shows no concentrated suspicious regions (wFUS@20% = {w20:.1%} ≈ random baseline).")
        lines.append("✅ Conclusion: Image appears clean with no detectable hidden data.")
    else:
        lines.append(f"🟡 Model is uncertain — P(stego) = {det_prob:.4f} is near the decision boundary.")
        lines.append("💡 Try adjusting the detection threshold or enabling grayscale preprocessing.")
        lines.append("⚠ Conclusion: Inconclusive — manual inspection recommended.")

    if ood_flags:
        lines.append(f"⚠ Note: {len(ood_flags)} out-of-distribution flag(s) detected. Results may be less reliable.")

    return lines


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px'>
      <p style='font-family:"Share Tech Mono",monospace;font-size:18px;color:#00c8ff;margin:0;letter-spacing:2px'>HYBRIDLOCNET</p>
      <p style='font-size:11px;color:#4a6080;margin:4px 0 0;letter-spacing:2px;text-transform:uppercase'>Forensic Steganalysis System</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    ckpt_path  = st.text_input("Checkpoint", value="checkpoints/best.pt",
                               label_visibility="visible")
    device_opt = st.selectbox("Device", ["cuda", "cpu"],
                               index=0 if torch.cuda.is_available() else 1)

    model, model_status, loaded_ok, ckpt_metrics = load_model(ckpt_path, device_opt)
    color = "#00ff88" if loaded_ok else "#ffaa00"
    st.markdown(f"<p style='color:{color};font-size:11px;font-family:\"Share Tech Mono\",monospace;margin:4px 0'>{model_status}</p>", unsafe_allow_html=True)

    if loaded_ok and ckpt_metrics:
        st.markdown(f"<p style='color:#4a6080;font-size:10px;font-family:\"Share Tech Mono\",monospace;margin:0'>val_acc={ckpt_metrics.get('det_acc',0):.3f} · wFUS={ckpt_metrics.get('wfus20',0):.3f}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='section-header'>detection settings</p>", unsafe_allow_html=True)
    threshold    = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    uncert_band  = st.slider("Uncertainty band (±)", 0.0, 0.4, 0.15, 0.05)
    use_gray     = st.toggle("Grayscale preprocessing", value=True,
                             help="Recommended — matches BOSSBase training distribution")

    st.markdown("---")
    st.markdown("""
    <p class='section-header'>model info</p>
    <p style='font-size:11px;color:#4a6080;font-family:"Share Tech Mono",monospace;line-height:1.8'>
    params: 2.09M<br>
    input: 256×256<br>
    streams: SRM + ResNet-18<br>
    fusion: SE attention<br>
    heads: detect · localize · payload<br>
    dataset: BOSSBase 10K
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='border-bottom:1px solid #1a2535;padding-bottom:20px;margin-bottom:24px'>
  <h1 style='font-family:"Rajdhani",sans-serif;font-size:32px;letter-spacing:4px;margin:0;color:#c8d8e8'>
    FORENSIC STEGANALYSIS
  </h1>
  <p style='color:#4a6080;font-size:13px;margin:6px 0 0;font-family:"Share Tech Mono",monospace;letter-spacing:1px'>
    DETECT · LOCALIZE · ESTIMATE  |  HybridLocNet v2.0  |  SRM + ResNet-18 + SE Fusion
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_single, tab_compare, tab_batch, tab_how, tab_limits = st.tabs([
    "◈  SINGLE ANALYSIS",
    "⇄  COMPARISON MODE",
    "▦  BATCH MODE",
    "?  HOW IT WORKS",
    "⚠  LIMITATIONS",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════

with tab_single:

    # Demo buttons
    st.markdown("<p class='section-header'>quick demo</p>", unsafe_allow_html=True)
    dcol1, dcol2, dcol3, _ = st.columns([1, 1, 1, 3])
    demo_cover = dcol1.button("▶ Cover Image Demo")
    demo_stego = dcol2.button("▶ Stego Image Demo")
    demo_clear = dcol3.button("✕ Clear")

    st.markdown("---")

    # File upload
    uploaded = st.file_uploader(
        "Upload image for analysis",
        type=["png", "jpg", "jpeg", "pgm", "bmp"],
        label_visibility="collapsed"
    )

    # Demo mode: load sample images
    demo_img_bytes = None
    demo_label = ""

    if demo_cover and Path("BOSSbase_1.01/1.pgm").exists():
        with open("BOSSbase_1.01/1.pgm", "rb") as f:
            demo_img_bytes = f.read()
        demo_label = "1.pgm (BOSSBase cover)"
        st.info("▶ Running demo on cover image: BOSSbase_1.01/1.pgm")

    elif demo_stego:
        # CHANGE: Bypass PNG quantization entirely — load directly from dataset pipeline.
        # Stego embedding lives in normalized float tensor space (sub-0.01 value changes).
        # Saving to PNG and reloading quantizes those changes away, so any PNG-based
        # stego demo always reads as COVER. Pull the tensor before it ever touches uint8.
        st.info("▶ Running demo on stego image (direct dataset pipeline — no PNG quantization)")
        stego_tensor, stego_loc = get_stego_tensor_direct()
        if stego_tensor is not None:
            with torch.no_grad():
                out = model(stego_tensor.to(device_opt))
            det_prob = torch.sigmoid(out['det']).item()
            loc_map  = torch.sigmoid(out['loc'])[0, 0].cpu().numpy()
            pay_map  = out['pay'][0, 0].cpu().numpy()

            # Reconstruct a display image by reversing normalization
            mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = ((stego_tensor[0] * std_t + mean_t) * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            img_display = Image.fromarray(img_display)

            verdict, v_color, v_class, v_icon = get_verdict(det_prob, threshold, uncert_band)
            confidence = det_prob if det_prob > 0.5 else (1 - det_prob)
            w20 = wfus(loc_map, 20)
            w10 = wfus(loc_map, 10)
            peak_pay = pay_map.max()
            ood_flags = []

            st.markdown(f"""
            <div class='{v_class}'>
              <p style='font-size:42px;font-weight:800;color:{v_color};margin:0;
                        font-family:"Rajdhani",sans-serif;letter-spacing:6px;line-height:1'>
                {v_icon} {verdict}
              </p>
              <p style='font-size:14px;color:#8aabbf;margin:10px 0 4px;font-family:"Share Tech Mono",monospace'>
                P(stego) = {det_prob:.6f} &nbsp;·&nbsp; confidence = {confidence:.1%} &nbsp;·&nbsp; threshold = {threshold}
              </p>
              <p style='font-size:11px;color:#4a6080;margin:0;font-family:"Share Tech Mono",monospace'>
                ✓ in-distribution &nbsp;·&nbsp; direct tensor (no PNG roundtrip) &nbsp;·&nbsp; BOSSBase val set
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<p class='section-header'>forensic metrics</p>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, val, lbl, clr in [
                (mc1, f"{confidence:.1%}",   "Confidence",          v_color),
                (mc2, f"{w20:.1%}",          "wFUS @ 20% pixels",   "#9966ff"),
                (mc3, f"{w10:.1%}",          "wFUS @ 10% pixels",   "#ffaa00"),
                (mc4, f"{peak_pay:.4f} bpp", "Peak Payload",        "#00ff88"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                      <p class='metric-val' style='color:{clr}'>{val}</p>
                      <p class='metric-lbl'>{lbl}</p>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<p class='section-header'>spatial analysis</p>", unsafe_allow_html=True)
            fig = fig_spatial(img_display, loc_map)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("---")
            pay_col, wfus_col = st.columns([1, 1])
            with pay_col:
                st.markdown("<p class='section-header'>payload density map</p>", unsafe_allow_html=True)
                fig2 = fig_payload(pay_map)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            with wfus_col:
                st.markdown("<p class='section-header'>forensic utility (wFUS)</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <p style='font-size:12px;color:#4a6080;margin-bottom:16px'>
                Fraction of embedding mass recovered when examining only the top-k% of pixels ranked by the model.
                </p>
                <p style='font-size:11px;color:#4a6080;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>RANDOM BASELINE (20% pixels → 20% mass)</p>
                <div class='progress-bg' style='margin-bottom:12px'>
                  <div class='progress-fill' style='width:20%;background:#2a3848'></div>
                </div>
                <p style='font-size:11px;color:#9966ff;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>HYBRIDLOCNET @ 20% pixels → {w20:.1%} mass</p>
                <div class='progress-bg' style='margin-bottom:12px'>
                  <div class='progress-fill' style='width:{min(int(w20*100),100)}%;background:linear-gradient(90deg,#4400aa,#9966ff)'></div>
                </div>
                <p style='font-size:11px;color:#ffaa00;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>HYBRIDLOCNET @ 10% pixels → {w10:.1%} mass</p>
                <div class='progress-bg' style='margin-bottom:16px'>
                  <div class='progress-fill' style='width:{min(int(w10*100),100)}%;background:linear-gradient(90deg,#664400,#ffaa00)'></div>
                </div>
                <p style='font-size:14px;color:#c8d8e8;font-family:"Rajdhani",sans-serif;letter-spacing:1px'>
                  <span style='color:#9966ff;font-size:22px;font-weight:700'>{w20/0.20:.1f}×</span>
                  improvement over random search
                </p>
                """, unsafe_allow_html=True)

            st.stop()
        else:
            st.error("Could not load stego sample from dataset. Check that ./BOSSbase_1.01 exists.")
            st.stop()

    elif demo_cover:
        st.warning("Demo images not found. Upload an image manually.")

    # Determine source
    if demo_img_bytes:
        img_bytes = demo_img_bytes
        fname = demo_label
        img_pil = Image.open(io.BytesIO(img_bytes))
    elif uploaded:
        img_bytes = uploaded.read()
        fname = uploaded.name
        img_pil = Image.open(io.BytesIO(img_bytes))
    else:
        st.markdown("""
        <div style='border:1px dashed #1a2535;border-radius:10px;padding:60px;text-align:center;margin-top:16px'>
          <p style='font-size:16px;color:#4a6080;font-family:"Share Tech Mono",monospace;letter-spacing:2px'>
            DROP AN IMAGE TO ANALYZE
          </p>
          <p style='font-size:12px;color:#2a3848;margin-top:8px'>
            Best results: BOSSBase .pgm files · stego_sample.png<br>
            Real-world images supported with OOD detection
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # OOD warnings
    ood_flags = detect_ood(fname, img_pil)
    if ood_flags:
        for flag in ood_flags:
            st.markdown(f"""
            <div class='ood-box'>
              <span style='color:#ffaa00;font-weight:600;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:1px'>⚠ OOD WARNING</span><br>
              <span style='color:#8aabbf;font-size:12px'>{flag}</span>
            </div>""", unsafe_allow_html=True)

    # Run inference
    with st.spinner("Analyzing image..."):
        det_prob, loc_map, pay_map = run_inference_cached(
            img_bytes, ckpt_path, device_opt, use_gray)

    img_display = img_pil.convert('L').convert('RGB') if use_gray else img_pil.convert('RGB')

    verdict, v_color, v_class, v_icon = get_verdict(det_prob, threshold, uncert_band)
    confidence = det_prob if det_prob > 0.5 else (1 - det_prob)
    w20 = wfus(loc_map, 20)
    w10 = wfus(loc_map, 10)
    peak_pay = pay_map.max()

    # ── VERDICT BANNER ─────────────────────────────────────────
    st.markdown(f"""
    <div class='{v_class}'>
      <p style='font-size:42px;font-weight:800;color:{v_color};margin:0;
                font-family:"Rajdhani",sans-serif;letter-spacing:6px;line-height:1'>
        {v_icon} {verdict}
      </p>
      <p style='font-size:14px;color:#8aabbf;margin:10px 0 4px;font-family:"Share Tech Mono",monospace'>
        P(stego) = {det_prob:.6f} &nbsp;·&nbsp; confidence = {confidence:.1%} &nbsp;·&nbsp; threshold = {threshold}
      </p>
      <p style='font-size:11px;color:#4a6080;margin:0;font-family:"Share Tech Mono",monospace'>
        {'⚠ OOD input' if ood_flags else '✓ in-distribution'} &nbsp;·&nbsp;
        {'grayscale mode' if use_gray else 'colour mode'} &nbsp;·&nbsp;
        {fname}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── METRICS ROW ────────────────────────────────────────────
    st.markdown("<p class='section-header'>forensic metrics</p>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, lbl, clr in [
        (mc1, f"{confidence:.1%}",   "Confidence",          v_color),
        (mc2, f"{w20:.1%}",          "wFUS @ 20% pixels",   "#9966ff"),
        (mc3, f"{w10:.1%}",          "wFUS @ 10% pixels",   "#ffaa00"),
        (mc4, f"{peak_pay:.4f} bpp", "Peak Payload",        "#00ff88"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <p class='metric-val' style='color:{clr}'>{val}</p>
              <p class='metric-lbl'>{lbl}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── SPATIAL ANALYSIS ───────────────────────────────────────
    st.markdown("<p class='section-header'>spatial analysis</p>", unsafe_allow_html=True)
    fig = fig_spatial(img_display, loc_map)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── PAYLOAD + wFUS BARS ────────────────────────────────────
    pay_col, wfus_col = st.columns([1, 1])

    with pay_col:
        st.markdown("<p class='section-header'>payload density map</p>", unsafe_allow_html=True)
        fig2 = fig_payload(pay_map)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with wfus_col:
        st.markdown("<p class='section-header'>forensic utility (wFUS)</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:12px;color:#4a6080;margin-bottom:16px'>
        Fraction of embedding mass recovered when examining only the top-k% of pixels ranked by the model.
        </p>

        <p style='font-size:11px;color:#4a6080;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>RANDOM BASELINE (20% pixels → 20% mass)</p>
        <div class='progress-bg' style='margin-bottom:12px'>
          <div class='progress-fill' style='width:20%;background:#2a3848'></div>
        </div>

        <p style='font-size:11px;color:#9966ff;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>HYBRIDLOCNET @ 20% pixels → {w20:.1%} mass</p>
        <div class='progress-bg' style='margin-bottom:12px'>
          <div class='progress-fill' style='width:{min(int(w20*100),100)}%;background:linear-gradient(90deg,#4400aa,#9966ff)'></div>
        </div>

        <p style='font-size:11px;color:#ffaa00;font-family:"Share Tech Mono",monospace;margin:0 0 4px'>HYBRIDLOCNET @ 10% pixels → {w10:.1%} mass</p>
        <div class='progress-bg' style='margin-bottom:16px'>
          <div class='progress-fill' style='width:{min(int(w10*100),100)}%;background:linear-gradient(90deg,#664400,#ffaa00)'></div>
        </div>

        <p style='font-size:14px;color:#c8d8e8;font-family:"Rajdhani",sans-serif;letter-spacing:1px'>
          <span style='color:#9966ff;font-size:22px;font-weight:700'>{w20/0.20:.1f}×</span>
          improvement over random search
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── EXPLANATION PANEL ──────────────────────────────────────
    st.markdown("<p class='section-header'>automated analysis explanation</p>", unsafe_allow_html=True)
    explanations = generate_explanation(verdict, det_prob, w20, w10, peak_pay, ood_flags)
    explanation_html = "".join(
        f"<p style='font-size:13px;color:#8aabbf;margin:6px 0;padding:8px 12px;"
        f"background:#0d1520;border-radius:4px;border-left:2px solid #1a2535'>{e}</p>"
        for e in explanations
    )
    st.markdown(explanation_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── FORENSIC REPORT SUMMARY ────────────────────────────────
    st.markdown("<p class='section-header'>forensic report summary</p>", unsafe_allow_html=True)

    conclusion = {
        "STEGO":     "Hidden data detected and spatially localized. Steganographic embedding confirmed.",
        "COVER":     "No steganographic embedding detected. Image appears unmodified.",
        "UNCERTAIN": "Inconclusive result. P(stego) is within uncertainty band. Manual review recommended.",
    }[verdict]

    st.markdown(f"""
    <div class='report-box'>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:0'>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>VERDICT</span>
          <span style='color:{v_color};font-size:14px;font-weight:700;font-family:"Rajdhani",monospace;letter-spacing:2px'>{v_icon} {verdict}</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>CONFIDENCE</span>
          <span style='color:#c8d8e8;font-size:13px;font-family:"Share Tech Mono",monospace'>{confidence:.2%}</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>P(STEGO)</span>
          <span style='color:#c8d8e8;font-size:13px;font-family:"Share Tech Mono",monospace'>{det_prob:.6f}</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>wFUS@20%</span>
          <span style='color:#9966ff;font-size:13px;font-family:"Share Tech Mono",monospace'>{w20:.2%}</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>wFUS@10%</span>
          <span style='color:#ffaa00;font-size:13px;font-family:"Share Tech Mono",monospace'>{w10:.2%}</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>PEAK PAYLOAD</span>
          <span style='color:#00ff88;font-size:13px;font-family:"Share Tech Mono",monospace'>{peak_pay:.6f} bpp</span></div>
        <div class='info-row'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>OOD FLAGS</span>
          <span style='color:{"#ffaa00" if ood_flags else "#00ff88"};font-size:13px;font-family:"Share Tech Mono",monospace'>{len(ood_flags)} detected</span></div>
        <div class='info-row' style='border:none'><span style='color:#4a6080;font-size:12px;font-family:"Share Tech Mono",monospace'>INPUT MODE</span>
          <span style='color:#c8d8e8;font-size:13px;font-family:"Share Tech Mono",monospace'>{'grayscale' if use_gray else 'colour'}</span></div>
      </div>
      <div style='margin-top:16px;padding-top:16px;border-top:1px solid #1a2535'>
        <p style='font-size:11px;color:#4a6080;font-family:"Share Tech Mono",monospace;margin:0 0 6px;letter-spacing:1px'>CONCLUSION</p>
        <p style='font-size:14px;color:#c8d8e8;margin:0;font-family:"Rajdhani",sans-serif;letter-spacing:0.5px'>{conclusion}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Technical expander
    with st.expander("Technical details"):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"""
**Detection**
- P(stego): `{det_prob:.8f}`
- Threshold: `{threshold}` | Band: `±{uncert_band:.2f}`
- Verdict: `{verdict}`

**Localization**
- Map range: `{loc_map.min():.4f}` → `{loc_map.max():.4f}`
- wFUS@10%: `{w10:.4f}` | wFUS@20%: `{w20:.4f}`
- Baseline improvement: `{w20/0.20:.2f}×`
""")
        with tc2:
            st.markdown(f"""
**Payload**
- Peak: `{peak_pay:.8f}` bpp
- Mean: `{pay_map.mean():.8f}` bpp

**Model**
- Architecture: `SRM + ResNet-18 + SE`
- Parameters: `2.09M`
- Training acc: `100%` | wFUS: `0.615`
- OOD flags: `{len(ood_flags)}`
""")


# ══════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON MODE
# ══════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown("<p class='section-header'>cover vs stego comparison</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a6080;font-size:12px'>Upload two images to compare side-by-side — ideal for demonstrating detection on matched cover/stego pairs.</p>", unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("<p style='color:#00ff88;font-size:12px;font-family:\"Share Tech Mono\",monospace'>COVER IMAGE</p>", unsafe_allow_html=True)
        cover_file = st.file_uploader("Cover image", type=["png","jpg","jpeg","pgm","bmp"], key="cov", label_visibility="collapsed")
    with cc2:
        st.markdown("<p style='color:#ff3355;font-size:12px;font-family:\"Share Tech Mono\",monospace'>STEGO IMAGE</p>", unsafe_allow_html=True)
        stego_file = st.file_uploader("Stego image", type=["png","jpg","jpeg","pgm","bmp"], key="stg", label_visibility="collapsed")

    if cover_file and stego_file:
        results = []
        for f, label in [(cover_file, "Cover"), (stego_file, "Stego")]:
            img_b = f.read()
            det, loc, pay = run_inference_cached(img_b, ckpt_path, device_opt, use_gray)
            results.append({'img': Image.open(io.BytesIO(img_b)), 'det': det, 'loc': loc, 'pay': pay, 'label': label})

        fig = fig_compare(results)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        st.markdown("<p class='section-header'>comparison summary</p>", unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        for col, r in zip([sc1, sc2], results):
            v, vc, _, icon = get_verdict(r['det'], threshold, uncert_band)
            w = wfus(r['loc'], 20)
            with col:
                st.markdown(f"""
                <div class='report-box' style='border-left-color:{vc}'>
                  <p style='color:{vc};font-family:"Rajdhani",sans-serif;font-size:20px;letter-spacing:3px;margin:0'>{icon} {v}</p>
                  <p style='color:#4a6080;font-size:11px;font-family:"Share Tech Mono",monospace;margin:4px 0'>{r["label"]}</p>
                  <p style='color:#c8d8e8;font-size:12px;font-family:"Share Tech Mono",monospace;margin:8px 0 2px'>P(stego): {r["det"]:.4f}</p>
                  <p style='color:#9966ff;font-size:12px;font-family:"Share Tech Mono",monospace;margin:2px 0'>wFUS@20%: {w:.2%} ({w/0.20:.1f}× random)</p>
                  <p style='color:#00ff88;font-size:12px;font-family:"Share Tech Mono",monospace;margin:2px 0'>Peak payload: {r["pay"].max():.4f} bpp</p>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='border:1px dashed #1a2535;border-radius:10px;padding:40px;text-align:center'>
          <p style='color:#4a6080;font-family:"Share Tech Mono",monospace'>Upload both images above to run comparison</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — BATCH MODE
# ══════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("<p class='section-header'>batch analysis</p>", unsafe_allow_html=True)
    batch_files = st.file_uploader("Upload multiple images",
                                   type=["png","jpg","jpeg","pgm","bmp"],
                                   accept_multiple_files=True,
                                   label_visibility="collapsed")

    if batch_files:
        batch_results = []
        prog = st.progress(0, text="Analyzing images...")
        for i, f in enumerate(batch_files):
            img_b = f.read()
            det, loc, pay = run_inference_cached(img_b, ckpt_path, device_opt, use_gray)
            v, vc, _, icon = get_verdict(det, threshold, uncert_band)
            w = wfus(loc, 20)
            batch_results.append({'name': f.name, 'det': det, 'verdict': v, 'color': vc, 'icon': icon, 'wfus20': w, 'peak': pay.max()})
            prog.progress((i+1)/len(batch_files), text=f"Analyzed {i+1}/{len(batch_files)}")
        prog.empty()

        # Summary table
        st.markdown("<p class='section-header'>batch results</p>", unsafe_allow_html=True)
        header = "<div style='display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr;gap:0;background:#0d1520;border:1px solid #1a2535;border-radius:8px;overflow:hidden'>"
        header += "".join(f"<div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:#4a6080;font-family:\"Share Tech Mono\",monospace;font-size:10px;letter-spacing:1px'>{h}</div>"
                          for h in ["FILENAME", "VERDICT", "P(STEGO)", "wFUS@20%", "PEAK BPP"])

        rows = ""
        for r in batch_results:
            rows += f"""
            <div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:#8aabbf;font-size:12px;font-family:"Share Tech Mono",monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{r["name"]}</div>
            <div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:{r["color"]};font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:1px'>{r["icon"]} {r["verdict"]}</div>
            <div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:#8aabbf;font-family:"Share Tech Mono",monospace;font-size:11px'>{r["det"]:.4f}</div>
            <div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:#9966ff;font-family:"Share Tech Mono",monospace;font-size:11px'>{r["wfus20"]:.1%}</div>
            <div style='padding:10px 14px;border-bottom:1px solid #1a2535;color:#00ff88;font-family:"Share Tech Mono",monospace;font-size:11px'>{r["peak"]:.4f}</div>"""

        n_stego = sum(1 for r in batch_results if r['verdict'] == 'STEGO')
        n_cover = sum(1 for r in batch_results if r['verdict'] == 'COVER')
        n_unsure = sum(1 for r in batch_results if r['verdict'] == 'UNCERTAIN')

        st.markdown(header + rows + "</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:16px;display:flex;gap:12px'>
          <div class='metric-card' style='flex:1'><p class='metric-val' style='color:#ff3355'>{n_stego}</p><p class='metric-lbl'>STEGO detected</p></div>
          <div class='metric-card' style='flex:1'><p class='metric-val' style='color:#00ff88'>{n_cover}</p><p class='metric-lbl'>COVER (clean)</p></div>
          <div class='metric-card' style='flex:1'><p class='metric-val' style='color:#ffaa00'>{n_unsure}</p><p class='metric-lbl'>UNCERTAIN</p></div>
          <div class='metric-card' style='flex:1'><p class='metric-val' style='color:#c8d8e8'>{len(batch_results)}</p><p class='metric-lbl'>Total analyzed</p></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════

with tab_how:
    st.markdown("<p class='section-header'>architecture overview</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:24px'>

      <div class='report-box' style='border-left-color:#00c8ff'>
        <p style='color:#00c8ff;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>01 · SRM STREAM</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>30 fixed Fridrich high-pass kernels. Mathematically designed to suppress image content and amplify sub-pixel statistical deviations caused by LSB embedding. Non-learnable — prevents convergence to generic edge detectors.</p>
      </div>

      <div class='report-box' style='border-left-color:#9966ff'>
        <p style='color:#9966ff;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>02 · CNN STREAM</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>ResNet-18 backbone (layers 1–2, stride-8). Random initialization — no ImageNet pretraining. ImageNet features encode semantic objects which are anti-correlated with stego signal. Random init forces learning stego-specific residuals.</p>
      </div>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>03 · SE FUSION</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Squeeze-and-Excitation channel attention on each stream. Learnable scalar alpha (α≈0.53) mixes contributions. Neither stream fully suppressed. Produces fused feature map [B, 256, 32, 32].</p>
      </div>

      <div class='report-box' style='border-left-color:#00ff88'>
        <p style='color:#00ff88;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>04 · DETECTION HEAD</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Global Average Pooling → Linear(256→128) → BN → ReLU → Dropout(0.3) → Linear(128→1). Outputs raw logit for BCEWithLogitsLoss. Achieves 100% test accuracy on BOSSBase 2-bit LSB.</p>
      </div>

      <div class='report-box' style='border-left-color:#ff3355'>
        <p style='color:#ff3355;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>05 · LOCALIZATION HEAD</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>UNet-lite decoder: 3× bilinear upsample (32×32 → 256×256). Full-resolution SRM skip connection at final stage routes precise spatial signal. Trained vs continuous rho maps. wFUS@20% = 61.5%.</p>
      </div>

      <div class='report-box' style='border-left-color:#34d399'>
        <p style='color:#34d399;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>06 · PAYLOAD HEAD</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Same UNet-lite decoder. ReLU output (density ≥ 0). Trained vs rho×rate maps using Huber loss (δ=0.1). Estimates bits-per-pixel embedding density. Test MAE = 0.0079 bpp.</p>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-box' style='margin-bottom:16px'>
      <p style='color:#00c8ff;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 12px'>TRAINING STRATEGY</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Stage 1 (epochs 1–10):</span> Detection-only warmup. Loss = BCEWithLogitsLoss only. Allows detection head to reach ~100% before spatial supervision.</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Stage 2 (epochs 11–40):</span> Multi-task training. Loss = L_cls + λ₁·L_loc + λ₂·L_pay. Lambda ramps 0→0.3 over 5 epochs to prevent gradient shock.</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Optimizer:</span> Adam (lr=5e-4, weight_decay=1e-4) with CosineAnnealingLR. AMP enabled for 1.6× speedup.</p>
    </div>

    <div class='report-box'>
      <p style='color:#9966ff;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 12px'>METRICS EXPLAINED</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Accuracy / AUC / F1:</span> Standard classification metrics. AUC=1.000 means perfect separation of cover and stego.</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Soft IoU (0.037):</span> Structurally penalized against spatially diffuse rho maps. Even a perfect predictor scores low on continuous ground truth. Not the primary metric.</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>wFUS@k%:</span> Primary localization metric. Fraction of embedding mass recovered by examining only the top-k% of pixels. Random baseline = k/100. Our 61.5% = 3.1× improvement over random search.</p>
      <p style='color:#8aabbf;font-size:13px;margin:4px 0'><span style='color:#c8d8e8'>Payload MAE (0.0079 bpp):</span> Mean absolute error in bits-per-pixel density estimation. Sub-1% error.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — LIMITATIONS
# ══════════════════════════════════════════════════════════════

with tab_limits:
    st.markdown("<p class='section-header'>model limitations & scope</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>⚠ GRAYSCALE ONLY</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Trained exclusively on BOSSBase grayscale .pgm images. Colour images introduce channel distribution shift. Enable grayscale preprocessing to reduce (not eliminate) this effect.</p>
      </div>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>⚠ JPEG SENSITIVITY</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>JPEG DCT quantization artifacts create high-frequency noise that resembles LSB embedding. May produce false positives on compressed images. Use uncompressed PNG/BMP/PGM for reliable results.</p>
      </div>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>⚠ SYNTHETIC EMBEDDING ONLY</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Trained on synthetic 2-bit LSB embedding guided by local variance. May not generalize to real-world steganographic tools (F5, WOW, HUGO, S-UNIWARD). Domain adaptation needed for production.</p>
      </div>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>⚠ SOFT IoU CEILING</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Soft IoU (0.037) appears low because continuous rho maps are spatially diffuse. Even a perfect predictor scores low on this metric. wFUS@20% (0.615) is the meaningful localization metric.</p>
      </div>

      <div class='report-box' style='border-left-color:#ffaa00'>
        <p style='color:#ffaa00;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>⚠ PNG QUANTIZATION (FILE UPLOADS)</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Stego embedding lives in normalized float tensor space — sub-0.01 value changes invisible in uint8. Saving stego to PNG and re-uploading destroys the signal. The Stego Demo button bypasses this by pulling tensors directly from the dataset pipeline. For your own stego images, use the dataset pipeline or embed at a higher bit depth.</p>
      </div>

      <div class='report-box' style='border-left-color:#1a2535'>
        <p style='color:#c8d8e8;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>✓ STRENGTHS</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>100% detection accuracy on in-distribution data. 3.1× wFUS improvement. Sub-1% payload estimation error. Multi-task unified inference. 2.09M parameters, runs in real-time.</p>
      </div>

      <div class='report-box' style='border-left-color:#1a2535'>
        <p style='color:#c8d8e8;font-family:"Share Tech Mono",monospace;font-size:11px;letter-spacing:2px;margin:0 0 8px'>→ FUTURE WORK</p>
        <p style='font-size:13px;color:#8aabbf;margin:0'>Domain adaptation for real-world stego tools. JPEG-aware training. Attention visualization. Higher payload rates. Real stego ground truth for localization. Cross-dataset evaluation.</p>
      </div>

    </div>
    """, unsafe_allow_html=True)