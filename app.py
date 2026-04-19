import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

st.set_page_config(page_title="CIFAR-10 Neural Network Classifier", layout="wide", page_icon="")

CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
CLASS_ICONS = ['✈', '🚗', '🐦', '🐱', '🦌', '🐶', '🐸', '🐴', '🚢', '🚛']
NUM_CLASSES = 10

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #050810; }

.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1526 50%, #0a0f1e 100%);
    border: 1px solid #1a2744;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, #1a3a8f15 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero-title span { color: #4d9fff; }
.hero-sub {
    color: #4a6080;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.upload-zone {
    background: #0a0f1e;
    border: 2px dashed #1a2744;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s;
}
.pred-card {
    background: linear-gradient(135deg, #0a0f1e, #0d1526);
    border: 1px solid #1a2744;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4d9fff, #a855f7, #4d9fff);
}
.pred-class {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.3rem 0;
    letter-spacing: -1px;
}
.pred-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4d9fff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.pred-label {
    color: #4a6080;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card {
    background: #0a0f1e;
    border: 1px solid #1a2744;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4d9fff;
}
.metric-lbl {
    color: #4a6080;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.2rem;
}
.top3-card {
    background: #0a0f1e;
    border: 1px solid #1a2744;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    color: #4a6080;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}
.status-pill {
    display: inline-block;
    background: #0d2210;
    border: 1px solid #1a4a20;
    color: #4ade80;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
}
.warn-pill {
    display: inline-block;
    background: #1a1000;
    border: 1px solid #4a3000;
    color: #fbbf24;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
}
div[data-testid="stFileUploader"] {
    background: #0a0f1e;
}
div[data-testid="stSidebar"] {
    background: #050810;
    border-right: 1px solid #1a2744;
}
</style>
""", unsafe_allow_html=True)


# ── Pure numpy CNN forward pass ───────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def batch_norm_inference(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def conv2d(x, W, b, padding=1):
    # x: (H, W, C_in)   W: (kH, kW, C_in, C_out)
    H, W_sz, C_in = x.shape
    kH, kW, _, C_out = W.shape
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)))
    oH = x.shape[0] - kH + 1
    oW = x.shape[1] - kW + 1
    out = np.zeros((oH, oW, C_out), dtype=np.float32)
    for co in range(C_out):
        for ci in range(C_in):
            for ki in range(kH):
                for kj in range(kW):
                    out[:, :, co] += x[ki:ki+oH, kj:kj+oW, ci] * W[ki, kj, ci, co]
        out[:, :, co] += b[co]
    return out

def maxpool2d(x):
    H, W, C = x.shape
    oH, oW = H // 2, W // 2
    out = np.zeros((oH, oW, C), dtype=np.float32)
    for i in range(oH):
        for j in range(oW):
            out[i, j, :] = x[2*i:2*i+2, 2*j:2*j+2, :].max(axis=(0, 1))
    return out

def predict_numpy(image, weights):
    """
    Runs the CNN forward pass using pure numpy.
    Matches the exact architecture built in the notebook.
    weights = model.get_weights() — list of arrays in Keras layer order.
    """
    x = image.astype(np.float32)  # (32, 32, 3)
    w = weights
    i = 0

    def conv_bn_relu(x, i):
        # Conv2D kernel, bias
        x = conv2d(x, w[i], w[i+1])
        i += 2
        # BN: gamma, beta, mean, var
        x = batch_norm_inference(x, w[i], w[i+1], w[i+2], w[i+3])
        i += 4
        x = relu(x)
        return x, i

    # Block 1
    x, i = conv_bn_relu(x, i)
    x, i = conv_bn_relu(x, i)
    x = maxpool2d(x)

    # Block 2
    x, i = conv_bn_relu(x, i)
    x, i = conv_bn_relu(x, i)
    x = maxpool2d(x)

    # Block 3
    x, i = conv_bn_relu(x, i)
    x, i = conv_bn_relu(x, i)
    x = maxpool2d(x)

    # Flatten
    x = x.flatten()

    def dense_bn_relu(x, i):
        x = w[i].T @ x + w[i+1]
        i += 2
        x = batch_norm_inference(x, w[i], w[i+1], w[i+2], w[i+3])
        i += 4
        x = relu(x)
        return x, i

    # Dense 512
    x, i = dense_bn_relu(x, i)
    # Dense 256
    x, i = dense_bn_relu(x, i)
    # Output
    x = w[i].T @ x + w[i+1]
    return softmax(x)


@st.cache_resource(show_spinner=False)
def load_weights():
    weights = np.load('model_weights.npy', allow_pickle=True)
    return list(weights)


def preprocess(pil_image):
    img = pil_image.convert('RGB').resize((32, 32), Image.LANCZOS)
    return np.array(img).astype('float32') / 255.0


def make_chart(probs):
    sorted_idx = np.argsort(probs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0a0f1e')

    bar_colors = ['#4d9fff' if i == sorted_idx[0] else '#1a2744' for i in range(NUM_CLASSES)]
    bars = ax.barh(range(NUM_CLASSES),
                   [probs[i] * 100 for i in sorted_idx],
                   color=[bar_colors[sorted_idx[j]] for j in range(NUM_CLASSES)],
                   height=0.6, edgecolor='none')

    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(
        [f"{CLASS_ICONS[sorted_idx[j]]}  {CLASS_NAMES[sorted_idx[j]]}" for j in range(NUM_CLASSES)],
        color='#8899aa', fontsize=10, fontfamily='monospace'
    )
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)', color='#4a6080', fontsize=9)
    ax.tick_params(colors='#4a6080', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1a2744')
    ax.set_xlim(0, 115)
    for j, bar in enumerate(bars):
        val = probs[sorted_idx[j]] * 100
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center',
                color='#4d9fff' if j == 0 else '#4a6080',
                fontsize=9, fontfamily='monospace',
                fontweight='bold' if j == 0 else 'normal')
    ax.set_title('Confidence Distribution', color='#4a6080',
                 fontsize=9, pad=10, fontfamily='monospace',
                 loc='left', style='italic')
    plt.tight_layout()
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <div style='font-family: IBM Plex Mono, monospace; color: #4d9fff;
                    font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;'>
            Model Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)

    arch_steps = [
        ("INPUT", "32 × 32 × 3"),
        ("BLOCK 1", "Conv64 + BN + ReLU ×2\nMaxPool + Dropout"),
        ("BLOCK 2", "Conv128 + BN + ReLU ×2\nMaxPool + Dropout"),
        ("BLOCK 3", "Conv256 + BN + ReLU ×2\nMaxPool + Dropout"),
        ("DENSE", "512 → 256 + BN + ReLU"),
        ("OUTPUT", "Softmax (10 classes)"),
    ]

    for label, detail in arch_steps:
        st.markdown(f"""
        <div style='background:#0a0f1e; border:1px solid #1a2744; border-radius:8px;
                    padding:0.6rem 0.8rem; margin:0.3rem 0;'>
            <div style='font-family: IBM Plex Mono, monospace; color:#4d9fff;
                        font-size:0.7rem; letter-spacing:1px;'>{label}</div>
            <div style='color:#8899aa; font-size:0.75rem; margin-top:2px;
                        white-space:pre-line;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; color: #4a6080;
                font-size: 0.7rem; line-height: 1.8;'>
        STAT222 · BSDS-02<br>
        Lab 11 — Open Ended Lab<br>
        Instructor: Ms. Ansar Shahzadi<br>
        Dataset: CIFAR-10
    </div>
    """, unsafe_allow_html=True)


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>CIFAR<span>-10</span> Neural Classifier</div>
    <div class='hero-sub'>Deep CNN · Keras / TensorFlow · Lab 11 · BSDS-02</div>
</div>
""", unsafe_allow_html=True)

# Load weights
with st.spinner("Loading neural network weights..."):
    try:
        weights = load_weights()
        st.markdown("<div class='status-pill'>Model loaded</div>", unsafe_allow_html=True)
        model_loaded = True
    except Exception as e:
        st.markdown("<div class='warn-pill'>Weight file not found — check model_weights.npy is in repo</div>",
                    unsafe_allow_html=True)
        model_loaded = False

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<div class='section-title'>Upload Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                 label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Original image", use_column_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            small = pil_img.convert('RGB').resize((32, 32))
            st.image(small, caption="Model input (32x32)", use_column_width=True)
        with col_b:
            w, h = pil_img.size
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val'>{w}×{h}</div>
                <div class='metric-lbl'>Original Size</div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style='border: 2px dashed #1a2744; border-radius: 16px;
                    padding: 4rem 2rem; text-align: center; color: #2a3a54;
                    font-family: IBM Plex Mono, monospace; font-size: 0.9rem;'>
            Upload an image to classify
        </div>
        """, unsafe_allow_html=True)

    elif not model_loaded:
        st.error("Model weights not loaded. Check that model_weights.npy is in the GitHub repo.")

    else:
        img_arr = preprocess(pil_img)

        with st.spinner("Running CNN forward pass..."):
            try:
                probs = predict_numpy(img_arr, weights)
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                top_name = CLASS_NAMES[top_idx]
                top_icon = CLASS_ICONS[top_idx]

                # Prediction card
                conf_color = "#4ade80" if top_conf > 0.7 else "#fbbf24" if top_conf > 0.4 else "#f87171"
                st.markdown(f"""
                <div class='pred-card'>
                    <div class='pred-label'>Prediction</div>
                    <div style='font-size:3rem; margin:0.3rem 0;'>{top_icon}</div>
                    <div class='pred-class'>{top_name}</div>
                    <div class='pred-conf'>{top_conf*100:.1f}%</div>
                    <div class='pred-label'>confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Top 3 Predictions</div>",
                            unsafe_allow_html=True)
                top3 = np.argsort(probs)[::-1][:3]
                medals = ["01", "02", "03"]
                for rank, idx in enumerate(top3):
                    bar_w = int(probs[idx] * 100)
                    st.markdown(f"""
                    <div class='top3-card'>
                        <div style='display:flex; align-items:center; gap:0.8rem;'>
                            <span style='font-family:IBM Plex Mono,monospace;
                                         color:#4a6080; font-size:0.75rem;'>#{medals[rank]}</span>
                            <span style='font-size:1.3rem;'>{CLASS_ICONS[idx]}</span>
                            <span style='color:#ffffff; font-weight:600;'>{CLASS_NAMES[idx]}</span>
                        </div>
                        <span style='font-family:IBM Plex Mono,monospace; color:#4d9fff;
                                     font-weight:700;'>{probs[idx]*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Confidence Distribution</div>",
                            unsafe_allow_html=True)
                fig = make_chart(probs)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("The weight file structure may not match. Make sure model_weights.npy was downloaded from the same notebook.")
