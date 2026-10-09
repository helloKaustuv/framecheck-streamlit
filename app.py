import sys
import subprocess
import os
import hashlib
import time
import io
import tempfile
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import streamlit as st

try:
    import torch
    from transformers import pipeline
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "pillow", "opencv-python-headless", "streamlit"])
    import torch
    from transformers import pipeline
    import cv2

# Optimize PyTorch CPU inference speed
torch.set_num_threads(4)

# Page Configuration
st.set_page_config(
    page_title="FrameCheck — AI Image & Video Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Mode CSS (High-Contrast Cyberpunk Glassmorphic Design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #090d16 !important;
        color: #f1f5f9 !important;
    }
    .stApp {
        background-color: #090d16 !important;
    }

    /* Fix Streamlit default typography colors for Dark Mode */
    p, span, label, div {
        color: #cbd5e1;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }
    
    /* Header Bar */
    .nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .logo-badge {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    .logo-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .logo-title span {
        color: #10b981;
    }
    .model-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399 !important;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Cards */
    .dark-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2.25rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
    }

    /* Hero Typography */
    .hero-tag {
        color: #10b981 !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-title {
        font-size: 3.75rem;
        font-weight: 800;
        line-height: 1.08;
        color: #ffffff !important;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
        letter-spacing: -0.03em;
    }
    .hero-title span {
        background: linear-gradient(135deg, #34d399 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #94a3b8 !important;
        line-height: 1.6;
    }

    /* Verdict Card (Dark Emerald for Real, Dark Rose for AI) */
    .verdict-box-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    .verdict-box-ai {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.4) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(251, 113, 133, 0.4);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 0 30px rgba(244, 63, 94, 0.15);
    }

    /* Terminal Console Box */
    .terminal-box {
        background-color: #020617;
        color: #f8fafc;
        border-radius: 16px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.7;
        border: 1px solid #1e293b;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6);
    }
    .terminal-header {
        color: #64748b;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.6rem;
        margin-bottom: 0.9rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
    }
    .highlight-label {
        color: #4ade80;
        font-weight: bold;
    }
    .highlight-score {
        color: #38bdf8;
        font-weight: bold;
    }

    /* Streamlit File Uploader Dark Mode Fix */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #10b981 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 12px;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.85rem;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

MODEL_ID = "Organika/sdxl-detector"

@st.cache_resource(show_spinner=False)
def get_detector_pipeline():
    """
    Cached initialization of Organika/sdxl-detector pipeline.
    Runs fast without reloading on every frame/rerun.
    """
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=MODEL_ID, device=device)

def run_model_inference(image: Image.Image):
    """Executes model inference with PyTorch no_grad for maximum speed."""
    detector = get_detector_pipeline()
    with torch.no_grad():
        predictions = detector(image)
    return predictions

# JPEG ERROR LEVEL ANALYSIS (ELA)
def generate_ela_image(image: Image.Image, quality: int = 85) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)

    ela = ImageChops.difference(image.convert('RGB'), compressed.convert('RGB'))
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela

# VIDEO KEYFRAME EXTRACTOR
def extract_video_frames(video_bytes, max_frames=10):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_bytes)
        temp_path = tfile.name

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0

    step = max(1, total_frames // max_frames)
    frames_data = []

    for i in range(0, total_frames, step):
        if len(frames_data) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        timestamp_sec = i / fps
        frames_data.append({
            "index": len(frames_data) + 1,
            "timestamp_sec": timestamp_sec,
            "timestamp_str": f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}.{int((timestamp_sec%1)*10)}",
            "image": pil_img
        })

    cap.release()
    try:
        os.remove(temp_path)
    except:
        pass

    return frames_data, duration, fps, total_frames

# Header
st.markdown("""
<div class="nav-header">
    <div class="logo-container">
        <div class="logo-badge">🛡️</div>
        <div class="logo-title">Frame<span>Check</span></div>
    </div>
    <div class="model-badge">
        <span style="width: 8px; height: 8px; background-color: #34d399; border-radius: 50%; display: inline-block;"></span>
        Model: Organika/sdxl-detector &nbsp; | &nbsp; PyTorch Neural Engine
    </div>
</div>
""", unsafe_allow_html=True)

# Split Grid Hero Section
col1, col2 = st.columns([7, 5], gap="large")

with col1:
    st.markdown("""
    <div class="hero-tag">● A CLEARER PICTURE OF WHAT YOU SEE</div>
    <h1 class="hero-title">Is it real, or <span>AI-made?</span></h1>
    <p class="hero-sub">
        Get an instant machine learning signal on any image or video with the <strong>Organika/sdxl-detector</strong> model.
        Fast, high-contrast inference on Streamlit Cloud.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; gap: 2rem; margin-top: 2rem; font-size: 0.9rem; font-weight: 700; color: #10b981;">
        <div>✳ Simple image check</div>
        <div>☁ Hosted processing</div>
        <div>⚡ Fast CPU/GPU inference</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 1rem;'>Check an Image or Video</h3>", unsafe_allow_html=True)
    
    upload_mode = st.radio("Media Type", ["Image", "Video NEW"], horizontal=True, label_visibility="collapsed")
    
    if upload_mode == "Image":
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded_file = st.file_uploader("Upload Video (MP4, WEBM, MOV)", type=["mp4", "webm", "mov", "avi"])

    st.markdown('<div style="margin-top: 1rem; text-align: center; font-size: 0.8rem; color: #64748b;">☁ Uploaded media is processed on Streamlit's server</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ANALYSIS ENGINE RUNNER
if uploaded_file is not None:
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Classification Results")

    if upload_mode == "Image":
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        # Fast Cached Model Inference
        with st.spinner("⚡ Running Organika/sdxl-detector neural model..."):
            start_time = time.time()
            predictions = run_model_inference(image)
            elapsed_ms = (time.time() - start_time) * 1000.0

        best = predictions[0]
        best_label = best['label'] # 'artificial' or 'human'
        best_score = best['score'] # 0.0 to 1.0
        confidence_pct = best_score * 100.0

        is_artificial = (best_label.lower() == 'artificial')
        
        # Display ONLY detected class score (AI score when AI, Human score when Human)
        v_class = "verdict-box-ai" if is_artificial else "verdict-box-real"
        v_color = "#f43f5e" if is_artificial else "#34d399"
        v_title = "ARTIFICIAL (AI-MADE)" if is_artificial else "HUMAN (REAL AUTHENTIC)"
        
        st.markdown(f"""
        <div class="{v_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 800; text-transform: uppercase; color: {v_color}; letter-spacing: 0.1em;">
                        {v_title}
                    </span>
                    <h2 style="font-size: 3rem; margin: 0.25rem 0; color: #ffffff !important;">
                        {confidence_pct:.1f}% <span style="font-size: 1.5rem; color: {v_color}; font-weight: 700;">{best_label.upper()}</span>
                    </h2>
                    <span style="font-size: 0.85rem; color: #94a3b8;">
                        File: {uploaded_file.name} &nbsp; | &nbsp; Resolution: {image.width} × {image.height} &nbsp; | &nbsp; Latency: {elapsed_ms:.0f}ms
                    </span>
                </div>
                <div style="text-align: right; min-width: 200px;">
                    <span style="font-size: 0.9rem; font-weight: 700; color: #ffffff;">{best_label.upper()} SCORE</span>
                    <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 999px; margin-top: 0.5rem; overflow: hidden;">
                        <div style="width: {confidence_pct}%; height: 100%; background: {v_color}; transition: width 0.6s ease;"></div>
                    </div>
                </div>
            </div>
        </div>
        <br/>
        """, unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["💻 Colab Terminal Console", "🔍 Metadata & EXIF", "🔬 ELA Forensics Studio"])

        with tab1:
            res_col1, res_col2 = st.columns([1, 1], gap="medium")
            with res_col1:
                st.image(image, caption=f"{best_label} — {confidence_pct:.1f}%", use_container_width=True)
            
            with res_col2:
                # Output Block matching Colab script
                st.markdown(f"""
                <div class="terminal-box">
                    <div class="terminal-header">
                        <span>Python Console Output (Organika/sdxl-detector)</span>
                        <span>PyTorch / Transformers</span>
                    </div>
                    <div>Model prediction: <span class="highlight-label">{best_label}</span></div>
                    <div>Confidence: <span class="highlight-score">{confidence_pct:.1f}%</span></div>
                    <br/>
                    <div>Single Score Output ({best_label}):</div>
                    <pre style="color: #f59e0b; background: transparent; padding: 0; margin: 0;">[{{'label': '{best_label}', 'score': {best_score}}}]</pre>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("##### File Attributes & Cryptographic SHA-256")
            st.json({
                "File Name": uploaded_file.name,
                "File Size": f"{len(file_bytes)/1024:.2f} KB",
                "Dimensions": f"{image.width} × {image.height}",
                "Color Format": image.mode,
                "SHA-256 Hash": sha256_hash,
                "Model Pipeline": MODEL_ID,
                "Inference Latency": f"{elapsed_ms:.1f} ms"
            })

            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif():
                raw_exif = image._getexif()
                for tag_id, val in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[str(tag)] = str(val)
                st.markdown("##### Camera Hardware EXIF Tags")
                st.json(exif_data)

        with tab3:
            st.markdown("##### JPEG Error Level Analysis (ELA) Heatmap")
            ela_img = generate_ela_image(image)
            ela_col1, ela_col2 = st.columns(2)
            with ela_col1:
                st.image(image, caption="Original Input Image", use_container_width=True)
            with ela_col2:
                st.image(ela_img, caption="JPEG ELA Compression Heatmap", use_container_width=True)

    else:
        # Video Processing
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        with st.spinner("⚡ Extracting video keyframes & running Organika/sdxl-detector..."):
            frames, duration, fps, total_frames = extract_video_frames(video_bytes, max_frames=10)

            timeline_results = []
            total_top_score = 0.0
            art_count = 0

            for frame_info in frames:
                preds = run_model_inference(frame_info["image"])
                top_pred = preds[0]
                label = top_pred["label"]
                score = top_pred["score"]

                if label.lower() == 'artificial':
                    art_count += 1

                total_top_score += score

                timeline_results.append({
                    "frame": frame_info["index"],
                    "timestamp": frame_info["timestamp_str"],
                    "label": label,
                    "confidence": f"{score*100:.1f}%",
                    "image": frame_info["image"]
                })

            is_video_art = (art_count >= len(timeline_results) / 2)
            avg_score_pct = (total_top_score / max(1, len(timeline_results))) * 100.0

        v_class = "verdict-box-ai" if is_video_art else "verdict-box-real"
        v_title = "ARTIFICIAL (AI GENERATED VIDEO)" if is_video_art else "HUMAN (REAL AUTHENTIC VIDEO)"
        
        st.markdown(f"""
        <div class="{v_class}">
            <span style="font-size: 0.8rem; font-weight: 800; text-transform: uppercase; color: #ffffff; letter-spacing: 0.1em;">
                {v_title}
            </span>
            <h2 style="font-size: 2.75rem; margin: 0.25rem 0; color: #ffffff !important;">
                {avg_score_pct:.1f}% <span style="font-size: 1.5rem;">{'AI Score' if is_video_art else 'Human Score'}</span>
            </h2>
            <span style="font-size: 0.85rem; color: #cbd5e1;">
                Duration: {duration:.1f}s &nbsp; | &nbsp; FPS: {fps:.1f} &nbsp; | &nbsp; Sampled Keyframes: {len(timeline_results)}
            </span>
        </div>
        <br/>
        """, unsafe_allow_html=True)

        st.markdown("##### Keyframe Timeline Inspection")
        cols = st.columns(len(timeline_results))
        for idx, item in enumerate(timeline_results):
            with cols[idx]:
                st.image(item["image"], caption=f"{item['timestamp']}\n{item['label']}\n{item['confidence']}", use_container_width=True)
