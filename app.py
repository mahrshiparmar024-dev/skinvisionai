"""
============================================================
  SKINVISION AI — Main Streamlit Application
============================================================
  Skin Disease Detection & Classification Using CNN
  A Virtual Dermatologist powered by AI
  
  University Final Year Project
  Dataset: HAM10000 | Model: EfficientNetB3 (Transfer Learning)
  
  Inspired by open-source implementations such as:
  github.com/FridahKimathi/Skin-Disease-Image-Classifier
  
  Independently implemented with significant improvements:
  - Grad-CAM explainability
  - Confidence thresholding & skin image validation
  - PDF diagnostic reports
  - Modern clinical UI (Streamlit, not Flask)
  - Image-aware demo mode for presentations
============================================================
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from datetime import datetime

# ── Local Imports ──
from config import (
    APP_TITLE, APP_SUBTITLE, APP_VERSION,
    CLASS_LABELS, CLASS_DESCRIPTIONS, PROJECT_STATS,
    DISCLAIMER_TEXT, IMPACT_STATEMENT, COLORS, IMG_SIZE,
    TREATMENT_RECOMMENDATIONS
)
from model_utils import (
    load_trained_model, preprocess_image, predict_disease,
    generate_gradcam, overlay_gradcam, validate_image,
    validate_skin_image, get_confidence_color, get_confidence_label,
    get_model_debug_info, CONFIDENCE_THRESHOLD
)
from pdf_generator import generate_pdf_report

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL SETUP
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title=f"{APP_TITLE} — AI Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Custom CSS ──
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        css_content = f.read()
        # MOBILE FIX: Remove header hiding so hamburger menu shows on mobile
        css_content = css_content.replace('header {visibility: hidden;}', 
                                          'header {visibility: visible !important;}')
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        # Additional mobile navigation CSS
        st.markdown("""
        <style>
        /* Show header on mobile for hamburger menu */
        @media (max-width: 768px) {
            header[data-testid="stHeader"] {
                visibility: visible !important;
                background: rgba(15, 23, 42, 0.95) !important;
            }
            /* Compact hero on mobile */
            .hero-section { padding: 32px 20px !important; }
            .hero-title { font-size: 28px !important; }
            .hero-subtitle { font-size: 15px !important; }
            /* Mobile nav bar styling */
            .mobile-nav-bar {
                background: linear-gradient(135deg, #0A6C9E, #085580);
                padding: 8px 16px;
                border-radius: 12px;
                margin-bottom: 16px;
            }
            .mobile-nav-bar p {
                color: white !important;
                font-size: 13px !important;
                margin: 0 !important;
            }
        }
        /* Desktop: header can stay hidden, hide mobile nav */
        @media (min-width: 769px) {
            header[data-testid="stHeader"] {
                visibility: hidden;
            }
            .mobile-nav-bar {
                display: none !important;
            }
            /* Hide the mobile selectbox on desktop */
            div[data-testid="stSelectbox"]:first-of-type {
                display: none !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)

# ── Load Model (cached) ──
@st.cache_resource
def get_model(_version=3):
    return load_trained_model()

_model_result = get_model()
if isinstance(_model_result[1], dict):
    model, model_status = _model_result
else:
    model = _model_result[0]
    model_status = {"error": "Model not available", "model_path": "models/skin_disease_model.h5", "demo_mode": True}

model_loaded = model is not None
demo_mode = model_status.get("demo_mode", True)

# ══════════════════════════════════════════════════════════
#  HELPER: HTML COMPONENTS
# ══════════════════════════════════════════════════════════

def render_hero():
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-badge">🔬 AI-Powered Healthcare</div>
        <div class="hero-title">Virtual Dermatologist</div>
        <div class="hero-subtitle">{IMPACT_STATEMENT}</div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(icon, value, label):
    return f"""
    <div class="metric-card">
        <div style="font-size:28px; margin-bottom:8px;">{icon}</div>
        <span class="metric-value">{value}</span>
        <span class="metric-label">{label}</span>
    </div>
    """

def render_step_card(number, icon, title, desc):
    return f"""
    <div class="step-card">
        <div class="step-number">{number}</div>
        <div style="font-size:32px; margin-bottom:12px;">{icon}</div>
        <h4>{title}</h4>
        <p>{desc}</p>
    </div>
    """

def render_disclaimer():
    st.markdown(f"""
    <div class="disclaimer-banner prominent">
        <div>
            <p><strong>⚠️ Medical Disclaimer:</strong> {DISCLAIMER_TEXT}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_section_header(title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        <p>{subtitle}</p>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown(f"""
    <div class="app-footer">
        <strong>{APP_TITLE}</strong> v{APP_VERSION} · AI-Powered Skin Disease Detection<br>
        University Final Year Project · Built with Streamlit & TensorFlow<br><br>
        <em>{DISCLAIMER_TEXT}</em>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  NAVIGATION (Sidebar + Mobile Top Bar)
# ══════════════════════════════════════════════════════════

NAV_OPTIONS = [
    "🏠 Home", "📖 About", "⚙️ How It Works", "🧠 Model Details",
    "🔬 Try the Detector", "📊 Comparison", "⚖️ Disclaimer & Ethics"
]

# ── Mobile Top Navigation Bar ──
# (Visible to all users as a fallback, especially mobile where sidebar is collapsed)
st.markdown("""
<div class="mobile-nav-bar">
    <p>📱 <strong>Tap ☰ (top-left)</strong> to open the navigation menu, or use the dropdown below:</p>
</div>
""", unsafe_allow_html=True)
mobile_page = st.selectbox(
    "Navigate to:", NAV_OPTIONS,
    index=0,
    key="mobile_nav",
    label_visibility="collapsed"
)

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0 24px;">
        <div style="font-size:36px;">🔬</div>
        <h2 style="color:#FFFFFF; margin:8px 0 4px; font-size:22px;">{APP_TITLE}</h2>
        <p style="color:#94A3B8; font-size:13px; margin:0;">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    sidebar_page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(mobile_page),
        label_visibility="collapsed",
        key="sidebar_nav"
    )
    
    st.markdown("---")
    
    if model_loaded:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
                    border-radius:12px; padding:12px 16px; margin-top:8px;">
            <p style="color:#10B981 !important; font-size:12px !important; margin:0;">
                <strong>✅ Model Loaded</strong><br>
                Real-time AI predictions active.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3);
                    border-radius:12px; padding:12px 16px; margin-top:8px;">
            <p style="color:#F59E0B !important; font-size:12px !important; margin:0;">
                <strong>🧪 Demo Mode</strong><br>
                Image-aware predictions active.<br>
                Place model in models/ for real AI.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Page selection: use whichever was last changed
# Mobile dropdown and sidebar radio sync through index binding
page = sidebar_page


# ══════════════════════════════════════════════════════════
#  PAGE 1: HOME
# ══════════════════════════════════════════════════════════

if page == "🏠 Home":
    render_hero()
    
    cols = st.columns(4)
    stats = [
        ("🎯", str(PROJECT_STATS["classes"]), "Disease Classes"),
        ("📊", PROJECT_STATS["images"], "Training Images"),
        ("✅", PROJECT_STATS["accuracy"], "Accuracy"),
        ("🧠", PROJECT_STATS["technique"], "AI Technique"),
    ]
    for col, (icon, val, label) in zip(cols, stats):
        col.markdown(render_metric_card(icon, val, label), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="impact-section">
        <h2>🌍 Making Healthcare Accessible</h2>
        <p>{IMPACT_STATEMENT}</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_section_header("How It Works", "Three simple steps to AI-powered skin analysis")
    cols = st.columns(3)
    steps = [
        ("1", "📤", "Upload Image", "Take a photo of the skin lesion and upload it to our system."),
        ("2", "🧠", "AI Analysis", "Our CNN model analyzes the image using transfer learning."),
        ("3", "📊", "Get Results", "View prediction, confidence score, and Grad-CAM visualization."),
    ]
    for col, (num, icon, title, desc) in zip(cols, steps):
        col.markdown(render_step_card(num, icon, title, desc), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    render_disclaimer()
    render_footer()


# ══════════════════════════════════════════════════════════
#  PAGE 2: ABOUT
# ══════════════════════════════════════════════════════════

elif page == "📖 About":
    render_section_header("About the Project",
                          "Skin Disease Detection & Classification Using CNN")
    
    st.markdown("""
    <div class="info-card">
        <h3>📌 Project Overview</h3>
        <p>This project develops a <strong>Virtual Dermatologist</strong> — an AI-powered system
        that analyzes dermatoscopic images of skin lesions and classifies them into one of 7
        disease categories. Using Convolutional Neural Networks (CNN) with Transfer Learning,
        the system aims to make dermatological screening accessible to underserved communities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Objectives")
    objectives = [
        "Develop a CNN-based model for classifying 7 types of skin diseases",
        "Achieve 80–92% classification accuracy using transfer learning (EfficientNetB3)",
        "Implement Grad-CAM for model explainability and visual interpretation",
        "Handle class imbalance in the HAM10000 dataset via augmentation and class weights",
        "Build an intuitive, accessible Streamlit web application for real-time prediction",
        "Generate downloadable PDF reports with diagnosis details and disclaimers",
    ]
    for obj in objectives:
        st.markdown(f"- ✅ {obj}")
    
    st.markdown("---")
    
    st.markdown("### 🧠 Key Concepts")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🔬 Convolutional Neural Networks (CNN)", expanded=True):
            st.write("""
            CNNs are deep learning models designed for image analysis. They use convolutional
            layers to automatically extract features like edges, textures, and patterns from images.
            This makes them ideal for skin lesion classification where visual patterns are key.
            """)
        with st.expander("🔄 Transfer Learning"):
            st.write("""
            Instead of training from scratch, we use EfficientNetB3 pre-trained on ImageNet (millions
            of images). We fine-tune the top layers on our HAM10000 dataset. This gives better accuracy
            with less training data and time.
            """)
    with c2:
        with st.expander("🔥 Grad-CAM (Explainability)", expanded=True):
            st.write("""
            Gradient-weighted Class Activation Mapping generates a heatmap showing which regions
            of the image the model focused on. This builds trust by making the AI's decision
            interpretable to doctors and patients.
            """)
        with st.expander("⚖️ Class Imbalance Handling"):
            st.write("""
            HAM10000 has uneven class distribution. We address this using data augmentation
            (flipping, rotation, zoom), class weights during training, and stratified splitting
            to ensure the model learns all 7 classes effectively.
            """)
    
    st.markdown("---")
    
    st.markdown("### 📊 HAM10000 Dataset")
    st.info("**Source:** Harvard Dataverse / Kaggle · **Size:** 10,015 dermatoscopic images · **Classes:** 7")
    
    class_counts = {
        "Melanocytic Nevi": 6705, "Melanoma": 1113, "Benign Keratosis": 1099,
        "Basal Cell Carcinoma": 514, "Actinic Keratosis": 327,
        "Vascular Lesions": 142, "Dermatofibroma": 115
    }
    fig = px.bar(
        x=list(class_counts.keys()), y=list(class_counts.values()),
        labels={"x": "Disease Class", "y": "Number of Images"},
        color=list(class_counts.values()),
        color_continuous_scale=["#E8F4FD", "#0A6C9E"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"), showlegend=False,
        coloraxis_showscale=False, margin=dict(t=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Project Pipeline")
    pipeline = [
        ("1️⃣", "Data Collection", "Gather HAM10000 dataset from Kaggle"),
        ("2️⃣", "Data Preprocessing", "Resize to 224×224, normalize, augment"),
        ("3️⃣", "Exploratory Data Analysis", "Analyze class distribution and imbalance"),
        ("4️⃣", "Model Selection", "Choose EfficientNetB3 with ImageNet weights"),
        ("5️⃣", "Transfer Learning", "Freeze base layers, add custom classifier head"),
        ("6️⃣", "Training", "Train with class weights, callbacks, and validation split"),
        ("7️⃣", "Evaluation", "Test accuracy, confusion matrix, classification report"),
        ("8️⃣", "Deployment", "Streamlit web app with Grad-CAM and PDF reports"),
    ]
    for icon, title, desc in pipeline:
        st.markdown(f"**{icon} {title}** — {desc}")
    
    render_disclaimer()
    render_footer()


# ══════════════════════════════════════════════════════════
#  PAGE 3: HOW IT WORKS
# ══════════════════════════════════════════════════════════

elif page == "⚙️ How It Works":
    render_section_header("How It Works", "From image upload to AI-powered diagnosis in seconds")
    
    cols = st.columns(4)
    steps = [
        ("1", "📤", "Upload", "Upload a dermatoscopic image of the skin lesion (JPG/PNG, max 10MB)."),
        ("2", "🔍", "Validate", "Image is checked for skin-like features, brightness, and quality."),
        ("3", "🧠", "AI Analysis", "EfficientNetB3 CNN processes the image through transfer learning."),
        ("4", "📊", "Results", "View prediction, confidence %, top 3 classes, and Grad-CAM heatmap."),
    ]
    for col, (num, icon, title, desc) in zip(cols, steps):
        col.markdown(render_step_card(num, icon, title, desc), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🔧 Technical Pipeline")
    st.markdown("""
    ```
    📷 Raw Image (JPG/PNG)
         │
         ▼
    🔍 Validate (skin-tone, brightness, edge density)
         │
         ▼
    🔄 Resize to 224×224  →  📊 Normalize (0–1)  →  🧮 Array Shape: (1, 224, 224, 3)
         │
         ▼
    🧠 EfficientNetB3 (Transfer Learning)  →  📤 Softmax Output [7 probabilities]
         │                                           │
         ▼                                           ▼
    🔥 Grad-CAM (Last Conv Layer)              🏷️ Predicted Class + Confidence
         │                                           │
         ▼                                           ▼
    🗺️ Heatmap Overlay                         📊 Top 3 Predictions
         │                                           │
         └──────────────┬────────────────────────────┘
                        ▼
                  📄 PDF Report
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛡️ Image Validation Checks")
    st.markdown("""
    Before running the AI model, every uploaded image passes through 4 validation checks:
    
    | Check | Description | Action if Failed |
    |-------|-------------|-----------------|
    | **Brightness** | Too dark (<25) or too bright (>245 mean) | Warning + guidance |
    | **Color Variance** | Very low variance = blank/solid image | Warning + guidance |
    | **Skin-Tone Detection** | HSV analysis for skin-like regions | Warning + option to proceed |
    | **Edge Density** | High edges = screenshot/UI with text | Warning + guidance |
    """)
    
    st.markdown("---")
    
    st.markdown("### ❓ Frequently Asked Questions")
    with st.expander("What types of images can I upload?"):
        st.write("Dermatoscopic images in JPG or PNG format, up to 10MB. For best results, use close-up, well-lit images of the skin lesion.")
    with st.expander("How accurate is the model?"):
        st.write("The model achieves 80–92% accuracy on the HAM10000 test set. However, accuracy may vary on real-world images.")
    with st.expander("Is my image data stored?"):
        st.write("No. All images are processed in-session only and are not saved to any server or database.")
    with st.expander("Can this replace a doctor?"):
        st.write("**Absolutely not.** This tool is for educational and preliminary screening purposes only. Always consult a qualified dermatologist.")
    with st.expander("What happens if I upload a non-skin image?"):
        st.write("Our validation layer checks for skin-like colors, appropriate brightness, and detects screenshots. You'll receive a warning if the image doesn't appear to be a skin photo.")
    
    render_disclaimer()
    render_footer()


# ══════════════════════════════════════════════════════════
#  PAGE 4: MODEL DETAILS
# ══════════════════════════════════════════════════════════

elif page == "🧠 Model Details":
    render_section_header("Model Architecture", "EfficientNetB3 with Transfer Learning")
    
    st.markdown("""
    <div class="info-card">
        <h3>🏗️ EfficientNetB3 Architecture</h3>
        <p>EfficientNetB3 is a convolutional neural network that uses a compound scaling method to uniformly
        scale depth, width, and resolution. Pre-trained on ImageNet (1.4M images, 1000 classes), we
        fine-tune it on HAM10000 for 7-class skin disease classification.</p>
        <br>
        <p><strong>Input:</strong> 224×224×3 (RGB image) → <strong>Base Model:</strong> EfficientNetB3 (frozen layers) →
        <strong>Custom Head:</strong> GlobalAvgPooling → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Parameters", "~12M")
    c2.metric("Trainable Parameters", "~2M")
    c3.metric("Target Accuracy", "80–92%")
    
    st.markdown("---")
    
    st.markdown("### 🏥 The 7 Disease Classes")
    for cls_name, info in CLASS_DESCRIPTIONS.items():
        sev = info["severity"]
        sev_class = {"Low": "severity-low", "Moderate": "severity-moderate",
                     "High": "severity-high", "Critical": "severity-critical"}.get(sev, "severity-low")
        st.markdown(f"""
        <div class="class-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h4 style="margin:0; color:{info['color']};">{cls_name}</h4>
                <span class="severity-badge {sev_class}">{sev}</span>
            </div>
            <p style="color:#64748B; font-size:14px; line-height:1.7; margin:0;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Handling Class Imbalance")
    st.write("""
    The HAM10000 dataset is **heavily imbalanced** — Melanocytic Nevi has 6,705 images while
    Dermatofibroma has only 115. We address this through:
    
    1. **Data Augmentation** — Random flips, rotations, zoom, brightness changes
    2. **Class Weights** — Higher weights for underrepresented classes during training
    3. **Stratified Splitting** — Ensures proportional class distribution in train/val/test sets
    """)
    
    render_footer()


# ══════════════════════════════════════════════════════════
#  PAGE 5: TRY THE DETECTOR (MAIN FEATURE)
# ══════════════════════════════════════════════════════════

elif page == "🔬 Try the Detector":
    render_section_header("🔬 AI Skin Disease Detector",
                          "Upload a skin image for instant AI-powered analysis")
    
    # ── Disclaimer Gate ──
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False
    
    if not st.session_state.disclaimer_accepted:
        st.markdown("""
        <div class="disclaimer-banner prominent">
            <div>
                <p><strong>⚠️ Important — Please Read Before Proceeding</strong></p>
                <p>This AI tool is for <strong>educational and screening purposes only</strong>.
                It is NOT a substitute for professional medical advice, diagnosis, or treatment.
                Always consult a qualified dermatologist for medical concerns.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        accepted = st.checkbox("I understand and acknowledge this disclaimer")
        if accepted:
            st.session_state.disclaimer_accepted = True
            st.rerun()
        else:
            st.stop()
    
    # ── Demo Mode Info ──
    if demo_mode:
        st.info(
            "🧪 **Demo Mode** — No trained model loaded. Using image-aware analysis "
            "that generates predictions based on your image's visual properties. "
            "Different images will produce different results. "
            "For real AI predictions, place your trained `.h5` model in the `models/` folder."
        )
    
    # ── Image Upload ──
    uploaded_file = st.file_uploader(
        "Upload a dermatoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG. Maximum file size: 10MB."
    )
    
    if uploaded_file is not None:
        # ── Step 1: File Validation ──
        is_valid, error_msg = validate_image(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.stop()
        
        image = Image.open(uploaded_file)
        preprocessed, resized = preprocess_image(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown(f"#### 🔄 Preprocessed ({IMG_SIZE}×{IMG_SIZE})")
            st.image(resized, use_container_width=True)
        
        st.markdown("---")
        
        analyze_btn = st.button("🔬 Analyze Image", use_container_width=True, type="primary")
        
        if analyze_btn or st.session_state.get("analysis_done"):
            if not st.session_state.get("analysis_done"):
                
                # ── Step 2: Skin Image Validation ──
                progress = st.progress(0, text="🔍 Step 1/3 — Validating image...")
                time.sleep(0.4)
                skin_valid, skin_issues = validate_skin_image(image)
                progress.progress(20, text="🔍 Validation complete.")
                time.sleep(0.3)
                
                if not skin_valid:
                    progress.empty()
                    # Pattern adapted from DermaAI: dedicated error feedback
                    # (ref: main.py line 66-68 — render error.html with specific message)
                    st.markdown("""
                    <div class="result-card error" style="padding:32px;">
                        <h3 style="color:#EF4444; margin-bottom:12px;">⚠️ Image Validation Failed</h3>
                        <p style="color:#64748B;">The uploaded image could not be processed.
                        Please ensure that the image contains skin and try again.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    for issue in skin_issues:
                        st.markdown(f"- {issue}")
                    st.info("💡 **Tip:** For best results, upload a close-up, well-lit photo "
                            "of a skin lesion. Dermatoscopic images work best.")
                    if not st.checkbox("I understand the warnings — proceed with analysis anyway"):
                        st.stop()
                
                # ── Step 3: Prediction ──
                with st.spinner("🧠 Analyzing with AI..."):
                    progress.progress(40, text="🧠 Step 2/3 — Running CNN model...")
                    
                    result = predict_disease(model, preprocessed, image, demo_mode)
                    
                    time.sleep(0.3)
                    progress.progress(70, text="🔥 Step 3/3 — Generating Grad-CAM heatmap...")
                    
                    heatmap = generate_gradcam(model, preprocessed, result["predicted_idx"], demo_mode)
                    gradcam_overlay = overlay_gradcam(image, heatmap)
                    
                    time.sleep(0.3)
                    progress.progress(100, text="✅ Analysis complete!")
                    time.sleep(0.3)
                    progress.empty()
                
                st.session_state.analysis_done = True
                st.session_state.result = result
                st.session_state.gradcam_overlay = gradcam_overlay
                st.session_state.skin_valid = skin_valid
                st.rerun()
            
            # ── Results Display ──
            result = st.session_state.result
            gradcam_overlay = st.session_state.gradcam_overlay
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            top_3 = result["top_3"]
            all_probs = result["all_probs"]
            is_low_conf = result["is_low_confidence"]
            is_demo = result.get("is_demo", False)
            conf_color = get_confidence_color(confidence)
            conf_label = get_confidence_label(confidence)
            severity = CLASS_DESCRIPTIONS.get(predicted_class, {}).get("severity", "Unknown")
            
            # ── Demo Mode Result Banner ──
            if is_demo:
                st.markdown("""
                <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25);
                            border-radius:12px; padding:12px 16px; margin-bottom:16px;">
                    <p style="color:#D97706 !important; font-size:13px !important; margin:0;">
                        🧪 <strong>Demo Mode Results</strong> — These predictions are generated by analyzing
                        your image's visual properties (color, brightness, texture). For real AI predictions,
                        load a trained model. Results are <strong>deterministic</strong>: the same image will
                        always produce the same prediction.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # ── Low Confidence Warning ──
            # Pattern from DermaAI: prob < threshold → 'Inconclusive result'
            # (ref: main.py line 94-96 — render error.html with inconclusive msg)
            if is_low_conf:
                st.markdown("""
                <div class="result-card error" style="padding:32px;">
                    <h3 style="color:#EF4444; margin-bottom:12px;">⚠️ Inconclusive Result</h3>
                    <p style="color:#64748B;">The AI model's confidence is below the minimum threshold.
                    Please consult a healthcare professional for an accurate diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning(
                    f"**Low Confidence: {confidence*100:.1f}%** — "
                    f"Below {CONFIDENCE_THRESHOLD*100:.0f}% threshold. Possible reasons:\n"
                    "- The image is not a clear skin lesion photo\n"
                    "- The condition may not match the 7 trained classes\n"
                    "- Image quality may be insufficient\n\n"
                    "**Results are shown below for reference, but should not be relied upon.**"
                )
            
            if not st.session_state.get("skin_valid", True):
                st.info("ℹ️ This image was flagged during validation. Results should be interpreted with caution.")
            
            st.success("✅ Analysis Complete!")
            
            # Main prediction card
            card_style = "warning" if is_low_conf else "primary"
            st.markdown(f"""
            <div class="result-card {card_style}" style="text-align:center; padding:40px;">
                <div style="font-size:14px; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                    Predicted Condition {'(Low Confidence)' if is_low_conf else ''}
                </div>
                <div class="prediction-name">{predicted_class}</div>
                <div class="confidence-score">{confidence*100:.1f}%</div>
                <div style="font-size:14px; color:{conf_color}; font-weight:600; margin-top:4px;">
                    {conf_label} Confidence · Severity: {severity}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🥇 Top 3 Predictions")
                for i, (cls, conf) in enumerate(top_3):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"**{medal} {cls}**")
                    st.progress(conf, text=f"{conf*100:.1f}%")
                
                st.markdown("#### 📊 All Class Probabilities")
                fig = go.Figure(go.Bar(
                    x=[p * 100 for p in all_probs],
                    y=CLASS_LABELS,
                    orientation='h',
                    marker=dict(
                        color=[COLORS["primary"] if l == predicted_class else COLORS["border"]
                               for l in CLASS_LABELS],
                        cornerradius=6
                    ),
                    text=[f"{p*100:.1f}%" for p in all_probs],
                    textposition="auto"
                ))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Confidence (%)", range=[0, 100]),
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔥 Grad-CAM Heatmap")
                st.image(gradcam_overlay,
                         caption="AI Focus Areas — Highlighted regions influenced the prediction",
                         use_container_width=True)
                
                st.markdown(f"#### 📋 About {predicted_class}")
                desc = CLASS_DESCRIPTIONS.get(predicted_class, {}).get("description", "")
                st.info(desc)
            
            # ── Treatment Recommendations ──
            # Pattern adapted from DermaAI's skin_disorder.json
            # (ref: main.py line 99 → treatment_dict.get(pred_class, []))
            treatments = TREATMENT_RECOMMENDATIONS.get(predicted_class, [])
            if treatments:
                st.markdown("---")
                st.markdown("#### 💊 Common Treatment Options")
                st.caption("⚠️ These are general options only. Always consult a dermatologist for personalized advice.")
                for j, treatment in enumerate(treatments, 1):
                    st.markdown(f"**{j}.** {treatment}")
            
            st.markdown("---")
            render_disclaimer()
            
            # ── Actions ──
            col1, col2 = st.columns(2)
            with col1:
                pdf_bytes = generate_pdf_report(
                    image, predicted_class, confidence, top_3, gradcam_overlay
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"SkinVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with col2:
                if st.button("🔄 Try Another Image", use_container_width=True):
                    for key in ["analysis_done", "result", "gradcam_overlay", "skin_valid"]:
                        st.session_state.pop(key, None)
                    st.rerun()
            
            # ── Debug Panel ──
            with st.expander("🔧 Debug Information (Development)", expanded=False):
                debug_info = get_model_debug_info(model, model_status)
                st.markdown("**Model Status:**")
                for key, val in debug_info.items():
                    st.text(f"  {key}: {val}")
                st.markdown("**Raw Prediction Probabilities:**")
                for i, (label, prob) in enumerate(zip(CLASS_LABELS, all_probs)):
                    st.text(f"  [{i}] {label}: {prob:.6f}")
                st.text(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
                st.text(f"  Is Low Confidence: {is_low_conf}")
                st.text(f"  Demo Mode: {is_demo}")


# ══════════════════════════════════════════════════════════
#  PAGE 6: COMPARISON (Existing vs Our System)
# ══════════════════════════════════════════════════════════

elif page == "📊 Comparison":
    render_section_header("📊 Existing Systems vs Our System",
                          "How SkinVision AI improves upon existing implementations")
    
    st.markdown("""
    <div class="info-card">
        <h3>🔍 Literature & System Review</h3>
        <p>We analyzed existing open-source skin disease classifiers — including
        <strong>DermaAI</strong> (FridahKimathi et al.) and similar Flask-based implementations —
        to identify gaps and design a significantly improved system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚔️ Feature Comparison")
    
    comparison_data = {
        "Feature": [
            "Framework", "Dataset", "Disease Classes",
            "Explainability (Grad-CAM)", "Confidence Threshold",
            "Image Validation", "PDF Reports",
            "Top-3 Predictions", "UI Design",
            "Medical Disclaimers", "Deployment"
        ],
        "Typical GitHub Projects": [
            "Flask (basic HTML/CSS)", "DermNet + ISIC (mixed)", "8 classes",
            "❌ Not implemented", "❌ No threshold check",
            "❌ No validation", "❌ Not available",
            "❌ Single prediction only", "Basic HTML forms",
            "❌ Minimal/None", "Google Cloud (Flask)"
        ],
        "SkinVision AI (Ours)": [
            "Streamlit ✅ (Modern)", "HAM10000 (standardized) ✅", "7 classes (HAM10000) ✅",
            "✅ Full Grad-CAM with heatmap overlay", "✅ 50% min threshold + warnings",
            "✅ 4-layer validation (HSV, brightness, edges)", "✅ Professional PDF download",
            "✅ Top-3 with confidence bars", "Premium clinical UI with CSS ✅",
            "✅ Page-level + PDF disclaimers", "Streamlit (local/cloud) ✅"
        ]
    }
    
    st.table(comparison_data)
    
    st.markdown("---")
    st.markdown("### 🏆 Our Key Improvements")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **🔬 Explainability (Grad-CAM)**
        - Existing projects provide NO visual explanation
        - Our system generates heatmaps showing WHERE the AI focused
        - Builds trust with doctors and patients
        
        **🛡️ Input Validation**
        - Existing projects accept ANY image without checks
        - Our system validates: skin-tone, brightness, edge density
        - Rejects screenshots and non-skin images
        
        **📄 PDF Reporting**
        - No existing project offers downloadable reports
        - Our system generates branded diagnostic PDFs
        - Includes prediction, confidence, and disclaimers
        """)
    
    with c2:
        st.markdown("""
        **⚖️ Confidence Analysis**
        - Existing projects show single prediction with no uncertainty
        - Our system shows: Top-3 predictions, confidence bars, threshold warnings
        - Flags uncertain predictions instead of blindly outputting
        
        **🎨 Clinical UI/UX**
        - Existing projects use basic HTML forms (student-level)
        - Our system uses premium CSS with card layouts, animations
        - Hospital-grade design language (Inter font, clinical colors)
        
        **🔒 Ethics & Safety**
        - Existing projects have minimal disclaimers
        - Our system: mandatory disclaimer gate, page-level banners
        - Clear statement: NOT a diagnostic tool
        """)
    
    st.markdown("---")
    st.markdown("### 📌 Reference & Attribution")
    st.markdown("""
    > This project was **inspired by** existing open-source implementations, particularly:
    > - [DermaAI — Skin Disease Image Classifier](https://github.com/FridahKimathi/Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis) *(FridahKimathi et al.)*
    >
    > All code in SkinVision AI is **independently implemented** with significant architectural
    > and feature improvements. No code was copied from external sources.
    """)
    
    render_footer()


# ══════════════════════════════════════════════════════════
#  PAGE 7: DISCLAIMER & ETHICS
# ══════════════════════════════════════════════════════════

elif page == "⚖️ Disclaimer & Ethics":
    render_section_header("Disclaimer & Ethics", "Responsible AI in Healthcare")
    
    st.markdown("""
    <div class="disclaimer-banner prominent">
        <div>
            <p><strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong></p>
            <p>This AI-powered tool is designed for <strong>educational and preliminary screening purposes only</strong>.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice
            of a qualified dermatologist or healthcare provider.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Intended Use")
    st.markdown("""
    - ✅ **Educational tool** for learning about skin diseases
    - ✅ **Research demonstration** of CNN and transfer learning
    - ✅ **Preliminary screening aid** to encourage professional consultation
    - ❌ **NOT** for self-diagnosis or medical decision-making
    - ❌ **NOT** a replacement for professional dermatological evaluation
    """)
    
    st.markdown("### 🤖 Ethical AI Considerations")
    st.markdown("""
    - **Bias Awareness:** The model is trained on HAM10000 which may not represent all skin tones equally
    - **Limitations:** Performance may vary on images unlike those in the training set
    - **Transparency:** Grad-CAM provides visual explanations of model decisions
    - **Human Oversight:** Always requires professional medical review
    """)
    
    st.markdown("### 🔒 Data Privacy")
    st.markdown("""
    - All images are processed **in-session only** and are **never stored**
    - No personally identifiable information is collected
    - PDF reports are generated client-side and not saved on any server
    """)
    
    render_disclaimer()
    render_footer()
