<p align="center">
  <img src="assets/logo.png" alt="SkinVision AI" width="120" height="120" style="border-radius:20px;">
</p>

<h1 align="center">🔬 SkinVision AI</h1>
<h3 align="center">Virtual Dermatologist — AI-Powered Skin Disease Detection</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.14+-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-1.50+-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Dataset-HAM10000-green?style=for-the-badge" alt="HAM10000">
  <img src="https://img.shields.io/badge/Accuracy-80--92%25-brightgreen?style=for-the-badge" alt="Accuracy">
</p>

---

## 📌 Project Overview

**SkinVision AI** is an AI-powered skin disease detection and classification system that analyzes dermatoscopic images and classifies them into **7 disease categories** using Convolutional Neural Networks (CNN) with Transfer Learning. Built as a university final-year project, it demonstrates the potential of deep learning in making dermatological screening accessible to underserved communities.

### 🎯 What Makes It Special

| Feature | Description |
|---------|-------------|
| 🧠 **EfficientNetB3** | Transfer learning with ImageNet pre-trained weights |
| 🔥 **Grad-CAM** | Visual explainability — see WHERE the AI focused |
| 📊 **Top-3 Predictions** | Confidence scores for top 3 disease matches |
| 🛡️ **Image Validation** | 4-layer skin image validation (HSV, brightness, edges) |
| ⚖️ **Confidence Threshold** | Flags uncertain predictions below 50% confidence |
| 📄 **PDF Reports** | Downloadable diagnostic reports with disclaimers |
| 🎨 **Clinical UI** | Premium hospital-grade design (not a student template) |

---

## 🏥 The 7 Disease Classes

| # | Disease | Severity | HAM10000 Count |
|---|---------|----------|----------------|
| 1 | Actinic Keratosis | ⚠️ Moderate | 327 |
| 2 | Basal Cell Carcinoma | 🔴 High | 514 |
| 3 | Benign Keratosis | 🟢 Low | 1,099 |
| 4 | Dermatofibroma | 🟢 Low | 115 |
| 5 | **Melanoma** | 🔴 **Critical** | 1,113 |
| 6 | Melanocytic Nevi | 🟢 Low | 6,705 |
| 7 | Vascular Lesions | 🟢 Low | 142 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow 2.14+ / Keras |
| Model | EfficientNetB3 (Transfer Learning) |
| Web App | Streamlit 1.50+ |
| Image Processing | OpenCV, Pillow |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Visualization | Plotly |
| PDF Generation | FPDF2 |
| Dataset | HAM10000 (10,015 images, 7 classes) |

---

## 📂 Project Structure

```
project/
├── app.py                  # Main Streamlit application (7 pages)
├── model_utils.py          # ML pipeline: load, preprocess, predict, Grad-CAM
├── pdf_generator.py        # PDF diagnostic report generator
├── config.py               # Central configuration (labels, colors, paths)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── models/                 # ← Place your trained .h5 model here
│   └── skin_disease_model.h5
│
├── styles/
│   └── custom.css          # Premium clinical theme CSS
│
├── assets/
│   └── class_images/       # Sample images for each class
│
└── reports/                # Generated PDF reports (auto-created)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/skinvision-ai.git
cd skinvision-ai
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Place Your Model (Optional)
```bash
# Place your trained .h5 model at:
# models/skin_disease_model.h5
# Without it, the app runs in Demo Mode with image-aware predictions
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧪 Demo Mode

When no trained model is available, SkinVision AI runs in **Demo Mode**:

- Analyzes actual image properties (color, brightness, texture)
- Generates deterministic predictions (same image → same result)
- Different images produce different predictions
- Clearly labeled as "Demo Mode" throughout the UI

This allows full UI demonstration during viva/presentation without the trained model.

---

## 📊 Application Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Hero section, impact stats, CTA |
| 📖 About | Project overview, objectives, dataset, pipeline |
| ⚙️ How It Works | Technical pipeline, validation checks, FAQ |
| 🧠 Model Details | Architecture, 7 classes, class imbalance handling |
| 🔬 Try the Detector | Upload → Validate → Predict → Grad-CAM → PDF |
| 📊 Comparison | Existing systems vs our improvements |
| ⚖️ Disclaimer | Ethics, privacy, intended use |

---

## 📌 Inspiration & References

This project was inspired by existing open-source skin disease classification implementations, particularly:

- **[DermaAI — Skin Disease Image Classifier](https://github.com/FridahKimathi/Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis)** by FridahKimathi et al.

### How We Improved Upon Existing Systems

| Aspect | Existing Projects | SkinVision AI |
|--------|-------------------|---------------|
| Framework | Flask (basic) | Streamlit (modern, interactive) |
| Explainability | None | Grad-CAM heatmaps |
| Reports | None | PDF diagnostic reports |
| Validation | None | 4-layer image validation |
| Predictions | Single class | Top-3 with confidence |
| UI | Basic HTML | Premium clinical design |
| Ethics | Minimal | Full disclaimer system |

> ⚠️ **Note:** All code in SkinVision AI is **independently implemented**. No code was copied from external sources. The above projects served as inspiration for understanding the problem domain.

---

## 📚 References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling. *ICML 2019*.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations. *ICCV 2017*.
4. Esteva, A., et al. (2017). Dermatologist-level classification. *Nature*, 542.

---

## 📄 License

This project is developed for **educational purposes** as part of a university final-year project. It is not intended for commercial or clinical use.

---

<p align="center">
  Built with ❤️ using Python, TensorFlow, and Streamlit
</p>
