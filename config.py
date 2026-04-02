"""
============================================================
  CONFIGURATION — Skin Disease Detection & Classification
============================================================
Central configuration file containing all constants,
class labels, descriptions, color tokens, and paths.
============================================================
"""

import os

# ──────────────────────────────────────────────────────────
# 📁 PATHS
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_disease_model.h5")  # ← REPLACE WITH YOUR MODEL PATH
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLES_DIR = os.path.join(BASE_DIR, "styles")

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "class_images"), exist_ok=True)
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────────────────
# 🏷️ CLASS LABELS (HAM10000 — 7 Classes)
# ──────────────────────────────────────────────────────────
CLASS_LABELS = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
]

# ──────────────────────────────────────────────────────────
# 📝 CLASS DESCRIPTIONS (for Model Details page)
# ──────────────────────────────────────────────────────────
CLASS_DESCRIPTIONS = {
    "Actinic Keratosis": {
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. "
                        "It is considered a pre-cancerous condition that can develop into squamous cell carcinoma.",
        "severity": "Moderate",
        "color": "#F59E0B",  # Warning amber
    },
    "Basal Cell Carcinoma": {
        "description": "The most common type of skin cancer. It appears as a slightly transparent bump on the skin, "
                        "often on sun-exposed areas. It rarely spreads but can cause disfigurement.",
        "severity": "High",
        "color": "#EF4444",  # Error red
    },
    "Benign Keratosis": {
        "description": "A non-cancerous skin growth that includes seborrheic keratoses, solar lentigines, "
                        "and lichen-planus like keratoses. They appear as waxy, brown, or black raised spots.",
        "severity": "Low",
        "color": "#10B981",  # Success green
    },
    "Dermatofibroma": {
        "description": "A common benign fibrous nodule most frequently found on the legs. "
                        "It feels like a hard lump under the skin and is usually harmless.",
        "severity": "Low",
        "color": "#10B981",
    },
    "Melanoma": {
        "description": "The most dangerous form of skin cancer. It develops from melanocytes and can spread rapidly. "
                        "Early detection is crucial for survival. Look for asymmetric moles with irregular borders.",
        "severity": "Critical",
        "color": "#DC2626",  # Deep red
    },
    "Melanocytic Nevi": {
        "description": "Commonly known as moles. These are benign growths of melanocytes that appear as small, dark spots. "
                        "Most are harmless, but changes in shape or color should be monitored.",
        "severity": "Low",
        "color": "#10B981",
    },
    "Vascular Lesions": {
        "description": "Abnormalities of blood vessels in the skin including angiomas, angiokeratomas, "
                        "pyogenic granulomas, and hemorrhages. Most are benign but may require monitoring.",
        "severity": "Low",
        "color": "#10B981",
    },
}

# ──────────────────────────────────────────────────────────
# 🎨 DESIGN TOKENS
# ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#0A6C9E",
    "primary_dark": "#085580",
    "primary_light": "#E8F4FD",
    "accent": "#00C4B4",
    "accent_light": "#E0FBF8",
    "bg_main": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "bg_dark": "#0F172A",
    "text_primary": "#1F2937",
    "text_secondary": "#64748B",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "border": "#E2E8F0",
}

# ──────────────────────────────────────────────────────────
# 📐 MODEL SETTINGS
# ──────────────────────────────────────────────────────────
IMG_SIZE = 224  # EfficientNetB3 input size
BATCH_SIZE = 32
MODEL_INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ──────────────────────────────────────────────────────────
# 📊 PROJECT STATS (for Home page)
# ──────────────────────────────────────────────────────────
PROJECT_STATS = {
    "classes": 7,
    "images": "10,000+",
    "accuracy": "80–92%",
    "technique": "Transfer Learning",
}

# ──────────────────────────────────────────────────────────
# ⚠️ DISCLAIMER TEXT
# ──────────────────────────────────────────────────────────
DISCLAIMER_TEXT = (
    "⚠️ This is not a substitute for professional medical advice. "
    "Always consult a qualified dermatologist. This tool is designed "
    "for educational and screening purposes only."
)

IMPACT_STATEMENT = (
    "Millions of people suffer from skin diseases but lack access to dermatologists, "
    "especially in rural areas. A smartphone app powered by AI can help detect skin "
    "conditions instantly — making healthcare accessible to everyone."
)

# ──────────────────────────────────────────────────────────
# 📋 APP METADATA
# ──────────────────────────────────────────────────────────
APP_TITLE = "SkinVision AI"
APP_SUBTITLE = "Virtual Dermatologist"
APP_VERSION = "1.0.0"

# ──────────────────────────────────────────────────────────
# 💊 TREATMENT RECOMMENDATIONS (per class)
# Inspired by DermaAI's skin_disorder.json treatment approach
# Adapted for our HAM10000 7-class label set
# ──────────────────────────────────────────────────────────
TREATMENT_RECOMMENDATIONS = {
    "Actinic Keratosis": [
        "Cryotherapy — using liquid nitrogen to freeze off the lesion.",
        "Topical medications (such as imiquimod or 5-fluorouracil) — stimulate the immune system or destroy abnormal cells.",
        "Chemical peels — applying a chemical solution to remove the damaged skin layer.",
        "Photodynamic therapy — applying a photosensitizing agent activated by light to destroy the lesion.",
    ],
    "Basal Cell Carcinoma": [
        "Surgical excision — removal of the tumor and a margin of surrounding tissue.",
        "Mohs surgery — layer-by-layer removal to minimize damage to healthy tissue.",
        "Radiation therapy — using high-energy radiation to destroy cancer cells.",
        "Topical chemotherapy (such as imiquimod cream) — for superficial cases.",
    ],
    "Benign Keratosis": [
        "Cryotherapy — freezing with liquid nitrogen for cosmetic removal.",
        "Electrodesiccation and curettage — scraping off the growth and using electric current.",
        "Shave excision — shaving the lesion flush with the skin surface.",
        "Generally no treatment needed — benign and harmless in most cases.",
    ],
    "Dermatofibroma": [
        "No treatment usually necessary — dermatofibromas are benign and harmless.",
        "Surgical excision — if the lesion is symptomatic or bothersome.",
        "Cryotherapy — freezing for cosmetic reasons.",
        "Monitoring — regular observation for any changes in size or appearance.",
    ],
    "Melanoma": [
        "Surgical excision with wide margins — primary treatment for melanoma.",
        "Immunotherapy (checkpoint inhibitors like pembrolizumab) — helps the immune system fight cancer.",
        "Targeted therapy (BRAF/MEK inhibitors) — for melanomas with specific genetic mutations.",
        "Radiation therapy — for advanced or metastatic cases.",
        "CRITICAL: Seek immediate dermatological consultation for any suspected melanoma.",
    ],
    "Melanocytic Nevi": [
        "No treatment needed — most moles are benign and harmless.",
        "Regular monitoring — watch for changes in size, shape, color, or border (ABCDE rule).",
        "Surgical excision — if the mole shows suspicious changes.",
        "Dermoscopic follow-up — periodic dermoscopy for atypical nevi.",
    ],
    "Vascular Lesions": [
        "Laser therapy — pulsed dye laser for cherry angiomas and hemangiomas.",
        "Cryotherapy — freezing small vascular lesions.",
        "Electrocautery — burning off small angiomas.",
        "Monitoring — most vascular lesions are benign and require observation only.",
    ],
}
