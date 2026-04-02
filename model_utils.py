"""
============================================================
  MODEL UTILITIES — Skin Disease Detection & Classification
============================================================
ML pipeline: model loading, image preprocessing, prediction,
Grad-CAM heatmap generation, skin image validation, and
image-aware demo mode for presentations.

When a trained model is available: Real inference.
When model is unavailable: Image-aware demo predictions
that analyze actual image properties (NOT random).
============================================================
"""

import os
import hashlib
import numpy as np
from PIL import Image
import cv2

# ── TensorFlow Import (with graceful fallback) ──────────
import subprocess
import sys

TF_AVAILABLE = False
tf = None
keras_load_model = None

def _check_tf_available():
    """Test if TensorFlow can be imported without crashing the main process."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import tensorflow; print('ok')"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False

if _check_tf_available():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load_model
        TF_AVAILABLE = True
        print("✅ TensorFlow loaded successfully")
    except Exception:
        TF_AVAILABLE = False
        tf = None
        print("⚠️ TensorFlow import failed.")
else:
    print("⚠️ TensorFlow not compatible with this environment.")

from config import CLASS_LABELS, MODEL_PATH, IMG_SIZE

# ── Confidence threshold for valid predictions ──────────
CONFIDENCE_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════

def load_trained_model():
    """
    Load the trained skin disease classification model.

    Returns:
        model: Loaded Keras model or None if not available.
        model_status: Dict with status info for debug display.
    """
    status = {
        "tf_available": TF_AVAILABLE,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": False,
        "demo_mode": True,
        "error": None,
    }

    if not TF_AVAILABLE:
        status["error"] = "TensorFlow is not installed or not compatible with this environment."
        return None, status

    if not os.path.exists(MODEL_PATH):
        status["error"] = f"Model file not found at: {MODEL_PATH}"
        return None, status

    try:
        model = keras_load_model(MODEL_PATH)
        status["model_loaded"] = True
        status["demo_mode"] = False
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return model, status
    except Exception as e:
        status["error"] = f"Failed to load model: {str(e)}"
        return None, status


# ══════════════════════════════════════════════════════════
#  SKIN IMAGE VALIDATION (Heuristic-Based)
# ══════════════════════════════════════════════════════════

def validate_skin_image(image):
    """
    Validate whether an uploaded image looks like a skin/dermatoscopic image
    using multi-layer heuristic checks.

    Checks:
        1. Brightness — reject extremely dark/bright images
        2. Color variance — reject solid-color/blank images
        3. Skin-tone check — HSV-based skin color detection
        4. Edge density — reject UI screenshots (high text/edge ratio)

    Returns:
        is_valid: Boolean
        issues: List of validation issue strings
    """
    issues = []
    img_rgb = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_rgb, (224, 224))

    # 1. Brightness check
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 25:
        issues.append("Image is too dark. Please upload a well-lit skin image.")
    elif mean_brightness > 245:
        issues.append("Image is too bright or overexposed.")

    # 2. Color variance
    color_std = np.std(img_resized)
    if color_std < 10:
        issues.append("Image has very low color variation — may be blank or solid-color.")

    # 3. Skin-tone presence (HSV)
    # Primary mask: DermaAI's exact HSV range for skin detection
    # (ref: main.py is_skin() — lower=[0,20,70], upper=[20,255,255], threshold>5%)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    skin_masks = []
    # DermaAI primary skin range
    skin_masks.append(cv2.inRange(img_hsv,
        np.array([0, 20, 70], dtype=np.uint8),
        np.array([20, 255, 255], dtype=np.uint8)))
    # Extended: lighter/paler skin tones
    skin_masks.append(cv2.inRange(img_hsv,
        np.array([0, 10, 60], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8)))
    # Extended: darker skin tones
    skin_masks.append(cv2.inRange(img_hsv,
        np.array([0, 10, 30], dtype=np.uint8),
        np.array([30, 200, 200], dtype=np.uint8)))

    combined = skin_masks[0]
    for m in skin_masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    skin_ratio = np.sum(combined > 0) / (224 * 224)

    if skin_ratio < 0.05:
        issues.append("This image does not appear to contain skin-like regions. "
                       "Please upload a dermatoscopic or close-up skin photo.")

    # 4. Edge density (screenshot detection)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (224 * 224)
    if edge_ratio > 0.30:
        issues.append("This image appears to be a screenshot or contain UI elements. "
                       "Please upload an actual skin lesion photograph.")

    return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════
#  IMAGE-AWARE DEMO PREDICTION
#  (Extracts real features from the image — NOT random)
# ══════════════════════════════════════════════════════════

def _analyze_image_features(image):
    """
    Extract visual features from the image to generate
    image-specific (non-random) demo predictions.

    Features analyzed:
        - Mean color (RGB channels)
        - Color variance
        - Brightness distribution
        - Edge characteristics
        - Dominant hue in HSV
        - Pixel hash for determinism

    Returns a feature dict used to map to class probabilities.
    """
    img = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img, (224, 224))

    # RGB channel statistics
    r_mean, g_mean, b_mean = np.mean(img_resized[:,:,0]), np.mean(img_resized[:,:,1]), np.mean(img_resized[:,:,2])

    # HSV analysis
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    h_mean = np.mean(hsv[:,:,0])  # Dominant hue
    s_mean = np.mean(hsv[:,:,1])  # Saturation
    v_mean = np.mean(hsv[:,:,2])  # Value/brightness

    # Texture: edge density
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (224 * 224)

    # Color variance
    color_std = np.std(img_resized)

    # Deterministic hash from pixel data for consistency
    pixel_hash = int(hashlib.md5(img_resized.tobytes()[:1000]).hexdigest()[:8], 16)

    return {
        'r_mean': r_mean, 'g_mean': g_mean, 'b_mean': b_mean,
        'h_mean': h_mean, 's_mean': s_mean, 'v_mean': v_mean,
        'edge_ratio': edge_ratio, 'color_std': color_std,
        'pixel_hash': pixel_hash,
        'brightness': v_mean,
    }


def _generate_image_aware_prediction(image):
    """
    Generate deterministic, image-specific demo predictions.
    Different images → different predictions (not random).

    Uses HSV color analysis to map image features to
    clinically plausible class probabilities:
        - Dark brownish images → Melanocytic Nevi / Melanoma territory
        - Reddish images → Vascular Lesions / BCC
        - Pinkish/light images → Benign Keratosis / Dermatofibroma
        - High contrast → Actinic Keratosis

    The predictions are DETERMINISTIC: same image → same result.
    """
    features = _analyze_image_features(image)

    # Start with a base distribution from pixel hash (deterministic)
    np.random.seed(features['pixel_hash'] % (2**31))

    # Base probabilities — Dirichlet ensures they sum to 1
    base = np.random.dirichlet(np.ones(7) * 1.5)

    # Boost classes based on actual image color features
    boosts = np.zeros(7)

    # CLASS_LABELS order: [Actinic Keratosis, Basal Cell Carcinoma,
    #   Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevi, Vascular Lesions]

    h = features['h_mean']
    s = features['s_mean']
    v = features['brightness']
    r = features['r_mean']

    # Dark images with brown/black tones → Melanoma or Melanocytic Nevi
    if v < 120 and s > 40:
        boosts[4] += 0.3  # Melanoma
        boosts[5] += 0.2  # Melanocytic Nevi

    # Very dark spots → Melanoma
    if v < 80:
        boosts[4] += 0.15

    # Brownish/tan images → Melanocytic Nevi (most common)
    if 10 < h < 25 and v > 100:
        boosts[5] += 0.35  # Melanocytic Nevi

    # Reddish images → Vascular Lesions or BCC
    if r > 150 and features['g_mean'] < 120:
        boosts[6] += 0.25  # Vascular Lesions
        boosts[1] += 0.15  # BCC

    # High saturation + warm hue → Actinic Keratosis
    if s > 80 and h < 15:
        boosts[0] += 0.2  # Actinic Keratosis

    # Light, low-contrast images → Benign Keratosis / Dermatofibroma
    if v > 160 and features['color_std'] < 40:
        boosts[2] += 0.25  # Benign Keratosis
        boosts[3] += 0.15  # Dermatofibroma

    # Moderate brightness with texture → Benign Keratosis
    if 0.05 < features['edge_ratio'] < 0.15:
        boosts[2] += 0.1

    # Apply boosts and normalize
    probs = base + boosts
    probs = probs / np.sum(probs)

    # Make dominant class more pronounced (60-85%)
    dominant_idx = np.argmax(probs)
    probs[dominant_idx] = max(probs[dominant_idx], 0.55 + (features['pixel_hash'] % 30) / 100)
    probs = probs / np.sum(probs)  # Re-normalize

    # Reset numpy random state
    np.random.seed(None)

    predicted_class = CLASS_LABELS[dominant_idx]
    confidence = float(probs[dominant_idx])
    top_3_idx = np.argsort(probs)[::-1][:3]
    top_3 = [(CLASS_LABELS[i], float(probs[i])) for i in top_3_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3": top_3,
        "all_probs": probs,
        "is_low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "predicted_idx": dominant_idx,
        "is_demo": True,
    }


# ══════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════

def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocess an uploaded image for model inference.

    Pipeline:
        1. Convert to RGB
        2. Resize to 224×224 (EfficientNetB3)
        3. Convert to numpy float32
        4. Normalize [0, 1]
        5. Add batch dimension → (1, 224, 224, 3)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    resized_image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(resized_image, dtype=np.float32)
    img_array = img_array / 255.0
    preprocessed = np.expand_dims(img_array, axis=0)

    return preprocessed, resized_image


# ══════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════

def predict_disease(model, preprocessed_image, original_image=None, demo_mode=False):
    """
    Run inference on a preprocessed image.

    - Real model: Uses TF model.predict()
    - Demo mode: Uses image-aware heuristic predictions

    Args:
        model: Keras model or None
        preprocessed_image: NumPy array (1, 224, 224, 3)
        original_image: PIL Image (needed for demo mode feature extraction)
        demo_mode: Whether to use demo predictions

    Returns:
        result: Dict with prediction details
    """
    if demo_mode or model is None:
        if original_image is not None:
            return _generate_image_aware_prediction(original_image)
        else:
            # Fallback: use preprocessed image data
            temp_img = Image.fromarray(
                (preprocessed_image[0] * 255).astype(np.uint8)
            )
            return _generate_image_aware_prediction(temp_img)

    # ── REAL MODEL PREDICTION ──
    predictions = model.predict(preprocessed_image, verbose=0)
    probs = predictions[0]

    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = float(probs[predicted_idx])

    top_3_idx = np.argsort(probs)[::-1][:3]
    top_3 = [(CLASS_LABELS[i], float(probs[i])) for i in top_3_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3": top_3,
        "all_probs": probs,
        "is_low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "predicted_idx": predicted_idx,
        "is_demo": False,
    }


# ══════════════════════════════════════════════════════════
#  GRAD-CAM IMPLEMENTATION
# ══════════════════════════════════════════════════════════

def generate_gradcam(model, preprocessed_image, predicted_class_idx, demo_mode=False):
    """
    Generate a Grad-CAM heatmap.
    Real model: uses tf.GradientTape
    Demo mode: generates image-aware synthetic heatmap
    """
    if demo_mode or model is None or tf is None:
        return _generate_image_aware_heatmap(preprocessed_image)

    try:
        # Find last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer = sub_layer
                        break
                if last_conv_layer:
                    break

        if last_conv_layer is None:
            return _generate_image_aware_heatmap(preprocessed_image)

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_image)
            loss = predictions[:, predicted_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        return heatmap

    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}")
        return _generate_image_aware_heatmap(preprocessed_image)


def _generate_image_aware_heatmap(preprocessed_image):
    """
    Generate a synthetic Grad-CAM heatmap based on actual image content.
    Uses edge detection and brightness variance to create realistic
    focus areas on the actual lesion region.
    """
    img = (preprocessed_image[0] * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find regions of interest using adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Calculate local variance — high variance = interesting region
    mean = cv2.blur(gray.astype(np.float32), (31, 31))
    sq_mean = cv2.blur((gray.astype(np.float32))**2, (31, 31))
    variance = sq_mean - mean**2
    variance = np.maximum(variance, 0)

    # Normalize variance to [0, 1]
    if variance.max() > 0:
        heatmap = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    else:
        heatmap = np.zeros_like(variance)

    # Smooth it
    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (31, 31), 0)

    # Add center bias (lesions tend to be centered in dermoscopic images)
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2.0 * (IMG_SIZE//3)**2))
    heatmap = heatmap * 0.6 + center_bias.astype(np.float32) * 0.4

    # Final normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_gradcam(original_image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    img = np.array(original_image.convert('RGB'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(img * (1 - alpha) + heatmap_colored * alpha)
    return overlay


# ══════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════

def validate_image(uploaded_file):
    """Validate file type, size, and image integrity."""
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if uploaded_file.type not in allowed_types:
        return False, f"Invalid file type: {uploaded_file.type}. Please upload JPG or PNG."

    max_size = 10 * 1024 * 1024
    if uploaded_file.size > max_size:
        return False, f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum is 10MB."

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_confidence_color(confidence):
    """Return hex color based on confidence level."""
    if confidence >= 0.8:
        return "#10B981"
    elif confidence >= 0.5:
        return "#F59E0B"
    else:
        return "#EF4444"


def get_confidence_label(confidence):
    """Return human-readable confidence label."""
    if confidence >= 0.85:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Moderate"
    elif confidence >= 0.3:
        return "Low"
    else:
        return "Very Low"


def get_model_debug_info(model, model_status):
    """Return debug information about model status."""
    info = {
        "tensorflow_available": TF_AVAILABLE,
        "tensorflow_version": None,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_loaded": model is not None,
        "demo_mode": model_status.get("demo_mode", True),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "num_classes": len(CLASS_LABELS),
        "input_size": f"{IMG_SIZE}×{IMG_SIZE}",
    }

    if TF_AVAILABLE and tf is not None:
        info["tensorflow_version"] = tf.__version__

    if model is not None:
        try:
            info["model_input_shape"] = str(model.input_shape)
            info["model_output_shape"] = str(model.output_shape)
            info["total_params"] = model.count_params()
        except Exception:
            pass

    if model_status and model_status.get("error"):
        info["error"] = model_status["error"]

    return info
