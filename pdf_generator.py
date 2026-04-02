"""
============================================================
  PDF REPORT GENERATOR — Skin Disease Detection
============================================================
Generates professional diagnostic reports in PDF format
using FPDF2 library. Includes:
  - Header with app branding
  - Patient image
  - Prediction results
  - Confidence scores
  - Top 3 predictions
  - Treatment recommendations
  - Grad-CAM heatmap
  - Medical disclaimer

IMPORTANT: All emoji/Unicode characters are stripped before
passing to FPDF — Helvetica only supports Latin-1.
============================================================
"""

import os
import re
import io
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("[WARNING] fpdf2 not installed. PDF generation disabled.")

from config import (
    APP_TITLE, DISCLAIMER_TEXT, CLASS_DESCRIPTIONS, TREATMENT_RECOMMENDATIONS,
    REPORTS_DIR, COLORS
)


# ══════════════════════════════════════════════════════════
#  EMOJI / UNICODE SANITIZER
#  (Helvetica cannot render emojis or special Unicode chars)
# ══════════════════════════════════════════════════════════

def _sanitize_for_pdf(text):
    """
    Remove all emoji and non-Latin-1 characters from text
    so FPDF's Helvetica font doesn't crash.

    Strips:
      - Emoji (Unicode blocks: emoticons, symbols, dingbats, etc.)
      - Variation selectors (U+FE00-FE0F)
      - Zero-width joiners (U+200D)
      - Any char outside Latin-1 (U+0000–U+00FF)
    """
    if not isinstance(text, str):
        return str(text)

    # Remove emoji and special Unicode characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols (⚠ lives here)
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # zero-width joiner
        "\U00002B50"             # star
        "]",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Remove any remaining non-Latin-1 characters
    text = text.encode('latin-1', errors='ignore').decode('latin-1')

    # Clean up extra whitespace from removed chars
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class SkinDiseaseReport(FPDF):
    """
    Custom PDF report class for skin disease diagnosis results.
    Inherits from FPDF and adds custom header/footer.
    All text is sanitized to remove emojis before rendering.
    """

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        """Custom header with app branding and blue accent bar."""
        # Blue accent bar at the top
        self.set_fill_color(10, 108, 158)  # Primary blue
        self.rect(0, 0, 210, 8, 'F')

        # App title
        self.set_y(14)
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(10, 108, 158)
        self.cell(0, 10, _sanitize_for_pdf(APP_TITLE), ln=True, align='C')

        # Subtitle
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, 'AI-Powered Skin Disease Detection Report', ln=True, align='C')

        # Divider line
        self.set_draw_color(226, 232, 240)
        self.line(15, 34, 195, 34)
        self.ln(12)

    def footer(self):
        """Custom footer with page number and disclaimer."""
        self.set_y(-25)

        # Disclaimer — sanitized to remove emojis
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(100, 116, 139)
        self.cell(0, 4, _sanitize_for_pdf(DISCLAIMER_TEXT), ln=True, align='C')

        # Page number
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        """Add a styled section title (sanitized)."""
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(10, 108, 158)
        self.cell(0, 10, _sanitize_for_pdf(title), ln=True)

        # Accent underline
        self.set_draw_color(0, 196, 180)
        self.set_line_width(0.8)
        x = self.get_x()
        y = self.get_y()
        self.line(x, y, x + 40, y)
        self.set_line_width(0.2)
        self.ln(6)

    def info_row(self, label, value):
        """Add a label-value row."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(31, 41, 55)
        self.cell(55, 7, _sanitize_for_pdf(label) + ':', align='L')

        self.set_font('Helvetica', '', 10)
        self.set_text_color(71, 85, 105)
        self.cell(0, 7, _sanitize_for_pdf(str(value)), ln=True)

    def safe_cell(self, w, h, txt, **kwargs):
        """Wrapper for cell() that sanitizes text."""
        self.cell(w, h, _sanitize_for_pdf(txt), **kwargs)

    def safe_multi_cell(self, w, h, txt, **kwargs):
        """Wrapper for multi_cell() that sanitizes text."""
        self.multi_cell(w, h, _sanitize_for_pdf(txt), **kwargs)


def generate_pdf_report(
    original_image,
    predicted_class,
    confidence,
    top_3,
    gradcam_overlay=None,
    report_id=None
):
    """
    Generate a professional PDF diagnostic report.

    Args:
        original_image: PIL Image of the uploaded skin image
        predicted_class: String name of the predicted disease
        confidence: Float confidence score (0-1)
        top_3: List of tuples [(class_name, confidence), ...]
        gradcam_overlay: NumPy array of Grad-CAM overlay image (optional)
        report_id: Optional report ID string

    Returns:
        pdf_bytes: Bytes of the generated PDF file
    """
    if not FPDF_AVAILABLE:
        return None

    # Generate report ID if not provided
    if report_id is None:
        report_id = f"SKD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Create PDF
    pdf = SkinDiseaseReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Report Metadata ────────────────────────────────────
    pdf.section_title('Report Information')
    pdf.info_row('Report ID', report_id)
    pdf.info_row('Date & Time', datetime.now().strftime('%B %d, %Y at %I:%M %p'))
    pdf.info_row('Model', 'EfficientNetB3 (Transfer Learning)')
    pdf.info_row('Dataset', 'HAM10000 (10,015 images, 7 classes)')
    pdf.ln(8)

    # ── Uploaded Image ─────────────────────────────────────
    pdf.section_title('Uploaded Image')

    # Save original image to temp file for embedding
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        original_image_resized = original_image.copy()
        original_image_resized.thumbnail((400, 400))
        original_image_resized.save(tmp.name, format='PNG')
        tmp_path = tmp.name

    # Center the image
    img_width = 70
    x_pos = (210 - img_width) / 2
    pdf.image(tmp_path, x=x_pos, w=img_width)
    os.unlink(tmp_path)  # Clean up temp file
    pdf.ln(8)

    # ── Prediction Results ─────────────────────────────────
    pdf.section_title('Prediction Results')

    # Main prediction — large and prominent
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(10, 108, 158)
    pdf.safe_cell(0, 10, f'Predicted Condition: {predicted_class}', ln=True)

    pdf.set_font('Helvetica', 'B', 14)
    conf_pct = confidence * 100
    if confidence >= 0.8:
        pdf.set_text_color(16, 185, 129)  # Green
    elif confidence >= 0.5:
        pdf.set_text_color(245, 158, 11)  # Amber
    else:
        pdf.set_text_color(239, 68, 68)   # Red
    pdf.safe_cell(0, 10, f'Confidence: {conf_pct:.1f}%', ln=True)
    pdf.ln(4)

    # Severity indicator
    severity = CLASS_DESCRIPTIONS.get(predicted_class, {}).get("severity", "Unknown")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(55, 7, 'Severity Level:', align='L')
    pdf.set_font('Helvetica', 'B', 10)
    if severity in ['Critical', 'High']:
        pdf.set_text_color(239, 68, 68)
    elif severity == 'Moderate':
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 7, severity, ln=True)
    pdf.ln(6)

    # ── Top 3 Predictions ──────────────────────────────────
    pdf.section_title('Top 3 Predictions')

    # Table header
    pdf.set_fill_color(248, 250, 252)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(10, 8, '#', border=1, align='C', fill=True)
    pdf.cell(95, 8, 'Condition', border=1, align='C', fill=True)
    pdf.cell(40, 8, 'Confidence', border=1, align='C', fill=True)
    pdf.cell(35, 8, 'Severity', border=1, align='C', fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font('Helvetica', '', 10)
    for i, (cls_name, cls_conf) in enumerate(top_3, 1):
        sev = CLASS_DESCRIPTIONS.get(cls_name, {}).get("severity", "-")
        pdf.set_text_color(71, 85, 105)
        pdf.cell(10, 8, str(i), border=1, align='C')
        pdf.safe_cell(95, 8, cls_name, border=1, align='L')
        pdf.cell(40, 8, f'{cls_conf * 100:.1f}%', border=1, align='C')
        pdf.cell(35, 8, sev, border=1, align='C')
        pdf.ln()
    pdf.ln(8)

    # ── Treatment Recommendations ──────────────────────────
    # (Inspired by DermaAI's skin_disorder.json — treatment per class)
    treatments = TREATMENT_RECOMMENDATIONS.get(predicted_class, [])
    if treatments:
        pdf.section_title('Treatment Recommendations')
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 116, 139)
        pdf.safe_cell(0, 6,
            'These are general treatment options. Always consult a dermatologist for personalized advice.',
            ln=True)
        pdf.ln(4)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(71, 85, 105)
        for j, treatment in enumerate(treatments, 1):
            pdf.safe_multi_cell(0, 6, f'{j}. {treatment}')
            pdf.ln(2)
        pdf.ln(6)

    # ── Grad-CAM Heatmap ──────────────────────────────────
    if gradcam_overlay is not None:
        pdf.section_title('Grad-CAM Explainability Heatmap')

        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, 'The heatmap highlights regions the AI model focused on for its prediction.', ln=True)
        pdf.ln(4)

        # Save gradcam overlay to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            overlay_img = Image.fromarray(gradcam_overlay.astype(np.uint8))
            overlay_img.save(tmp.name, format='PNG')
            tmp_path = tmp.name

        x_pos = (210 - img_width) / 2
        pdf.image(tmp_path, x=x_pos, w=img_width)
        os.unlink(tmp_path)
        pdf.ln(8)

    # ── Condition Description ──────────────────────────────
    pdf.section_title('About the Predicted Condition')
    description = CLASS_DESCRIPTIONS.get(predicted_class, {}).get(
        "description", "No description available."
    )
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    pdf.safe_multi_cell(0, 6, description)
    pdf.ln(8)

    # ── Prominent Disclaimer ───────────────────────────────
    pdf.add_page()
    pdf.section_title('Important Medical Disclaimer')

    # Warning box
    pdf.set_fill_color(254, 243, 199)
    pdf.set_draw_color(245, 158, 11)
    pdf.rect(15, pdf.get_y(), 180, 50, 'DF')

    pdf.set_xy(20, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 8, 'WARNING: This is NOT a medical diagnosis.', ln=True)

    pdf.set_x(20)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(146, 64, 14)
    disclaimer_extended = (
        "This AI-powered tool is designed for educational and preliminary screening purposes only. "
        "It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of a qualified dermatologist or healthcare provider with any questions "
        "you may have regarding a medical condition. Never disregard professional medical advice or delay "
        "in seeking it because of results generated by this tool."
    )
    pdf.multi_cell(170, 5, disclaimer_extended)
    pdf.ln(20)

    # ── About the System ───────────────────────────────────
    pdf.section_title('About SkinVision AI')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    about_text = (
        "SkinVision AI is a deep learning-based skin disease detection system developed as a "
        "university research project. It uses EfficientNetB3 architecture with transfer learning, "
        "trained on the HAM10000 dataset containing 10,015 dermatoscopic images across 7 disease "
        "categories. The system achieves 80-92% classification accuracy and uses Grad-CAM for "
        "explainable AI visualization, showing which regions of the skin image influenced the "
        "model's prediction."
    )
    pdf.multi_cell(0, 6, about_text)

    # ── Get PDF bytes ──────────────────────────────────────
    pdf_bytes = pdf.output()

    return bytes(pdf_bytes)
