"""
Green Land — Sand Grain Size Measurement App
=============================================
A Streamlit web application that allows users to measure the size of sand grains
(in millimetres) using a smartphone camera or uploaded images.

Workflow:
  1. User captures or uploads an image of sand grains beside a known reference object.
  2. The reference object (coin, credit card, etc.) is used to compute a
     Pixels-Per-Millimetre (PPM) calibration ratio.
  3. OpenCV image-processing pipeline (Grayscale → Gaussian Blur → Canny edges →
     Contours) isolates individual grains.
  4. Each grain's equivalent circular diameter is computed in mm.
  5. Results are displayed as an annotated image + distribution histogram + stats.

Author : Green Land CV Team
Version: 1.0
"""

import math
import io

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (Streamlit-safe)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Green Land — Sand Grain Analyser",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — clean, branded, professional dark-green theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background & text ── */
  .stApp {
    background: linear-gradient(160deg, #0B1A10 0%, #0E2016 50%, #0A1A14 100%);
    color: #D4E8D4;
    min-height: 100vh;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0D1C12 !important;
    border-right: 1px solid #1E3A28 !important;
  }
  section[data-testid="stSidebar"] * { color: #B0D0B4 !important; }

  /* ── Header banner ── */
  .gl-header {
    background: linear-gradient(135deg, #0F2D1A 0%, #1A4A2A 60%, #0F2D1A 100%);
    border: 1px solid #2A5A3A;
    border-radius: 14px;
    padding: 22px 30px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 4px 24px rgba(0,160,80,0.15);
  }
  .gl-logo { font-size: 2.6rem; line-height: 1; }
  .gl-brand { font-size: 1.75rem; font-weight: 700; color: #6EE09A; margin: 0; letter-spacing: -0.02em; }
  .gl-tagline { font-size: 0.82rem; color: #5A9070; margin: 3px 0 0; letter-spacing: 0.03em; }

  /* ── Cards ── */
  .gl-card {
    background: rgba(15, 40, 22, 0.7);
    border: 1px solid #1E3A28;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(4px);
  }
  .gl-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3EBD70;
    margin-bottom: 12px;
  }

  /* ── Metric tiles ── */
  .gl-metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
  .gl-metric {
    flex: 1;
    min-width: 90px;
    background: #0B1E12;
    border: 1px solid #1E3A28;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
  }
  .gl-metric-val { font-size: 1.45rem; font-weight: 700; color: #6EE09A; line-height: 1.1; }
  .gl-metric-lbl { font-size: 0.68rem; color: #4A7A5A; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

  /* ── Grain-class pills ── */
  .pill-very-fine { background:#0D2A1A; color:#34D399; border:1px solid #065F46;
                    border-radius:4px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .pill-fine      { background:#1A2F0D; color:#86EFAC; border:1px solid #166534;
                    border-radius:4px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .pill-medium    { background:#2A270D; color:#FDE68A; border:1px solid #92400E;
                    border-radius:4px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .pill-coarse    { background:#2A1A0D; color:#FCA5A5; border:1px solid #7F1D1D;
                    border-radius:4px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

  /* ── Step numbers ── */
  .gl-step {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: #1E4A2A; border: 1px solid #3EBD70;
    color: #6EE09A; font-size: 0.8rem; font-weight: 700; margin-right: 8px;
  }
  .gl-step-label { font-weight: 600; color: #A0D0B0; font-size: 0.9rem; }

  /* ── Info / warning boxes ── */
  .gl-info {
    background: #0D2A1A; border-left: 3px solid #34D399; border-radius: 6px;
    padding: 10px 14px; font-size: 0.8rem; color: #86EFAC; margin: 8px 0;
  }
  .gl-warn {
    background: #2A1A0D; border-left: 3px solid #F59E0B; border-radius: 6px;
    padding: 10px 14px; font-size: 0.8rem; color: #FDE68A; margin: 8px 0;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1A6B35 0%, #2A8A50 100%);
    color: #D4FFE4;
    border: 1px solid #2A8A50;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 10px 22px;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 2px 12px rgba(46,160,90,0.25);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2A8A50 0%, #3AAA65 100%);
    box-shadow: 0 4px 20px rgba(46,160,90,0.40);
    transform: translateY(-1px);
  }

  /* ── Sliders & selects ── */
  .stSlider > div { color: #B0D0B4 !important; }

  /* ── Table ── */
  .gl-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .gl-table th { background: #0B1E12; color: #3EBD70; font-weight: 600;
                 padding: 8px 12px; text-align: left; border-bottom: 1px solid #1E3A28; }
  .gl-table td { padding: 7px 12px; border-bottom: 1px solid #12281A; color: #A0C0AA; }
  .gl-table tr:hover td { background: #0F2416; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0B1A10; }
  ::-webkit-scrollbar-thumb { background: #1E3A28; border-radius: 10px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
  label { color: #7AAA8A !important; font-size: 0.82rem !important; }
  .stMarkdown hr { border-color: #1E3A28; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Reference objects with their known widths in millimetres.
# Users place one of these next to the sand to enable calibration.
REFERENCE_OBJECTS = {
    "1 Dirham coin  (20.0 mm)":        20.0,
    "2 Dirham coin  (23.0 mm)":        23.0,
    "5 Dirham coin  (25.0 mm)":        25.0,
    "1 Euro coin    (23.25 mm)":       23.25,
    "US Quarter     (24.26 mm)":       24.26,
    "Credit card — width (85.6 mm)":   85.6,
    "Credit card — height (53.98 mm)": 53.98,
    "Standard ruler — 10 mm mark":     10.0,
    "Custom (enter below) …":          None,
}

# Wentworth grain-size scale (simplified)
GRAIN_CLASSES = [
    (0.0,   0.063, "Very Fine / Silt",  "pill-very-fine"),
    (0.063, 0.25,  "Fine Sand",         "pill-fine"),
    (0.25,  0.5,   "Medium Sand",       "pill-fine"),
    (0.5,   2.0,   "Coarse Sand",       "pill-medium"),
    (2.0,   64.0,  "Gravel",            "pill-coarse"),
]


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_image(uploaded_file) -> np.ndarray:
    """Convert a Streamlit uploaded file / camera image to a BGR NumPy array."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """
    Step 1 — Grayscale conversion.
    Single-channel images are far cheaper to process and edge detectors
    require them.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Step 2 — Gaussian blur.
    Reduces high-frequency noise so that Canny does not chase pixel-level
    artefacts.  ksize must be odd.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def apply_canny(blurred: np.ndarray, low_thresh: int, high_thresh: int) -> np.ndarray:
    """
    Step 3 — Canny edge detection.
    Finds the outlines of grains.  low/high thresholds are exposed in the UI
    so the user can tune for different sand colours (silica vs dark basalt, etc.).
    """
    return cv2.Canny(blurred, low_thresh, high_thresh)


def close_edges(edges: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Step 3b — Morphological closing.
    Bridges small gaps in edges so that grain outlines form closed loops,
    which is required for findContours to work properly.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def detect_reference_coin(gray: np.ndarray):
    """
    Auto-detect a circular reference object (coin) using Hough Circle Transform.
    Returns (cx, cy, radius) in pixels, or None if not found.
    """
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gray.shape[0] // 4,   # minimum distance between circle centres
        param1=60,                     # Canny high threshold inside Hough
        param2=35,                     # accumulator threshold (lower → more circles found)
        minRadius=20,
        maxRadius=min(gray.shape) // 3,
    )
    if circles is not None:
        c = np.uint16(np.around(circles[0][0]))
        return int(c[0]), int(c[1]), int(c[2])   # cx, cy, r
    return None


def find_grains(
    closed_edges: np.ndarray,
    px_per_mm: float,
    min_mm: float,
    max_mm: float,
) -> list[dict]:
    """
    Step 4 — Contour detection & grain measurement.

    For each closed contour:
    • Compute the contour area in pixels².
    • Convert to an equivalent circular diameter in mm:
        diameter_mm = (2 / px_per_mm) × sqrt(area_px / π)
    • Filter by min/max mm to exclude noise and the reference object itself.
    """
    # Find all external contours in the edge map
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert size limits from mm to pixel-area limits
    min_area_px = math.pi * ((px_per_mm * min_mm) / 2) ** 2
    max_area_px = math.pi * ((px_per_mm * max_mm) / 2) ** 2

    grains = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # Equivalent circular diameter
        diam_px = 2 * math.sqrt(area_px / math.pi)
        diam_mm = diam_px / px_per_mm

        # Centroid (for label placement)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        grains.append({
            "diameter_mm": round(diam_mm, 3),
            "area_px": area_px,
            "centroid": (cx, cy),
            "contour": cnt,
        })

    return grains


def classify_grain(d_mm: float) -> tuple[str, str]:
    """Return (label, css_pill_class) for a grain diameter in mm."""
    for lo, hi, label, pill in GRAIN_CLASSES:
        if lo <= d_mm < hi:
            return label, pill
    return "Very Coarse / Gravel", "pill-coarse"


def annotate_image(
    img_bgr: np.ndarray,
    grains: list[dict],
    ref_circle=None,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw:
    • Coloured contours around each detected grain.
    • Optional diameter label at the grain centroid.
    • Green circle + "REF" label around the reference object (if auto-detected).
    """
    out = img_bgr.copy()

    # Colour map per grain class
    color_map = {
        "Very Fine / Silt":  (52,  211, 153),   # teal-green
        "Fine Sand":         (110, 224, 154),    # light green
        "Medium Sand":       (253, 224, 71),     # amber
        "Coarse Sand":       (251, 146, 60),     # orange
        "Gravel":            (248, 113, 113),    # red
        "Very Coarse / Gravel": (220, 80, 80),
    }

    for g in grains:
        label, _ = classify_grain(g["diameter_mm"])
        color = color_map.get(label, (200, 200, 200))
        # Draw the grain outline (BGR colour)
        cv2.drawContours(out, [g["contour"]], -1, color[::-1], 1)

        if show_labels:
            cx, cy = g["centroid"]
            text = f"{g['diameter_mm']:.2f}"
            cv2.putText(
                out, text, (cx - 14, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                color[::-1], 1, cv2.LINE_AA,
            )

    # Draw the reference circle if auto-detected
    if ref_circle:
        rx, ry, rr = ref_circle
        cv2.circle(out, (rx, ry), rr, (52, 211, 153), 2)
        cv2.putText(
            out, "REF", (rx - 16, ry - rr - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (52, 211, 153), 2, cv2.LINE_AA,
        )

    return out


def compute_stats(grains: list[dict]) -> dict | None:
    """Summarise grain diameter distribution."""
    if not grains:
        return None

    diams = sorted(g["diameter_mm"] for g in grains)
    arr   = np.array(diams)

    cls_counts: dict[str, int] = {}
    for g in grains:
        lbl, _ = classify_grain(g["diameter_mm"])
        cls_counts[lbl] = cls_counts.get(lbl, 0) + 1

    dominant = max(cls_counts, key=cls_counts.get)

    return {
        "count":    len(diams),
        "min":      round(float(arr.min()), 3),
        "max":      round(float(arr.max()), 3),
        "mean":     round(float(arr.mean()), 3),
        "median":   round(float(np.median(arr)), 3),
        "std":      round(float(arr.std()), 3),
        "d10":      round(float(np.percentile(arr, 10)), 3),
        "d50":      round(float(np.percentile(arr, 50)), 3),
        "d90":      round(float(np.percentile(arr, 90)), 3),
        "dominant": dominant,
        "cls_counts": cls_counts,
        "diams":    diams,
    }


def make_histogram(diams: list[float], bins: int = 25) -> io.BytesIO:
    """
    Generate a styled Matplotlib histogram of grain diameters and return
    it as a PNG byte-stream (for st.image).
    """
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#0B1A10")
    ax.set_facecolor("#0E2016")

    # Histogram bars with green gradient effect
    n, bin_edges, patches = ax.hist(
        diams, bins=bins,
        color="#34D399", edgecolor="#0B1A10", linewidth=0.5, alpha=0.9,
    )

    # Colour-code bars by grain class (tonal gradient)
    for patch, left in zip(patches, bin_edges[:-1]):
        mid = left + (bin_edges[1] - bin_edges[0]) / 2
        _, pill = classify_grain(mid)
        cm = {
            "pill-very-fine": "#34D399",
            "pill-fine":      "#6EE09A",
            "pill-medium":    "#FBBF24",
            "pill-coarse":    "#F87171",
        }
        patch.set_facecolor(cm.get(pill, "#34D399"))

    # Vertical lines for D10, D50, D90
    d10, d50, d90 = (
        float(np.percentile(diams, 10)),
        float(np.percentile(diams, 50)),
        float(np.percentile(diams, 90)),
    )
    for val, lbl, col in [(d10, "D10", "#A78BFA"), (d50, "D50", "#F472B6"), (d90, "D90", "#FB923C")]:
        ax.axvline(val, color=col, linewidth=1.2, linestyle="--", alpha=0.85)
        ax.text(val + 0.002, ax.get_ylim()[1] * 0.92, lbl,
                color=col, fontsize=7, fontweight="bold")

    ax.set_xlabel("Grain Diameter (mm)", color="#5A9070", fontsize=9)
    ax.set_ylabel("Count", color="#5A9070", fontsize=9)
    ax.set_title("Grain Size Distribution", color="#6EE09A", fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors="#4A7A5A", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3A28")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Legend
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color="#A78BFA", label="D10"),
        mpatches.Patch(color="#F472B6", label="D50 (median)"),
        mpatches.Patch(color="#FB923C", label="D90"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              fontsize=7, facecolor="#0B1A10",
              edgecolor="#1E3A28", labelcolor="#B0D0B4")

    fig.tight_layout(pad=1.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "px_per_mm":  None,
    "grains":     [],
    "stats":      None,
    "annotated":  None,
    "processed":  False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gl-header">
  <span class="gl-logo">🌿</span>
  <div>
    <p class="gl-brand">Green Land</p>
    <p class="gl-tagline">Sand Grain Size Measurement · Powered by OpenCV</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Green Land Settings")
    st.markdown("---")

    # ── Reference object ──────────────────────────────────────────────────────
    st.markdown("### 📏 Reference Object")
    st.markdown(
        '<div class="gl-info">Place the reference object <strong>flat</strong> '
        'in the same plane as the sand grains before taking the photo.</div>',
        unsafe_allow_html=True,
    )
    ref_choice = st.selectbox("Known object", list(REFERENCE_OBJECTS.keys()), key="ref_choice")
    ref_mm = REFERENCE_OBJECTS[ref_choice]
    if ref_mm is None:
        ref_mm = st.number_input(
            "Known diameter / width (mm)", value=25.0, min_value=1.0, step=0.1, format="%.2f"
        )

    calib_mode = st.radio(
        "Calibration method",
        ["🔵 Auto-detect circle (Hough)", "✏️ Enter pixel diameter manually"],
        help="Auto works best for round coins on a contrasting background.",
    )
    manual_px_diam = None
    if calib_mode == "✏️ Enter pixel diameter manually":
        manual_px_diam = st.number_input(
            "Reference diameter in pixels", min_value=1, value=150, step=1
        )

    st.markdown("---")

    # ── Edge-detection thresholds ─────────────────────────────────────────────
    st.markdown("### 🎛️ Detection Tuning")
    st.markdown(
        '<div class="gl-info">Lower thresholds detect more edges '
        '(better for light/silica sand). Raise them for dark sand.</div>',
        unsafe_allow_html=True,
    )
    canny_low  = st.slider("Canny — low threshold",  10,  200,  40, 5,
                            help="Weak edges below this are discarded.")
    canny_high = st.slider("Canny — high threshold", 30,  400, 120, 5,
                            help="Strong edges above this are always kept.")
    blur_k     = st.slider("Gaussian blur kernel",    3,   15,   5, 2,
                            help="Larger kernel → smoother image → fewer false edges.")
    close_iter = st.slider("Morphological closing iterations", 1, 5, 2, 1,
                            help="Closes gaps in grain outlines (increase for broken edges).")

    st.markdown("---")

    # ── Grain size range ──────────────────────────────────────────────────────
    st.markdown("### 🔍 Grain Size Filter (mm)")
    min_grain_mm = st.slider("Minimum grain size",  0.01,  2.0,  0.05, 0.01, format="%.2f")
    max_grain_mm = st.slider("Maximum grain size",   0.5, 100.0, 20.0,  0.5, format="%.1f")

    st.markdown("---")

    # ── Display options ───────────────────────────────────────────────────────
    st.markdown("### 🖼️ Display Options")
    show_labels = st.checkbox("Show diameter labels on grains", value=True)
    hist_bins   = st.slider("Histogram bins", 10, 60, 25, 5)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.73rem;color:#2A5A3A;text-align:center">'
        'Green Land v1.0 · OpenCV Granulometry</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — INPUT & CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Step 1 : Image input ──────────────────────────────────────────────────
    st.markdown(
        '<div class="gl-card"><div class="gl-card-title">'
        '<span class="gl-step">1</span>'
        '<span class="gl-step-label">Capture or Upload Image</span></div>',
        unsafe_allow_html=True,
    )
    input_mode = st.radio(
        "Image source", ["📷 Camera (mobile/webcam)", "📁 Upload file"],
        horizontal=True, label_visibility="collapsed",
    )
    img_file = None
    if input_mode == "📷 Camera (mobile/webcam)":
        st.markdown(
            '<div class="gl-info">💡 Tip: Photograph the sand from directly above '
            '(top-down) in good, even lighting. Place the reference object in the '
            'same plane as the sample.</div>',
            unsafe_allow_html=True,
        )
        img_file = st.camera_input("Take a photo", label_visibility="collapsed")
    else:
        img_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 2 : Preview & calibration ───────────────────────────────────────
    if img_file:
        img_bgr  = load_image(img_file)
        gray     = to_grayscale(img_bgr)
        blurred  = apply_blur(gray, ksize=blur_k)

        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">'
            '<span class="gl-step">2</span>'
            '<span class="gl-step-label">Image Preview & Calibration</span></div>',
            unsafe_allow_html=True,
        )

        # Show the original image
        img_rgb_preview = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb_preview, caption="Loaded image", use_container_width=True)

        # Perform calibration
        ref_circle = None
        ppm = None   # pixels per mm

        if calib_mode == "🔵 Auto-detect circle (Hough)":
            ref_circle = detect_reference_coin(gray)
            if ref_circle:
                _, _, r = ref_circle
                ref_px_diam = r * 2
                ppm = ref_px_diam / ref_mm
                st.markdown(
                    f'<div class="gl-info">✅ Reference detected — diameter: '
                    f'<strong>{ref_px_diam} px</strong> → '
                    f'<strong>{ref_mm:.2f} mm</strong> &nbsp;|&nbsp; '
                    f'Scale: <strong>{ppm:.2f} px / mm</strong></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="gl-warn">⚠️ Could not auto-detect a circle. '
                    'Switch to <em>Manual pixel input</em> in the sidebar, or try '
                    'a photo with better contrast around the coin.</div>',
                    unsafe_allow_html=True,
                )
        else:
            if manual_px_diam:
                ppm = manual_px_diam / ref_mm
                st.markdown(
                    f'<div class="gl-info">📐 Manual calibration — '
                    f'{manual_px_diam} px = {ref_mm:.2f} mm → '
                    f'<strong>{ppm:.2f} px / mm</strong></div>',
                    unsafe_allow_html=True,
                )

        if ppm:
            st.session_state.px_per_mm = ppm

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Step 3 : Run analysis ─────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">'
            '<span class="gl-step">3</span>'
            '<span class="gl-step-label">Run Analysis</span></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.px_per_mm:
            st.markdown(
                '<div class="gl-warn">⚠️ Calibration required before analysis '
                'can run. Please calibrate the reference object above.</div>',
                unsafe_allow_html=True,
            )

        run_btn = st.button(
            "🌿 Measure Sand Grains",
            disabled=(st.session_state.px_per_mm is None),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Process ───────────────────────────────────────────────────────────
        if run_btn and st.session_state.px_per_mm:
            with st.spinner("Processing image… detecting grains…"):
                # Full pipeline
                blurred_run = apply_blur(gray, ksize=blur_k)
                edges       = apply_canny(blurred_run, canny_low, canny_high)
                closed      = close_edges(edges, close_iter)

                grains = find_grains(
                    closed,
                    st.session_state.px_per_mm,
                    min_grain_mm,
                    max_grain_mm,
                )

                st.session_state.grains    = grains
                st.session_state.stats     = compute_stats(grains)
                st.session_state.annotated = annotate_image(
                    img_bgr, grains, ref_circle, show_labels
                )
                st.session_state.processed = True

    else:
        # ── Onboarding placeholder ────────────────────────────────────────────
        st.markdown("""
        <div class="gl-card" style="min-height:340px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;gap:14px;text-align:center">
          <span style="font-size:3.5rem">🌿</span>
          <p style="color:#3EBD70;font-size:1.05rem;font-weight:600;margin:0">
            Welcome to Green Land
          </p>
          <p style="color:#2A5A3A;font-size:0.83rem;margin:0;max-width:280px">
            Measure sand grain sizes precisely using your phone camera or an uploaded image.
          </p>
          <div style="background:#0B1A10;border-radius:10px;padding:16px 20px;
                      text-align:left;width:100%;max-width:300px;margin-top:8px">
            <p style="color:#3EBD70;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:.1em;margin:0 0 10px">
              Quick Start Guide
            </p>
            <p style="color:#4A7A5A;font-size:0.8rem;margin:0 0 6px">
              1️⃣ &nbsp;Place sand on a flat, contrasting surface
            </p>
            <p style="color:#4A7A5A;font-size:0.8rem;margin:0 0 6px">
              2️⃣ &nbsp;Put a coin (e.g. 1 Dirham) beside the sample
            </p>
            <p style="color:#4A7A5A;font-size:0.8rem;margin:0 0 6px">
              3️⃣ &nbsp;Take a clear top-down photo in good lighting
            </p>
            <p style="color:#4A7A5A;font-size:0.8rem;margin:0 0 6px">
              4️⃣ &nbsp;Select the coin type in the sidebar
            </p>
            <p style="color:#4A7A5A;font-size:0.8rem;margin:0">
              5️⃣ &nbsp;Click <em>Measure Sand Grains</em> → instant results!
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    if st.session_state.processed and st.session_state.stats:
        stats = st.session_state.stats

        # ── Annotated image ───────────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">🔬 Detected Grains</div>',
            unsafe_allow_html=True,
        )
        ann_rgb = cv2.cvtColor(st.session_state.annotated, cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, caption="Grain outlines coloured by size class", use_container_width=True)

        # Colour legend
        st.markdown("""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px">
          <span class="pill-very-fine">Very Fine / Silt &lt;0.063 mm</span>
          <span class="pill-fine">Fine / Medium Sand 0.063–0.5 mm</span>
          <span class="pill-medium">Coarse Sand 0.5–2 mm</span>
          <span class="pill-coarse">Gravel &gt;2 mm</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">📊 Size Summary</div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        for col_widget, val, lbl in [
            (m1, stats["count"],             "Grains"),
            (m2, f"{stats['min']} mm",       "Minimum"),
            (m3, f"{stats['max']} mm",       "Maximum"),
            (m4, f"{stats['mean']} mm",      "Average"),
            (m5, f"{stats['median']} mm",    "Median"),
        ]:
            col_widget.markdown(
                f'<div class="gl-metric">'
                f'<div class="gl-metric-val">{val}</div>'
                f'<div class="gl-metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Percentiles ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">📐 Percentile Statistics</div>',
            unsafe_allow_html=True,
        )
        p1, p2, p3, p4 = st.columns(4)
        for col_widget, val, lbl in [
            (p1, f"{stats['d10']} mm", "D10 (10th pct)"),
            (p2, f"{stats['d50']} mm", "D50 (median)"),
            (p3, f"{stats['d90']} mm", "D90 (90th pct)"),
            (p4, f"{stats['std']} mm", "Std Deviation"),
        ]:
            col_widget.markdown(
                f'<div class="gl-metric">'
                f'<div class="gl-metric-val">{val}</div>'
                f'<div class="gl-metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        dominant_lbl = stats["dominant"]
        pill_class   = next(
            (p for lo, hi, lbl, p in GRAIN_CLASSES if lbl == dominant_lbl), "pill-fine"
        )
        st.markdown(
            f'<p style="font-size:0.82rem;color:#5A9070;margin-top:10px">'
            f'Dominant class: &nbsp;<span class="{pill_class}">{dominant_lbl}</span></p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Histogram ─────────────────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">📈 Grain Size Histogram</div>',
            unsafe_allow_html=True,
        )
        hist_buf = make_histogram(stats["diams"], bins=hist_bins)
        st.image(hist_buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Class distribution table ──────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">📋 Class Distribution</div>',
            unsafe_allow_html=True,
        )
        total = stats["count"]
        rows_html = ""
        for lo, hi, label, pill in GRAIN_CLASSES:
            cnt = stats["cls_counts"].get(label, 0)
            pct = round(cnt / total * 100, 1) if total else 0
            bar_w = max(2, int(pct * 1.5))
            rows_html += (
                f"<tr>"
                f"<td><span class='{pill}'>{label}</span></td>"
                f"<td>{lo}–{hi} mm</td>"
                f"<td>{cnt}</td>"
                f"<td>{pct}%</td>"
                f"<td><div style='width:{bar_w}px;height:7px;"
                f"background:linear-gradient(90deg,#1A6B35,#6EE09A);"
                f"border-radius:4px'></div></td>"
                f"</tr>"
            )
        st.markdown(
            f"<table class='gl-table'><thead><tr>"
            f"<th>Class</th><th>Size Range</th><th>Count</th><th>%</th><th>Bar</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown(
            '<div class="gl-card"><div class="gl-card-title">⬇️ Export Results</div>',
            unsafe_allow_html=True,
        )
        import json
        from datetime import datetime

        # Build CSV
        csv_lines = ["diameter_mm,class"]
        for g in st.session_state.grains:
            lbl, _ = classify_grain(g["diameter_mm"])
            csv_lines.append(f"{g['diameter_mm']},{lbl}")
        csv_data = "\n".join(csv_lines)

        # Build JSON
        report = {
            "generated_at": datetime.now().isoformat(),
            "calibration_px_per_mm": round(st.session_state.px_per_mm, 4),
            "reference_object_mm": ref_mm,
            "grain_count": stats["count"],
            "statistics_mm": {
                "min": stats["min"], "max": stats["max"],
                "mean": stats["mean"], "median": stats["median"],
                "std": stats["std"],
                "D10": stats["d10"], "D50": stats["d50"], "D90": stats["d90"],
            },
            "class_distribution": stats["cls_counts"],
            "grain_diameters_mm": stats["diams"],
        }

        # Annotated image bytes
        _, buf_jpg = cv2.imencode(".jpg", st.session_state.annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "📄 CSV", data=csv_data,
            file_name=f"greenland_grains_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )
        dl2.download_button(
            "📦 JSON", data=json.dumps(report, indent=2),
            file_name=f"greenland_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json", use_container_width=True,
        )
        dl3.download_button(
            "🖼️ Image", data=buf_jpg.tobytes(),
            file_name=f"greenland_annotated_{datetime.now().strftime('%Y%m%d_%H%M')}.jpg",
            mime="image/jpeg", use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.processed and not st.session_state.stats:
        # Analysis ran but no grains found
        st.markdown("""
        <div class="gl-card" style="min-height:300px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;gap:12px;text-align:center">
          <span style="font-size:2.5rem">🔍</span>
          <p style="color:#F59E0B;font-size:1rem;font-weight:600;margin:0">
            No grains detected
          </p>
          <p style="color:#4A7A5A;font-size:0.82rem;margin:0;max-width:280px">
            Try lowering the Canny thresholds or adjusting the minimum grain size
            in the sidebar settings. Ensure the image is sharp and well-lit.
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # No results yet — visual placeholder
        st.markdown("""
        <div class="gl-card" style="min-height:400px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;gap:14px;text-align:center">
          <span style="font-size:3.5rem">📊</span>
          <p style="color:#3EBD70;font-size:1rem;font-weight:600;margin:0">
            Results will appear here
          </p>
          <p style="color:#2A5A3A;font-size:0.82rem;margin:0;max-width:260px">
            Upload or capture an image, calibrate the reference object, then click
            <strong>Measure Sand Grains</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 8px;color:#1E3A28;font-size:0.72rem">
  🌿 Green Land · Sand Grain Measurement · OpenCV + Streamlit · v1.0
</div>
""", unsafe_allow_html=True)
