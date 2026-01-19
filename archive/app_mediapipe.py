# app.py
# Streamlit POC — IA frugale vision (MediaPipe + OpenCV)
# 1) Comptage / présence (visages) via MediaPipe Face Detection
# 2) Détection de structure visuelle (tables/colonnes/zones) via OpenCV (heuristiques)
# 3) Contrôle qualité visuel (comparaison à un gabarit) via ORB + homographie + SSIM
#
# Install:
#   pip install streamlit opencv-python mediapipe numpy scikit-image
#
# Run:
#   streamlit run app.py

import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim

import mediapipe as mp


# -----------------------------
# Helpers
# -----------------------------

def read_image_bytes(file) -> Optional[np.ndarray]:
    if file is None:
        return None
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))

def draw_label(img: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# -----------------------------
# 1) Presence / counting (MediaPipe Face Detection)
# -----------------------------

mp_face = mp.solutions.face_detection

def count_faces_mediapipe(bgr: np.ndarray, min_conf: float = 0.5) -> Tuple[np.ndarray, int]:
    rgb = to_rgb(bgr)
    out = bgr.copy()

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_conf) as fd:
        res = fd.process(rgb)

    count = 0
    h, w = out.shape[:2]
    if res.detections:
        for det in res.detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < min_conf:
                continue
            box = det.location_data.relative_bounding_box
            x1 = clamp_int(box.xmin * w, 0, w - 1)
            y1 = clamp_int(box.ymin * h, 0, h - 1)
            x2 = clamp_int((box.xmin + box.width) * w, 0, w - 1)
            y2 = clamp_int((box.ymin + box.height) * h, 0, h - 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            draw_label(out, f"face {score:.2f}", (x1, max(20, y1 - 8)))
            count += 1

    draw_label(out, f"Faces detectees: {count}", (10, 30))
    return out, count


# -----------------------------
# 2) Visual structure detection (heuristics)
#   - Detect table/grid lines or zones via morphology + Hough
# -----------------------------

@dataclass
class StructureParams:
    binarize_block: int = 31
    binarize_c: int = 10
    min_line_len_ratio: float = 0.35
    hough_thresh: int = 120

def detect_structure(bgr: np.ndarray, p: StructureParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      overlay_bgr: original with detected lines/boxes overlaid
      mask: binary mask used to infer lines (for debug)
    """
    img = bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle varying lighting
    block = p.binarize_block if p.binarize_block % 2 == 1 else p.binarize_block + 1
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, p.binarize_c
    )

    h, w = th.shape
    # Morphological extraction of horizontal and vertical lines
    horiz_ksize = max(10, w // 40)
    vert_ksize = max(10, h // 40)

    horiz = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_ksize, 1)))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_ksize, 1)))

    vert = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_ksize)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_ksize)))

    mask = cv2.bitwise_or(horiz, vert)

    # Hough lines for a cleaner overlay
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=p.hough_thresh,
                            minLineLength=int(min(h, w) * p.min_line_len_ratio),
                            maxLineGap=15)

    overlay = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Also propose "zones" as bounding boxes from connected components in mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (h * w) * 0.003:  # ignore tiny artifacts
            continue
        cv2.rectangle(overlay, (x, y), (x + ww, y + hh), (255, 255, 255), 2)

    draw_label(overlay, "Structure: lignes + zones (heuristiques)", (10, 30))
    return overlay, mask


# -----------------------------
# 3) Quality control (template vs sample)
#   - Align sample to template using ORB features + homography
#   - Compute SSIM + diff map
# -----------------------------

@dataclass
class QCParams:
    max_width: int = 900
    orb_features: int = 1200
    ssim_threshold: float = 0.90  # lower => more differences accepted
    diff_highlight: int = 35      # threshold on diff image for contouring

def resize_to_max(bgr: np.ndarray, max_w: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / float(w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def align_sample_to_template(template_bgr: np.ndarray, sample_bgr: np.ndarray, p: QCParams):
    tpl = resize_to_max(template_bgr, p.max_width)
    smp = resize_to_max(sample_bgr, p.max_width)

    tpl_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    smp_g = cv2.cvtColor(smp, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=p.orb_features)
    k1, d1 = orb.detectAndCompute(tpl_g, None)
    k2, d2 = orb.detectAndCompute(smp_g, None)

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, "Pas assez de points ORB pour aligner (image trop uniforme ou floue)."

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    if len(matches) < 12:
        return None, "Pas assez de matches pour calculer une homographie."

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        return None, "Homographie introuvable (angles/taille trop différents)."

    h, w = tpl.shape[:2]
    aligned = cv2.warpPerspective(smp, H, (w, h))
    return aligned, None

def qc_compare(template_bgr: np.ndarray, sample_bgr: np.ndarray, p: QCParams):
    tpl = resize_to_max(template_bgr, p.max_width)
    aligned, err = align_sample_to_template(tpl, sample_bgr, p)
    if err:
        return None, err

    tpl_g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    ali_g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(tpl_g, ali_g, full=True)
    diff_u8 = (diff * 255).astype(np.uint8)
    # Lower values => more difference (because SSIM map)
    inv = 255 - diff_u8

    # Threshold + contours for highlighting
    _, thr = cv2.threshold(inv, p.diff_highlight, 255, cv2.THRESH_BINARY)
    thr = cv2.medianBlur(thr, 5)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = tpl.copy()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < (vis.shape[0] * vis.shape[1]) * 0.0008:
            continue
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return {
        "template": tpl,
        "aligned": aligned,
        "highlight": vis,
        "diff_mask": thr,
        "ssim_score": float(score),
        "pass": bool(score >= p.ssim_threshold),
    }, None


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="POC Vision frugale", layout="wide")
st.title("POC — IA frugale vision (MediaPipe + heuristiques)")

tab1, tab2, tab3 = st.tabs(["Comptage / présence", "Structure visuelle", "Contrôle qualité (gabarit)"])

with tab1:
    st.subheader("Comptage / présence (visages) — MediaPipe")
    st.caption("Frugal: détection de visages pour compter des personnes dans une image. Pas du 'people tracking' vidéo.")
    left, right = st.columns(2)

    with left:
        img_file = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"], key="faces_file")
        cam = st.camera_input("Ou prendre une photo", key="faces_cam")
        min_conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.5, 0.05)

    img = read_image_bytes(cam) if cam is not None else read_image_bytes(img_file)
    if img is not None:
        out, n = count_faces_mediapipe(img, min_conf=min_conf)
        with right:
            st.image(to_rgb(out), caption=f"Détections — {n} visage(s)", use_container_width=True)
    else:
        st.info("Charge une image ou prends une photo.")

with tab2:
    st.subheader("Détection de structure visuelle — tables / colonnes / zones (heuristiques)")
    st.caption("Frugal: pas besoin de 'comprendre' le contenu; on détecte des lignes et blocs via morphologie + Hough.")
    left, right = st.columns(2)

    with left:
        img_file = st.file_uploader("Image (doc/tableau/screenshot)", type=["jpg", "jpeg", "png"], key="struct_file")
        cam = st.camera_input("Ou prendre une photo", key="struct_cam")
        block = st.slider("Adaptive threshold: block size", 11, 71, 31, 2)
        cval = st.slider("Adaptive threshold: C", 0, 25, 10, 1)
        min_ratio = st.slider("Longueur min des lignes (ratio)", 0.10, 0.70, 0.35, 0.05)
        hough = st.slider("Hough threshold", 50, 250, 120, 10)

    img = read_image_bytes(cam) if cam is not None else read_image_bytes(img_file)
    if img is not None:
        p = StructureParams(binarize_block=int(block), binarize_c=int(cval),
                            min_line_len_ratio=float(min_ratio), hough_thresh=int(hough))
        overlay, mask = detect_structure(img, p)
        with right:
            st.image(to_rgb(overlay), caption="Overlay structure", use_container_width=True)
            with st.expander("Masque (debug)"):
                st.image(mask, caption="Masque lignes/zones", use_container_width=True)
    else:
        st.info("Charge une image (photo de tableau, doc scanné, screenshot).")

with tab3:
    st.subheader("Contrôle qualité visuel — comparaison à un gabarit")
    st.caption("Frugal: on aligne l'échantillon sur le gabarit, puis on met en évidence les différences visuelles.")
    colA, colB = st.columns(2)

    with colA:
        tpl_file = st.file_uploader("Gabarit (template)", type=["jpg", "jpeg", "png"], key="qc_tpl")
        tpl_cam = st.camera_input("Ou photo gabarit", key="qc_tpl_cam")

    with colB:
        smp_file = st.file_uploader("Échantillon à contrôler", type=["jpg", "jpeg", "png"], key="qc_smp")
        smp_cam = st.camera_input("Ou photo échantillon", key="qc_smp_cam")

    tpl = read_image_bytes(tpl_cam) if tpl_cam is not None else read_image_bytes(tpl_file)
    smp = read_image_bytes(smp_cam) if smp_cam is not None else read_image_bytes(smp_file)

    p = QCParams(
        max_width=st.slider("Largeur max (resize)", 400, 1600, 900, 50),
        orb_features=st.slider("ORB features", 300, 3000, 1200, 100),
        ssim_threshold=st.slider("Seuil SSIM (pass/fail)", 0.70, 0.99, 0.90, 0.01),
        diff_highlight=st.slider("Seuil mise en évidence diff", 5, 80, 35, 1),
    )

    if tpl is not None and smp is not None:
        res, err = qc_compare(tpl, smp, p)
        if err:
            st.error(err)
        else:
            st.metric("SSIM", f"{res['ssim_score']:.3f}", "PASS" if res["pass"] else "FAIL")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.image(to_rgb(res["template"]), caption="Gabarit (redimensionné)", use_container_width=True)
            with r2:
                st.image(to_rgb(res["aligned"]), caption="Échantillon aligné", use_container_width=True)
            with r3:
                st.image(to_rgb(res["highlight"]), caption="Différences détectées (rectangles)", use_container_width=True)

            with st.expander("Masque différences (debug)"):
                st.image(res["diff_mask"], caption="Masque binaire des différences", use_container_width=True)
    else:
        st.info("Charge un gabarit et un échantillon.")

st.divider()
st.caption(
    "Notes POC: "
    "1) Comptage = visages (simple, frugal). "
    "2) Structure = heuristiques (lignes/zones). "
    "3) QC = alignement ORB + SSIM (différences simples). "
    "Pour du vrai temps réel vidéo, ajoute streamlit-webrtc; ici on reste frugal en mode image."
)
