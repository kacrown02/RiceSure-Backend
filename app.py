# ===========================================================
# RiceSure Backend — v11.7
#  • Balanced 5-class SavedModel backend
#  • Stronger separation of dominant variety:
#       - Probability sharpening (gamma) for per-grain
#         confidence (top-1 ↑, others ↓)
#       - NEW soft-dominance smoothing when:
#            * softmax says tray is very close to pure
#            * but raw majority fraction is only ~60–70%
#         (borderline grains are pulled to dominant variety)
#       - Existing single-variety smoothing when tray is
#         already very pure (≥86% by counts)
#       - Purity = 90% majority vote + 10% softmax confidence
#  • Gentle aspect-based refinement for trio:
#         {7-TONNER, JASMINE, SINANDOMENG}
#       - SHORT  (aspect ≤ 2.0)  → favor 7-TONNER
#       - MEDIUM (2.0–2.8)      → favor JASMINE
#       - LONG   (aspect > 2.8) → favor SINANDOMENG
#  • No hard anti-7T routing, no global per-class bias
#  • Overlay: green box + label per grain
#
# Endpoints:
#   GET  /api/health
#   GET  /api/ping
#   GET  /api/check_mapping
#   POST /api/analyze
#   POST /api/diag_once
#   GET  /uploads/<file>
#   GET  /debug/<file>
# ===========================================================

import os
import io
import json
import uuid
import time
import logging
import contextlib

from contextlib import contextmanager
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
from PIL import Image, ImageOps

import tensorflow as tf

# -------------------- Quiet TF / env -----------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only by default

# -------------------- Paths --------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DEBUG_DIR  = os.path.join(BASE_DIR, "debug")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR,  exist_ok=True)

# -------------------- Config -------------------------------
CONFIG = {
    "SAVE_DEBUG_MASKS": True,
    "MIN_GRAINS": 10,
    "PURITY_THRESHOLD": 92.0,
}

# Sharpening + smoothing hyper-params
#   • Higher gamma => top-1 probability more dominant
PROB_GAMMA = float(os.environ.get("PROB_GAMMA", "3.0"))

# NEW soft-dominance smoothing (for trays that *look* pure by softmax)
DOM_SOFT_MIN_GRAINS     = 18    # need enough grains
DOM_SOFT_MIN_MAJ_FRAC   = 0.60  # majority by counts at least 60%
DOM_SOFT_MIN_SOFT_FRAC  = 0.85  # majority by softmax ≥ 85%
DOM_SOFT_MIN_MAJ_PROB   = 0.50  # per-grain prob for majority class
DOM_SOFT_MARGIN         = 0.05  # p_maj ≥ p_curr - 0.05

# Single-variety smoothing: more aggressive when already very pure
SINGLE_VAR_MIN_FRAC       = 0.86   # majority fraction≥86%
SINGLE_VAR_MIN_MAJ_PROB   = 0.40   # per-grain prob for majority
SINGLE_VAR_MAX_OTHER_PROB = 0.70   # max prob for current label
SINGLE_VAR_PROB_MARGIN    = 0.20   # p_majority ≥ p_other - margin
SINGLE_VAR_MIN_GRAINS     = 18

# -------------------- URL helper ---------------------------
def build_public_url(rel_path: str) -> str:
    host = request.headers.get("Host", "").strip()
    xf_proto = request.headers.get("X-Forwarded-Proto", "").strip()
    scheme = xf_proto or getattr(request, "scheme", "http") or "http"
    if not host:
        return rel_path
    return f"{scheme}://{host}{rel_path}"

# Segmentation / contour filters (class-agnostic)
PAD            = 6
MIN_AREA_FRAC  = 0.00045
MIN_AREA_PX    = 90
MAX_AREA_FRAC  = 0.33
ASPECT_MIN     = 0.18
ASPECT_MAX     = 7.2
SOLIDITY_MIN   = 0.80
EXTENT_MIN     = 0.30
EXTENT_MAX     = 0.97

SEG_MAX_DIM = 1700

ROT_LIST_FEW  = [0, 1, 2, 3]
ROT_LIST_MID  = [0, 2]
ROT_LIST_MANY = [0]
CROWD_MID     = 40
CROWD_MANY    = 80

os.environ.setdefault("RESIZE_ONLY", "1")
PREPROC_MODE = os.environ.get("PREPROC_MODE", "eff").strip().lower()

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preproc  # type: ignore
except Exception:
    eff_preproc = None

# ===========================================================
# Model & labels auto-discovery
# ===========================================================

def _has_savedmodel(dirpath: str) -> bool:
    if not os.path.isfile(os.path.join(dirpath, "saved_model.pb")):
        return False
    vdir = os.path.join(dirpath, "variables")
    return os.path.isdir(vdir) and any(f.startswith("variables.") for f in os.listdir(vdir))

def _pick_labels_file(files: set, dirpath: str):
    if "label_map.json" in files:
        return os.path.join(dirpath, "label_map.json")
    if "label_names.txt" in files:
        return os.path.join(dirpath, "label_names.txt")
    return None

def find_model_bundle(root: str):
    best = None
    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        has_sm = _has_savedmodel(dirpath)
        if not has_sm:
            continue
        labels_fp = _pick_labels_file(files, dirpath)
        name_bonus = 30 if dirpath.lower().endswith("_savedmodel") else 0
        score = 100 + name_bonus + (3 if labels_fp else 0)
        cand = {
            "dir": dirpath,
            "has_savedmodel": has_sm,
            "labels_fp": labels_fp,
            "mtime": os.path.getmtime(dirpath),
            "score": score,
        }
        if (best is None or
            cand["score"] > best["score"] or
            (cand["score"] == best["score"] and cand["mtime"] > best["mtime"])):
            best = cand
    if best is None:
        raise FileNotFoundError(f"No SavedModel found under {root}")
    return best

_BUNDLE   = find_model_bundle(MODELS_DIR)
SAVED_DIR = _BUNDLE["dir"]
LABELS_FP = _BUNDLE["labels_fp"]

print("Resolved model bundle:")
print("  DIR       :", SAVED_DIR)
print("  SavedModel:", True)
print("  labels    :", LABELS_FP)

def resolve_labels_path():
    env_p = os.environ.get("LABELS_PATH", "").strip()
    if env_p and os.path.isfile(env_p):
        return env_p
    if LABELS_FP and os.path.isfile(LABELS_FP):
        return LABELS_FP
    candidate = os.path.join(SAVED_DIR, "label_map.json")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError("label_map.json / label_names.txt not found near SavedModel.")

# ===========================================================
# Preprocess helpers
# ===========================================================

def _gray_world_norm(img):
    eps = 1e-6
    avg = np.mean(img.reshape(-1, 3), axis=0) + eps
    scale = np.mean(avg) / avg
    out = np.clip(img * scale, 0, 255).astype(np.uint8)
    return out

def _crop_norm(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _gray_world_norm(rgb2)

def apply_preprocess(x_uint8_or_float):
    if PREPROC_MODE == "eff" and eff_preproc is not None:
        x = x_uint8_or_float.astype("float32")
        return eff_preproc(x)
    elif PREPROC_MODE == "none":
        return x_uint8_or_float.astype("float32")
    else:
        return (x_uint8_or_float.astype("float32") / 255.0)

def _letterbox_rgb(img_rgb, size):
    h, w = img_rgb.shape[:2]
    scale_ = min(size / h, size / w)
    nh, nw = int(round(h * scale_)), int(round(w * scale_))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    if out.shape[:2] != (size, size):
        out = cv2.resize(out, (size, size))
    return out

# ===========================================================
# Segmentation helpers (class-agnostic)
# ===========================================================

def _downscale_for_seg(bgr):
    H, W = bgr.shape[:2]
    long_side = max(H, W)
    if long_side > SEG_MAX_DIM:
        scale = long_side / SEG_MAX_DIM
        return cv2.resize(bgr, (int(W / scale), int(H / scale)), cv2.INTER_AREA), scale
    return bgr, 1.0

def _score_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Hs, Ws = mask.shape
    imgA = Hs * Ws
    good = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if 0.0005 * imgA < a < 0.06 * imgA:
            x, y, w, h = cv2.boundingRect(c)
            asp = w / max(h, 1)
            if ASPECT_MIN <= asp <= ASPECT_MAX:
                good += 1
    return good

def segment_mask(bgr, dbg_prefix=None):
    small, scale = _downscale_for_seg(bgr)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_otsu = 255 - otsu
    adap = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )

    cand = [
        (otsu, _score_mask(otsu)),
        (inv_otsu, _score_mask(inv_otsu)),
        (adap, _score_mask(adap)),
    ]
    cand.sort(key=lambda t: t[1], reverse=True)
    bw = cand[0][0]

    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kc = int(max(3, min(5, round(max(small.shape) // 500))))
    kerC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))

    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kerC)

    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    big = any(s[cv2.CC_STAT_AREA] > 0.07 * bw.size for s in stats[1:])
    if num <= 3 or big:
        try:
            dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
            _, peaks = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
            peaks = peaks.astype(np.uint8)
            markers, _ = cv2.connectedComponents(peaks)
            markers = cv2.watershed(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), markers.astype(np.int32))
            bw = np.where(markers > 1, 255, 0).astype(np.uint8)
        except Exception:
            pass

    if CONFIG["SAVE_DEBUG_MASKS"] and dbg_prefix:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{dbg_prefix}_mask.png"), bw)

    return bw, scale

def keep_contour(c, W, H):
    x, y, ww, hh = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    imgA = W * H

    if area < max(MIN_AREA_FRAC * imgA, MIN_AREA_PX):
        return None
    if area > MAX_AREA_FRAC * imgA:
        return None

    aspect = ww / max(hh, 1)
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return None

    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull) + 1e-6
    if area / ha < SOLIDITY_MIN:
        return None

    extent = area / float(ww * hh + 1e-6)
    if not (EXTENT_MIN <= extent <= EXTENT_MAX):
        return None

    x0 = max(0, x - PAD)
    y0 = max(0, y - PAD)
    x1 = min(W, x + ww + PAD)
    y1 = min(H, y + hh + PAD)
    return (x0, y0, x1, y1)

def oriented_aspect_ratio(contour) -> float:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w <= 0 or h <= 0:
        return 1.0
    return float(max(w, h) / max(min(w, h), 1.0))

# ===========================================================
# Labels & model wrapper
# ===========================================================

def _load_class_names(path):
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            try:
                idxs = sorted(int(k) for k in obj.keys())
            except Exception:
                idxs = list(obj.keys())
            return [str(obj[str(i)]) for i in idxs]
        raise ValueError("Unrecognized label_map.json structure")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

_MODEL = None
_CLASS = None
_CLASS_IDX = None
_IMG_SIZE = None

@contextmanager
def silent_io():
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield

def _build_from_savedmodel(saved_dir: str):
    sm = tf.saved_model.load(saved_dir)
    keys = list(sm.signatures.keys())
    endpoint = "serving_default" if "serving_default" in keys else keys[0]
    fn = sm.signatures[endpoint]

    in_name, in_spec = next(iter(fn.structured_input_signature[1].items()))
    h = int(in_spec.shape[1])
    w = int(in_spec.shape[2])
    if h != w:
        raise ValueError(f"Non-square SavedModel input: {in_spec.shape}")

    layer = tf.keras.layers.TFSMLayer(saved_dir, call_endpoint=endpoint)

    def _call_tfsm(x):
        try:
            out = layer(**{in_name: x})
        except TypeError:
            out = layer(x)
        return next(iter(out.values())) if isinstance(out, dict) else out

    x = tf.keras.Input(shape=(h, w, 3), dtype=tf.float32)
    y = _call_tfsm(x)
    model = tf.keras.Model(x, y, name="rice_cls_savedmodel")
    return model, h

def ensure_model(force=False):
    global _MODEL, _CLASS, _IMG_SIZE, _CLASS_IDX
    if _MODEL is not None and _CLASS is not None and _IMG_SIZE is not None and not force:
        return

    labels_path = resolve_labels_path()
    print("Using labels:", labels_path)
    _CLASS = _load_class_names(labels_path)
    _CLASS_IDX = {name: i for i, name in enumerate(_CLASS)}
    print("Label order:", _CLASS)

    with silent_io():
        _MODEL, _IMG_SIZE = _build_from_savedmodel(SAVED_DIR)

    try:
        _MODEL.predict(np.zeros((1, _IMG_SIZE, _IMG_SIZE, 3), dtype=np.float32), verbose=0)
    except Exception:
        pass

    print("✅ Model ready. IMG_SIZE:", _IMG_SIZE, "Classes:", _CLASS)

def get_img_size():
    ensure_model()
    return int(_IMG_SIZE)

# ===========================================================
# Prediction helpers (balanced + sharpening + smoothing)
# ===========================================================

def _predict_batch(batch_np):
    with silent_io():
        preds = _MODEL.predict(batch_np, verbose=0)
    return np.asarray(preds)

def _sharpen_probs(probs: np.ndarray, gamma: float = PROB_GAMMA) -> np.ndarray:
    p = np.maximum(probs, 1e-8)
    p = p ** float(gamma)
    return p / (p.sum() + 1e-8)

def _refine_jas_sin_7t(probs: np.ndarray, aspect: float) -> np.ndarray:
    """
    Gentle aspect-based refinement for the trio {7-TONNER, JASMINE, SINANDOMENG}
    using your description:
      - 7-TONNER   : quite short
      - JASMINE    : medium length
      - SINANDOMENG: slightly longer

    Only used when none already has > 0.9 prob.
    """
    if _CLASS is None or _CLASS_IDX is None:
        return probs

    tri_names = ["7-TONNER", "JASMINE", "SINANDOMENG"]
    if any(name not in _CLASS_IDX for name in tri_names):
        return probs

    idx7  = _CLASS_IDX["7-TONNER"]
    idxJ  = _CLASS_IDX["JASMINE"]
    idxSi = _CLASS_IDX["SINANDOMENG"]

    tri_idx = [idx7, idxJ, idxSi]
    tri_p = probs[tri_idx].copy()

    if tri_p.max() > 0.90:
        return probs

    if aspect <= 2.0:
        mul7, mulJ, mulSi = 1.08, 0.97, 0.95
    elif aspect <= 2.8:
        mul7, mulJ, mulSi = 0.97, 1.06, 0.97
    else:
        mul7, mulJ, mulSi = 0.95, 0.97, 1.08

    tri_p[0] *= mul7
    tri_p[1] *= mulJ
    tri_p[2] *= mulSi

    probs_ref = probs.copy()
    probs_ref[tri_idx] = tri_p
    probs_ref = probs_ref / (probs_ref.sum() + 1e-8)
    return probs_ref

def _classify_crop(crop_rgb, size, rot_list, aspect=None):
    crop_rgb = _crop_norm(crop_rgb)

    if os.environ.get("RESIZE_ONLY") == "1":
        img_in = cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA)
    else:
        img_in = _letterbox_rgb(crop_rgb, size)

    img_in = apply_preprocess(img_in)

    tta_imgs = []
    for krot in rot_list:
        rimg = np.rot90(img_in, krot)
        tta_imgs.append(rimg)
        tta_imgs.append(rimg[:, ::-1, :])
    tta_imgs.append(img_in[::-1, :, :])

    batch = np.stack(tta_imgs, axis=0)
    preds = _predict_batch(batch)
    avg_probs = preds.mean(axis=0)
    probs = _sharpen_probs(avg_probs, PROB_GAMMA)

    if aspect is not None:
        probs = _refine_jas_sin_7t(probs, aspect)

    return probs

def detect_grains_and_predict(pil_img: Image.Image, want_debug=False):
    ensure_model()
    dbg_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    size = get_img_size()

    rgb_full = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))
    H, W = rgb_full.shape[:2]
    bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)

    mask_small, scale = segment_mask(bgr_full, dbg_prefix=dbg_id)
    cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > CROWD_MANY:
        rot_list = ROT_LIST_MANY
    elif len(cnts) > CROWD_MID:
        rot_list = ROT_LIST_MID
    else:
        rot_list = ROT_LIST_FEW

    grain_records = []

    for c_small in cnts:
        c_full = (c_small * scale).astype(np.int32)
        r = keep_contour(c_full, W, H)
        if r is None:
            continue
        x0, y0, x1, y1 = r
        crop = rgb_full[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        asp = oriented_aspect_ratio(c_full)
        probs = _classify_crop(crop, size, rot_list, aspect=asp)

        k = int(np.argmax(probs))

        grain_records.append({
            "box": (int(x0), int(y0), int(x1), int(y1)),
            "aspect": float(asp),
            "probs": probs,
            "chosen_k": k,
            "smoothed_from": None,
        })

    total = len(grain_records)
    if total < CONFIG["MIN_GRAINS"]:
        return {
            "verdict": "INSUFFICIENT_GRAINS",
            "total_grains": total,
            "counts_per_class": {cls: 0 for cls in _CLASS},
            "debug_overlay": None,
        }

    grain_labels = np.array([rec["chosen_k"] for rec in grain_records], dtype=np.int32)
    probs_matrix = np.stack([rec["probs"] for rec in grain_records], axis=0)
    votes = np.bincount(grain_labels, minlength=len(_CLASS))
    soft_votes = probs_matrix.sum(axis=0)

    maj_idx = int(np.argmax(votes))
    maj_frac = votes[maj_idx] / max(total, 1)
    soft_frac_maj = float(soft_votes[maj_idx]) / float(soft_votes.sum() + 1e-8)

    # -------------------------------------------------------
    # Stage 1: NEW soft-dominance smoothing
    #   • Activates when softmax says tray is almost pure
    #     even if counts are only ~60–70% majority.
    #   • Reassigns borderline grains to the majority class.
    # -------------------------------------------------------
    if (
        total >= DOM_SOFT_MIN_GRAINS and
        maj_frac >= DOM_SOFT_MIN_MAJ_FRAC and
        soft_frac_maj >= DOM_SOFT_MIN_SOFT_FRAC
    ):
        for i, rec in enumerate(grain_records):
            k_curr = int(grain_labels[i])
            if k_curr == maj_idx:
                continue
            p_curr = float(rec["probs"][k_curr])
            p_maj  = float(rec["probs"][maj_idx])
            if (
                p_maj >= DOM_SOFT_MIN_MAJ_PROB and
                p_maj >= p_curr - DOM_SOFT_MARGIN
            ):
                if rec["smoothed_from"] is None:
                    rec["smoothed_from"] = k_curr
                grain_labels[i] = maj_idx

        votes = np.bincount(grain_labels, minlength=len(_CLASS))
        maj_idx = int(np.argmax(votes))
        maj_frac = votes[maj_idx] / max(total, 1)

    # -------------------------------------------------------
    # Stage 2: existing single-variety smoothing when tray
    #          is already very pure by counts (≥86%).
    # -------------------------------------------------------
    if (total >= SINGLE_VAR_MIN_GRAINS) and (maj_frac >= SINGLE_VAR_MIN_FRAC):
        for i, rec in enumerate(grain_records):
            k_curr = int(grain_labels[i])
            if k_curr == maj_idx:
                continue
            p_curr = float(rec["probs"][k_curr])
            p_maj  = float(rec["probs"][maj_idx])
            if (
                p_maj >= SINGLE_VAR_MIN_MAJ_PROB and
                p_curr < SINGLE_VAR_MAX_OTHER_PROB and
                p_maj >= p_curr - SINGLE_VAR_PROB_MARGIN
            ):
                if rec["smoothed_from"] is None:
                    rec["smoothed_from"] = k_curr
                grain_labels[i] = maj_idx

        votes = np.bincount(grain_labels, minlength=len(_CLASS))
        maj_idx = int(np.argmax(votes))
        maj_frac = votes[maj_idx] / max(total, 1)

    maj_name = _CLASS[maj_idx]
    purity_vote = 100.0 * votes[maj_idx] / max(total, 1)
    purity_soft = 100.0 * float(soft_votes[maj_idx]) / float(soft_votes.sum() + 1e-8)

    # Stronger emphasis on majority vote
    purity = 0.9 * purity_vote + 0.1 * purity_soft

    eff_threshold = CONFIG["PURITY_THRESHOLD"]
    verdict = "PURE" if purity >= eff_threshold else "ADULTERATED"

    overlay = rgb_full.copy()
    for rec, k_final in zip(grain_records, grain_labels):
        x0, y0, x1, y1 = rec["box"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            overlay, _CLASS[int(k_final)], (x0, max(10, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
        )

    overlay_name = f"{dbg_id}_overlay.jpg"
    cv2.imwrite(
        os.path.join(DEBUG_DIR, overlay_name),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    rel  = f"/debug/{overlay_name}"
    full = build_public_url(rel)

    counts_dict = {cls: int(v) for cls, v in zip(_CLASS, votes)}

    result = {
        "counts_per_class": counts_dict,
        "total_grains": int(total),
        "majority_variety": maj_name,
        "purity_percent": round(purity, 2),
        "purity_detail": {
            "vote": round(purity_vote, 2),
            "soft": round(purity_soft, 2),
            "metric": "0.9_vote_0.1_soft_sharpened_smoothing",
            "majority_fraction": round(maj_frac * 100.0, 2),
            "majority_soft_fraction": round(soft_frac_maj * 100.0, 2),
            "prob_gamma": PROB_GAMMA,
        },
        "verdict": verdict,
        "purity_threshold": CONFIG["PURITY_THRESHOLD"],
        "effective_threshold": eff_threshold,
        "debug_overlay": overlay_name,
        "debug_overlay_path": rel,
        "debug_overlay_url": full,
        "elapsed_ms": elapsed_ms,
    }

    if want_debug:
        per_grain = []
        for rec, k_final in zip(grain_records, grain_labels):
            probs = rec["probs"]
            order = np.argsort(probs)
            k1, k2 = int(order[-1]), int(order[-2])
            info = {
                "box": [int(v) for v in rec["box"]],
                "aspect": float(round(rec["aspect"], 3)),
                "top1": {
                    "k": k1,
                    "name": _CLASS[k1],
                    "p": float(round(probs[k1], 4)),
                },
                "top2": {
                    "k": k2,
                    "name": _CLASS[k2],
                    "p": float(round(probs[k2], 4)),
                },
                "chosen": {
                    "k": int(k_final),
                    "name": _CLASS[int(k_final)],
                },
            }
            if rec["smoothed_from"] is not None:
                info["smoothed_from"] = int(rec["smoothed_from"])
            per_grain.append(info)
        result["per_grain_top2"] = per_grain[:160]

    return result

# ===========================================================
# Flask app & routes
# ===========================================================

logging.getLogger("werkzeug").disabled = True
app = Flask(__name__)
app.logger.disabled = True
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

@app.get("/api/health")
def health():
    return jsonify({"ok": True}), 200

@app.get("/api/ping")
def ping():
    try:
        ensure_model()
        return jsonify({
            "pong": True,
            "classes": _CLASS,
            "model_img_size": get_img_size(),
            "backend": "TF/SavedModel",
            "purity_threshold": CONFIG["PURITY_THRESHOLD"],
            "balanced": True,
            "prob_gamma": PROB_GAMMA,
        }), 200
    except Exception as e:
        return jsonify({"pong": False, "error": str(e)}), 500

@app.get("/api/check_mapping")
def check_mapping():
    try:
        ensure_model()
        return jsonify({
            "backend": "TF/SavedModel",
            "classes_in_backend_order": _CLASS,
            "model_img_size": get_img_size(),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/diag_once")
def diag_once():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no_file"}), 400

        f = request.files["file"]
        safe = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
        dst = os.path.join(UPLOAD_DIR, safe)
        f.save(dst)

        with Image.open(dst) as pil:
            result = detect_grains_and_predict(pil, want_debug=True)

        return jsonify(result), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "diag_failed", "detail": str(e)}), 500

@app.get("/uploads/<path:filename>")
def get_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.get("/debug/<path:filename>")
def get_debug(filename):
    return send_from_directory(DEBUG_DIR, filename, as_attachment=False)

@app.post("/api/analyze")
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no_file"}), 400

        f = request.files["file"]
        safe = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
        dst = os.path.join(UPLOAD_DIR, safe)
        f.save(dst)

        with Image.open(dst) as pil:
            result = detect_grains_and_predict(pil, want_debug=False)

        rel_img = f"/uploads/{safe}"
        full_img = build_public_url(rel_img)

        result.update({
            "original_image": full_img,
            "original_image_path": rel_img,
        })

        return jsonify(result), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "analysis_failed", "detail": str(e)}), 500

# ===========================================================
# Entrypoint
# ===========================================================

if __name__ == "__main__":  
    ensure_model(force=True)
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
