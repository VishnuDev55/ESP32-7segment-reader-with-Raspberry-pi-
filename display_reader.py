"""
display_reader.py
─────────────────────────────────────────────────────────────────────────────
High-accuracy 7-segment display digitiser for ESP32-CAM MJPEG streams.

TWO recognition engines run in parallel for every digit:
  1. Segment-state pixel counting  (fast, ~2 ms/digit)
  2. Synthetic template matching   (robust to threshold noise, ~10 ms/digit)
  If they agree → high confidence. If they disagree → template wins.

Majority voting across 3 consecutive frames eliminates single-frame errors.
Runs fully headless — auto-detects ROI, prints values to terminal, logs CSV.
Automatically falls back to headless mode if no display is available for --gui.

Usage:
  python display_reader.py                              # headless, auto-detect ROI
  python display_reader.py --gui                        # with OpenCV debug windows
  python display_reader.py --roi 100,150,200,80         # manual ROI override (x,y,w,h)
  python display_reader.py --url http://IP:81/stream    # custom ESP32-CAM IP
  python display_reader.py --interval 5                 # read every 5 s (default)
  python display_reader.py --log readings.csv           # custom log file
  python display_reader.py --once                       # single read, then exit
  python display_reader.py --max-digits 4               # limit expected digit count

Requirements:  pip install opencv-python numpy
"""

import cv2
import numpy as np
import csv
import json
import os
import platform
import threading
import argparse
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="7-Segment Display Digitiser")
    p.add_argument("--url",       default="http://192.168.137.219:81/stream",
                   help="ESP32-CAM MJPEG stream URL")
    p.add_argument("--log",       default="display_log.csv",
                   help="Output CSV file path")
    p.add_argument("--interval",  type=float, default=5.0,
                   help="Seconds between reads (default 5.0)")
    p.add_argument("--roi",       default=None,
                   help="Manual ROI as x,y,w,h  e.g. 100,150,200,80")
    p.add_argument("--roi-file",  default="display_roi.json",
                   help="JSON file to cache detected ROI between runs")
    p.add_argument("--gui",       action="store_true",
                   help="Show OpenCV debug windows (use on laptop for setup)")
    p.add_argument("--debug-dir", default=".",
                   help="Directory to save debug snapshots when pressing 's'")
    p.add_argument("--value-min", type=float, default=0.0,
                   help="Minimum plausible reading value (default 0)")
    p.add_argument("--value-max", type=float, default=95.0,
                   help="Maximum plausible reading value (default 95)")
    p.add_argument("--max-digits", type=int, default=5,
                   help="Maximum number of digits expected on the display (default 5)")
    p.add_argument("--once", action="store_true",
                   help="Take a single reading and exit (no voting, useful for testing)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
SCALE_FACTOR       = 3        # upsample ROI before processing
CLAHE_CLIP         = 3.0
CLAHE_GRID         = (4, 4)
BLUR_KERNEL        = (3, 3)
MORPH_CLOSE_K      = 3        # fills broken segment gaps
MORPH_OPEN_K       = 2        # removes noise dots
BORDER_TRIM        = 0.06     # fraction trimmed from each edge before contour search
MIN_DIGIT_H_FRAC   = 0.25     # digit must be > 25% of ROI height to be considered
MIN_AREA_FRAC      = 0.0002   # minimum contour area fraction (noise floor)
NARROW_ASPECT      = 0.40     # pre-classify as '1' if width/height < this
SEG_THRESHOLD      = 0.25     # pixel density to call a segment ON
SEG_THRESHOLD_MID  = 0.45     # middle segment (g) is stricter — prevents 0 → 8
MAX_FUZZY_DIFF     = 1        # max segment mismatches for fuzzy fallback (strict)
TEMPLATE_W         = 42       # digit template width  (pixels)
TEMPLATE_H         = 72       # digit template height (pixels)
TEMPLATE_MIN_SCORE = 0.45     # minimum TM_CCOEFF_NORMED score to trust template
VOTE_WINDOW        = 3        # number of frames to buffer for majority vote
VOTE_MIN_AGREE     = 2        # frames that must agree to trigger a log entry
BUF_MAX_BYTES      = 65536    # stream byte-buffer cap (prevents memory leak)
ROI_REDETECT_SEC   = 60       # seconds between automatic ROI re-detection attempts
ROI_DRIFT_FAIL_LIMIT = 6      # consecutive failed/out-of-range reads before re-checking ROI position


# ─────────────────────────────────────────────────────────────────
# 7-SEGMENT LOOKUP TABLE
# Tuple key: (a, b, c, d, e, f, g)
#   a = top          b = top-right     c = bottom-right
#   d = bottom       e = bottom-left   f = top-left    g = middle
# ─────────────────────────────────────────────────────────────────
SEG_MAP: dict[tuple, str] = {
    (1,1,1,1,1,1,0): '0',
    (0,1,1,0,0,0,0): '1',   # standard right-bar 1
    (0,0,0,0,1,1,0): '1',   # left-bar only 1
    (0,1,1,0,0,1,0): '1',   # double-bar 1 (no middle)
    (0,1,1,0,1,1,0): '1',   # double-bar 1 variant
    (0,0,1,0,0,1,0): '1',   # minimal double-bar
    (0,0,1,0,1,0,0): '1',   # another variant
    (1,1,0,1,1,0,1): '2',
    (1,1,1,1,0,0,1): '3',
    (0,1,1,0,0,1,1): '4',
    (1,0,1,1,0,1,1): '5',
    (1,0,1,1,1,1,1): '6',
    (1,1,1,0,0,0,0): '7',
    (1,1,1,1,1,1,1): '8',
    (1,1,1,1,0,1,1): '9',
    (0,0,0,0,0,0,0): ' ',
}

SEG_ZONES: dict[str, tuple] = {
    'a': (0.12, 0.00, 0.88, 0.18),   # top
    'b': (0.72, 0.12, 1.00, 0.48),   # top-right
    'c': (0.72, 0.52, 1.00, 0.88),   # bottom-right
    'd': (0.12, 0.82, 0.88, 1.00),   # bottom
    'e': (0.00, 0.52, 0.28, 0.88),   # bottom-left
    'f': (0.00, 0.12, 0.28, 0.48),   # top-left
    'g': (0.25, 0.42, 0.75, 0.58),   # middle
}

SEG_THRESHOLDS: dict[str, float] = {
    seg: (SEG_THRESHOLD_MID if seg == 'g' else SEG_THRESHOLD)
    for seg in 'abcdefg'
}


# ─────────────────────────────────────────────────────────────────
# SYNTHETIC TEMPLATE GENERATOR
# Renders clean reference digits using OpenCV line drawing.
# No external files needed — fully self-contained.
# ─────────────────────────────────────────────────────────────────
def _generate_templates(w: int = TEMPLATE_W, h: int = TEMPLATE_H) -> dict[str, np.ndarray]:
    """
    Generate synthetic 7-segment digit templates (0–9).
    Each template is a (h, w) uint8 image: white background, black segments.
    """
    thickness = max(3, w // 8)

    # Segment endpoint definitions as fractions of (w, h)
    seg_coords = {
        'a': ((0.15, 0.03), (0.85, 0.03)),   # top horizontal
        'b': ((0.92, 0.06), (0.92, 0.46)),   # top-right vertical
        'c': ((0.92, 0.54), (0.92, 0.94)),   # bottom-right vertical
        'd': ((0.15, 0.97), (0.85, 0.97)),   # bottom horizontal
        'e': ((0.08, 0.54), (0.08, 0.94)),   # bottom-left vertical
        'f': ((0.08, 0.06), (0.08, 0.46)),   # top-left vertical
        'g': ((0.15, 0.50), (0.85, 0.50)),   # middle horizontal
    }

    # Active segments for each digit
    digit_segs = {
        '0': 'abcdef',
        '1': 'bc',
        '2': 'abdeg',
        '3': 'abcdg',
        '4': 'bcfg',
        '5': 'acdfg',
        '6': 'acdefg',
        '7': 'abc',
        '8': 'abcdefg',
        '9': 'abcdfg',
    }

    templates: dict[str, np.ndarray] = {}
    for digit, active in digit_segs.items():
        img = np.full((h, w), 255, dtype=np.uint8)
        for seg in active:
            (x0f, y0f), (x1f, y1f) = seg_coords[seg]
            p0 = (int(x0f * w), int(y0f * h))
            p1 = (int(x1f * w), int(y1f * h))
            cv2.line(img, p0, p1, 0, thickness)
        # Dilate slightly to make matching more forgiving
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
        img = cv2.erode(img, k, iterations=1)
        templates[digit] = img
    return templates


TEMPLATES: dict[str, np.ndarray] = _generate_templates()


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────
def preprocess(crop: np.ndarray) -> np.ndarray:
    """
    crop (BGR) → binary (uint8, white bg, black segments).
    Steps: grayscale → 3× upsample → CLAHE → Gaussian blur →
           Otsu threshold → morphological close + open.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * SCALE_FACTOR, h * SCALE_FACTOR),
                      interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray  = clahe.apply(gray)
    gray  = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Guarantee white background / black segments
    if cv2.countNonZero(binary) < binary.size * 0.5:
        binary = cv2.bitwise_not(binary)

    # Close: bridges tiny gaps inside segments (broken LCD lines)
    ck = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ck)

    # Open: removes isolated noise dots without affecting segments
    ok = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_OPEN_K, MORPH_OPEN_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ok)

    return binary


# ─────────────────────────────────────────────────────────────────
# ENGINE 1 — SEGMENT-STATE PIXEL COUNTING
# ─────────────────────────────────────────────────────────────────
def _read_segment_state(binary: np.ndarray) -> tuple:
    """Sample the 7 zones and return (a,b,c,d,e,f,g) ON/OFF tuple."""
    h, w = binary.shape
    inv  = cv2.bitwise_not(binary)
    on   = {}
    for seg, (x0f, y0f, x1f, y1f) in SEG_ZONES.items():
        x0 = max(0, int(x0f * w));  y0 = max(0, int(y0f * h))
        x1 = max(x0+1, int(x1f*w)); y1 = max(y0+1, int(y1f*h))
        zone  = inv[y0:y1, x0:x1]
        ratio = cv2.countNonZero(zone) / zone.size if zone.size > 0 else 0
        on[seg] = 1 if ratio > SEG_THRESHOLDS[seg] else 0
    return (on['a'], on['b'], on['c'], on['d'], on['e'], on['f'], on['g'])


def engine_segment(binary: np.ndarray) -> tuple[str, float]:
    """
    Returns (character, confidence).
    1.0 = exact lookup match, 0.5 = fuzzy match, 0.0 = no match.
    """
    h, w = binary.shape

    # Aspect-ratio shortcut: very narrow contour must be '1'
    if h > 0 and (w / float(h)) < NARROW_ASPECT:
        return '1', 0.95

    segs = _read_segment_state(binary)
    char = SEG_MAP.get(segs)
    if char is not None:
        return char, 1.0

    # Fuzzy fallback: find closest match by Hamming distance
    best_char  = '?'
    best_score = 99
    for pattern, digit in SEG_MAP.items():
        if digit == ' ':
            continue
        dist = sum(a != b for a, b in zip(segs, pattern))
        if dist < best_score:
            best_score = dist
            best_char  = digit

    if best_score <= MAX_FUZZY_DIFF:
        return best_char, max(0.3, 0.5 - best_score * 0.1)
    return '?', 0.0


# ─────────────────────────────────────────────────────────────────
# ENGINE 2 — SYNTHETIC TEMPLATE MATCHING
# ─────────────────────────────────────────────────────────────────
def engine_template(binary: np.ndarray) -> tuple[str, float]:
    """
    Resize digit to (TEMPLATE_W × TEMPLATE_H) and compare against all
    10 synthetic templates using TM_CCOEFF_NORMED.
    Returns (best_digit_char, score).
    """
    resized    = cv2.resize(binary, (TEMPLATE_W, TEMPLATE_H),
                            interpolation=cv2.INTER_AREA)
    best_char  = '?'
    best_score = -1.0

    for digit, tmpl in TEMPLATES.items():
        result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score  = float(result[0][0])
        if score > best_score:
            best_score = score
            best_char  = digit

    if best_score >= TEMPLATE_MIN_SCORE:
        return best_char, best_score
    return '?', max(0.0, best_score)


# ─────────────────────────────────────────────────────────────────
# DUAL ENGINE MERGE
# ─────────────────────────────────────────────────────────────────
def decode_digit(digit_binary: np.ndarray) -> tuple[str, str]:
    """
    Run both engines and return (final_char, human-readable method string).

    Merge strategy:
      • Both agree                        → use agreed result (high confidence)
      • Disagree, template score ≥ min    → trust template (more robust)
      • Disagree, template uncertain      → trust segment if confidence ≥ 0.5
      • Both uncertain                    → return '?'
    """
    seg_char, seg_conf    = engine_segment(digit_binary)
    tmpl_char, tmpl_score = engine_template(digit_binary)

    if seg_char == tmpl_char and seg_char != '?':
        return seg_char, f"agree seg={seg_conf:.2f} tmpl={tmpl_score:.2f}"

    if tmpl_score >= TEMPLATE_MIN_SCORE and tmpl_char != '?':
        return tmpl_char, f"tmpl_wins({tmpl_char} vs seg={seg_char}) tmpl={tmpl_score:.2f}"

    if seg_conf >= 0.5 and seg_char != '?':
        return seg_char, f"seg_fallback({seg_char}) conf={seg_conf:.2f}"

    return '?', (f"unknown seg={seg_char}/{seg_conf:.2f} "
                 f"tmpl={tmpl_char}/{tmpl_score:.2f}")


# ─────────────────────────────────────────────────────────────────
# DIGIT FINDER
# ─────────────────────────────────────────────────────────────────
def find_digits(binary: np.ndarray, max_digits: int = 5):
    """
    Locate individual digit bounding boxes in a preprocessed binary image.
    Returns (digit_boxes, decimal_x_centers).
    digit_boxes: list of (x, y, w, h) sorted left-to-right.
    """
    img_h, img_w = binary.shape
    tx = int(img_w * BORDER_TRIM)
    ty = int(img_h * BORDER_TRIM)
    trimmed = binary[ty:img_h-ty, tx:img_w-tx]
    t_h, t_w = trimmed.shape

    inv      = cv2.bitwise_not(trimmed)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple] = []
    decimals:   set[int]    = set()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area   = w * h
        aspect = w / float(h) if h > 0 else 0

        # Skip noise
        if area < t_h * t_w * MIN_AREA_FRAC:
            continue

        # Decimal point: small, roughly square, in the lower third of the image
        if h < t_h * 0.22 and 0.2 < aspect < 4.0 and (y + h) > t_h * 0.35:
            decimals.add((x + tx) + w // 2)
            continue

        # Digit candidate: tall enough relative to ROI height
        if h > t_h * MIN_DIGIT_H_FRAC:
            candidates.append((x + tx, y + ty, w, h))

    # Keep the N tallest, then sort left-to-right
    candidates.sort(key=lambda d: d[3], reverse=True)
    top = candidates[:max_digits]
    top.sort(key=lambda d: d[0])

    # Deduplicate horizontally-overlapping contours (real digit vs edge noise)
    deduped: list[tuple] = []
    for d in top:
        dx, dy, dw, dh = d
        if deduped and dx < deduped[-1][0] + deduped[-1][2] * 0.6:
            if dh > deduped[-1][3]:
                deduped[-1] = d
        else:
            deduped.append(d)

    deduped.sort(key=lambda d: d[0])
    return deduped, decimals


# ─────────────────────────────────────────────────────────────────
# FULL DISPLAY DECODER
# ─────────────────────────────────────────────────────────────────
def decode_display(binary: np.ndarray, max_digits: int = 5) -> tuple[str | None, np.ndarray, list]:
    """
    Decode all digits in the binary image.
    Returns (value_string | None, annotated_vis_BGR, debug_info_list).
    """
    vis    = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    digits, decimals = find_digits(binary, max_digits=max_digits)
    debug  = []

    if not digits:
        return None, vis, ['no digits found']

    result: list[tuple[int, str]] = []

    for (x, y, w, h) in digits:
        # Pad the crop slightly to avoid edge clipping
        px0 = max(0, x - 3);            py0 = max(0, y - 3)
        px1 = min(binary.shape[1], x+w+3); py1 = min(binary.shape[0], y+h+3)
        roi = binary[py0:py1, px0:px1]

        char, method = decode_digit(roi)
        result.append((x, char))
        debug.append(f"x={x}: '{char}' | {method}")

        colour = (0, 200, 0) if char != '?' else (0, 0, 200)
        cv2.rectangle(vis, (x, y), (x+w, y+h), colour, 2)
        cv2.putText(vis, char, (x+2, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    # Assemble final string, injecting decimal points at correct positions
    full_str = ""
    for i, (x, ch) in enumerate(result):
        full_str += ch
        next_x = result[i+1][0] if i+1 < len(result) else 9999
        for dx in decimals:
            if x < dx < next_x:
                full_str += '.'
                cv2.circle(vis, (dx, int(binary.shape[0] * 0.85)),
                           6, (0, 100, 255), -1)
                break

    full_str = full_str.strip().replace(' ', '').rstrip('.')
    if not any(c.isdigit() for c in full_str):
        return None, vis, debug

    return full_str, vis, debug


# ─────────────────────────────────────────────────────────────────
# FRAME RESIZE (aspect-ratio preserving)
# Scales to a target width instead of forcing a fixed 4:3 box,
# so non-4:3 camera resolutions don't get stretched/distorted.
# ─────────────────────────────────────────────────────────────────
def resize_frame(frame: np.ndarray, target_w: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / float(w)
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────
# AUTO-ROI DETECTOR
# Finds the largest bright rectangle in the frame — works well
# when the display is the most prominent lit object in view.
# ─────────────────────────────────────────────────────────────────
def auto_detect_roi(frame: np.ndarray) -> tuple | None:
    """
    Try to locate the display automatically.
    Returns (x, y, w, h) or None if detection fails.
    """
    h, w  = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best       = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < h * w * 0.005 or area > h * w * 0.70:
            continue

        cx, cy, cw, ch_ = cv2.boundingRect(c)
        aspect = cw / float(ch_) if ch_ > 0 else 0

        # 7-segment displays are typically wider than tall
        if not (1.2 < aspect < 9.0):
            continue

        score = area * min(aspect / 3.0, 1.0)
        if score > best_score:
            best_score = score
            margin = 5
            best = (max(0, cx-margin), max(0, cy-margin),
                    min(w, cw+2*margin), min(h, ch_+2*margin))

    return best


# ─────────────────────────────────────────────────────────────────
# ROI JSON CACHE
# ─────────────────────────────────────────────────────────────────
def load_roi_cache(path: str) -> tuple | None:
    try:
        with open(path) as f:
            d = json.load(f)
        roi = tuple(d.get("roi", []))
        if len(roi) == 4:
            return roi
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def save_roi_cache(path: str, roi: tuple):
    with open(path, 'w') as f:
        json.dump({"roi": list(roi)}, f, indent=2)


# ─────────────────────────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────────────────────────
class CSVLogger:
    def __init__(self, path: str):
        self._path = path
        self._f    = None
        self._w    = None
        self._lock = threading.Lock()

    def open(self):
        self._f = open(self._path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow(["Timestamp", "Value", "Confidence", "Score"])
        self._f.flush()

    def log(self, value: str, confidence: str = "", score: float | None = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        score_str = f"{score:.2f}" if score is not None else ""
        with self._lock:
            self._w.writerow([ts, value, confidence, score_str])
            self._f.flush()
        # Safe ASCII-only terminal print
        print(f"  [LOG] [{ts}]  Value = {value}  ({confidence})")

    def close(self):
        if self._f:
            self._f.close()


# ─────────────────────────────────────────────────────────────────
# ZERO-LAG MJPEG STREAM CAPTURE
# Reads raw bytes from socket — no OpenCV buffering lag.
# ─────────────────────────────────────────────────────────────────
class StreamCapture:
    def __init__(self, url: str):
        self.url     = url
        self._frame  = None
        self._ret    = False
        self._lock   = threading.Lock()
        self._active = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        retry = 0.1
        while self._active:
            try:
                stream = urllib.request.urlopen(self.url, timeout=5)
                buf    = b''
                retry  = 0.1
                while self._active:
                    chunk = stream.read(4096)
                    # Cap buffer to prevent unbounded memory growth
                    if len(buf) > BUF_MAX_BYTES:
                        a = buf.rfind(b'\xff\xd8')
                        buf = buf[a:] if a != -1 else b''
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.find(b'\xff\xd8')
                    b = buf.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg   = buf[a:b+2]
                        buf   = buf[b+2:]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8),
                            cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._ret   = True
                                self._frame = frame
            except Exception as e:
                print(f"  [Stream] {e} – retry in {retry:.1f}s")
                time.sleep(retry)
                retry = min(retry * 2, 5.0)

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def release(self):
        self._active = False


# ─────────────────────────────────────────────────────────────────
# OCR WORKER  (runs in a background daemon thread)
# ─────────────────────────────────────────────────────────────────
def ocr_worker(crop: np.ndarray, state: dict, lock: threading.Lock,
               logger: CSVLogger, args, save_snap: bool = False,
               skip_vote: bool = False):
    """
    skip_vote: log the very first valid reading immediately instead of
    waiting for majority agreement across VOTE_WINDOW frames. Used by
    `--once` for a single confidence-free test read.
    """
    binary        = preprocess(crop)
    val, vis, dbg = decode_display(binary, max_digits=args.max_digits)

    if save_snap:
        cv2.imwrite(str(Path(args.debug_dir) / "snap_raw.jpg"),
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(Path(args.debug_dir) / "snap_processed.jpg"), binary)
        print(f"  [SNAP] Saved to {args.debug_dir}/snap_*.jpg")

    # Sanity-range check
    valid = False
    if val:
        try:
            fval  = float(val)
            valid = args.value_min <= fval <= args.value_max
            if not valid:
                print(f"  [SKIP] Out-of-range: {val} "
                      f"(expected {args.value_min}–{args.value_max})")
        except ValueError:
            print(f"  [SKIP] Non-numeric: {val}")

    do_log      = False
    log_val     = val
    confidence  = ""
    score       = None

    with lock:
        state['vis'] = vis
        state['dbg'] = dbg
        if valid:
            state['fail_streak'] = 0
            if skip_vote:
                state['value'] = val
                log_val        = val
                confidence     = "single read, no voting"
                do_log         = True
            else:
                state['vote_buf'].append(val)
                buf = state['vote_buf']
                if len(buf) >= VOTE_WINDOW:
                    counts        = Counter(buf[-VOTE_WINDOW:])
                    winner, freq  = counts.most_common(1)[0]
                    state['vote_buf'] = []          # reset window after decision
                    if freq >= VOTE_MIN_AGREE:
                        state['value'] = winner
                        log_val        = winner
                        score          = freq / VOTE_WINDOW
                        confidence     = f"{freq}/{VOTE_WINDOW} frames agree"
                        do_log         = True
                    else:
                        print(f"  [VOTE] No consensus {dict(counts)} – skipping")
        else:
            state['fail_streak'] = state.get('fail_streak', 0) + 1
        state['running'] = False

    if do_log:
        logger.log(log_val, confidence, score=score)
    elif not val:
        print(f"  [X] Decode failed. Debug: {dbg}")


# ─────────────────────────────────────────────────────────────────
# DISPLAY AVAILABILITY CHECK
# Some OpenCV/Qt builds call abort() instead of raising a catchable
# cv2.error when no display server is present (e.g. SSH into a
# headless Raspberry Pi). A try/except around cv2.namedWindow can't
# recover from that, so check for a display *before* calling any
# cv2 GUI function at all.
# ─────────────────────────────────────────────────────────────────
def gui_available() -> bool:
    if platform.system() == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # Windows / macOS normally have a GUI session available


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve GUI availability up front so a missing display fails fast
    # with a clear message instead of crashing (or aborting) mid-run.
    gui_active = args.gui
    if gui_active and not gui_available():
        print("  [WARN] No display detected (DISPLAY not set).")
        print("  [WARN] Falling back to headless mode.")
        gui_active = False
    elif gui_active:
        try:
            cv2.namedWindow("Display Reader", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Display Reader")
        except cv2.error as e:
            print(f"  [WARN] Could not open a display window ({e}).")
            print("  [WARN] Falling back to headless mode.")
            gui_active = False

    print("\n  7-Segment Display Reader")
    print(f"  Stream   : {args.url}")
    print(f"  Interval : {args.interval}s")
    print(f"  Log      : {args.log}")
    print(f"  Mode     : {'GUI' if gui_active else 'Headless'}")
    print()

    # Connect to stream
    print(f"  Connecting -> {args.url} ...")
    stream = StreamCapture(args.url)
    time.sleep(2.0)

    ret, test = stream.read()
    if not ret or test is None:
        alt = args.url.rsplit(':', 1)[0] + "/"
        print(f"  Stream not ready – trying {alt}")
        stream.release()
        stream = StreamCapture(alt)
        time.sleep(2.0)
        ret, test = stream.read()
        if not ret or test is None:
            print("  ERROR: Cannot open stream. Check IP/power.")
            sys.exit(1)
    print("  Stream OK!\n")

    # ── Resolve ROI (CLI > cache file > auto-detect) ─────────────
    roi = None
    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(','))
            print(f"  [ROI] From CLI: {roi}")
        except ValueError:
            print("  [WARN] Invalid --roi format. Falling back to auto-detect.")

    if roi is None:
        roi = load_roi_cache(args.roi_file)
        if roi:
            print(f"  [ROI] Loaded from cache ({args.roi_file}): {roi}")

    if roi is None:
        print("  [ROI] No cached ROI – attempting auto-detection...")
        _, frm = stream.read()
        if frm is not None:
            frm = resize_frame(frm)
            roi = auto_detect_roi(frm)
            if roi:
                print(f"  [ROI] Auto-detected: {roi}")
                save_roi_cache(args.roi_file, roi)
            else:
                print("  [ROI] Auto-detection failed.")
                if not gui_active:
                    print("  Hint: run once with --gui to draw the ROI manually.")
    # ─────────────────────────────────────────────────────────────

    logger = CSVLogger(args.log)
    logger.open()

    lock  = threading.Lock()
    state = {
        'value':       '--',
        'running':     False,
        'vis':         None,
        'dbg':         [],
        'vote_buf':    [],
        'fail_streak': 0,
    }

    # ── Single-shot mode: one read, no voting, then exit ─────────
    if args.once:
        if not roi:
            print("  ERROR: No ROI available for a single read "
                  "(auto-detect failed and no --roi given).")
            logger.close()
            stream.release()
            sys.exit(1)

        ret, frame = stream.read()
        if not ret or frame is None:
            print("  ERROR: Could not grab a frame from the stream.")
            logger.close()
            stream.release()
            sys.exit(1)

        frame      = resize_frame(frame)
        x, y, w, h = roi
        crop       = frame[y:y+h, x:x+w].copy()
        with lock:
            state['running'] = True
        ocr_worker(crop, state, lock, logger, args, save_snap=False, skip_vote=True)

        with lock:
            result = state['value']
        print(f"  Reading: {result}")
        logger.close()
        stream.release()
        return
    # ─────────────────────────────────────────────────────────────

    last_ocr        = 0.0
    last_roi_detect = time.time()

    if gui_active:
        print("""  +----------------------------------------------------+
  |  GUI Controls                                      |
  |  1 -> Draw ROI around display digits               |
  |  r -> Re-run auto ROI detection                    |
  |  s -> Save debug snapshot                          |
  |  q -> Quit                                         |
  +----------------------------------------------------+""")
    else:
        print("  Running headless. Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = resize_frame(frame)
            now   = time.time()

            # Periodic auto-ROI re-detection if ROI still unknown
            if roi is None and (now - last_roi_detect) > ROI_REDETECT_SEC:
                roi = auto_detect_roi(frame)
                if roi:
                    print(f"  [ROI] Auto-detected: {roi}")
                    save_roi_cache(args.roi_file, roi)
                last_roi_detect = now

            # Re-check ROI position if reads keep failing/out-of-range —
            # most likely the camera or display has physically shifted.
            with lock:
                fail_streak = state['fail_streak']
            if roi and fail_streak >= ROI_DRIFT_FAIL_LIMIT:
                print(f"  [ROI] {fail_streak} consecutive bad reads – "
                      f"re-checking display position...")
                detected = auto_detect_roi(frame)
                if detected:
                    roi = detected
                    save_roi_cache(args.roi_file, roi)
                    print(f"  [ROI] Re-detected: {roi}")
                with lock:
                    state['fail_streak'] = 0

            # Trigger OCR on interval
            with lock:
                already = state['running']

            if roi and not already and (now - last_ocr) > args.interval:
                x, y, w, h = roi
                crop = frame[y:y+h, x:x+w].copy()
                with lock:
                    state['running'] = True
                threading.Thread(
                    target=ocr_worker,
                    args=(crop, state, lock, logger, args, False),
                    daemon=True
                ).start()
                last_ocr = now

            # ── GUI mode ────────────────────────────────────────
            if gui_active:
                try:
                    display = frame.copy()

                    if roi:
                        x, y, w, h = roi
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 0), 2)
                        with lock:
                            val_txt = state['value']
                        cv2.putText(display, f"Value: {val_txt}", (x, max(y-10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
                    else:
                        cv2.putText(display, "Press '1' to draw ROI", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

                    with lock:
                        vis_img = state['vis']
                    if vis_img is not None:
                        cv2.imshow("7-Seg Debug", vis_img)
                    cv2.imshow("Display Reader", display)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    elif key == ord('1'):
                        print("\n  Draw a box around the digits, then press ENTER.")
                        r = cv2.selectROI("Display Reader", display,
                                          fromCenter=False, showCrosshair=True)
                        if r[2] > 10 and r[3] > 10:
                            roi = r
                            save_roi_cache(args.roi_file, roi)
                            print(f"  [ROI] Set to: {roi}")
                        else:
                            print("  Cancelled.")

                    elif key == ord('r'):
                        detected = auto_detect_roi(frame)
                        if detected:
                            roi = detected
                            save_roi_cache(args.roi_file, roi)
                            print(f"  [ROI] Re-detected: {roi}")
                        else:
                            print("  [ROI] Auto-detection failed.")

                    elif key == ord('s'):
                        if roi:
                            x, y, w, h = roi
                            snap_crop = frame[y:y+h, x:x+w].copy()
                            snap_bin  = preprocess(snap_crop)
                            cv2.imwrite(str(Path(args.debug_dir) / "snap_raw.jpg"),
                                        cv2.cvtColor(snap_crop, cv2.COLOR_BGR2GRAY))
                            cv2.imwrite(str(Path(args.debug_dir) / "snap_processed.jpg"),
                                        snap_bin)
                            print(f"  [SNAP] Saved to {args.debug_dir}/snap_*.jpg")
                        else:
                            cv2.imwrite(str(Path(args.debug_dir) / "snap_raw.jpg"),
                                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                except cv2.error as e:
                    print(f"  [WARN] Display window lost ({e}). Switching to headless mode.")
                    gui_active = False

            else:
                # Headless: sleep to yield CPU
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n  Stopped.")

    finally:
        stream.release()
        logger.close()
        if gui_active:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print(f"\n  Done. All readings saved to: {args.log}")


if __name__ == "__main__":
    main()
