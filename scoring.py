
import cv2
import numpy as np

# ----------------------------
# Utility
# ----------------------------
def normalize_score(raw: float, low: float, high: float) -> float:
    """Map raw value to [0,1] using low/high bounds."""
    raw = float(raw)
    if raw <= low:
        return 0.0
    if raw >= high:
        return 1.0
    return (raw - low) / (high - low)

def _moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    if x.size == 0:
        return x
    window = int(max(1, window))
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")

def _read_video_frames(path: str, seconds: float = 8.0, max_frames: int = 240):
    """Read up to `seconds` from a video file, capped by max_frames for speed."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0
    target = int(min(max_frames, max(8, round(seconds * fps))))
    frames = []
    while len(frames) < target:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 8:
        raise ValueError("Video is too short or unreadable. Please upload a clearer video (at least ~1–2 seconds).")
    return frames, fps

def _center_crop(frame, frac=0.6):
    """Simple, detector-free ROI: center crop."""
    h, w = frame.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x0 = max(0, (w - cw)//2)
    y0 = max(0, (h - ch)//2)
    roi = frame[y0:y0+ch, x0:x0+cw]
    return roi

# ----------------------------
# Layer 1: Physiological vitality (rPPG-like)
# ----------------------------
def compute_vitality(frames):
    """
    Extract a simple green-channel time series from a face-ish ROI (center crop),
    smooth it, then use variability as vitality_raw.
    """
    signal = []
    for f in frames:
        roi = _center_crop(f, frac=0.55)
        # green channel mean
        g = roi[:, :, 1].astype(np.float32)
        signal.append(float(g.mean()))
    signal = np.array(signal, dtype=np.float32)

    # preprocess
    centered = signal - float(signal.mean())
    smoothed = _moving_average(centered, window=7)
    vitality_raw = float(np.std(smoothed))

    # Normalize (MVP bounds)
    vitality_score = normalize_score(vitality_raw, low=0.5, high=3.0)
    return vitality_raw, vitality_score

# ----------------------------
# Layer 2: Visual anti-spoof (motion, texture, integrity)
# ----------------------------
def compute_motion(frames):
    """
    Motion consistency proxy (MVP):
    - compute mean absolute frame-to-frame difference (grayscale) on ROI
    - map to a raw motion value in a roughly 0..~3 range for normalize(low=0.1, high=2.0)
    """
    diffs = []
    prev = None
    for f in frames:
        roi = _center_crop(f, frac=0.55)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    diffs = np.array(diffs, dtype=np.float32)
    # raw: average diff scaled
    mean_diff = float(np.mean(diffs)) if diffs.size else 0.0
    motion_raw = mean_diff * 20.0  # scaling so typical webcam movement lands ~0.1..2.0
    motion_score = normalize_score(motion_raw, low=0.1, high=2.0)
    return motion_raw, motion_score

def compute_texture(frames):
    """
    Texture analysis proxy (MVP):
    - compute Laplacian variance on ROI from a representative frame (middle frame)
    - normalize with low=50, high=300 (as agreed)
    """
    mid = frames[len(frames)//2]
    roi = _center_crop(mid, frac=0.55)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture_score = normalize_score(texture_var, low=50, high=300)
    return texture_var, texture_score

def compute_integrity(frames):
    """
    Face integrity proxy (MVP):
    - use sharpness (Laplacian variance) on ROI; this acts as 'base quality metrics'
    - normalize with low=50, high=200 (as agreed)
    """
    mid = frames[len(frames)//2]
    roi = _center_crop(mid, frac=0.55)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    integrity_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    integrity_score = normalize_score(integrity_raw, low=50, high=200)
    return integrity_raw, integrity_score

def compute_visual(frames):
    motion_raw, motion_score = compute_motion(frames)
    texture_var, texture_score = compute_texture(frames)
    integrity_raw, integrity_score = compute_integrity(frames)

    # agreed weights
    visual_score = 0.40 * motion_score + 0.35 * texture_score + 0.25 * integrity_score
    visual_output = 1 if visual_score >= 0.75 else 0
    return {
        "motion_raw": motion_raw,
        "motion_score": motion_score,
        "texture_raw": texture_var,
        "texture_score": texture_score,
        "integrity_raw": integrity_raw,
        "integrity_score": integrity_score,
        "visual_score": visual_score,
        "visual_output": visual_output,
    }

# ----------------------------
# Layer 3: Contextual scoring (MVP simulation)
# ----------------------------
_CONTEXT_MAP = {
    "Trusted device (known / secure)": 3,
    "New device (not seen before)": 2,
    "Risky device (emulator / rooted / unknown)": 0,
    "Same city / normal network": 3,
    "Same country / slightly unusual": 2,
    "Different country / suspicious network": 0,
    "Normal transaction behavior": 3,
    "Somewhat unusual": 2,
    "Highly unusual": 0,
}

def compute_context(device_choice: str, geo_choice: str, behavior_choice: str):
    device_raw = _CONTEXT_MAP.get(device_choice, 2)
    geo_raw = _CONTEXT_MAP.get(geo_choice, 2)
    behavior_raw = _CONTEXT_MAP.get(behavior_choice, 2)

    device_score = normalize_score(device_raw, low=0, high=3)
    geo_score = normalize_score(geo_raw, low=0, high=3)
    behavior_score = normalize_score(behavior_raw, low=0, high=3)

    # agreed example weights
    context_score = 0.40 * behavior_score + 0.35 * device_score + 0.25 * geo_score
    contextual_output = 1 if context_score >= 0.70 else 0

    return {
        "device_raw": device_raw,
        "geo_raw": geo_raw,
        "behavior_raw": behavior_raw,
        "device_score": device_score,
        "geo_score": geo_score,
        "behavior_score": behavior_score,
        "context_score": context_score,
        "contextual_output": contextual_output,
    }

# ----------------------------
# Final fusion (40/40/20)
# ----------------------------
def run_mvp(video_path: str, device_choice: str, geo_choice: str, behavior_choice: str):
    frames, fps = _read_video_frames(video_path, seconds=8.0, max_frames=240)

    vitality_raw, vitality_score = compute_vitality(frames)
    visual = compute_visual(frames)
    context = compute_context(device_choice, geo_choice, behavior_choice)

    final_score = 0.40 * vitality_score + 0.40 * visual["visual_score"] + 0.20 * context["context_score"]
    final_output = 1 if final_score >= 0.75 else 0

    return {
        "fps": fps,
        "vitality_raw": vitality_raw,
        "vitality_score": vitality_score,
        **visual,
        **context,
        "final_score": final_score,
        "final_output": final_output,
    }
