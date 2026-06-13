from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
OUTPUTS_DIR   = ROOT / "outputs"
SIGNALS_DIR   = OUTPUTS_DIR / "signals"
FEATURES_DIR  = OUTPUTS_DIR / "features"
MODELS_DIR    = OUTPUTS_DIR / "models"
RESULTS_DIR   = OUTPUTS_DIR / "results"

# ── Source data ──────────────────────────────────────────────────────────────
RAW_INDEX_CSV   = Path("/data2/sandeep/HB_METER_REBUILD/features/feature_index.csv")
VIDEO_ROOT      = Path("/data2/sandeep/AIIMS_DATA/ALL_VIDEOS")
INDEX_CSV       = DATA_DIR / "index.csv"          # rebuilt index with protocol tag
USABLE_CSV      = DATA_DIR / "usable.csv"         # after saturation filter

# ── Video / recording constants ──────────────────────────────────────────────
# 6-second protocol: ~30 fps, 3 × 2 s segments
PROTO_6S_MAX_DURATION  = 8.0    # seconds — videos shorter than this are 6s protocol
PROTO_30S_MIN_DURATION = 20.0   # seconds — videos longer than this are 30s protocol

# LED segment order: 0=RED(660nm)  1=ORANGE(610nm)  2=YELLOW(590nm)
N_SEGMENTS = 3

# ROI: center crop fraction of the frame
ROI_Y = (0.30, 0.70)
ROI_X = (0.30, 0.70)

# Frame quality thresholds
MIN_BRIGHTNESS = 5.0     # discard near-black frames
MAX_BRIGHTNESS = 253.0   # discard fully saturated frames

# R-channel saturation flag: if mean R > this in any segment → video is saturated
R_SAT_THRESHOLD = 240.0

# ── Feature extraction ───────────────────────────────────────────────────────
WINDOW_FRAMES   = 40    # number of frames per segment window
# PPG / heart-rate frequency band (Hz)
HR_BAND_LOW  = 0.5
HR_BAND_HIGH = 3.0

# Channels extracted per frame from ROI
CHANNELS = ["R", "G", "B", "RC", "GC", "RGN"]
#   RC  = R / (R+G+B)   chromaticity red
#   GC  = G / (R+G+B)   chromaticity green
#   RGN = (R-G)/(R+G)   normalised redness

# ── Augmentation ────────────────────────────────────────────────────────────
AUG_TEMPORAL_WINDOWS_6S  = 5   # how many windows to sample per 6s video
AUG_TEMPORAL_WINDOWS_30S = 7   # how many windows to sample per 30s video
AUG_BRIGHTNESS_RANGE     = (0.85, 1.15)
AUG_NOISE_SIGMA_FRAC     = 0.01   # sigma = this × mean_channel_value
AUG_JITTER_FRAMES        = 3      # ± frames to shift segment boundary
AUG_COPIES_PER_VIDEO     = 3      # noise+brightness copies in addition to temporal

# ── Training ─────────────────────────────────────────────────────────────────
CV_FOLDS    = 5
RANDOM_SEED = 42

# WHO 2011 anemia thresholds (g/dL) — used for labels, not model input
# keys: (sex, age_group) — 'F'=female, 'M'=male, 'C'=child (<12 yr), 'P'=pregnant
WHO_THRESHOLDS = {
    "child":    11.0,
    "pregnant": 11.0,
    "female":   12.0,
    "male":     13.0,
}
DEFAULT_ANEMIA_THRESHOLD = 12.0   # used when sex/age unknown

# ── Evaluation ────────────────────────────────────────────────────────────────
# Clinical accuracy bands for regression (g/dL)
ACCURACY_BANDS = [0.5, 1.0, 1.5, 2.0]
# HB ranges for per-bucket error analysis
HB_BUCKETS = [(0, 7), (7, 10), (10, 12), (12, 25)]
