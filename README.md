# 🚗 Driver Fatigue Assessment System

A real-time, webcam-based driver fatigue detection system that uses computer vision and machine learning to assess whether you're safe to drive — before you get behind the wheel.

---

## Overview

This system monitors eye behavior and facial expressions through your webcam, computing industry-standard fatigue metrics (PERCLOS, microsleeps, blink rate) and generating a **fatigue score** with a clear driving recommendation. Session history is stored locally and a separate analysis script detects chronic fatigue patterns over time using scikit-learn.

---

## Features

- **Real-time EAR tracking** — Eye Aspect Ratio computed per frame with hysteresis to reduce noise
- **PERCLOS measurement** — Percentage of Eye Closure, tracked independently for left and right eyes
- **Microsleep detection** — Eyes closed for 15+ consecutive frames triggers a microsleep event
- **Blink rate monitoring** — Blinks per minute, with valid blink window (2–15 frames)
- **Emotion analysis** — DeepFace runs asynchronously every 2 seconds to detect dominant mood
- **Fatigue scoring** — 0–7 score maps to one of three verdicts: Ready / Take a Break / Do Not Drive
- **Session history** — All sessions saved to a local SQLite database with a browsable in-app history screen
- **Longitudinal analysis** — Separate script performs trend detection, anomaly detection, circadian pattern analysis, and a personalized Random Forest classifier

---

## Two Detection Backends

| File | Backend | Notes |
|---|---|---|
| `fatigue_dlib.py` | **dlib** (gradient boosting trees) | Faster, CPU-friendly, requires `.dat` model file |
| `fatigue_pytorch.py` | **face_alignment** (FAN, PyTorch CNN) | More robust to lighting/pose, GPU-accelerated |

Both files share identical downstream logic — EAR, blink state machine, PERCLOS, and scoring are the same.

---

## Scoring System

| Score | Verdict | Meaning |
|---|---|---|
| 0–1 | ✅ READY TO DRIVE | Normal fatigue levels |
| 2–3 | ⚠️ TAKE A BREAK FIRST | Elevated fatigue detected |
| 4–7 | 🚫 DO NOT DRIVE | Dangerous fatigue levels |

**Score breakdown:**

- `+3` if worst-eye PERCLOS > 25%
- `+1` if worst-eye PERCLOS > 15%
- `+1–3` for microsleep events (capped at 3)
- `+1` if dominant emotion is `sad`, `fear`, or `angry`

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/driver-fatigue-assessment.git
cd driver-fatigue-assessment
```

### 2. Install dependencies

```bash
pip install opencv-python deepface dlib scipy numpy pandas scikit-learn
```

For the PyTorch backend, also install:

```bash
pip install torch face-alignment
```

### 3. Download the dlib shape predictor (dlib backend only)

```bash
# Download from dlib's model zoo
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place `shape_predictor_68_face_landmarks.dat` in the same directory as `fatigue_dlib.py`.

---

## Usage

### Run the assessment (dlib backend)

```bash
python fatigue_dlib.py
```

### Run the assessment (PyTorch backend)

```bash
python fatigue_pytorch.py
```

### Run longitudinal analysis (requires 5+ sessions)

```bash
python fatigue_analysis.py
```

### Controls

| Key | Action |
|---|---|
| `1` `2` `3` `4` | Select assessment duration (30s / 60s / 90s / 120s) |
| `Enter` | Start assessment |
| `R` | Retry after results |
| `H` | View session history |
| `Q` | Quit |

Mouse clicks also work on all buttons.

---

## Fatigue Analysis Script

`fatigue_analysis.py` runs independently and performs six analyses on your session history:

1. **Summary stats** — Overall score distribution and averages
2. **Trend detection** — Linear regression to detect worsening or improving patterns
3. **Circadian analysis** — Average fatigue by time of day and day of week
4. **Anomaly detection** — Isolation Forest flags statistical outlier sessions
5. **Personal baseline classifier** — Random Forest trained on your own data; flags sessions that are unusual *for you specifically*, not against a fixed threshold
6. **Chronic risk score** — Composite risk score with actionable recommendations

> **Note:** The personal classifier requires 8+ sessions to train. Anomaly detection requires 10+. The more sessions you log, the more personalized the model becomes.

---

## How It Works

```
Webcam frame
    │
    ├─ dlib / face_alignment → 68 facial landmarks
    │        │
    │        ├─ LEFT_EYE_IDX  (36–41) → EAR_L → open/closed state
    │        └─ RIGHT_EYE_IDX (42–47) → EAR_R → open/closed state
    │
    ├─ EAR < 0.20 → eye closed
    │  EAR > 0.25 → eye open  (hysteresis band)
    │
    ├─ Closed 2–15 frames → BLINK
    │  Closed 15+ frames  → MICROSLEEP
    │
    ├─ PERCLOS = 1 - (open_frames / face_frames)
    │
    ├─ DeepFace [background thread, every 2s] → dominant_emotion
    │
    └─ compute_score() → verdict → results screen → saved to SQLite
```

---

## Project Structure

```
driver-fatigue-assessment/
├── fatigue_dlib.py          # Main app — dlib backend
├── fatigue_pytorch.py       # Main app — PyTorch FAN backend
├── fatigue_analysis.py      # Longitudinal analysis script
├── shape_predictor_68_face_landmarks.dat   # dlib model (download separately)
└── fatigue_sessions.db      # SQLite database (auto-created on first run)
```

---

## Requirements

- Python 3.8+
- Webcam
- ~500 MB disk space (for PyTorch models on first run)

### Key dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture and rendering |
| `dlib` | Face detection + 68-point landmark regression |
| `face_alignment` | PyTorch FAN landmark detection (alternative backend) |
| `deepface` | Emotion recognition |
| `scipy` | Euclidean distance for EAR computation |
| `scikit-learn` | Linear regression, Isolation Forest, Random Forest |
| `pandas` | Session data manipulation |
| `sqlite3` | Local session storage (stdlib) |

---

## Thresholds Reference

| Parameter | Value | Meaning |
|---|---|---|
| `EAR_CLOSED` | 0.20 | Eye is considered closed |
| `EAR_OPEN` | 0.25 | Eye is considered open (hysteresis) |
| `PERCLOS_TIRED` | 15% | Mild fatigue threshold |
| `PERCLOS_VERY_TIRED` | 25% | Severe fatigue threshold |
| `BLINK_MIN_FRAMES` | 2 | Minimum frames for a valid blink |
| `BLINK_MAX_FRAMES` | 15 | Maximum frames before classified as microsleep |
| `MICROSLEEP_FRAMES` | 15 | Frames of eye closure = microsleep |
| `MIN_FACE_COVERAGE` | 40% | Minimum % of session with face detected |
| `DEEPFACE_INTERVAL` | 2.0s | How often emotion analysis runs |

---

## Disclaimer

This tool is for informational purposes only. It is **not a medical device** and should not be used as the sole basis for any safety-critical decision. Always exercise personal judgment about your fitness to drive.

---

## License

MIT