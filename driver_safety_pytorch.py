import cv2
import time
import threading
import numpy as np
from deepface import DeepFace
import face_alignment
from scipy.spatial import distance as dist

# ═══════════════════════════════════════════════════════════════════
#  PyTorch version — uses face_alignment (Fan model, PyTorch backend)
#
#  face_alignment runs a deep CNN (Face Alignment Network) trained
#  on 300W dataset to regress 68 landmark coordinates directly from
#  the image. This is fundamentally different from dlib which uses
#  a gradient boosting regression tree — the CNN is more robust to
#  lighting, partial occlusion, and head pose.
#
#  Everything downstream (EAR, hysteresis, blink state machine,
#  PERCLOS, fatigue scoring) is identical to the dlib version.
#
#  Landmark indices (same convention as dlib 68-point model):
#    Left eye:  36–41
#    Right eye: 42–47
# ═══════════════════════════════════════════════════════════════════

LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

EAR_CLOSED = 0.20
EAR_OPEN   = 0.25

MIN_FACE_COVERAGE  = 0.40
PERCLOS_TIRED      = 0.15
PERCLOS_VERY_TIRED = 0.25
BLINK_MIN_FRAMES   = 2
BLINK_MAX_FRAMES   = 15
MICROSLEEP_FRAMES  = 15
DEEPFACE_INTERVAL  = 2.0

DURATION_OPTIONS = [30, 60, 90, 120]

GREEN  = (0, 220, 100);  ORANGE = (0, 165, 255);  RED    = (0, 0, 220)
YELLOW = (0, 200, 255);  WHITE  = (255, 255, 255); GRAY   = (160, 160, 160)
BLACK  = (0, 0, 0);      DARK   = (25, 25, 35)

# Initialise the PyTorch face alignment model
# device='cuda' if you have a GPU, otherwise 'cpu'
# On Apple Silicon MPS is available: device='mps'
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device=DEVICE,
    flip_input=False,
)

deepface_result  = []
deepface_lock    = threading.Lock()
deepface_running = False

L, R = 0, 1


def compute_ear(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices], dtype=np.float32)
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def draw_eye(frame, landmarks, indices, color):
    pts = np.array([landmarks[i] for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 1)


def run_deepface(frame):
    global deepface_result, deepface_running
    try:
        r = DeepFace.analyze(frame, actions=["emotion"],
                             enforce_detection=False, silent=True)
        with deepface_lock:
            deepface_result = r
    except Exception:
        pass
    finally:
        deepface_running = False


# ── Scoring ──────────────────────────────────────────────────────
def compute_score(perclos_l, perclos_r, microsleeps_total, dominant_emotion):
    worst = max(perclos_l, perclos_r)
    score = 0
    if worst > PERCLOS_VERY_TIRED:
        score += 3
    elif worst > PERCLOS_TIRED:
        score += 1
    score += min(microsleeps_total, 3)
    if dominant_emotion in ("sad", "fear", "angry"):
        score += 1
    return score


def verdict_from_score(score):
    if score <= 1:
        return "READY TO DRIVE",     GREEN
    elif score <= 3:
        return "TAKE A BREAK FIRST", ORANGE
    else:
        return "DO NOT DRIVE",       RED


def perclos_color(p):
    if p < PERCLOS_TIRED:
        return GREEN
    elif p < PERCLOS_VERY_TIRED:
        return ORANGE
    return RED


# ── Drawing helpers ───────────────────────────────────────────────
def progress_bar(frame, x, y, w, h, pct, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y),
                  (x + int(w * min(max(pct, 0.0), 1.0)), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BLACK, 1)


def centered_text(canvas, text, y, font_scale, color, thickness=1):
    w = canvas.shape[1]
    (tw, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(canvas, text, (w // 2 - tw // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def hdivider(canvas, y, color=(80, 80, 80)):
    cv2.line(canvas, (40, y), (canvas.shape[1] - 40, y), color, 1)


# ── Duration selection screen ─────────────────────────────────────
def duration_select_screen(cap):
    selected = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        centered_text(frame, "DRIVER FATIGUE ASSESSMENT", h // 2 - 130, 0.95, WHITE, 2)
        centered_text(frame, "Select assessment duration:", h // 2 - 80, 0.65, GRAY)

        box_w, box_h = 120, 55
        gap     = 20
        total_w = len(DURATION_OPTIONS) * box_w + (len(DURATION_OPTIONS) - 1) * gap
        start_x = (w - total_w) // 2
        btn_y   = h // 2 - 40

        for i, secs in enumerate(DURATION_OPTIONS):
            bx     = start_x + i * (box_w + gap)
            is_sel = (i == selected)
            col    = GREEN if is_sel else (60, 60, 60)
            cv2.rectangle(frame, (bx, btn_y),
                          (bx + box_w, btn_y + box_h), col, -1 if is_sel else 2)
            cv2.rectangle(frame, (bx, btn_y),
                          (bx + box_w, btn_y + box_h), WHITE if is_sel else GRAY, 1)
            label = (f"{secs}s" if secs < 60
                     else f"{secs // 60}m" if secs % 60 == 0 else f"{secs}s")
            tc = BLACK if is_sel else WHITE
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.putText(frame, label,
                        (bx + box_w // 2 - tw // 2, btn_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, tc, 2)
            hint = f"[{i + 1}]"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, hint,
                        (bx + box_w // 2 - hw // 2, btn_y + box_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        centered_text(frame, "ENTER = start     Q = quit",
                      btn_y + box_h + 60, 0.55, GRAY)
        cv2.imshow("Driver Fatigue Assessment", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            return None
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            selected = key - ord("1")
        elif key in (13, 10):
            return DURATION_OPTIONS[selected]


# ── Results screen ────────────────────────────────────────────────
def results_screen(shape, stats, dominant_emotion, duration):
    h, w   = shape[:2]
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)

    pl, pr  = stats["perclos"]
    bl, br  = stats["blinks"]
    ml, mr  = stats["microsleeps"]
    elapsed = stats["elapsed"]
    bpm_l   = bl / max(elapsed / 60, 1 / 60)
    bpm_r   = br / max(elapsed / 60, 1 / 60)
    ms_tot  = ml + mr

    score            = compute_score(pl, pr, ms_tot, dominant_emotion)
    verdict, v_color = verdict_from_score(score)

    cv2.rectangle(canvas, (0, 0), (w, 80), v_color, -1)
    (tw, _), _ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(canvas, verdict, (w // 2 - tw // 2, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
    cv2.putText(canvas, f"{duration}s assessment",
                (w - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    col_label = 30;  col_l = 270;  col_r = 410;  bar_w = 110
    y = 108

    cv2.putText(canvas, "Metric",    (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GRAY,  1)
    cv2.putText(canvas, "LEFT EYE",  (col_l, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 1)
    cv2.putText(canvas, "RIGHT EYE", (col_r, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 1)
    y += 8;  hdivider(canvas, y);  y += 22

    cv2.putText(canvas, "PERCLOS", (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GRAY, 1)
    cv2.putText(canvas, f"{pl*100:.1f}%", (col_l, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, perclos_color(pl), 2)
    cv2.putText(canvas, f"{pr*100:.1f}%", (col_r, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, perclos_color(pr), 2)
    y += 12
    progress_bar(canvas, col_l, y, bar_w, 7, pl / 0.40, perclos_color(pl))
    progress_bar(canvas, col_r, y, bar_w, 7, pr / 0.40, perclos_color(pr))
    y += 28

    ml_col = GREEN if ml == 0 else (ORANGE if ml < 3 else RED)
    mr_col = GREEN if mr == 0 else (ORANGE if mr < 3 else RED)
    cv2.putText(canvas, "Microsleeps", (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GRAY, 1)
    cv2.putText(canvas, str(ml), (col_l, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, ml_col, 2)
    cv2.putText(canvas, str(mr), (col_r, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, mr_col, 2)
    y += 12
    progress_bar(canvas, col_l, y, bar_w, 7, ml / 6, ml_col)
    progress_bar(canvas, col_r, y, bar_w, 7, mr / 6, mr_col)
    y += 28

    cv2.putText(canvas, "Blinks/min",   (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GRAY,  1)
    cv2.putText(canvas, f"{bpm_l:.1f}", (col_l, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
    cv2.putText(canvas, f"{bpm_r:.1f}", (col_r, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
    y += 34

    cv2.putText(canvas, "Blinks total", (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GRAY,  1)
    cv2.putText(canvas, str(bl),         (col_l, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
    cv2.putText(canvas, str(br),         (col_r, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
    y += 34

    hdivider(canvas, y);  y += 22

    score_col = GREEN if score <= 1 else (ORANGE if score <= 3 else RED)
    for label, value, col in [
        ("Dominant mood", dominant_emotion, WHITE),
        ("Fatigue score", f"{score} / 7",  score_col),
    ]:
        cv2.putText(canvas, label + ":", (col_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GRAY, 1)
        cv2.putText(canvas, value,       (col_l, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.72, col,  2)
        y += 36

    worst     = "LEFT" if pl >= pr else "RIGHT"
    worst_col = perclos_color(max(pl, pr))
    cv2.putText(canvas, f"Score uses worst eye  ({worst}  {max(pl,pr)*100:.1f}%)",
                (col_label, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, worst_col, 1)

    centered_text(canvas, "R = retry    Q = quit", h - 20, 0.5, GRAY)
    return canvas


# ── Main ──────────────────────────────────────────────────────────
def run():
    global deepface_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Driver Fatigue Assessment — PyTorch FAN landmark detection")

    while True:

        chosen_duration = duration_select_screen(cap)
        if chosen_duration is None:
            break

        eye_open_frames   = [0, 0]
        eye_blink_count   = [0, 0]
        eye_microsleeps   = [0, 0]
        eye_consec_closed = [0, 0]
        eye_is_open       = [True, True]

        total_frames       = 0
        face_frames        = 0
        emotion_counts     = {}
        last_emotion       = "neutral"
        last_deepface_time = 0.0
        start_time         = time.time()
        ear_vals           = [None, None]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            if elapsed >= chosen_duration:
                break

            total_frames += 1

            # ── PyTorch landmark detection ────────────────────────
            # face_alignment expects RGB
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_lmarks = fa.get_landmarks(rgb)  # list of (68,2) arrays or None

            face_detected = all_lmarks is not None and len(all_lmarks) > 0
            prev_open     = list(eye_is_open)

            if face_detected:
                face_frames += 1
                lmarks = all_lmarks[0]  # first face, shape (68, 2)

                ear_l = compute_ear(lmarks, LEFT_EYE_IDX)
                ear_r = compute_ear(lmarks, RIGHT_EYE_IDX)
                ear_vals = [ear_l, ear_r]

                # Hysteresis
                for side, e in ((L, ear_l), (R, ear_r)):
                    if e < EAR_CLOSED:
                        eye_is_open[side] = False
                    elif e > EAR_OPEN:
                        eye_is_open[side] = True

                # PERCLOS numerator
                for side in (L, R):
                    if eye_is_open[side]:
                        eye_open_frames[side] += 1

                # Blink / microsleep state machine
                for side in (L, R):
                    if not eye_is_open[side]:
                        eye_consec_closed[side] += 1
                        if eye_consec_closed[side] == MICROSLEEP_FRAMES:
                            eye_microsleeps[side] += 1
                    else:
                        if not prev_open[side]:
                            cc = eye_consec_closed[side]
                            if BLINK_MIN_FRAMES <= cc <= BLINK_MAX_FRAMES:
                                eye_blink_count[side] += 1
                        eye_consec_closed[side] = 0

                # Draw landmarks and eye outlines
                for side, indices, ear_val in [
                    (L, LEFT_EYE_IDX,  ear_l),
                    (R, RIGHT_EYE_IDX, ear_r),
                ]:
                    col = GREEN if eye_is_open[side] else RED
                    draw_eye(frame, lmarks, indices, col)
                    pt  = (int(lmarks[indices[0]][0]),
                           int(lmarks[indices[0]][1]) - 8)
                    lbl = f"{'L' if side == L else 'R'} {ear_val:.2f}"
                    cv2.putText(frame, lbl, pt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

                # Draw all 68 landmarks as tiny dots
                for i, (x, y_pt) in enumerate(lmarks):
                    cv2.circle(frame, (int(x), int(y_pt)), 1, GRAY, -1)

            else:
                for side in (L, R):
                    eye_consec_closed[side] = 0
                    eye_is_open[side]       = True
                ear_vals = [None, None]

            # DeepFace background thread
            now = time.time()
            if now - last_deepface_time >= DEEPFACE_INTERVAL and not deepface_running:
                last_deepface_time = now
                deepface_running   = True
                threading.Thread(target=run_deepface,
                                 args=(frame.copy(),), daemon=True).start()

            with deepface_lock:
                if deepface_result:
                    last_emotion = deepface_result[0].get("dominant_emotion", "neutral")
                    emotion_counts[last_emotion] = \
                        emotion_counts.get(last_emotion, 0) + 1

            # HUD
            sf  = max(face_frames, 1)
            pl  = 1.0 - (eye_open_frames[L] / sf)
            pr  = 1.0 - (eye_open_frames[R] / sf)
            rem = int(chosen_duration - elapsed)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), DARK, -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, f"ASSESSING — {rem}s remaining",
                        (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
            progress_bar(frame, 0, 56, frame.shape[1], 9,
                         elapsed / chosen_duration, GREEN)

            if not face_detected:
                cv2.putText(frame, "Keep face in frame", (12, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
            else:
                bpm_l  = eye_blink_count[L] / max(elapsed / 60, 1 / 60)
                bpm_r  = eye_blink_count[R] / max(elapsed / 60, 1 / 60)
                el_str = f"{ear_vals[L]:.2f}" if ear_vals[L] is not None else "---"
                er_str = f"{ear_vals[R]:.2f}" if ear_vals[R] is not None else "---"
                hud = [
                    (f"EAR          L:{el_str}  R:{er_str}", WHITE),
                    (f"PERCLOS      L:{pl*100:.0f}%  R:{pr*100:.0f}%",
                     perclos_color(max(pl, pr))),
                    (f"Microsleeps  L:{eye_microsleeps[L]}  R:{eye_microsleeps[R]}",
                     RED if (eye_microsleeps[L] + eye_microsleeps[R]) else GREEN),
                    (f"Blinks total L:{eye_blink_count[L]}  R:{eye_blink_count[R]}", WHITE),
                    (f"Blinks/min   L:{bpm_l:.0f}  R:{bpm_r:.0f}", WHITE),
                    (f"Mood: {last_emotion}", WHITE),
                    (f"Device: {DEVICE}", GRAY),
                ]
                pad      = 6
                row_h    = 28
                panel_x1 = 8
                panel_y1 = 70
                panel_x2 = 320
                panel_y2 = panel_y1 + len(hud) * row_h + pad
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (panel_x1, panel_y1),
                              (panel_x2, panel_y2), DARK, -1)
                cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

                yh = panel_y1 + row_h - pad
                for txt, col in hud:
                    cv2.putText(frame, txt, (panel_x1 + 6, yh),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
                    yh += row_h

            cv2.imshow("Driver Fatigue Assessment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Session end
        if face_frames < total_frames * MIN_FACE_COVERAGE:
            blank = np.full(frame.shape, 20, dtype=np.uint8)
            centered_text(blank, "Not enough face data — press R to retry",
                          blank.shape[0] // 2, 0.75, YELLOW, 2)
            cv2.imshow("Driver Fatigue Assessment", blank)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            continue

        elapsed_total = time.time() - start_time
        perclos_final = [
            1.0 - (eye_open_frames[L] / max(face_frames, 1)),
            1.0 - (eye_open_frames[R] / max(face_frames, 1)),
        ]
        dominant_emotion = (max(emotion_counts, key=emotion_counts.get)
                            if emotion_counts else "neutral")
        stats = {
            "perclos":     perclos_final,
            "blinks":      eye_blink_count,
            "microsleeps": eye_microsleeps,
            "face_frames": face_frames,
            "elapsed":     elapsed_total,
        }
        results = results_screen(frame.shape, stats, dominant_emotion, chosen_duration)

        while True:
            cv2.imshow("Driver Fatigue Assessment", results)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("r"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run()