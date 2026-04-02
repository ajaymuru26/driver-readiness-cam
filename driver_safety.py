import cv2
import time
import threading
import numpy as np
from deepface import DeepFace
import dlib
import sqlite3
import pandas as pd
from datetime import datetime
from scipy.spatial import distance as dist

# ═══════════════════════════════════════════════════════════════════
#  EAR (Eye Aspect Ratio) via dlib 68-point landmarks
# ═══════════════════════════════════════════════════════════════════

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

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
BLUE   = (200, 130, 50)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

deepface_result  = []
deepface_lock    = threading.Lock()
deepface_running = False

L, R = 0, 1

# ── Mouse state ───────────────────────────────────────────────────
mouse_x, mouse_y   = 0, 0
mouse_clicked      = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


def consume_click():
    global mouse_clicked
    clicked = mouse_clicked
    mouse_clicked = False
    return clicked


def draw_button(canvas, label, x1, y1, x2, y2, color, text_color=BLACK):
    """Draw a filled button and return True if it was clicked."""
    hovered = x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2
    fill = tuple(min(c + 30, 255) for c in color) if hovered else color
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), WHITE if hovered else GRAY, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cx = x1 + (x2 - x1) // 2 - tw // 2
    cy = y1 + (y2 - y1) // 2 + th // 2
    cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 1)
    if hovered and consume_click():
        return True
    return False


# ── Core helpers ──────────────────────────────────────────────────
def compute_ear(shape, indices):
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in indices],
                   dtype=np.float32)
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def draw_eye(frame, shape, indices, color):
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in indices],
                   dtype=np.int32)
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


def score_color(score):
    if score <= 1:
        return GREEN
    elif score <= 3:
        return ORANGE
    return RED


# ── Database ──────────────────────────────────────────────────────
def save_session(stats, dominant_emotion, score, duration):
    conn = sqlite3.connect("fatigue_sessions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            duration INTEGER,
            perclos_l REAL,
            perclos_r REAL,
            blinks_l INTEGER,
            blinks_r INTEGER,
            microsleeps_l INTEGER,
            microsleeps_r INTEGER,
            emotion TEXT,
            score INTEGER
        )
    """)
    conn.execute("""
        INSERT INTO sessions VALUES (NULL,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        duration,
        stats["perclos"][0], stats["perclos"][1],
        stats["blinks"][0],  stats["blinks"][1],
        stats["microsleeps"][0], stats["microsleeps"][1],
        dominant_emotion,
        score
    ))
    conn.commit()
    conn.close()


def load_sessions():
    try:
        conn = sqlite3.connect("fatigue_sessions.db")
        df = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


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


# ── History screen ────────────────────────────────────────────────
def history_screen(frame_shape):
    df = load_sessions()
    h, w = frame_shape[:2]
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (w, 60), (40, 40, 55), -1)
    centered_text(canvas, "SESSION HISTORY", 42, 0.95, WHITE, 2)

    # Back button
    back_clicked = draw_button(canvas, "< Back", w - 120, 10, w - 10, 50,
                               (60, 60, 80), WHITE)

    if df.empty:
        centered_text(canvas, "No sessions recorded yet.", h // 2, 0.7, GRAY)
        return canvas, back_clicked

    avg_score   = df["score"].mean()
    best_score  = int(df["score"].min())
    worst_score = int(df["score"].max())
    total       = len(df)

    sy = 80
    stats_items = [
        (f"Sessions: {total}",           WHITE),
        (f"Avg Score: {avg_score:.1f}",  score_color(int(avg_score))),
        (f"Best: {best_score}/7",        score_color(best_score)),
        (f"Worst: {worst_score}/7",      score_color(worst_score)),
    ]
    col_w = w // len(stats_items)
    for i, (txt, col) in enumerate(stats_items):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cx = i * col_w + col_w // 2
        cv2.putText(canvas, txt, (cx - tw // 2, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

    hdivider(canvas, sy + 14)

    cols = {
        "#":          30,
        "Date":       70,
        "Time":       190,
        "Dur":        290,
        "PERCLOS L":  345,
        "PERCLOS R":  435,
        "Microsleep": 525,
        "Mood":       620,
        "Score":      720,
    }
    hy = sy + 38
    for label, x in cols.items():
        cv2.putText(canvas, label, (x, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, GRAY, 1)
    hdivider(canvas, hy + 8)

    row_h    = 30
    start_y  = hy + 28
    max_rows = (h - start_y - 40) // row_h

    for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
        ry = start_y + i * row_h
        if i % 2 == 0:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (20, ry - 18), (w - 20, ry + 10),
                          (35, 35, 45), -1)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        sc    = int(row["score"])
        ts    = str(row["timestamp"])
        date  = ts[:10]
        ttime = ts[11:16]
        pl_pct = f"{row['perclos_l']*100:.0f}%"
        pr_pct = f"{row['perclos_r']*100:.0f}%"
        ms_tot = int(row["microsleeps_l"]) + int(row["microsleeps_r"])
        dur    = f"{int(row['duration'])}s"

        row_data = [
            (str(int(row["id"])), cols["#"],          WHITE),
            (date,                cols["Date"],        GRAY),
            (ttime,               cols["Time"],        GRAY),
            (dur,                 cols["Dur"],         WHITE),
            (pl_pct,              cols["PERCLOS L"],   perclos_color(row["perclos_l"])),
            (pr_pct,              cols["PERCLOS R"],   perclos_color(row["perclos_r"])),
            (str(ms_tot),         cols["Microsleep"],  GREEN if ms_tot == 0 else (ORANGE if ms_tot < 3 else RED)),
            (str(row["emotion"]), cols["Mood"],        WHITE),
            (f"{sc}/7",           cols["Score"],       score_color(sc)),
        ]
        for txt, x, col in row_data:
            cv2.putText(canvas, txt, (x, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    return canvas, back_clicked


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

        clicked_duration = None
        for i, secs in enumerate(DURATION_OPTIONS):
            bx     = start_x + i * (box_w + gap)
            is_sel = (i == selected)
            label  = (f"{secs}s" if secs < 60
                      else f"{secs // 60}m" if secs % 60 == 0 else f"{secs}s")
            col    = GREEN if is_sel else (60, 60, 70)
            if draw_button(frame, label, bx, btn_y, bx + box_w, btn_y + box_h,
                           col, BLACK if is_sel else WHITE):
                clicked_duration = secs
            hint = f"[{i + 1}]"
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, hint,
                        (bx + box_w // 2 - hw // 2, btn_y + box_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

        # History button
        hist_clicked = draw_button(frame, "View History",
                                   w // 2 - 80, btn_y + box_h + 40,
                                   w // 2 + 80, btn_y + box_h + 80,
                                   (50, 60, 80), WHITE)

        centered_text(frame, "ENTER = start     Q = quit",
                      btn_y + box_h + 100, 0.5, GRAY)

        cv2.imshow("Driver Fatigue Assessment", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            return None
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            selected = key - ord("1")
        elif key in (13, 10):
            return DURATION_OPTIONS[selected]

        if clicked_duration is not None:
            return clicked_duration
        if hist_clicked:
            return "history"


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

    # Buttons at bottom
    bw, bh = 140, 38
    by = h - 55
    draw_button(canvas, "Retry",        col_label,         by, col_label + bw,     by + bh, (60, 60, 60), WHITE)
    draw_button(canvas, "View History", col_label + bw + 10, by, col_label + bw*2 + 10, by + bh, (50, 60, 80), WHITE)
    draw_button(canvas, "Quit",         col_label + bw*2 + 20, by, col_label + bw*3 + 20, by + bh, (50, 30, 30), WHITE)

    return canvas, score


# ── Main ──────────────────────────────────────────────────────────
def run():
    global deepface_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cv2.namedWindow("Driver Fatigue Assessment")
    cv2.setMouseCallback("Driver Fatigue Assessment", mouse_callback)

    print("Driver Fatigue Assessment — dlib EAR blink detection")

    while True:

        chosen = duration_select_screen(cap)
        if chosen is None:
            break

        # History screen
        if chosen == "history":
            ret, frame = cap.read()
            fshape = frame.shape if ret else (480, 640, 3)
            while True:
                hist, back = history_screen(fshape)
                cv2.imshow("Driver Fatigue Assessment", hist)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or back:
                    break
            continue

        chosen_duration = chosen

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces         = detector(gray, 0)
            face_detected = len(faces) > 0
            prev_open     = list(eye_is_open)

            if face_detected:
                face_frames += 1
                shape = predictor(gray, faces[0])

                ear_l = compute_ear(shape, LEFT_EYE_IDX)
                ear_r = compute_ear(shape, RIGHT_EYE_IDX)
                ear_vals = [ear_l, ear_r]

                for side, e in ((L, ear_l), (R, ear_r)):
                    if e < EAR_CLOSED:
                        eye_is_open[side] = False
                    elif e > EAR_OPEN:
                        eye_is_open[side] = True

                for side in (L, R):
                    if eye_is_open[side]:
                        eye_open_frames[side] += 1

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

                for side, indices, ear_val in [
                    (L, LEFT_EYE_IDX,  ear_l),
                    (R, RIGHT_EYE_IDX, ear_r),
                ]:
                    col = GREEN if eye_is_open[side] else RED
                    draw_eye(frame, shape, indices, col)
                    pt  = (shape.part(indices[0]).x, shape.part(indices[0]).y - 8)
                    lbl = f"{'L' if side == L else 'R'} {ear_val:.2f}"
                    cv2.putText(frame, lbl, pt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

                d = faces[0]
                cv2.rectangle(frame, (d.left(), d.top()),
                              (d.right(), d.bottom()), GRAY, 1)

            else:
                for side in (L, R):
                    eye_consec_closed[side] = 0
                    eye_is_open[side]       = True
                ear_vals = [None, None]

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
                ]
                pad      = 6
                row_h    = 28
                panel_x1 = 8
                panel_y1 = 70
                panel_x2 = 310
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

        score = compute_score(stats["perclos"][0], stats["perclos"][1],
                              stats["microsleeps"][0] + stats["microsleeps"][1],
                              dominant_emotion)
        save_session(stats, dominant_emotion, score, chosen_duration)

        # Results loop with clickable buttons
        col_label = 30
        bw = 140
        by = frame.shape[0] - 55
        retry_x1,   retry_x2   = col_label,          col_label + bw
        hist_x1,    hist_x2    = col_label + bw + 10, col_label + bw * 2 + 10
        quit_x1,    quit_x2    = col_label + bw*2+20, col_label + bw * 3 + 20

        while True:
            canvas, _ = results_screen(frame.shape, stats, dominant_emotion, chosen_duration)
            cv2.imshow("Driver Fatigue Assessment", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or (mouse_x in range(quit_x1, quit_x2) and
                                    mouse_y in range(by, by + 38) and consume_click()):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("r") or (mouse_x in range(retry_x1, retry_x2) and
                                      mouse_y in range(by, by + 38) and consume_click()):
                break
            elif key == ord("h") or (mouse_x in range(hist_x1, hist_x2) and
                                      mouse_y in range(by, by + 38) and consume_click()):
                while True:
                    hist, back = history_screen(frame.shape)
                    cv2.imshow("Driver Fatigue Assessment", hist)
                    if cv2.waitKey(30) & 0xFF == ord("q") or back:
                        break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run()