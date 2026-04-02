"""
fatigue_analysis.py
───────────────────
Run this separately from the main detection script to analyze
your session history and detect chronic fatigue patterns.

Usage:
    python fatigue_analysis.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline


DB_PATH = "fatigue_sessions.db"

# ── Thresholds ────────────────────────────────────────────────────
MIN_SESSIONS_FOR_TREND      = 5
MIN_SESSIONS_FOR_CLASSIFIER = 8   # need more data for a personal model
CHRONIC_SCORE_THRESHOLD     = 3.0
WORSENING_SLOPE_THRESHOLD   = 0.1
ANOMALY_CONTAMINATION       = 0.15


# ── Load & clean data ─────────────────────────────────────────────
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM sessions ORDER BY timestamp ASC", conn)
        conn.close()
    except Exception as e:
        print(f"Could not load database: {e}")
        return None

    if df.empty:
        print("No sessions found. Run some assessments first.")
        return None

    df["timestamp"]       = pd.to_datetime(df["timestamp"])
    df["date"]            = df["timestamp"].dt.date
    df["hour"]            = df["timestamp"].dt.hour
    df["day_of_week"]     = df["timestamp"].dt.day_name()
    df["perclos_worst"]   = df[["perclos_l", "perclos_r"]].max(axis=1)
    df["perclos_avg"]     = df[["perclos_l", "perclos_r"]].mean(axis=1)
    df["perclos_asym"]    = (df["perclos_l"] - df["perclos_r"]).abs()  # asymmetry
    df["microsleeps_total"] = df["microsleeps_l"] + df["microsleeps_r"]
    df["blinks_total"]    = df["blinks_l"] + df["blinks_r"]
    df["session_index"]   = range(len(df))

    return df


# ── 1. Basic summary ──────────────────────────────────────────────
def summary_stats(df):
    print("\n" + "═" * 55)
    print("  FATIGUE HISTORY SUMMARY")
    print("═" * 55)
    print(f"  Total sessions    : {len(df)}")
    print(f"  Date range        : {df['date'].min()}  →  {df['date'].max()}")
    print(f"  Avg fatigue score : {df['score'].mean():.2f} / 7")
    print(f"  Best session      : {df['score'].min()} / 7")
    print(f"  Worst session     : {df['score'].max()} / 7")
    print(f"  Avg PERCLOS (worst eye) : {df['perclos_worst'].mean()*100:.1f}%")
    print(f"  Total microsleeps : {df['microsleeps_total'].sum()}")

    verdict_dist = df["score"].apply(
        lambda s: "Ready" if s <= 1 else ("Caution" if s <= 3 else "Do Not Drive")
    ).value_counts()
    print("\n  Verdict breakdown:")
    for v, count in verdict_dist.items():
        pct = count / len(df) * 100
        print(f"    {v:<15} {count:>3} sessions  ({pct:.0f}%)")


# ── 2. Trend detection via linear regression ──────────────────────
def trend_analysis(df):
    print("\n" + "═" * 55)
    print("  TREND ANALYSIS")
    print("═" * 55)

    if len(df) < MIN_SESSIONS_FOR_TREND:
        print(f"  Need at least {MIN_SESSIONS_FOR_TREND} sessions for trend analysis.")
        print(f"  You have {len(df)}. Keep running assessments!")
        return

    X = df[["session_index"]].values
    y = df["score"].values

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2    = model.score(X, y)

    print(f"  Score trend slope : {slope:+.4f} per session")
    print(f"  R² fit            : {r2:.3f}")

    if slope > WORSENING_SLOPE_THRESHOLD:
        print("\n  ⚠  WORSENING TREND DETECTED")
        print(f"     Your fatigue score is increasing by ~{slope:.2f} points per session.")
        print("     This may indicate accumulating sleep debt.")
    elif slope < -WORSENING_SLOPE_THRESHOLD:
        print("\n  ✓  IMPROVING TREND")
        print(f"     Your fatigue score is improving by ~{abs(slope):.2f} points per session.")
    else:
        print("\n  →  STABLE TREND")
        print("     No significant change in fatigue over time.")

    if len(df) >= 7:
        df["rolling_avg"] = df["score"].rolling(window=7).mean()
        recent_avg = df["rolling_avg"].iloc[-1]
        print(f"\n  7-session rolling avg : {recent_avg:.2f} / 7")
        if recent_avg >= CHRONIC_SCORE_THRESHOLD:
            print("  ⚠  CHRONIC FATIGUE FLAG: Recent average score is elevated.")
        else:
            print("  ✓  Recent average is within normal range.")


# ── 3. Circadian pattern analysis ────────────────────────────────
def circadian_analysis(df):
    print("\n" + "═" * 55)
    print("  CIRCADIAN PATTERN ANALYSIS")
    print("═" * 55)

    if len(df) < MIN_SESSIONS_FOR_TREND:
        print(f"  Need at least {MIN_SESSIONS_FOR_TREND} sessions.")
        return

    df["time_bucket"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"],
        right=False
    )
    hourly = df.groupby("time_bucket", observed=True)["score"].agg(["mean", "count"])
    hourly.columns = ["avg_score", "sessions"]

    print("\n  Avg fatigue by time of day:")
    for period, row in hourly.iterrows():
        if row["sessions"] == 0:
            continue
        bar  = "█" * int(row["avg_score"] * 3)
        flag = "  ⚠" if row["avg_score"] >= CHRONIC_SCORE_THRESHOLD else ""
        print(f"    {str(period):<22} {row['avg_score']:.2f}/7  {bar}{flag}  (n={int(row['sessions'])})")

    worst_period = hourly["avg_score"].idxmax()
    print(f"\n  You are most fatigued during: {worst_period}")

    if len(df) >= 7:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df.groupby("day_of_week")["score"].mean().reindex(dow_order).dropna()
        print("\n  Avg fatigue by day of week:")
        for day, avg in dow.items():
            bar  = "█" * int(avg * 3)
            flag = "  ⚠" if avg >= CHRONIC_SCORE_THRESHOLD else ""
            print(f"    {day:<12} {avg:.2f}/7  {bar}{flag}")


# ── 4. Anomaly detection (Isolation Forest) ───────────────────────
def anomaly_detection(df):
    print("\n" + "═" * 55)
    print("  ANOMALY DETECTION  (Isolation Forest)")
    print("═" * 55)

    if len(df) < 10:
        print("  Need at least 10 sessions for anomaly detection.")
        return

    features  = ["perclos_worst", "microsleeps_total", "score"]
    X         = df[features].values
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)

    iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
    df["anomaly"] = iso.fit_predict(X_scaled)

    anomalies = df[df["anomaly"] == -1]

    if anomalies.empty:
        print("  ✓  No anomalous sessions detected.")
    else:
        print(f"  ⚠  {len(anomalies)} anomalous session(s) detected:\n")
        for _, row in anomalies.iterrows():
            print(f"    Session #{int(row['id'])}  |  {str(row['timestamp'])[:16]}")
            print(f"      Score: {int(row['score'])}/7  |  "
                  f"PERCLOS: {row['perclos_worst']*100:.0f}%  |  "
                  f"Microsleeps: {int(row['microsleeps_total'])}")
        print("\n  These sessions were statistical outliers vs your baseline.")


# ── 5. Personal baseline classifier ──────────────────────────────
def personal_baseline_classifier(df):
    print("\n" + "═" * 55)
    print("  PERSONAL BASELINE CLASSIFIER")
    print("═" * 55)

    if len(df) < MIN_SESSIONS_FOR_CLASSIFIER:
        print(f"  Need at least {MIN_SESSIONS_FOR_CLASSIFIER} sessions to train")
        print(f"  your personal model. You have {len(df)}.")
        print("  The more sessions you log, the more personalized")
        print("  and accurate this model becomes.")
        return

    # ── Feature engineering ───────────────────────────────────────
    # These are all the signals the model learns from.
    # perclos_asym catches one-eye drooping (neurological signal).
    # hour captures your circadian pattern.
    # blinks_total captures overall eye activity rate.
    features = [
        "perclos_worst",      # worst eye closure rate
        "perclos_avg",        # average across both eyes
        "perclos_asym",       # left-right asymmetry
        "microsleeps_total",  # total microsleep events
        "blinks_total",       # total blinks (low blinks = zoned out)
        "hour",               # time of day
    ]

    X = df[features].values

    # ── Label: is this session above YOUR personal average? ───────
    # Instead of using fixed thresholds (score > 3 = bad),
    # the model learns what's abnormal FOR YOU specifically.
    # If your baseline is naturally high (e.g. avg score 5),
    # a score of 4 would be flagged as "below your norm" not "dangerous".
    personal_avg   = df["score"].mean()
    personal_std   = df["score"].std()
    threshold      = personal_avg + (0.5 * personal_std)
    df["above_baseline"] = (df["score"] > threshold).astype(int)

    y         = df["above_baseline"].values
    n_above   = y.sum()
    n_below   = len(y) - n_above

    print(f"\n  Your personal fatigue baseline : {personal_avg:.2f} / 7")
    print(f"  Abnormal threshold (mean+0.5σ) : {threshold:.2f} / 7")
    print(f"  Sessions above baseline        : {n_above}")
    print(f"  Sessions at/below baseline     : {n_below}")

    # Need at least one session in each class to train
    if n_above == 0 or n_below == 0:
        print("\n  All sessions are the same — need more variance to train.")
        return

    # ── Train Random Forest ───────────────────────────────────────
    # Random Forest builds many decision trees, each trained on a
    # random subset of your data and features, then votes.
    # It handles small datasets better than a single decision tree
    # and naturally gives us feature importances.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100,    # 100 trees
            max_depth=3,         # shallow trees to avoid overfitting on small data
            class_weight="balanced",  # handles imbalanced classes
            random_state=42
        ))
    ])

    pipeline.fit(X, y)

    # ── Cross-validation ──────────────────────────────────────────
    # With small data, Leave-One-Out CV is most reliable —
    # train on all sessions except one, test on that one, repeat.
    if len(df) >= 10:
        loo    = LeaveOneOut()
        scores = cross_val_score(pipeline, X, y, cv=loo, scoring="accuracy")
        cv_acc = scores.mean()
        print(f"\n  Leave-One-Out CV accuracy : {cv_acc*100:.0f}%")
        if cv_acc >= 0.75:
            print("  ✓  Model is learning your personal patterns well.")
        elif cv_acc >= 0.55:
            print("  →  Model is learning but needs more sessions to improve.")
        else:
            print("  ⚠  Model needs more data — log more sessions.")
    else:
        print("\n  (Need 10+ sessions for cross-validation accuracy)")

    # ── Feature importances ───────────────────────────────────────
    # Which signals matter most for YOUR fatigue specifically?
    # This is different person to person — for some people PERCLOS
    # dominates, for others microsleeps are the key signal.
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_imp    = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    print("\n  What drives YOUR fatigue (feature importances):")
    for feat, imp in feat_imp:
        bar   = "█" * int(imp * 40)
        label = feat.replace("_", " ").title()
        print(f"    {label:<22} {imp:.3f}  {bar}")

    top_feature = feat_imp[0][0].replace("_", " ")
    print(f"\n  → Your fatigue is most strongly driven by: {top_feature}")

    # ── Predict on most recent session ───────────────────────────
    latest      = df.iloc[-1]
    X_latest    = latest[features].values.reshape(1, -1)
    pred        = pipeline.predict(X_latest)[0]
    prob        = pipeline.predict_proba(X_latest)[0][1]  # prob of above baseline

    print(f"\n  Most recent session prediction:")
    print(f"    Probability above YOUR baseline : {prob*100:.0f}%")

    if pred == 1:
        print(f"    ⚠  This session was ABOVE your personal norm")
        print(f"       (your baseline avg is {personal_avg:.1f}, this session scored {int(latest['score'])})")
    else:
        print(f"    ✓  This session was WITHIN your personal norm")
        print(f"       (your baseline avg is {personal_avg:.1f}, this session scored {int(latest['score'])})")

    # ── What your model learned ───────────────────────────────────
    print(f"\n  What this model does differently from fixed thresholds:")
    print(f"    Fixed threshold : score > 3 = bad  (same for everyone)")
    print(f"    Your model      : score > {threshold:.1f} = bad  (calibrated to you)")
    print(f"    If your natural baseline is high, the model adjusts")
    print(f"    so it only flags sessions that are unusual FOR YOU.")


# ── 6. Chronic sleep deprivation risk score ───────────────────────
def chronic_risk_score(df):
    print("\n" + "═" * 55)
    print("  CHRONIC SLEEP DEPRIVATION RISK")
    print("═" * 55)

    if len(df) < MIN_SESSIONS_FOR_TREND:
        print(f"  Need at least {MIN_SESSIONS_FOR_TREND} sessions.")
        return

    risk    = 0
    reasons = []

    avg = df["score"].mean()
    if avg >= 4.0:
        risk += 3
        reasons.append(f"High average fatigue score ({avg:.1f}/7)")
    elif avg >= 2.5:
        risk += 1
        reasons.append(f"Moderate average fatigue score ({avg:.1f}/7)")

    if len(df) >= MIN_SESSIONS_FOR_TREND:
        X     = df[["session_index"]].values
        slope = LinearRegression().fit(X, df["score"].values).coef_[0]
        if slope > WORSENING_SLOPE_THRESHOLD:
            risk += 2
            reasons.append(f"Worsening trend (+{slope:.2f} per session)")

    ms_rate = df["microsleeps_total"].mean()
    if ms_rate >= 3:
        risk += 2
        reasons.append(f"High average microsleeps ({ms_rate:.1f} per session)")
    elif ms_rate >= 1:
        risk += 1
        reasons.append(f"Moderate microsleeps ({ms_rate:.1f} per session)")

    danger_pct = (df["score"] > 3).mean()
    if danger_pct >= 0.5:
        risk += 2
        reasons.append(f"{danger_pct*100:.0f}% of sessions flagged 'Do Not Drive'")

    avg_perclos = df["perclos_worst"].mean()
    if avg_perclos >= 0.25:
        risk += 2
        reasons.append(f"Elevated PERCLOS baseline ({avg_perclos*100:.0f}%)")

    max_risk = 11
    risk     = min(risk, max_risk)
    pct      = risk / max_risk

    if pct < 0.3:
        level, symbol = "LOW",      "✓"
    elif pct < 0.6:
        level, symbol = "MODERATE", "⚠"
    else:
        level, symbol = "HIGH",     "✗"

    bar_len = 30
    filled  = int(bar_len * pct)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print(f"\n  Risk Score : {risk} / {max_risk}")
    print(f"  [{bar}]  {level} RISK  {symbol}")

    if reasons:
        print("\n  Contributing factors:")
        for r in reasons:
            print(f"    • {r}")

    if level == "HIGH":
        print("\n  ⚠  Recommendation: You may be experiencing chronic sleep")
        print("     deprivation. Consider speaking with a healthcare provider.")
    elif level == "MODERATE":
        print("\n  →  Recommendation: Monitor your sleep quality. Aim for")
        print("     7-9 hours per night and reduce late-night screen time.")
    else:
        print("\n  ✓  Your fatigue patterns look healthy. Keep it up.")

    print("\n  NOTE: This is not a medical diagnosis. Consult a doctor")
    print("  for any health concerns.")


# ── Main ──────────────────────────────────────────────────────────
def run():
    print("\n  DRIVER FATIGUE — LONGITUDINAL ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    df = load_data()
    if df is None:
        return

    summary_stats(df)
    trend_analysis(df)
    circadian_analysis(df)
    anomaly_detection(df)
    personal_baseline_classifier(df)
    chronic_risk_score(df)

    print("\n" + "═" * 55 + "\n")


if __name__ == "__main__":
    run()