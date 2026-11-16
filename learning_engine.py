import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression

class RecursiveLearningEngine:
    def __init__(self, db_path="statsline_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS optical_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                side TEXT,
                mean_brightness REAL,
                std_contrast REAL,
                auto_left_px INTEGER,
                auto_right_px INTEGER,
                true_left_px INTEGER,
                true_right_px INTEGER,
                optimal_lum_thresh INTEGER
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS psa_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                card_id TEXT,
                front_worst_axis REAL,
                back_worst_axis REAL,
                corner_whitening_score REAL,
                actual_psa_grade INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log_caliper_override(self, img_gray, side, auto_l, auto_r, true_l, true_r):
        """Records manual caliper overrides and reverse-engineers the ideal luminance threshold."""
        mean_b = float(np.mean(img_gray))
        std_c = float(np.std(img_gray))

        center_row = img_gray[700, :]
        left_boundary_lum = int(center_row[min(true_l, 999)])
        right_boundary_lum = int(center_row[max(1000 - true_r, 0)])
        ideal_thresh = int(np.clip((left_boundary_lum + right_boundary_lum) / 2, 140, 245))

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO optical_feedback 
            (timestamp, side, mean_brightness, std_contrast, auto_left_px, auto_right_px, true_left_px, true_right_px, optimal_lum_thresh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), side, mean_b, std_c, auto_l, auto_r, true_l, true_r, ideal_thresh))
        conn.commit()
        conn.close()
        return ideal_thresh

    def predict_optimal_lum_threshold(self, img_gray, side="BACK", default_thresh=200):
        """Uses Ridge Regression on past manual overrides to auto-tune the scanner for current lighting."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM optical_feedback WHERE side = ?", conn, params=(side,))
        conn.close()

        if len(df) < 5:
            return default_thresh, False, len(df)

        X = df[["mean_brightness", "std_contrast"]].values
        y = df["optimal_lum_thresh"].values

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        curr_features = np.array([[float(np.mean(img_gray)), float(np.std(img_gray))]])
        predicted_thresh = int(np.clip(model.predict(curr_features)[0], 140, 245))
        return predicted_thresh, True, len(df)

    def log_psa_outcome(self, card_id, front_worst, back_worst, corner_score, psa_grade):
        """Logs a returned PSA slab grade (10, 9, 8) to update the gem probability model."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO psa_outcomes
            (timestamp, card_id, front_worst_axis, back_worst_axis, corner_whitening_score, actual_psa_grade)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), card_id, front_worst, back_worst, corner_score, psa_grade))
        conn.commit()
        conn.close()

    def predict_gem_probability(self, front_worst, back_worst, corner_score=0.0):
        """Calculates empirical PSA 10 probability using Logistic Regression or analytical prior."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM psa_outcomes", conn)
        conn.close()

        if len(df) < 10 or len(df["actual_psa_grade"].unique()) < 2:
            front_penalty = max(0.0, (front_worst - 53.0) * 0.45)
            back_penalty = max(0.0, (back_worst - 57.0) * 0.25)
            logit = 2.2 - front_penalty - back_penalty - (corner_score * 0.5)
            prob = 1.0 / (1.0 + np.exp(-logit))
            return round(float(prob * 100), 1), False, len(df)

        X = df[["front_worst_axis", "back_worst_axis", "corner_whitening_score"]].values
        y = (df["actual_psa_grade"] == 10).astype(int).values

        clf = LogisticRegression()
        clf.fit(X, y)

        prob = clf.predict_proba([[front_worst, back_worst, corner_score]])[0][1]
        return round(float(prob * 100), 1), True, len(df)

    def get_calibration_stats(self):
        """Returns summary metrics and history tables from the recursive memory bank."""
        conn = sqlite3.connect(self.db_path)
        opt_df = pd.read_sql_query("SELECT * FROM optical_feedback ORDER BY id DESC", conn)
        psa_df = pd.read_sql_query("SELECT * FROM psa_outcomes ORDER BY id DESC", conn)
        conn.close()

        mean_pixel_error = 0.0
        if len(opt_df) > 0:
            err_l = np.abs(opt_df["auto_left_px"] - opt_df["true_left_px"])
            err_r = np.abs(opt_df["auto_right_px"] - opt_df["true_right_px"])
            mean_pixel_error = float(np.mean((err_l + err_r) / 2.0))

        return {
            "optical_samples": len(opt_df),
            "mean_pixel_error": round(mean_pixel_error, 2),
            "psa_returns_logged": len(psa_df),
            "historical_gem_rate": round(float((psa_df["actual_psa_grade"] == 10).mean() * 100), 1) if len(psa_df) > 0 else 0.0,
            "opt_df": opt_df,
            "psa_df": psa_df
        }
