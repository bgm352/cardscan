import cv2
import numpy as np

class CardSieveEngine:
    def __init__(self, front_max_tolerance=55.0, back_max_tolerance=60.0):
        # Standard normalized card canvas (2.5" x 3.5" aspect ratio)
        self.W = 1000
        self.H = 1400
        self.front_gate = front_max_tolerance
        self.back_gate = back_max_tolerance

    def load_and_normalize(self, file_bytes, auto_crop=True):
        """Decodes image bytes and normalizes to a 1000x1400 pixel grid."""
        arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")

        if auto_crop:
            img = self._attempt_card_isolation(img)

        resized = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return resized

    def _attempt_card_isolation(self, img):
        """Finds largest rectangular contour when a card is photographed on a dark background."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (img.shape[0] * img.shape[1] * 0.25):
            x, y, w, h = cv2.boundingRect(largest)
            return img[y:y+h, x:x+w]
        return img

    def analyze_card_side(self, img_bgr, side="BACK", lum_threshold=200, manual_left_px=None, manual_right_px=None):
        """
        Measures left vs. right border/anchor symmetry and generates the visual HUD overlay.
        side: 'BACK' (Checklist white margin) or 'FRONT' (Anchor/Border distance)
        """
        overlay = img_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        scan_y_positions = range(450, 950, 25)
        detected_lefts = []
        detected_rights = []

        for y in scan_y_positions:
            row = gray[y, :]
            if side == "BACK":
                inner_row = row[15:self.W - 15]
                dark_pixels = np.where(inner_row < lum_threshold)[0]
                if len(dark_pixels) > 0:
                    detected_lefts.append(dark_pixels[0] + 15)
                    detected_rights.append(self.W - (dark_pixels[-1] + 15))
            else:
                grad = np.abs(np.diff(row.astype(np.float32)))
                spikes = np.where(grad[20:self.W - 20] > 28)[0]
                if len(spikes) > 0:
                    detected_lefts.append(spikes[0] + 20)
                    detected_rights.append(self.W - (spikes[-1] + 20))

        left_px = int(np.median(detected_lefts)) if detected_lefts else 50
        right_px = int(np.median(detected_rights)) if detected_rights else 50

        if manual_left_px is not None:
            left_px = int(manual_left_px)
        if manual_right_px is not None:
            right_px = int(manual_right_px)

        total_margin = max(left_px + right_px, 1)
        left_pct = (left_px / total_margin) * 100.0
        right_pct = (right_px / total_margin) * 100.0
        worst_axis = max(left_pct, right_pct)

        gate_limit = self.front_gate if side == "FRONT" else self.back_gate
        passed = worst_axis <= gate_limit

        center_scanline = gray[700, :].tolist()
        overlay = self._draw_hud(overlay, left_px, right_px, left_pct, right_pct, passed, side)

        return {
            "left_px": left_px,
            "right_px": right_px,
            "left_pct": round(left_pct, 1),
            "right_pct": round(right_pct, 1),
            "ratio_str": f"{left_pct:.1f} / {right_pct:.1f}",
            "worst_axis": round(worst_axis, 1),
            "passed": passed,
            "gate_limit": gate_limit,
            "scanline": center_scanline,
            "overlay_rgb": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        }

    def _draw_hud(self, img, left_px, right_px, left_pct, right_pct, passed, side):
        """Draws target crosshairs, caliper lines, and Pass/Fail telemetry badges."""
        cv2.line(img, (self.W // 2, 0), (self.W // 2, self.H), (255, 220, 0), 2)
        cv2.line(img, (0, self.H // 2), (self.W, self.H // 2), (255, 220, 0), 2)

        status_color = (129, 185, 16) if passed else (68, 68, 239)

        right_x = self.W - right_px
        cv2.line(img, (left_px, 150), (left_px, self.H - 150), status_color, 4)
        cv2.line(img, (right_x, 150), (right_x, self.H - 150), status_color, 4)

        cv2.arrowedLine(img, (0, 700), (left_px, 700), status_color, 4, tipLength=0.25)
        cv2.arrowedLine(img, (self.W, 700), (right_x, 700), status_color, 4, tipLength=0.25)

        cv2.rectangle(img, (20, 20), (self.W - 20, 120), (20, 23, 30), -1)
        cv2.rectangle(img, (20, 20), (self.W - 20, 120), status_color, 3)

        verdict_txt = f"PASS ({side} GATE)" if passed else f"REJECT: {side} SKEW"
        cv2.putText(img, verdict_txt, (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        detail_txt = f"Ratio: {left_pct:.1f}/{right_pct:.1f}  |  L: {left_px}px  R: {right_px}px"
        cv2.putText(img, detail_txt, (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2)

        return img
