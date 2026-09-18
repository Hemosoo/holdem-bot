"""
screen_reader.py — OpenCV-based poker screen reader (research/learning).

Captures a screen region and attempts to detect:
  - Card rank and suit (via contour detection + OCR)
  - Pot and stack sizes (via OCR on configured regions)
  - Community cards vs hole cards (via region config)

Usage:
    python3 screen_reader.py               # live debug window, auto-detect cards
    python3 screen_reader.py --calibrate   # interactive region setup tool
    python3 screen_reader.py --once        # single capture, print detected state

Dependencies:
    pip install opencv-python mss pytesseract
    brew install tesseract   (macOS)  or  apt install tesseract-ocr  (Linux)

Config is saved/loaded from screen_config.json in the same directory.
"""

import cv2
import numpy as np
import mss
import json
import argparse
import time
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[warn] pytesseract not installed — OCR disabled, rank detection limited")

try:
    import Quartz
    import Cocoa
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False

CONFIG_PATH = Path(__file__).parent / "screen_config.json"

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Region:
    """A named screen rectangle: (x, y, width, height)."""
    name: str
    x: int
    y: int
    w: int
    h: int

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Region(**d)

    @property
    def bbox(self):
        return {"left": self.x, "top": self.y, "width": self.w, "height": self.h}


@dataclass
class DetectedGameState:
    hole_cards: list[str] = field(default_factory=list)
    community_cards: list[str] = field(default_factory=list)
    pot: float | None = None
    my_stack: float | None = None
    villain_stack: float | None = None
    raw_text: dict = field(default_factory=dict)  # region name -> raw OCR string


# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "monitor": 1,       # which monitor (1 = primary)
    "regions": [
        # Adjust these to match your poker client's layout.
        # Use --calibrate to set them interactively.
        {"name": "hole_cards",      "x": 700,  "y": 800, "w": 300, "h": 120},
        {"name": "community_cards", "x": 400,  "y": 500, "w": 700, "h": 130},
        {"name": "pot",             "x": 550,  "y": 460, "w": 200, "h":  40},
        {"name": "my_stack",        "x": 750,  "y": 930, "w": 150, "h":  35},
        {"name": "villain_stack",   "x": 750,  "y": 200, "w": 150, "h":  35},
    ]
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[config] Saved to {CONFIG_PATH}")


# ── Screen capture ─────────────────────────────────────────────────────────────

def capture_region(sct: mss.mss, region: Region) -> np.ndarray:
    """Capture a screen region and return as a BGR numpy array."""
    raw = sct.grab(region.bbox)
    img = np.array(raw)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def capture_full_monitor(sct: mss.mss, monitor_num: int = 1) -> np.ndarray:
    mon = sct.monitors[monitor_num]
    raw = sct.grab(mon)
    img = np.array(raw)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ── Quartz window capture (macOS fallback) ─────────────────────────────────────

def list_windows() -> list[dict]:
    """Return all on-screen windows with their id, name, and owning app."""
    if not QUARTZ_AVAILABLE:
        return []
    wins = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    results = []
    for w in wins:
        results.append({
            "id":    w.get("kCGWindowNumber", 0),
            "app":   w.get("kCGWindowOwnerName", ""),
            "title": w.get("kCGWindowName", ""),
            "layer": w.get("kCGWindowLayer", 0),
        })
    return results


def _cgimage_to_numpy(cg_img) -> np.ndarray | None:
    """Convert a CGImageRef to a BGR numpy array."""
    if cg_img is None:
        return None
    width  = Quartz.CGImageGetWidth(cg_img)
    height = Quartz.CGImageGetHeight(cg_img)
    bpr    = Quartz.CGImageGetBytesPerRow(cg_img)
    bpp    = Quartz.CGImageGetBitsPerPixel(cg_img) // 8
    data   = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg_img))
    arr    = np.frombuffer(data, dtype=np.uint8)
    n_rows = min(height, len(arr) // bpr)
    # Stride by bpr (may include row padding), then crop to actual pixel width
    arr    = arr[:n_rows * bpr].reshape((n_rows, bpr))
    arr    = arr[:, :width * bpp].reshape((n_rows, width, bpp))
    if bpp == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def capture_window_quartz(window_id: int) -> np.ndarray | None:
    """Capture a single window by Quartz window ID. Returns BGR numpy array or None."""
    if not QUARTZ_AVAILABLE:
        return None
    cg_img = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBestResolution,
    )
    return _cgimage_to_numpy(cg_img)


def capture_display_quartz() -> np.ndarray | None:
    """Capture the main display framebuffer directly. Bypasses window-level protections."""
    if not QUARTZ_AVAILABLE:
        return None
    display_id = Quartz.CGMainDisplayID()
    cg_img = Quartz.CGDisplayCreateImage(display_id)
    return _cgimage_to_numpy(cg_img)


def find_poker_window(keyword: str = "wpt") -> int | None:
    """Find the first window whose app or title contains keyword (case-insensitive)."""
    for w in list_windows():
        if keyword.lower() in w["app"].lower() or keyword.lower() in w["title"].lower():
            return w["id"]
    return None


# ── Card detection ─────────────────────────────────────────────────────────────

RANK_CHARS = set("A23456789TJQK")

SUIT_COLORS = {
    # (lower_HSV, upper_HSV, suit_char)
    "h": (np.array([0,   80, 80]),  np.array([10, 255, 255])),   # hearts   — red
    "d": (np.array([0,   80, 80]),  np.array([10, 255, 255])),   # diamonds — red (same hue)
    "c": (np.array([35,  40, 40]),  np.array([85, 255, 255])),   # clubs    — green tint / dark
    "s": (np.array([0,    0,  0]),  np.array([180,  50, 80])),   # spades   — near black
}


def _detect_suit_by_color(card_img: np.ndarray) -> str:
    """
    Rough suit detection by dominant color in the lower half of the card image.
    Returns 'h', 'd', 'c', or 's'. Falls back to 's' if uncertain.
    """
    h, w = card_img.shape[:2]
    lower_half = card_img[h // 2:, :]
    hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

    # Count red pixels (hearts/diamonds)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])) |
        cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
    )
    red_pixels = cv2.countNonZero(red_mask)

    # Count very dark pixels (spades/clubs)
    dark_mask = cv2.inRange(card_img, np.array([0, 0, 0]), np.array([80, 80, 80]))
    dark_pixels = cv2.countNonZero(dark_mask)

    total = lower_half.shape[0] * lower_half.shape[1]
    red_ratio = red_pixels / total
    dark_ratio = dark_pixels / total

    if red_ratio > 0.05:
        # Distinguish hearts (filled bottom) from diamonds (rotated square)
        # Simple heuristic: hearts have more red in center column
        center_col = lower_half[:, w // 3: 2 * w // 3]
        center_hsv = cv2.cvtColor(center_col, cv2.COLOR_BGR2HSV)
        center_red = cv2.inRange(center_hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        return "h" if cv2.countNonZero(center_red) > 0 else "d"
    elif dark_ratio > 0.08:
        # Spades have a distinctive stem; clubs are rounder — hard to distinguish without templates
        return "s"
    return "c"


def _ocr_rank(rank_img: np.ndarray) -> str:
    """Run OCR on the rank region of a card. Returns a single rank char or ''."""
    if not OCR_AVAILABLE:
        return "?"

    # Upscale and threshold for better OCR
    scale = 4
    big = cv2.resize(rank_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 10 -c tessedit_char_whitelist=A23456789TJQK"
    text = pytesseract.image_to_string(thresh, config=config).strip().upper()

    # Keep only valid rank chars
    text = "".join(c for c in text if c in RANK_CHARS)
    return text[0] if text else "?"


def detect_cards_in_region(img: np.ndarray, debug: bool = False) -> list[str]:
    """
    Find card-like rectangles in img, extract rank + suit, return list of card strings.
    Cards are expected to be light-colored rectangles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to find light (card face) regions
    _, light_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    card_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:      # too small
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        # Cards are taller than wide (aspect ~0.65–0.85) or rotated
        if not (0.4 < aspect < 1.2):
            continue
        # Ignore if suspiciously large (likely background)
        if w > img.shape[1] * 0.4 or h > img.shape[0] * 0.9:
            continue

        card_img = img[y:y+h, x:x+w]

        # Rank is in the top-left ~30% of the card
        rank_h = max(int(h * 0.35), 10)
        rank_w = max(int(w * 0.45), 10)
        rank_region = card_img[:rank_h, :rank_w]

        rank = _ocr_rank(rank_region)
        suit = _detect_suit_by_color(card_img)
        card_str = f"{rank}{suit}" if rank not in ("", "?") else f"?{suit}"

        cards.append(card_str)
        card_contours.append((x, y, w, h, card_str))

    # Sort left-to-right
    card_contours.sort(key=lambda c: c[0])
    cards = [c[4] for c in card_contours]

    if debug:
        debug_img = img.copy()
        for x, y, w, h, label in card_contours:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return cards, debug_img

    return cards


# ── Number / text OCR ──────────────────────────────────────────────────────────

def _ocr_number(img: np.ndarray) -> float | None:
    """Extract a dollar/chip amount from a region. Returns float or None."""
    if not OCR_AVAILABLE:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 3
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789.,$£"
    text = pytesseract.image_to_string(thresh, config=config).strip()

    # Strip currency symbols, keep digits and decimal
    clean = re.sub(r"[^\d.]", "", text)
    try:
        return float(clean)
    except ValueError:
        return None


# ── Main reader class ──────────────────────────────────────────────────────────

class PokerScreenReader:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.regions: dict[str, Region] = {
            r["name"]: Region.from_dict(r)
            for r in self.config.get("regions", [])
        }
        self.monitor_num = self.config.get("monitor", 1)
        self._sct = mss.mss()

    def read(self, debug: bool = False) -> DetectedGameState:
        state = DetectedGameState()
        debug_frames = {}

        for name, region in self.regions.items():
            img = capture_region(self._sct, region)

            if name in ("hole_cards", "community_cards"):
                if debug:
                    cards, dbg = detect_cards_in_region(img, debug=True)
                    debug_frames[name] = dbg
                else:
                    cards = detect_cards_in_region(img, debug=False)

                if name == "hole_cards":
                    state.hole_cards = cards
                else:
                    state.community_cards = cards

            elif name == "pot":
                val = _ocr_number(img)
                state.pot = val
                state.raw_text["pot"] = str(val)
                if debug:
                    debug_frames[name] = _annotate(img, f"pot: {val}")

            elif name == "my_stack":
                val = _ocr_number(img)
                state.my_stack = val
                state.raw_text["my_stack"] = str(val)
                if debug:
                    debug_frames[name] = _annotate(img, f"stack: {val}")

            elif name == "villain_stack":
                val = _ocr_number(img)
                state.villain_stack = val
                state.raw_text["villain_stack"] = str(val)
                if debug:
                    debug_frames[name] = _annotate(img, f"stack: {val}")

        if debug:
            return state, debug_frames
        return state

    def close(self):
        self._sct.close()


def _annotate(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


# ── Calibration tool ───────────────────────────────────────────────────────────

def calibrate(window_keyword: str = "wpt"):
    """
    Interactive calibration. Captures the full monitor (or a specific window via
    Quartz on macOS) and lets you drag rectangles to define each region.
    Press 's' to save, 'n' for next region, 'q' to quit without saving.
    """
    full = None

    if QUARTZ_AVAILABLE:
        # 1. Try raw display framebuffer — captures hardware-accelerated content
        print("[calibrate] Trying CGDisplayCreateImage (display framebuffer)...")
        full = capture_display_quartz()
        if full is not None and full.mean() > 5:
            print("[calibrate] Display capture OK.")
        else:
            print("[calibrate] Display capture blank, trying window capture...")
            full = None

        # 2. Fall back to window-specific Quartz capture
        if full is None:
            win_id = find_poker_window(window_keyword)
            if win_id is not None:
                print(f"[calibrate] Found poker window (id={win_id}), capturing via Quartz...")
                full = capture_window_quartz(win_id)
                if full is not None and full.mean() > 5:
                    print("[calibrate] Window capture OK.")
                else:
                    print("[calibrate] Window capture blank, falling back to mss.")
                    full = None

    if full is None:
        print("[calibrate] Capturing full screen via mss in 3 seconds — switch to your poker client...")
        time.sleep(3)
        with mss.mss() as sct:
            mon = sct.monitors[1]
            raw = sct.grab(mon)
            full = np.array(raw)
            full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)

    # Downscale for display if very large
    max_w = 1400
    scale = min(1.0, max_w / full.shape[1])
    display = cv2.resize(full, None, fx=scale, fy=scale)

    region_names = ["hole_cards", "community_cards", "pot", "my_stack", "villain_stack"]
    regions = []
    drawing = False
    start = (0, 0)
    rect = [0, 0, 0, 0]
    current_img = [display.copy()]

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp = current_img[0].copy()
            cv2.rectangle(tmp, start, (x, y), (0, 255, 0), 2)
            cv2.imshow("calibrate", tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect[0] = int(min(start[0], x) / scale)
            rect[1] = int(min(start[1], y) / scale)
            rect[2] = int(abs(x - start[0]) / scale)
            rect[3] = int(abs(y - start[1]) / scale)
            tmp = current_img[0].copy()
            cv2.rectangle(tmp, start, (x, y), (0, 255, 0), 2)
            current_img[0] = tmp
            cv2.imshow("calibrate", tmp)

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", mouse_cb)

    for name in region_names:
        current_img[0] = display.copy()
        cv2.putText(current_img[0], f"Draw: {name}  |  n=skip  s=save  q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("calibrate", current_img[0])
        print(f"  Draw a rectangle around: {name}  (n=skip, s=save, q=quit)")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n"):
                break
            elif key == ord("s"):
                if rect[2] > 0 and rect[3] > 0:
                    regions.append({"name": name, "x": rect[0], "y": rect[1],
                                    "w": rect[2], "h": rect[3]})
                    print(f"    Saved {name}: {rect}")
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    if regions:
        cfg = {"monitor": 1, "regions": regions}
        save_config(cfg)
        print(f"[calibrate] Done. Saved {len(regions)} regions.")
    else:
        print("[calibrate] No regions saved.")


# ── Live debug loop ────────────────────────────────────────────────────────────

def run_live(reader: PokerScreenReader, fps: int = 2):
    """
    Run a live debug window showing detected regions and overlaid labels.
    Press 'q' to quit.
    """
    print("[live] Starting — press 'q' in the debug window to quit")
    interval = 1.0 / fps

    with mss.mss() as sct:
        mon = sct.monitors[reader.monitor_num]
        mon_region = Region("full", mon["left"], mon["top"], mon["width"], mon["height"])

        while True:
            t0 = time.time()
            full_img = capture_region(sct, mon_region)
            state, debug_frames = reader.read(debug=True)

            # Draw all configured regions on the full screenshot
            overlay = full_img.copy()
            for name, region in reader.regions.items():
                x, y, w, h = region.x, region.y, region.w, region.h
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 180, 0), 2)
                cv2.putText(overlay, name, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 1)

            # Overlay detected values
            info_lines = [
                f"Hole:      {' '.join(state.hole_cards) or '---'}",
                f"Community: {' '.join(state.community_cards) or '---'}",
                f"Pot:       {state.pot if state.pot is not None else '---'}",
                f"My stack:  {state.my_stack if state.my_stack is not None else '---'}",
                f"Villain:   {state.villain_stack if state.villain_stack is not None else '---'}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(overlay, line, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

            # Show individual region debug frames in small panels
            panel_y = overlay.shape[0] - 160
            panel_x = 10
            for name, frame in debug_frames.items():
                if frame is None:
                    continue
                thumb = cv2.resize(frame, (200, 120))
                try:
                    overlay[panel_y:panel_y+120, panel_x:panel_x+200] = thumb
                    cv2.putText(overlay, name, (panel_x, panel_y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    panel_x += 210
                except ValueError:
                    pass  # region out of bounds

            # Downscale for display
            max_w = 1400
            scale = min(1.0, max_w / overlay.shape[1])
            display = cv2.resize(overlay, None, fx=scale, fy=scale)
            cv2.imshow("PokerScreenReader", display)

            elapsed = time.time() - t0
            wait_ms = max(1, int((interval - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenCV poker screen reader")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive region setup tool")
    parser.add_argument("--once", action="store_true",
                        help="Single capture, print detected state, exit")
    parser.add_argument("--fps", type=int, default=2,
                        help="Capture rate for live mode (default: 2)")
    parser.add_argument("--list-windows", action="store_true",
                        help="List all on-screen windows (macOS Quartz)")
    parser.add_argument("--window", type=str, default="wpt",
                        help="Keyword to match poker window app/title (default: wpt)")
    args = parser.parse_args()

    if args.list_windows:
        if not QUARTZ_AVAILABLE:
            print("[error] pyobjc not installed — run: pip install pyobjc-framework-Quartz")
            return
        windows = list_windows()
        print(f"{'ID':>8}  {'Layer':>6}  {'App':<30}  Title")
        print("-" * 70)
        for w in windows:
            print(f"{w['id']:>8}  {w['layer']:>6}  {w['app']:<30}  {w['title']}")
        return

    if args.calibrate:
        calibrate(window_keyword=args.window)
        return

    reader = PokerScreenReader()

    if args.once:
        state = reader.read(debug=False)
        print("\n── Detected Game State ──")
        print(f"  Hole cards  : {state.hole_cards or '(none)'}")
        print(f"  Community   : {state.community_cards or '(none)'}")
        print(f"  Pot         : {state.pot}")
        print(f"  My stack    : {state.my_stack}")
        print(f"  Villain     : {state.villain_stack}")
        reader.close()
        return

    # Default: live debug window
    try:
        run_live(reader, fps=args.fps)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
