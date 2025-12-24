import time
import threading
import queue
from pathlib import Path
import re
import unicodedata
import traceback

import cv2
import pyttsx3
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = "./runs/detect/train/weights/best.pt"
CAMERA_SOURCE = 0  # 0 webcam OR "rtsp://..."

# class mapping (your confirmed)
PERSON_CLASS = 6
HELMET_CLASS = 0
VEST_CLASS = 2
CLASSES_FILTER = [PERSON_CLASS, HELMET_CLASS, VEST_CLASS]

YOLO_CONF = 0.15
IMGSZ = 640
DEVICE = ""

PERSON_MIN_CONF = 0.20
HELMET_MIN_CONF = 0.6 #0.65
VEST_MIN_CONF = 0.6 # 0.60  # vest < 0.60 => treat as NOT OK => red

HEAD_Y_RATIO = 0.35
TORSO_Y1_RATIO = 0.25
TORSO_Y2_RATIO = 0.85

ALERT_COOLDOWN_SEC = 5.0
DRAW_ALL_DETECTIONS = True

NO_PERSON_FRAMES_TO_WARN = 10

# robustness
MAX_READ_FAIL_STREAK = 30
REOPEN_SLEEP_SEC = 0.5

# TTS
ENABLE_TTS = False              # set False to check if crash comes from TTS
PREFERRED_VOICE_NAME = "Linh"  # try pick voice contains "Linh" first
# ==================================================


def to_ascii_upper(s: str) -> str:
    """Convert Vietnamese w/ diacritics -> ASCII, then UPPERCASE."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # special-case Đ/đ
    s = s.replace("Đ", "D").replace("đ", "d")
    return s.upper()


def safe_log(msg: str):
    # Minimal file logging to diagnose crashy behaviors
    with open("runtime.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


class TTSPlayer:
    """
    Offline TTS via pyttsx3.
    - Display text: YOU handle (we show overlay separately)
    - Speak text: can keep Vietnamese with diacritics
    """
    def __init__(self, rate=170, volume=1.0, prefer_name="Linh"):
        self.q = queue.Queue(maxsize=50)
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        chosen = self._select_voice(prefer_name=prefer_name)
        if chosen:
            print(f"[TTS] Using voice: {chosen}")
            safe_log(f"TTS voice selected: {chosen}")
        else:
            print("[TTS] No suitable Vietnamese voice found. Using default voice.")
            safe_log("TTS voice selected: DEFAULT")
            self._print_voices()

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _print_voices(self):
        try:
            voices = self.engine.getProperty("voices") or []
            print("[TTS] Available voices:")
            for i, v in enumerate(voices):
                print(f"  {i:02d}. name={getattr(v,'name','')} | id={getattr(v,'id','')}")
        except Exception:
            pass

    def _select_voice(self, prefer_name="Linh"):
        """
        Selection priority:
        1) name/id contains preferred name (e.g., "Linh")
        2) name/id/languages indicates Vietnamese (vi / vi-VN / vietnam)
        3) else: return None (keep default)
        """
        try:
            voices = self.engine.getProperty("voices") or []

            def norm(x): return (x or "").lower()

            def contains_vi_token(s: str) -> bool:
                s = norm(s)
                return bool(re.search(r'(^|[^a-z0-9])vi([-_]?vn)?([^a-z0-9]|$)', s))

            pref = norm(prefer_name)

            # 1) Preferred name
            if pref:
                for v in voices:
                    name = norm(getattr(v, "name", ""))
                    vid = norm(getattr(v, "id", ""))
                    if pref in name or pref in vid:
                        self.engine.setProperty("voice", v.id)
                        return getattr(v, "name", v.id)

            # 2) Vietnamese indicators
            candidates = []
            for v in voices:
                name = norm(getattr(v, "name", ""))
                vid = norm(getattr(v, "id", ""))
                langs = norm(str(getattr(v, "languages", "") or ""))

                if ("vietnam" in name or "vietnam" in vid or "vietnam" in langs or
                    "vietnamese" in name or "vietnamese" in vid or "vietnamese" in langs or
                    contains_vi_token(name) or contains_vi_token(vid) or contains_vi_token(langs)):
                    candidates.append(v)

            if not candidates:
                return None

            # Prefer strongest id match
            def score(v):
                name = norm(getattr(v, "name", ""))
                vid = norm(getattr(v, "id", ""))
                sc = 0
                if "vietnamese" in vid: sc += 100
                if "vietnam" in vid: sc += 90
                if contains_vi_token(vid): sc += 80
                if "vietnamese" in name: sc += 60
                if "vietnam" in name: sc += 50
                if contains_vi_token(name): sc += 40
                return sc

            best = max(candidates, key=score)
            self.engine.setProperty("voice", best.id)
            return getattr(best, "name", best.id)

        except Exception:
            return None

    def say(self, text: str):
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        while True:
            text = self.q.get()
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                safe_log(f"TTS error: {repr(e)}")


def draw_label(img, text, x, y, color_bgr, scale=1.0, thickness=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 10)
    x0 = max(0, x)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 16, y0 + th + baseline + 12), color_bgr, -1)
    cv2.putText(img, text, (x0 + 8, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_big_warning(img, text):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = 90
    cv2.rectangle(img, (x - 20, y - th - baseline - 25), (x + tw + 20, y + baseline + 25), (0, 0, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def center_in_region(box, region) -> bool:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1, ry1, rx2, ry2 = region
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def open_camera(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    # Helps some RTSP/backends; harmless for webcam
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}

    print("Model classes (id -> name):")
    for k in sorted(names.keys()):
        print(f"  {k}: {names[k]}")
    print(f"\n[Using] PERSON={PERSON_CLASS}, HELMET={HELMET_CLASS}, VEST={VEST_CLASS}\n")

    cap = open_camera(CAMERA_SOURCE)
    if cap is None:
        raise RuntimeError(f"Cannot open camera/source: {CAMERA_SOURCE}")

    tts = TTSPlayer(rate=170, volume=1.0, prefer_name=PREFERRED_VOICE_NAME) if ENABLE_TTS else None

    win = "PPE Realtime"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_alert_at = {"NO_HELMET": 0.0, "NO_VEST": 0.0, "NO_BOTH": 0.0}
    no_person_streak = 0
    read_fail_streak = 0

    while True:
        try:
            # If window closed manually, exit cleanly
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                safe_log("Window closed by user.")
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                read_fail_streak += 1
                if read_fail_streak >= MAX_READ_FAIL_STREAK:
                    safe_log("Camera read failing. Reopening camera...")
                    cap.release()
                    time.sleep(REOPEN_SLEEP_SEC)
                    cap = open_camera(CAMERA_SOURCE)
                    read_fail_streak = 0
                    if cap is None:
                        safe_log("Reopen camera failed. Retrying...")
                        time.sleep(1.0)
                continue
            read_fail_streak = 0

            annotated = frame.copy()

            # Predict (wrap to avoid Python exception killing loop)
            try:
                results = model.predict(
                    source=frame,
                    classes=CLASSES_FILTER,
                    conf=YOLO_CONF,
                    imgsz=IMGSZ,
                    device=DEVICE if DEVICE else None,
                    verbose=False,
                )
            except Exception as e:
                safe_log("Predict exception:\n" + traceback.format_exc())
                draw_big_warning(annotated, "ERROR PREDICT")
                cv2.imshow(win, annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            r = results[0]

            persons, helmets, vests = [], [], []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.detach().cpu().numpy()
                cls = r.boxes.cls.detach().cpu().numpy().astype(int)
                conf = r.boxes.conf.detach().cpu().numpy()

                # Debug draw all detections (yellow boxes)
                if DRAW_ALL_DETECTIONS:
                    for b, c, s in zip(xyxy, cls, conf):
                        x1, y1, x2, y2 = map(int, b.tolist())
                        label = f"{names.get(c, c)}:{float(s):.2f}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        draw_label(annotated, label, x1, y1, (0, 255, 255), scale=0.7, thickness=2)

                # Collect for PPE logic
                for b, c, s in zip(xyxy, cls, conf):
                    box = b.tolist()
                    score = float(s)
                    if c == PERSON_CLASS and score >= PERSON_MIN_CONF:
                        persons.append((box, score))
                    elif c == HELMET_CLASS:
                        helmets.append((box, score))
                    elif c == VEST_CLASS:
                        vests.append((box, score))

            # -------- MULTI-PERSON EVALUATION --------
            miss_helmet_cnt = 0
            miss_vest_cnt = 0
            miss_both_cnt = 0
            banner_text = None  # will be ASCII UPPER

            if len(persons) == 0:
                no_person_streak += 1
                if no_person_streak >= NO_PERSON_FRAMES_TO_WARN:
                    banner_text = "CANH BAO: KHONG PHAT HIEN PERSON"
            else:
                no_person_streak = 0

                for p_box, _ in persons:
                    px1, py1, px2, py2 = p_box
                    h = max(1.0, py2 - py1)

                    head_region = [px1, py1, px2, py1 + HEAD_Y_RATIO * h]
                    torso_region = [px1, py1 + TORSO_Y1_RATIO * h, px2, py1 + TORSO_Y2_RATIO * h]

                    best_helmet = 0.0
                    for hb, hc in helmets:
                        if center_in_region(hb, head_region):
                            best_helmet = max(best_helmet, hc)

                    best_vest = 0.0
                    for vb, vc in vests:
                        if center_in_region(vb, torso_region):
                            best_vest = max(best_vest, vc)

                    helmet_ok = best_helmet >= HELMET_MIN_CONF
                    vest_ok = best_vest >= VEST_MIN_CONF  # vest < 0.60 => red

                    missing_msgs = []
                    missing_helmet = False
                    missing_vest = False

                    # Overlay messages: VIETNAMESE UPPER NO DIACRITICS
                    if best_helmet <= 0.0:
                        missing_msgs.append("KHONG THAY MU")
                        missing_helmet = True
                    elif best_helmet < HELMET_MIN_CONF:
                        missing_msgs.append(f"MU<{HELMET_MIN_CONF:.2f}")
                        missing_helmet = True

                    if best_vest <= 0.0:
                        missing_msgs.append("KHONG THAY AO")
                        missing_vest = True
                    elif best_vest < VEST_MIN_CONF:
                        missing_msgs.append(f"AO<{VEST_MIN_CONF:.2f}")
                        missing_vest = True

                    x1, y1, x2, y2 = map(int, p_box)

                    if (not helmet_ok) or (not vest_ok):
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        draw_label(
                            annotated,
                            "CANH BAO: " + " | ".join(missing_msgs),
                            x1, y1,
                            (0, 0, 255),
                            scale=1.0,
                            thickness=4,
                        )

                        if missing_helmet and missing_vest:
                            miss_both_cnt += 1
                        elif missing_helmet:
                            miss_helmet_cnt += 1
                        elif missing_vest:
                            miss_vest_cnt += 1
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        draw_label(annotated, "PPE OK", x1, y1, (0, 255, 0), scale=0.8, thickness=2)

                # Global banner summary (ASCII UPPER)
                if miss_both_cnt or miss_helmet_cnt or miss_vest_cnt:
                    parts = []
                    if miss_both_cnt:
                        parts.append(f"{miss_both_cnt} NGUOI THIEU MU+AO")
                    if miss_helmet_cnt:
                        parts.append(f"{miss_helmet_cnt} NGUOI THIEU MU")
                    if miss_vest_cnt:
                        parts.append(f"{miss_vest_cnt} NGUOI THIEU AO")
                    banner_text = " | ".join(parts)

            if banner_text:
                draw_big_warning(annotated, banner_text)

            # -------- AUDIO (Vietnamese WITH diacritics) --------
            now = time.time()

            def can_alert(key): 
                return (now - last_alert_at[key]) >= ALERT_COOLDOWN_SEC

            if ENABLE_TTS and tts is not None:
                # severity: both > helmet > vest
                if miss_both_cnt > 0 and can_alert("NO_BOTH"):
                    tts.say("Cảnh báo. Có người chưa đội mũ và chưa mặc áo phản quang.")
                    last_alert_at["NO_BOTH"] = now
                elif (miss_helmet_cnt > 0 or miss_both_cnt > 0) and can_alert("NO_HELMET"):
                    tts.say("Cảnh báo. Có người chưa đội mũ bảo hộ.")
                    last_alert_at["NO_HELMET"] = now
                elif (miss_vest_cnt > 0 or miss_both_cnt > 0) and can_alert("NO_VEST"):
                    tts.say("Cảnh báo. Có người chưa mặc áo phản quang.")
                    last_alert_at["NO_VEST"] = now

            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        except Exception:
            # Catch Python exceptions (won't catch segfault hard-crash)
            safe_log("Main loop exception:\n" + traceback.format_exc())
            # Keep running instead of exiting
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
