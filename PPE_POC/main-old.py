import time
from collections import defaultdict, deque
import os
import cv2
import numpy as np
import simpleaudio as sa
from ultralytics import YOLO

# ================== CONFIG ==================
MODEL_PATH = "runs/detect/train/weights/best.pt"    
SOURCE = 0                # 0 = webcam; hoặc RTSP như "rtsp://user:pass@ip:554/stream"
IMG_SIZE = 640

CONF = 0.35
IOU_ASSOC = 0.05          # overlap tối thiểu để gán PPE vào người
WINDOW = 15               # số frame voting
MISSING_RATIO = 0.70      # >=70% frame thiếu -> vi phạm
COOLDOWN_SEC = 15         # chống spam loa theo từng người

# ROI: polygon khu vực "đang đi vào cổng" (sửa theo camera)
ROI_POLY = np.array([(200, 200), (1100, 200), (1100, 900), (200, 900)], dtype=np.int32)

AUDIO_HELMET = "helmet_missing.wav"
AUDIO_VEST = "vest_missing.wav"
AUDIO_BOTH = "ppe_missing.wav"

# ================== UTIL ==================
def play_wav(path: str):
    if not os.path.exists(path):
        print(f"[audio] missing file: {path}")
        return
    try:
        sa.WaveObject.from_wave_file(path).play()  # non-blocking
    except Exception as e:
        print(f"[audio] failed: {e}")

def point_in_poly(x, y, poly) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def get_class_id(names: dict, target: str):
    # names: {id: "class_name"}
    target_lower = target.lower()
    for k, v in names.items():
        if str(v).lower() == target_lower:
            return int(k)
    return None

# ================== MAIN ==================
def main():
    model = YOLO(MODEL_PATH)
    names = model.names  # dict
    print("[model classes]", names)

    # Try to detect person class from model (Construction-PPE có 'Person' theo docs) :contentReference[oaicite:1]{index=1}
    person_id = get_class_id(names, "person") or get_class_id(names, "Person")
    helmet_id = get_class_id(names, "helmet") or get_class_id(names, "safety-helmet")
    vest_id = get_class_id(names, "vest") or get_class_id(names, "reflective-vest")

    no_helmet_id = get_class_id(names, "no_helmet")  # nếu bạn train theo Construction-PPE
    # no_vest thường không có trong Construction-PPE (theo list trong docs) :contentReference[oaicite:2]{index=2}

    if person_id is None:
        # fallback: dùng model COCO để detect person
        person_model = YOLO("yolov8n.pt")
        print("[mode] 2-model (person from COCO + PPE from your model)")
    else:
        person_model = None
        print("[mode] 1-model (your model includes Person)")

    hist = defaultdict(lambda: deque(maxlen=WINDOW))  # track_id -> [(helmet_ok, vest_ok, in_roi)]
    last_alert = defaultdict(lambda: 0.0)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    # tracker state persists inside model.track(persist=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        cv2.polylines(frame, [ROI_POLY], True, (255, 255, 255), 2)

        now = time.time()

        # --- PERSON TRACK ---
        if person_model is None:
            # track all classes but we'll filter persons using person_id
            res_track = model.track(frame, imgsz=IMG_SIZE, conf=CONF, persist=True, stream=False, verbose=False)[0]
            boxes = res_track.boxes
            if boxes is None or len(boxes) == 0:
                cv2.imshow("PPE POC", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # build person list with track ids
            persons = []
            for b in boxes:
                cls = int(b.cls.item())
                if cls != person_id:
                    continue
                tid = int(b.id.item()) if b.id is not None else -1
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                persons.append((tid, (x1, y1, x2, y2)))

            # PPE detections from SAME results (no extra inference)
            ppe_boxes = []
            for b in boxes:
                cls = int(b.cls.item())
                if cls in [helmet_id, vest_id, no_helmet_id]:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    ppe_boxes.append((cls, (x1, y1, x2, y2)))
        else:
            # 2-model approach: track persons with COCO model
            res_p = person_model.track(frame, imgsz=IMG_SIZE, conf=CONF, persist=True, stream=False, verbose=False)[0]
            persons = []
            if res_p.boxes is not None:
                for b in res_p.boxes:
                    cls = int(b.cls.item())
                    if cls != 0:  # COCO person=0
                        continue
                    tid = int(b.id.item()) if b.id is not None else -1
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    persons.append((tid, (x1, y1, x2, y2)))

            # PPE predict (separate inference)
            res_ppe = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
            ppe_boxes = []
            if res_ppe.boxes is not None:
                for b in res_ppe.boxes:
                    cls = int(b.cls.item())
                    if cls in [helmet_id, vest_id, no_helmet_id]:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        ppe_boxes.append((cls, (x1, y1, x2, y2)))

        # --- ASSOC PPE -> PERSON + VOTING ---
        for tid, pb in persons:
            x1, y1, x2, y2 = pb
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            in_roi = point_in_poly(cx, cy, ROI_POLY)

            # Determine helmet/vest presence by overlap with person bbox
            helmet_ok = False
            vest_ok = False
            no_helmet_hit = False

            for cls, bb in ppe_boxes:
                if iou(pb, bb) < IOU_ASSOC:
                    continue
                if cls == helmet_id:
                    helmet_ok = True
                elif cls == vest_id:
                    vest_ok = True
                elif no_helmet_id is not None and cls == no_helmet_id:
                    no_helmet_hit = True

            # nếu model có "no_helmet" thì ưu tiên coi là thiếu helmet ngay
            if no_helmet_hit:
                helmet_ok = False

            hist[tid].append((helmet_ok, vest_ok, in_roi))

            # voting
            if len(hist[tid]) >= WINDOW and in_roi:
                hmiss = [not h for (h, v, r) in hist[tid] if r]
                vmiss = [not v for (h, v, r) in hist[tid] if r]
                miss_h = (sum(hmiss) / max(1, len(hmiss))) >= MISSING_RATIO
                miss_v = (sum(vmiss) / max(1, len(vmiss))) >= MISSING_RATIO

                if (miss_h or miss_v) and (now - last_alert[tid] >= COOLDOWN_SEC):
                    if miss_h and miss_v:
                        play_wav(AUDIO_BOTH)
                    elif miss_h:
                        play_wav(AUDIO_HELMET)
                    else:
                        play_wav(AUDIO_VEST)

                    last_alert[tid] = now
                    print(f"[ALERT] tid={tid} miss_helmet={miss_h} miss_vest={miss_v}")

            # draw
            label = f"ID {tid} H:{'Y' if helmet_ok else 'N'} V:{'Y' if vest_ok else 'N'}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(frame, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("PPE POC", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
