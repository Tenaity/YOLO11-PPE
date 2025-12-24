from ultralytics import YOLO
import cv2
import math
from .app.alerts.tts import TTS
from .app.ui.overlay import draw_alert

def main():
    # Use absolute path for the model to avoid CWD issues
    model_path = "/Users/tenaity/Documents/YOLO-PPE/runs/detect/train/weights/best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to standard yolo11n.pt if trained weights not found/loadable, 
        # though the user specifically asked for the notebook logic.
        model = YOLO("yolo11n.pt") 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Set camera resolution for better performance/quality if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tts = TTS(min_gap_sec=3.0)
    tts.speak("Bắt đầu POC. Hệ thống đang khởi động.")

    # Class names mapping based on the notebook/YAML
    # 0: helmet, 2: vest, 4: goggles
    class_names = {0: "HELMET", 2: "VEST", 4: "GOGGLES"}

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Run inference on the frame
        # Filter for classes: 0 (helmet), 2 (vest), 4 (goggles)
        results = model.predict(frame, classes=[0, 2, 4], verbose=False)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Process detections for alerts/overlay
        counts = {0: 0, 2: 0, 4: 0}
        
        # Check if boxes exist
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id in counts:
                    counts[cls_id] += 1

        # Prepare overlay text
        lines = []
        for cls_id, name in class_names.items():
            count = counts[cls_id]
            status = "CO" if count > 0 else "KHONG"
            lines.append(f"{name}: {status} ({count})")

        # Example TTS Logic: Speak if something is detected (optional)
        # or alert if missing. For now, we update the status lines.
        
        # Draw the custom alert overlay
        draw_alert(annotated_frame, lines)

        cv2.imshow("PPE POC - Realtime Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
