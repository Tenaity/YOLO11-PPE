import cv2
from .app.alerts.tts import TTS
from .app.ui.overlay import draw_alert

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    tts = TTS(min_gap_sec=3.0)
    tts.speak("Bắt đầu POC. Đây là test cảnh báo và hiển thị.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        lines = [
            "CANH BAO: KHONG DOI MU BAO HO",
            "CANH BAO: KHONG MAC AO PHAN QUANG / BAO HO",
        ]
        draw_alert(frame, lines)

        cv2.imshow("PPE POC - Overlay + TTS", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
