import cv2

def draw_alert(frame, lines):
    if not lines:
        return

    x, y, w = 20, 20, 860
    pad, line_h = 12, 30
    h = pad * 2 + line_h * len(lines)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    for i, t in enumerate(lines):
        ty = y + pad + line_h * (i + 1) - 8
        cv2.putText(frame, t, (x + 12, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
