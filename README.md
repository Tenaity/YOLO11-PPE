# YOLO-PPE — PPE_POC (Camera POC: Helmet + Vest + Alert)

POC realtime từ camera (webcam/RTSP) để kiểm tra PPE:
- Detect vi phạm (thiếu mũ / thiếu áo phản quang) (logic/model sẽ tích hợp tiếp theo)
- Hiển thị **banner đỏ** + text cảnh báo trên video
- Phát cảnh báo bằng **TTS offline** qua `pyttsx3` (tạm thời)

> Hiện tại POC tập trung vào bộ khung run-time (camera loop + overlay + TTS). Phần YOLO inference/rules sẽ được nối vào sau.

---

## 1) Yêu cầu hệ thống

- Python: **3.10+** (khuyến nghị 3.11)
- Webcam (để test nhanh) hoặc RTSP camera (sẽ cấu hình sau)
- macOS / Windows đều chạy được

---

## 2) Clone & Setup

### macOS / Linux
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd YOLO-PPE

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
