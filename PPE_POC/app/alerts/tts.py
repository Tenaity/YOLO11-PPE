import time
import pyttsx3

class TTS:
    """
    TTS offline qua pyttsx3 (macOS dùng NSSpeechSynthesizer).
    Có throttle để tránh nói spam.
    """
    def __init__(self, rate=185, volume=1.0, min_gap_sec=2 .5, prefer_voice_contains=("Mai", "Vietnam")):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        self.min_gap_sec = min_gap_sec
        self._last_spoken_at = 0.0

        # cố gắng chọn voice tiếng Việt nếu có
        try:
            voices = self.engine.getProperty("voices")
            chosen = None
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                vid = (getattr(v, "id", "") or "").lower()
                if any(k.lower() in name or k.lower() in vid for k in prefer_voice_contains):
                    chosen = v.id
                    break
            if chosen:
                self.engine.setProperty("voice", chosen)
        except Exception:
            pass

    def speak(self, text: str):
        now = time.time()
        if now - self._last_spoken_at < self.min_gap_sec:
            return
        self._last_spoken_at = now

        # pyttsx3 là blocking nếu runAndWait() trực tiếp.
        # Với POC: gọi nhanh 1 câu/lần nhờ throttle thì ổn.
        self.engine.say(text)
        self.engine.runAndWait()
