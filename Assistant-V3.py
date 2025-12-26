import base64
import os
import platform
import threading
import cv2
from io import BytesIO
from PIL import Image
from gtts import gTTS
import speech_recognition as sr

from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
import soundfile as sf
import torch


# =========================
# GLOBAL STATES (DURUM BİLGİLERİ)
# =========================
frame = None
running = True


# =========================
# IMAGE → BASE64 (GÖRÜNTÜ -> METİN)
# =========================
def convert_to_base64(pil_image):
    """
    PIL formatındaki görüntüyü Base64 dizesine çevirir.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# =========================
# SPEECH TO TEXT (SES TANIMA)
# =========================
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel    

from faster_whisper import WhisperModel

whisper_model = WhisperModel(
    "small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)
def speechToText():
    """
    Mikrofondan ses kaydeder ve Whisper modelini kullanarak metne çevirir.
    """
    SAMPLE_RATE = 16000
    DURATION = 4  # saniye

    print("🎤 Konuş...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = audio.squeeze()

    segments, info = whisper_model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=5
    )

    text = " ".join(seg.text for seg in segments).strip()

    if not text:
        print("❌ Konuşma algılanmadı")
        return ""

    print("📝 Sonuç:", text)
    return text



# =========================
# TEXT TO SPEECH (METNİ SESE ÇEVİRME)
# =========================
# def speech(text):
#     if not text:
#         return

#     tts = gTTS(text, lang="en")
#     tts.save("response.mp3")

#     if platform.system() == "Windows":
#         os.system("start response.mp3")
#     elif platform.system() == "Darwin":
#         os.system("afplay response.mp3")
#     else:
#         os.system("xdg-open response.mp3")

import numpy as np

def speech(text):
    """
    Metni Facebook MMS TTS modelini kullanarak sese dönüştürür ve oynatır.
    """
    if not text:
        return

    print("🔊 Generating speech...")

    output = tts_pipe(text)

    audio = output["audio"]
    sample_rate = output["sampling_rate"]

    # 🛠️ FIX: ensure mono + float32
    audio = np.array(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.squeeze()

    sf.write(
        "response.wav",
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16"
    )

    if platform.system() == "Windows":
        os.system("start response.wav")
    elif platform.system() == "Darwin":
        os.system("afplay response.wav")
    else:
        os.system("xdg-open response.wav")





# =========================
# LLM (BÜYÜK DİL MODELİ)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tts_pipe = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng",
    device=device
)





llm = ChatOllama(model="gemma3")


def prompt_func(data):
    """
    Ollama için prompt yapısını hazırlar (Görsel + Metin).
    """
    return [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{data['image']}",
                },
                {
                    "type": "text",
                    "text": data["text"],
                },
            ]
        )
    ]


chain = prompt_func | llm | StrOutputParser()


# =========================
# CAPTURE HANDLER (YAKALAMA İŞLEYİCİSİ)
# =========================
def handle_capture():
    """
    'q' tuşuna basıldığında çağrılır.
    Görüntüyü kaydeder, sesi alır, LLM'e gönderir ve cevabı seslendirir.
    """
    global frame

    if frame is None:
        return

    cv2.imwrite("captured.jpeg", frame)
    pil_image = Image.open("captured.jpeg")
    image_b64 = convert_to_base64(pil_image)

    user_text = speechToText()
    if not user_text:
        return

    response = chain.invoke(
        {"text": user_text, "image": image_b64}
    )

    print("🤖 LLM:", response)
    speech(response)


# =========================
# CAMERA THREAD (KAMERA İŞ PARÇACIĞI)
# =========================
def camera_thread():
    """
    Kamera akışını sürekli ekranda gösteren ve tuşları dinleyen ana döngü.
    """
    global frame, running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        running = False
        return

    print("✅ Camera started (q: capture | e: exit)")

    while running:
        ret, frm = cap.read()
        if not ret:
            continue

        frame = frm
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            running = False
        elif key == ord('q'):
            threading.Thread(
                target=handle_capture,
                daemon=True
            ).start()

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MAIN (ANA PROGRAM)
# =========================
if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_thread)
    cam_thread.start()
    cam_thread.join()
