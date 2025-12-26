# =========================
# LIBRARY IMPORTS
# =========================
import os
import time
import base64
import threading
import platform
import cv2  # OpenCV for camera
import numpy as np
import sounddevice as sd  # Audio recording
import soundfile as sf
import torch

from io import BytesIO
from PIL import Image
from faster_whisper import WhisperModel  # Speech-to-text
from transformers import pipeline  # TTS
from langchain_ollama import ChatOllama  # LLM
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# =========================
# GLOBAL STATES & CONFIGURATION
# =========================
frame = None  # Current camera frame
running = True  # Control flag
speech_buffer = []
is_speaking = False
last_speech_time = 0
llm_busy = False  # Prevent overlapping requests

# =========================
# LLM INTENT CLASSIFICATION
# Determines if the user wants Text, Vision or nothing
# =========================

intent_llm = ChatOllama(model="gemma3")

def classify_intent(text: str) -> str:
    """
    Returns one of:
    - IGNORE
    - TEXT
    - VISION
    """

    prompt = f"""
You are an intent classifier for a voice assistant.

User said:
"{text}"

Classify the intent.

Rules:
- If the user is NOT talking to the assistant → IGNORE
- If the user is talking to like herself/himself  → IGNORE
- If the user asks a general question or gives a command WITHOUT needing the camera → TEXT
- If the user asks about what they are seeing, the camera, an object, or says things like
  "look", "see", "what is this", "ne bu", "bak", "gör" → VISION

Reply with ONLY ONE WORD:
IGNORE
TEXT
VISION
"""

    response = intent_llm.invoke(prompt)

    if hasattr(response, "content"):
        result = response.content.strip().upper()
    else:
        result = str(response).strip().upper()

    print("🧠 Intent:", result)
    return result

# =========================
# IMAGE UTILITIES (PIL -> Base64)
# =========================
def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# =========================
# WHISPER MODEL
# =========================
whisper_model = WhisperModel(
    "small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)

# =========================
# TTS (TEXT-TO-SPEECH)
# Uses Facebook MMS-TTS model
# =========================
tts_pipe = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng",
    device=0 if torch.cuda.is_available() else -1
)

def speak(text):
    if not text:
        return

    output = tts_pipe(text)
    audio = np.array(output["audio"], dtype=np.float32).squeeze()
    sr = output["sampling_rate"]

    # Save to file then play (V4 specific implementation)
    sf.write("response.wav", audio, sr)

    if platform.system() == "Windows":
        os.system("start response.wav")
    elif platform.system() == "Darwin":
        os.system("afplay response.wav")
    else:
        os.system("xdg-open response.wav")

# =========================
# LLM SETUP
# =========================
llm = ChatOllama(model="gemma3")

def prompt_func(data):
    system_prompt = """
You are a real-time multimodal AI assistant.

You can SEE the image provided from the user's camera.
The image is a LIVE camera frame, not a description.

Rules:
- Give short, clear answers.
- Do not explain unless asked.
- Maximum 2–3 sentences.
- NEVER say you cannot see the image.
- NEVER ask the user to describe the image.
- ALWAYS assume the image is what the user is currently looking at.
- Answer naturally, as a helpful assistant.
- If the user asks what you see, describe the image clearly.
- If the question is unrelated to the image, still answer normally.
"""

    return [
        HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{data['image']}"
                },
                {"type": "text", "text": data["text"]}
            ]
        )
    ]


chain = prompt_func | llm | StrOutputParser()



# =========================
# SPEECH PROCESSING
# Transcribe -> Intent -> Respond
# =========================
def process_speech(audio):
    global llm_busy

    if llm_busy:
        print("⏳ LLM is busy, ignoring input.")
        return

    llm_busy = True
    try:
        segments, _ = whisper_model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=5
        )

        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            return

        print(f"\n🧑 User: {text}")

        intent = classify_intent(text)

        if intent == "IGNORE":
            print("🤖 Status: IGNORED")
            return

        if intent == "VISION":
            handle_llm_with_image(text)
            return

        if intent == "TEXT":
            response = llm.invoke(text)

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            print("🤖:", response_text)
            speak(response_text)

    finally:
        llm_busy = False



# =========================
# MICROPHONE LISTENER
# Continuously listens for audio
# =========================
def mic_listener():
    global speech_buffer, is_speaking, last_speech_time

    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01
    SILENCE_TIMEOUT = 1.0

    def callback(indata, frames, time_info, status):
        global speech_buffer, is_speaking, last_speech_time

        volume = np.linalg.norm(indata)

        if volume > SILENCE_THRESHOLD:
            is_speaking = True
            last_speech_time = time.time()
            speech_buffer.append(indata.copy())
        else:
            if is_speaking and time.time() - last_speech_time > SILENCE_TIMEOUT:
                is_speaking = False
                audio = np.concatenate(speech_buffer, axis=0).squeeze()
                speech_buffer = []
                threading.Thread(
                    target=process_speech,
                    args=(audio,),
                    daemon=True
                ).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback
    ):
        while running:
            time.sleep(0.1)

# =========================
# CAMERA THREAD
# Captures video frames
# =========================
def camera_thread():
    global frame, running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        running = False
        return

    print("📷 Camera active")

    while running:
        ret, frm = cap.read()
        if ret:
            frame = frm
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("e"):
            running = False

    cap.release()
    cv2.destroyAllWindows()

# =========================
# VISION HANDLER
# Sends image + text to LLM
# =========================
def handle_llm_with_image(text):
    if frame is None:
        return

    pil_image = Image.fromarray(frame)
    image_b64 = convert_to_base64(pil_image)

    response = chain.invoke({
        "text": text,
        "image": image_b64
    })

    print("🤖:", response)
    speak(str(response))

# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":
    print("🚀 Assistant started (press 'e' to exit)")
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=mic_listener, daemon=True).start()

    while running:
        time.sleep(0.5)
