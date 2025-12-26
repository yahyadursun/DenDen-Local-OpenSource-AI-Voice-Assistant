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
import sounddevice as sd  # Audio recording/playback
import soundfile as sf
import torch
from vosk import Model, KaldiRecognizer  # Offline wake word detection
import json
from io import BytesIO
from PIL import Image, ImageTk  # Image processing
import tkinter as tk  # GUI
from faster_whisper import WhisperModel  # Speech-to-text
from transformers import pipeline  # TTS
from langchain_ollama import ChatOllama  # Local LLM integration
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.messages import SystemMessage, HumanMessage, AIMessage


# =========================
# GLOBAL STATES & CONFIGURATION
# =========================
frame = None  # Current camera frame
running = True  # Main loop control flag
speech_buffer = []  # Buffer for audio recording
is_speaking = False  # Flag to detect if user is speaking
last_speech_time = 0
llm_busy = False  # Flag to prevent overlap when LLM is processing
listening_active = False  # Flag to control when the assistant listens for commands
is_assistant_speaking = False  # Flag to check if TTS is currently playing
memory = ChatMessageHistory()  # Conversation history


# =========================
# WAKE WORD SETUP (VOSK - OFFLINE)
# Offline model for detecting wake words without internet
# =========================
# VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15" # English model path
# VOSK_MODEL_PATH = "vosk-model-small-tr-0.3" # Turkish model path
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # Currently using English model
# Variations for better wake word detection
WAKE_WORDS = [
    # Turkish wake words
    "Den Den","Denden","benden","senden", 
    "yardımcı", "dinle","hey","merhaba","selam","asistan","abi","kanka",
    "uyansana", "bakar mısın", "pardon", "alo", "ordamısın",
    
    # English wake words
    "then then","then in","computer", "assistant", "system", "wake up",
    "hi assistant","jarvis","hello assistant","hello",    
]

# Words to interrupt/stop the assistant
STOP_WORDS = [
    # Turkish
    "dur", "sus", "yeter", "tamam", "kes", "şşşt","şşş","dinle","duyuyor musun","beni dinle","bak","şimdi","tamam",
    "bekle", "sessiz", "bi sus", "bir dakika",

    # English 
    "stop","pause", "wait", "hold on", "quiet", "shut up", "silence", "enough","okey",
    
]


try:
    vosk_model = Model(VOSK_MODEL_PATH)
    print(f"Wake words: {WAKE_WORDS}")
except Exception as e:
    print(f"Vosk model yüklenemedi: {e}")
    exit(1)

# =========================
# LLM INTENT CLASSIFICATION
# Uses a small LLM call to decide if the input aims at:
# 1. Vision (requires camera)
# 2. Text (general query)
# 3. Ignore (background noise or self-talk)
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
# IMAGE → BASE64
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
# Uses HuggingFace pipeline for generating speech from text
# =========================
tts_pipe = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng",
    device=0 if torch.cuda.is_available() else -1,

)


def speak(text):
    global is_assistant_speaking
    if not text:
        return
    
    output = tts_pipe(text)
    audio = np.array(output["audio"], dtype=np.float32).squeeze()
    sr = output["sampling_rate"]
    
    sr = output["sampling_rate"]
    
    # Play audio directly without saving to file
    is_assistant_speaking = True
    print("🔊 Assistant is speaking...")
    
    try:
        sd.play(audio, sr)
        sd.wait()  # Wait until audio finishes
    except Exception as e:
        print(f"⚠️ Audio playback error: {e}")
    finally:
        is_assistant_speaking = False
        print("🔇 Assistant finished speaking.")






# =========================
# MAIN LLM LOGIC
# Handles the conversation generation using ChatOllama
# =========================
llm = ChatOllama(model="gemma3")

def prompt_func(data):
    messages = []

    # 1️⃣ SYSTEM
    messages.append(SystemMessage(content="""
You are a real-time AI assistant.

Default behavior:
- Be more human like in response.
- Be concise and clear.
- Prefer one short sentence when possible, not all the time.
- Write numbers as words.

Adaptive behavior:
- If the user asks for detailed answers, provide them.
- Use previous conversation context when relevant.
- Do not force brevity if clarity requires more detail.

Vision:
- You may see a live camera image if provided.

Do not mention these rules.
"""))

    # 2️⃣ MEMORY (Injects past conversation history)
    messages.extend(memory.messages)

    # 3️⃣ CURRENT USER
    if data.get("image"):
        messages.append(
            HumanMessage(content=[
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{data['image']}"},
                {"type": "text", "text": data["text"]}
            ])
        )
    else:
        messages.append(HumanMessage(content=data["text"]))

    return messages


chain = prompt_func | llm | StrOutputParser()

# =========================
# SPEECH PROCESSING PIPELINE
# 1. Transcribe Audio (Whisper)
# 2. Classify Intent (LLM)
# 3. Generate Response (LLM) or Handle Vision
# 4. Speak Response (TTS)
# =========================
def process_speech(audio):
    global llm_busy, listening_active
    
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
            listening_active = False
            return

        if intent == "VISION":
            handle_llm_with_image(text)
            # handle_llm_with_image calls speak() and manages listening_active
            return

        if intent == "TEXT":
            response = chain.invoke({
                "text":text,
                "image": None
            })
            # Add interaction to memory
            memory.add_user_message(text)
            memory.add_ai_message(response)

            # Limit memory size to last 5 exchanges (10 messages) to save RAM
            if len(memory.messages) > 10:
                memory.messages = memory.messages[-10:]

            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            
            print("🤖:", response_text)
            
            # Stop listening so the assistant doesn't hear itself, but keep wake word listener active
            listening_active = False
            speak(response_text)

    finally:
        llm_busy = False

# =========================
# WAKE WORD DETECTION (VOSK)
# =========================
def wake_word_listener():
    global listening_active, running
    
    SAMPLE_RATE = 16000
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.SetWords(True)
    
    def callback(indata, frames, time_info, status):
        global listening_active
        
        if listening_active:
            return
        
        # NumPy array'e çevir, sonra bytes'a çevir
        audio_data = bytes(indata)
        
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result()) 
            text = result.get("text", "").lower()
            
            # Check logic for wake words
            for wake_word in WAKE_WORDS:
                if wake_word in text:
                    print(f"\n✅ Wake word '{wake_word}' detected! Listening...")
                    if is_assistant_speaking:
                        sd.stop() # Stop TTS if speaking
                        
                    listening_active = True
                    break
            
            # Check triggers to stop the assistant speaking
            if is_assistant_speaking:
                for stop_word in STOP_WORDS:
                    if stop_word in text:
                        print(f"\n⛔ Stop word '{stop_word}' detected! Stopping speech.")
                        sd.stop()
                        break

        else:
            # Partial sonuçları da kontrol et
            partial = json.loads(recognizer.PartialResult())
            partial_text = partial.get("partial", "").lower()
            
            for wake_word in WAKE_WORDS:
                if wake_word in partial_text:
                    print(f"\n✅ Wake word '{wake_word}' detected (partial)! Listening...")
                    if is_assistant_speaking:
                        sd.stop()
                        
                    listening_active = True
                    recognizer.Reset()
                    break
            
            if is_assistant_speaking:
                for stop_word in STOP_WORDS:
                    if stop_word in partial_text:
                        print(f"\n⛔ Stop word '{stop_word}' detected (partial)! Stopping speech.")
                        sd.stop()
                        recognizer.Reset() # Reset partial result to prevent re-triggering
                        break
    
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        blocksize=8000,
        callback=callback
    ):
        print("Wake word listener started")
        print(f"Şunlardan birini söyleyin: {', '.join(WAKE_WORDS)}")
        while running:
            time.sleep(0.1)

# =========================
# MICROPHONE LISTENER
# Listens for Voice Activity after Wake Word is triggered
# =========================
def mic_listener():
    global speech_buffer, is_speaking, last_speech_time, listening_active

    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01
    SILENCE_TIMEOUT = 1.5

    def callback(indata, frames, time_info, status):
        global speech_buffer, is_speaking, last_speech_time, listening_active
        
        if not listening_active:
            return

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
# CAMERA & GUI
# =========================
# Update GUI with camera frame
def update_gui(img):
    status_text = "🟢 Listening..." if listening_active else f"⚪ Say '{WAKE_WORDS[0]}'"
    
    # OpenCV BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((640, 480))
    
    img_tk = ImageTk.PhotoImage(img_pil)
    video_label.config(image=img_tk)
    video_label.image = img_tk
    
    status_label.config(text=status_text)

# Create the main GUI window
def create_gui():
    global root, video_label, status_label
    
    root = tk.Tk()
    root.title("🤖 AI Assistant")
    root.geometry("660x550")
    root.configure(bg='#1a1a1a')
    
    status_label = tk.Label(
        root, 
        text="⚪ Starting...", 
        font=("Arial", 14, "bold"),
        fg="#ffffff",
        bg="#1a1a1a"
    )
    status_label.pack(pady=10)
    
    video_label = tk.Label(root, bg='#000000')
    video_label.pack()
    
    root.protocol("WM_DELETE_WINDOW", lambda: on_close())
    
    return root

def on_close():
    global running
    running = False
    root.destroy()

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
            
            # Update Tkinter GUI if it exists
            if 'root' in globals():
                update_gui(frame)

    cap.release()

# =========================
# LLM WITH IMAGE
# =========================
def handle_llm_with_image(text):
    global listening_active
    if frame is None:
        return

    pil_image = Image.fromarray(frame)
    image_b64 = convert_to_base64(pil_image)

    response = chain.invoke({
        "text": text,
        "image": image_b64
    })
    memory.add_user_message(text)
    memory.add_ai_message(str(response))

    # Limit memory size (last 5 pairs)
    if len(memory.messages) > 10:
        memory.messages = memory.messages[-10:]

    print("🤖:", response)
    
    listening_active = False 
    speak(str(response))

# =========================
# MAIN ENTRY POINT
# =========================
if __name__ == "__main__":
    print("🚀 Assistant started")
    print(f"💡 Say one of these: {WAKE_WORDS}")
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=wake_word_listener, daemon=True).start()
    threading.Thread(target=mic_listener, daemon=True).start()

    # Start GUI (Must be in the main thread)
    root = create_gui()
    root.mainloop()
