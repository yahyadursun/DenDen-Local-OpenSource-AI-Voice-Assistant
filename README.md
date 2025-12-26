# Den Den - Local OpenSource AI Voice Assistant (V5) 🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-green)

[🇹🇷 Türkçe README için tıklayın](./README_TR.md)

A powerful, **fully offline** AI assistant that combines real-time voice interaction, computer vision, and local LLM intelligence. Designed to run entirely on your local machine, ensuring complete privacy and zero data leakage.

## 🌟 Key Features (V5)

*   **100% Local & Private:** No data is sent to the cloud. Powered by [Ollama](https://ollama.com) and local models.
*   **Real-time Voice Interaction:**
    *   **Wake Word Detection:** Uses `Vosk` for offline, low-latency wake word listening (e.g., "Den Den", "Hello",).
    *   **Speech-to-Text:** High-accuracy transcription using `faster-whisper`.
    *   **Text-to-Speech:** Natural sounding voice via local TTS pipelines.
*   **Vision Capabilities 👁️:** Can see and analyze the world through your webcam. Ask "What is this?" or "Look at this" to trigger visual analysis.
*   **Smart Intent Classification:** Intelligently distinguishes between irrelevant background noise, general questions, and vision-related requests.
*   **GUI Interface:** A clean visual interface showing camera feed and assistant status.

## 📜 Version History

*   **V5 (Latest):** The most advanced release. Introduces GUI, Vision support (multimodal), optimized threading for performance, and improved intent classification.
*   **V4:** Stability improvements and initial integration of local LLM chains.
*   **V3 & Earlier:** Initial prototypes establishing the core voice-to-text loop.

## ⚙️ How it Works

The assistant follows a locally hosted pipeline data flow:

1.  **Wake Word Detection (Vosk):** The system consistently listens for specific keywords (e.g., "Den Den") using a lightweight offline model. No audio is recorded until a wake word is detected.
2.  **Speech Capture:** Once triggered, it records your voice until it detects silence.
3.  **Transcription (Faster-Whisper):** The recorded audio is converted to text using the Whisper model running on your GPU.
4.  **Intent Classification (Ollama):** A small, fast LLM prompt analyzes your text to decide what to do:
    *   **TEXT:** General conversation (routed to Gemma3).
    *   **VISION:** If you ask "What do you see?", it captures a frame from your webcam and sends it to the multimodal model.
    *   **IGNORE:** If it hears background noise or self-talk, it simply ignores it.
5.  **Response Generation:** The LLM generates a text response.
6.  **Text-to-Speech:** The response is converted back to audio and played through your speakers.

## � Design Decisions & Performance Notes

### 🗣️ Text-to-Speech (TTS) Strategy
I rigorously tested various TTS models before settling on the current implementation.
*   **Why not cloud APIs?** Many high-quality voices require online APIs (OpenAI, Google, etc.). I rejected these to maintain the strict **100% Offline** policy.
*   **Why not heavier local models?** Some high-end local models (like XTTS or StyleTTS with full configs) proved too resource-intensive, causing significant delays on consumer hardware.
*   **The Solution:** I balanced quality and speed, ensuring the assistant speaks quickly without freezing your system.

### 👁️ Vision Performance (Gemma3)
The vision capabilities rely on multimodal LLMs (like `gemma3`).
*   **Performance Warning:** Processing speed and accuracy can heavily depend on **image quality** and resolution.
*   Low-light or blurry images may reduce the model's ability to correctly identify objects, and high-resolution images might slightly increase processing time.

## �🛠️ Installation & Setup

### Prerequisites
*   **Python 3.11.9** (Recommended).
    > **Note:** Newer versions of Python (e.g., 3.12+) may cause compatibility issues with some dependencies. It is strongly advised to use **Python 3.11.9** to ensure stability.
*   **[Ollama](https://ollama.com)** installed and running.
*   **CUDA capable GPU** (Recommended for faster performance).

### Step 1: Install Ollama Model
Pull the model used by the assistant (default is `gemma3`, but you can change it in the code):
```bash
ollama pull gemma3
```

### Step 2: Clone Repository
```bash
git clone https://github.com/yahyadursun/DenDen-Local-OpenSource-AI-Voice-Assistant.git
cd DenDen-Local-OpenSource-AI-Voice-Assistant
```

### ❗ Step 3: Create a Virtual Environment (Crucial)
**Why is this important?**
*   **Isolation:** It prevents conflicts between your system's Python packages and this project's dependencies.
*   **Stability:** Ensure the version of a library used here doesn't break other Python apps on your PC.
*   **Cleanliness:** Keeps your global Python installation clean.

**How to create and activate:**
```bash
python -m venv venv
```

*   **Windows:**
    ```powershell
    .\venv\Scripts\activate
    ```
*   **Linux/Mac:**
    ```bash
    source venv/bin/activate
    ```

### Step 4: Install Python Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```
*(Note: You may need to install platform-specific dependencies for `PyAudio` or `sounddevice` depending on your OS).*

### Step 5: Download Vosk Model
1.  Download a lightweight Vosk model (e.g., `vosk-model-small-en-us-0.15` or `vosk-model-small-tr-0.3`) from the [Vosk Models page](https://alphacephei.com/vosk/models).
2.  Extract the folder into the project root.
3.  Ensure the folder name matches `VOSK_MODEL_PATH` in `Assıstant-V5-latest.py` (Line ~48).

## 🚀 Usage

Run the latest version:
```bash
python Assıstant-V5-latest.py
```

### Voice Commands
*   **Wake Words:** "Den Den", "Jarvis", "Assistant", "Hey", "Merhaba".
*   **Vision Triggers:** "Look", "What is this", "Bak", "Gör".
*   **Stop Commands:** "Stop", "Dur", "Sus", "Enough".

### Controls
*   **'e' key:** Press while the camera window is focused to exit the application.

---
