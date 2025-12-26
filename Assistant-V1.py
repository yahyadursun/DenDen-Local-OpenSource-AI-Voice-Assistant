import base64
import os
from io import BytesIO
from gtts import gTTS
import cv2
from IPython.display import HTML, display
from PIL import Image
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import speech_recognition as sr
import threading

# Gerekli kütüphaneleri içe aktarıyoruz
# base64: Görüntüleri metin formatına çevirmek için
# os: İşletim sistemi işlemleri için (dosya açma vb.)
# BytesIO: Bellekte dosya benzeri nesneler oluşturmak için
# gTTS: Google Text-to-Speech servisi
# cv2: OpenCV kütüphanesi (kamera işlemleri için)
# IPython.display: Notebook ortamında gösterim için (burada kullanılmıyor olabilir ama import edilmiş)
# PIL: Python Imaging Library (görüntü işleme)
# langchain_*: LLM (Büyük Dil Modelleri) entegrasyonu için
# speech_recognition: Sesi metne çevirmek için
# threading: Eşzamanlı işlemler için


""" base64 converter """
def convert_to_base64(pil_image):
    """
    PIL görüntülerini Base64 kodlanmış dizelere dönüştürür

    :param pil_image: PIL görüntüsü
    :return: Yeniden boyutlandırılmış Base64 dizesi
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # İsterseniz formatı değiştirebilirsiniz
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
"""
file_path = "foto.jpeg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)
plt_img_base64(image_b64)
"""


""" Speech to text """
""" Ses tanıma (Speech to Text) """
def speechrec():
    """
    Mikrofondan gelen sesi dinler ve Google Speech Recognition kullanarak
    metne dönüştürür.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source,phrase_time_limit=3)
        print("Time over, thanks")
        # recognize_() metodu, API'ye ulaşılamazsa bir istek hatası
        # fırlatacaktır, bu yüzden hata yakalama (exception handling) kullanıyoruz
        
        try:
            # Google speech recognition kullanılıyor
            text = r.recognize_google(audio_text)
            print("Text: "+ text)
            
        except:
            print("Üzgünüm, dediğinizi anlayamadım")
    return text



""" Text to speech """

""" Metni sese çevirme (Text to Speech) """

def speech(output):
    """
    Verilen metni gTTS kullanarak ses dosyasına (mp3) çevirir ve çalar.
    """

    tts = gTTS(output, lang="en")
    tts.save("test.mp3")
    os.system("start test.mp3")




""" LLM """

""" LLM (Büyük Dil Modeli) Ayarları """

# Ollama üzerinden 'gemma3' modelini kullanıyoruz
llm = ChatOllama(model="gemma3")

def prompt_func(data):
    """
    Model için gerekli olan prompt yapısını hazırlar.
    Hem metin hem de görsel veriyi birleştirir.
    """
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

"""
chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "Can you explain this image with a brief ", "image": image_b64}
)

print(query_chain)
"""

""" Ana Döngü - Kamera Yakalama ve LLM Modeli """






cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Hata: Kameraya erişilemedi")
else:
    print("Kameraya başarıyla erişildi!")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Captured Frame",frame)
    else:
        print("Hata: Bir kare yakalanamadı")
    # 'q' tuşuna basılırsa işlem yap
    if cv2.waitKey(1)==ord('q'):

        cv2.imshow("Captured Frame",frame)

        # Anlık görüntüyü kaydet
        cv2.imwrite('savedimg.jpeg', frame) 

        file_path = "savedimg.jpeg"
        pil_image = Image.open(file_path)

        image_b64 = convert_to_base64(pil_image)
        # Zinciri çalıştır: Prompt -> LLM -> String Parser -> Speech
        chain = prompt_func | llm | StrOutputParser() | speech
        chain.invoke(
            {"text": speechrec(), "image": image_b64}
        )
    # 'e' tuşuna basılırsa çıkış yap
    elif cv2.waitKey(1)==ord('e'):
        cv2.destroyAllWindows() 
        break


cap.release()


# chain = prompt_func | llm | StrOutputParser() | speech
# chain.invoke(
#     {"text": "Can you explain this image with a brief ", "image": image_b64}
# )