# Den Den - Yerel Açık Kaynak Yapay Zeka Sesli Asistan (V5) 🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-green)

Gerçek zamanlı sesli etkileşim, bilgisayarlı görü ve yerel LLM zekasını birleştiren güçlü ve **tamamen çevrimdışı** bir yapay zeka asistanı. Tamamen yerel makinenizde çalışacak şekilde tasarlanmıştır, böylece tam gizlilik sağlar ve dışarıya hiçbir veri sızdırmaz.

## 🌟 Temel Özellikler (V5)

*   **%100 Yerel ve Gizli:** Buluta hiçbir veri gönderilmez. [Ollama](https://ollama.com) ve yerel modeller tarafından desteklenir.
*   **Gerçek Zamanlı Sesli Etkileşim:**
    *   **Uyandırma Kelimesi Algılama:** Çevrimdışı, düşük gecikmeli uyandırma kelimesi dinleme için `Vosk` kullanır (örneğin, "Den Den", "Jarvis").
    *   **Konuşmadan Metne:** `faster-whisper` kullanarak yüksek doğruluklu deşifre.
    *   **Metinden Sese:** Yerel TTS hatları aracılığıyla doğal seslendirme.
*   **Görü Yetenekleri 👁️:** Web kameranız aracılığıyla dünyayı görebilir ve analiz edebilir. Görsel analiz tetiklemek için "Bu nedir?" veya "Buna bak" diyebilirsiniz.
*   **Akıllı Niyet Sınıflandırması:** Arka plan gürültüsü, genel sorular ve görüyle ilgili istekleri akıllıca ayırt eder.
*   **GUI Arayüzü:** Kamera görüntüsü ve asistan durumunu gösteren temiz bir görsel arayüz.

## 📜 Sürüm Geçmişi

*   **V5 (En Son):** En gelişmiş sürüm. GUI, Görü desteği (multimodal), performans için optimize edilmiş iş parçacığı ve geliştirilmiş niyet sınıflandırması sunar.
*   **V4:** Kararlılık iyileştirmeleri ve yerel LLM zincirlerinin ilk entegrasyonu.
*   **V3 ve Öncesi:** Temel ses-metin döngüsünü oluşturan ilk prototipler.

## ⚙️ Nasıl Çalışır?

Asistan, yerel olarak barındırılan bir veri akışını takip eder:

1.  **Uyandırma Kelimesi Algılama (Vosk):** Sistem, belirli anahtar kelimeler (örneğin, "Den Den") için sürekli olarak hafif bir çevrimdışı modeli dinler. Bir uyandırma kelimesi algılanana kadar ses kaydedilmez.
2.  **Ses Yakalama:** Tetiklendiğinde, sessizlik algılayana kadar sesinizi kaydeder.
3.  **Deşifre (Faster-Whisper):** Kaydedilen ses, GPU'nuzda çalışan Whisper modeli kullanılarak metne dönüştürülür.
4.  **Niyet Sınıflandırması (Ollama):** Küçük, hızlı bir LLM istemi, ne yapılacağına karar vermek için metninizi analiz eder:
    *   **METİN (TEXT):** Genel sohbet (Gemma3'e yönlendirilir).
    *   **GÖRÜ (VISION):** "Ne görüyorsun?" diye sorarsanız, web kameranızdan bir kare yakalar ve bunu multimodal modele gönderir.
    *   **YOK SAY (IGNORE):** Arka plan gürültüsü veya kendi kendine konuşma duyarsa, bunu görmezden gelir.
5.  **Yanıt Üretimi:** LLM bir metin yanıtı üretir.
6.  **Metinden Sese:** Yanıt tekrar sese dönüştürülür ve hoparlörlerinizden çalınır.

## 💡 Tasarım Kararları ve Performans Notları

### 🗣️ Metinden Sese (TTS) Stratejisi
Mevcut uygulamada karar kılmadan önce çeşitli TTS modellerini titizlikle test ettim.
*   **Neden bulut API'leri değil?** Birçok yüksek kaliteli ses, çevrimiçi API'ler gerektirir (OpenAI, Google vb.). Katı **%100 Çevrimdışı** politikasını korumak için bunları reddettim.
*   **Neden daha ağır yerel modeller değil?** Bazı üst düzey yerel modeller (tam yapılandırmalı XTTS veya StyleTTS gibi) çok fazla kaynak tüketerek tüketici donanımlarında önemli gecikmelere neden oldu.
*   **Çözüm:** Kalite ve hızı dengeledim, böylece asistan sisteminizi dondurmadan hızlı bir şekilde konuşabiliyor.

### 👁️ Görü Performansı (Gemma3)
Görü yetenekleri multimodal LLM'lere (`gemma3` gibi) dayanır.
*   **Performans Uyarısı:** İşleme hızı ve doğruluğu, **görüntü kalitesine** ve çözünürlüğe büyük ölçüde bağlı olabilir.
*   Düşük ışıklı veya bulanık görüntüler, modelin nesneleri doğru tanımlama yeteneğini azaltabilir ve yüksek çözünürlüklü görüntüler işleme süresini biraz artırabilir.

## 🛠️ Kurulum ve Ayarlar

### Gereksinimler
*   **Python 3.11.9** (Önerilen).
    > **Not:** Python'un daha yeni sürümleri (örneğin, 3.12+), bazı bağımlılıklarla uyumluluk sorunlarına neden olabilir. Kararlılığı sağlamak için **Python 3.11.9** kullanılması şiddetle tavsiye edilir.
*   **[Ollama](https://ollama.com)** kurulu ve çalışıyor olmalı.
*   **CUDA destekleyen GPU** (Daha hızlı performans için önerilir).

### Adım 1: Ollama Modelini Yükleyin
Asistan tarafından kullanılan modeli çekin (varsayılan `gemma3`'tür, ancak kod içinde değiştirebilirsiniz):
```bash
ollama pull gemma3
```

### Adım 2: Depoyu Klonlayın
```bash
git clone https://github.com/yahyadursun/DenDen-Local-OpenSource-AI-Voice-Assistant.git
cd DenDen-Local-OpenSource-AI-Voice-Assistant
```

### ❗ Adım 3: Sanal Ortam Oluşturun (Önemli)
**Bu neden önemli?**
*   **İzolasyon:** Sisteminizin Python paketleri ile bu projenin bağımlılıkları arasındaki çakışmaları önler.
*   **Kararlılık:** Burada kullanılan bir kütüphane sürümünün PC'nizdeki diğer Python uygulamalarını bozmamasını sağlar.
*   **Temizlik:** Genel Python kurulumunuzu temiz tutar.

**Nasıl oluşturulur ve etkinleştirilir:**
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

### Adım 4: Python Bağımlılıklarını Yükleyin
Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
*(Not: İşletim sisteminize bağlı olarak `PyAudio` veya `sounddevice` için platforma özgü bağımlılıkları yüklemeniz gerekebilir).*

### Adım 5: Vosk Modelini İndirin
1.  [Vosk Modelleri sayfasından](https://alphacephei.com/vosk/models) hafif bir Vosk modeli indirin (örneğin, `vosk-model-small-en-us-0.15` veya `vosk-model-small-tr-0.3`).
2.  Klasörü proje kök dizinine çıkartın.
3.  Klasör adının `Assıstant-V5-latest.py` (Satır ~48) içindeki `VOSK_MODEL_PATH` ile eşleştiğinden emin olun.

## 🚀 Kullanım

En son sürümü çalıştırın:
```bash
python Assıstant-V5-latest.py
```

### Sesli Komutlar
*   **Uyandırma Kelimeleri:** "Den Den", "Jarvis", "Assistant", "Hey", "Merhaba".
*   **Görü Tetikleyicileri:** "Look", "What is this", "Bak", "Gör".
*   **Durdurma Komutları:** "Stop", "Dur", "Sus", "Enough".

### Kontroller
*   **'e' tuşu:** Uygulamadan çıkmak için kamera penceresi odaktayken basın.

---
*Açık Kaynak Topluluğu için ❤️ ile oluşturulmuştur.*
