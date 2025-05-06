
````markdown
# 🧠 Sanctuary

**Sanctuary** es un sistema conversacional emocional diseñado para escuchar, pensar y hablar como un acompañante humano. No es solo un chatbot: es una voz con identidad. Inspirado en el concepto simbólico de *DAEM*, Sanctuary busca crear un vínculo íntimo y empático con el usuario, integrando tecnologías avanzadas de procesamiento de voz y lenguaje natural.

---

## ⚙️ Arquitectura General

```plaintext
[ Usuario habla ]
       ↓
[ STT - Whisper (Transcripción) ]
       ↓
[ LLM - Qwen/Mistral (Generación de respuesta) ]
       ↓
[ TTS - XTTS/Tacotron (Síntesis de voz) ]
       ↓
[ Reproducción de audio ]
````

Cada componente está diseñado como un módulo desacoplado, permitiendo su reemplazo sin afectar el resto del sistema. El flujo está orquestado por un pipeline central.

---

## 🚀 Cómo ejecutar Sanctuary

### 1. Clonar el repositorio

```bash
git clone https://github.com/zquintero246/Sanctuary.git
cd Sanctuary
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Asegúrate de tener `ffmpeg` y `CUDA` configurados si planeas usar Whisper con aceleración por GPU.

### 3. Ejecutar test del sistema

```bash
python main.py
```

Esto procesará el archivo de prueba ubicado en:

```
/Services/sanctuary_stt/test_audios/audio_test2.mp4
```

---

## 🧩 Estructura del proyecto

```plaintext
Sanctuary/
├── Services/
│   ├── sanctuary_core/      # Coordinador del flujo general (pipeline)
│   ├── xtts_tts/            # Generación de voz (TTS)
│   ├── whisper_stt/         # Reconocimiento de voz (STT)
├── main.py                  # Script que ejecuta el pipeline completo
├── clear_gpu.sh            # Limpieza opcional de VRAM (solo NVIDIA)
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
```

---



## 🛠 Roadmap inicial

* [ ] ✅ Estructura inicial del proyecto
* [ ] Finalizar y probar flujo STT → LLM → TTS
* [ ] Exponer el sistema como API REST o WebSocket
* [ ] Crear interfaz web básica para probar interacciones
* [ ] Añadir persistencia de contexto conversacional (memoria)
* [ ] Implementar logging robusto y manejo de errores
* [ ] Soporte para múltiples voces / identidades

---

## 📄 Licencia
Este proyecto está licenciado bajo los términos de la Licencia Apache 2.0.
Consulta el archivo LICENSE para más detalles.
---



