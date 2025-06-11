import sounddevice as sd
from scipy.io.wavfile import write
import os

sd.default.samplerate = 44100

def audio_capture():
    duracion = 5
    frecuencia_muestreo = sd.default.samplerate

    print("Grabando...")

    audio = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=2, blocking=True)

    print("Grabación finalizada.")

    if "grabacion_test.wav" in os.listdir():
        os.remove("grabacion_test.wav")
        write("grabacion_test.wav", frecuencia_muestreo, audio)


