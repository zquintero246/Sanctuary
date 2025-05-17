# Text to speech
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Torch: seguridad extendida para evitar errores de unpickling en ciertos modelos
from torch.serialization import add_safe_globals

add_safe_globals([
    "TTS.tts.models.vits.ViTS",
    "TTS.utils.radam.RAdam"
])


# Generador de voz a partir de texto
def voice_gen(text_tts):

    #Se utiliza el modelo en español Tacotron2
    tts = TTS("tts_models/es/css10/vits").to("cpu")

    #Se le pasa el sample de voz a clonar
    wav = tts.tts(text=text_tts, speaker_wav=r"/home/caffeineaddict__/PycharmProjects/Sanctuary/sanctuary_tts/cloning_audio/Agata.wav")

    #Se reproduce la voz
    print("Reproduciendo voz de Sanctuary...")
    sd.play(np.array(wav), samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()