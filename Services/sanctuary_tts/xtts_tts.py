# Text to speech
from TTS.api import TTS
import sounddevice as sd
import torch

# Torch: seguridad extendida para evitar errores de unpickling en ciertos modelos
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.radam import RAdam
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

add_safe_globals([
    XttsConfig,
    Xtts,
    XttsAudioConfig,
    XttsArgs,
    RAdam,
    BaseDatasetConfig
])




# Generador de voz a partir de texto
def voice_gen(text_tts):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Se utiliza el modelo en español Tacotron2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    #Se le pasa el sample de voz a clonar
    wav = tts.tts(text=text_tts, speaker_wav=r"C:\Users\Zabdiel Julian\Downloads\Sanctuary_prod\Sanctuary\Services\sanctuary_tts\cloning_audio\female.wav", language="es")

    #Se reproduce la voz
    print("Reproduciendo voz de Sanctuary...")
    sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()