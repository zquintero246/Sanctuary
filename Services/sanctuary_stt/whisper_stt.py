import torch
import whisper
import time


# Se utiliza cuda para la aceleracion del modelo
def cuda_works():
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))



def generate_text(audio_path):

    # Se llama al modelo y el audio a transcribir
    model = whisper.load_model("base")
    result = model.transcribe(audio = audio_path)


    return result['text']

