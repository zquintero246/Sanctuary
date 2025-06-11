#Coordinador del pipeline
from Services.sanctuary_core.llm_core import answer_generation
from Services.sanctuary_tts.xtts_tts import voice_gen
from Services.sanctuary_stt.whisper_stt import generate_text
from Services.sanctuary_stt.audio_capture import audio_capture


# Ejecuta el programa
if __name__ == "__main__":
    voice_gen(
        answer_generation(
            generate_text(r"C:\Users\Zabdiel Julian\Downloads\Sanctuary_prod\Sanctuary\Services\sanctuary_stt\grabacion_test.wav")
        )
    )
