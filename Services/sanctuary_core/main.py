#Coordinador del pipeline
from Services.sanctuary_core.llm_core import answer_generation
from Services.sanctuary_tts.xtts_tts import voice_gen
from Services.sanctuary_stt.whisper_stt import generate_text


# Ejecuta el programa
if __name__ == "__main__":
    voice_gen(
        answer_generation(
            generate_text(r"/Services/sanctuary_stt/test_audios/audio_test2.mp4")
        )
    )
