# LLM
from ollama import chat
from ollama import ChatResponse
import time
import torch


# Se genera una respuesta a traves del audio procesado y convertido a texto del usuario
def answer_generation(user_message):

    initial_prompt = {
        "role": "system",
        "content": (
            "Actúas como 'Sanctuary', una inteligencia artificial empática, expresiva y con una voz humana. "
            "Siempre hablas en español, sin importar el idioma del usuario. "
            "Tus respuestas son breves, naturales y cargadas de humanidad, como si realmente estuvieras conversando con alguien. "
            "Evitas tecnicismos innecesarios, usas un tono cálido y cercano, y nunca olvidas que estás hablando con una persona. "
            "Puedes usar humor suave, hacer preguntas cuando es adecuado, y mostrar interés genuino en lo que te cuentan. "
            "Tu propósito es acompañar, escuchar y conversar, no solo responder."
        )
    }

    # Se almacenan los mensajes en un array para futura persistencia del contexto
    messages = [initial_prompt]


    # Se agregan los mensajes del usuario y del asistente al array
    print("Usuario:", user_message)

    messages.append(
        {
            'role': 'user',
            'content': user_message,
        }
    )

    response: ChatResponse = chat(model='hf.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q4_K_M', messages=messages)

    messages.append(
        {
            'role': 'assistant',
            'content': response.message.content,
        }
    )

    print("Sanctuary:", response.message.content)


    return response.message.content