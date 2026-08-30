"""
Asistente de voz local - esqueleto semana 1
Pensado para GTX 1650 (4GB VRAM): LLM en GPU, todo lo demas en CPU.

Instalacion:
    pip install sounddevice numpy faster-whisper requests silero-vad
    # TTS: elige uno
    pip install kokoro-onnx        # mejor voz
    pip install piper-tts          # mas rapido, mas robotico

    # Ollama aparte, desde ollama.com
    ollama pull qwen3:4b

Uso:
    python asistente.py
    Enter para hablar, Enter otra vez para cortar. Ctrl+C para salir.
    El wake word viene despues, primero hay que tener esto andando.
"""

import json
import queue
import sys
import threading

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# ---------------------------------------------------------------- configuracion

OLLAMA = "http://localhost:11434/api/chat"
MODELO = "qwen2.5:3b"  # medido: 100% GPU, 2.3GB, 4/4 en tool calling
SAMPLE_RATE = 16000

SISTEMA = """Eres un asistente de voz. Respondes en espanol, en frases cortas,
como hablaria una persona. Nada de listas ni markdown: esto se lee en voz alta.
Si necesitas una herramienta, usala sin anunciarlo."""


# ------------------------------------------------------------------ herramientas
# Cada herramienta es una funcion normal mas su descripcion en formato OpenAI.
# El enrutador de abajo decide cual subconjunto se le muestra al modelo.

def obtener_hora() -> str:
    from datetime import datetime
    return datetime.now().strftime("Son las %H:%M del %d de %B")


def tomar_nota(texto: str) -> str:
    with open("notas.txt", "a", encoding="utf-8") as f:
        f.write(texto + "\n")
    return "Nota guardada"


def abrir_app(nombre: str) -> str:
    import subprocess
    # Lista blanca a proposito. Nunca dejes que el modelo arme el comando.
    # Rutas absolutas de Windows: el modelo elige una clave, no escribe el path.
    permitidas = {
        "navegador": [r"C:\Program Files\Mozilla Firefox\firefox.exe"],
        "terminal": [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        ],
        "editor": [
            r"C:\Users\Administrator\AppData\Local\Programs"
            r"\Microsoft VS Code\Code.exe"
        ],
        "notas": [
            r"C:\Users\Administrator\AppData\Local\Programs"
            r"\Obsidian\Obsidian.exe"
        ],
        "explorador": [r"C:\Windows\explorer.exe"],
    }
    if nombre not in permitidas:
        return f"No conozco la app {nombre}"
    try:
        subprocess.Popen(permitidas[nombre])
    except OSError as e:
        return f"No pude abrir {nombre}: {e}"
    return f"Abriendo {nombre}"


HERRAMIENTAS = {
    "obtener_hora": {
        "fn": obtener_hora,
        "categoria": "general",
        "schema": {
            "type": "function",
            "function": {
                "name": "obtener_hora",
                "description": "Devuelve la fecha y hora actual",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    "tomar_nota": {
        "fn": tomar_nota,
        "categoria": "notas",
        "schema": {
            "type": "function",
            "function": {
                "name": "tomar_nota",
                "description": "Guarda una nota de texto para despues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "texto": {"type": "string", "description": "Contenido de la nota"}
                    },
                    "required": ["texto"],
                },
            },
        },
    },
    "abrir_app": {
        "fn": abrir_app,
        "categoria": "pc",
        "schema": {
            "type": "function",
            "function": {
                "name": "abrir_app",
                "description": "Abre una aplicacion del computador",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nombre": {
                            "type": "string",
                            "enum": ["navegador", "terminal", "editor"],
                        }
                    },
                    "required": ["nombre"],
                },
            },
        },
    },
}


def enrutar(texto: str) -> list:
    """Devuelve solo las herramientas relevantes. Un modelo de 4B se confunde
    si le muestras todo el catalogo, asi que filtramos por palabras clave.
    Cuando crezca el catalogo, esto pasa a ser una llamada corta al LLM."""
    t = texto.lower()
    categorias = {"general"}
    if any(p in t for p in ["nota", "anota", "apunta", "recuerda"]):
        categorias.add("notas")
    if any(p in t for p in ["abre", "abrir", "ejecuta", "lanza"]):
        categorias.add("pc")
    return [h["schema"] for h in HERRAMIENTAS.values() if h["categoria"] in categorias]


# ------------------------------------------------------------------------ audio

def grabar() -> np.ndarray:
    """Graba hasta que el usuario apreta Enter. El VAD viene en la semana 2."""
    buffer = queue.Queue()
    detener = threading.Event()

    def callback(indata, frames, time_info, status):
        buffer.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    with stream:
        print("  [grabando... Enter para cortar]")
        input()
        detener.set()

    trozos = []
    while not buffer.empty():
        trozos.append(buffer.get())
    if not trozos:
        return np.array([], dtype=np.float32)
    return np.concatenate(trozos).flatten()


def hablar(texto: str):
    """Reemplazable. Empieza con lo que tengas a mano y despues cambia a Kokoro."""
    try:
        from kokoro_onnx import Kokoro
        global _kokoro
        if "_kokoro" not in globals():
            _kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        audio, sr = _kokoro.create(texto, voice="ef_dora", lang="es")
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print(f"  [TTS no disponible: {e}]")
        print(f"  ASISTENTE: {texto}")


# -------------------------------------------------------------------------- LLM

def preguntar(historial: list, herramientas: list) -> dict:
    r = requests.post(
        OLLAMA,
        json={
            "model": MODELO,
            "messages": historial,
            "tools": herramientas,
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.7},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]


def ejecutar_herramientas(mensaje: dict) -> list:
    resultados = []
    for llamada in mensaje.get("tool_calls", []):
        nombre = llamada["function"]["name"]
        args = llamada["function"].get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        print(f"  [herramienta: {nombre}({args})]")
        if nombre not in HERRAMIENTAS:
            salida = f"Error: no existe la herramienta {nombre}"
        else:
            try:
                salida = HERRAMIENTAS[nombre]["fn"](**args)
            except Exception as e:
                salida = f"Error ejecutando {nombre}: {e}"
        resultados.append({"role": "tool", "content": str(salida), "name": nombre})
    return resultados


# ------------------------------------------------------------------------ bucle

def main():
    print("Cargando Whisper...")
    # int8 en CPU deja los 4GB de la GPU libres para el LLM.
    # Si tu CPU sufre, cambia a device="cuda", compute_type="int8_float16".
    whisper = WhisperModel("small", device="cpu", compute_type="int8")

    historial = [{"role": "system", "content": SISTEMA}]
    print("Listo. Enter para hablar, Ctrl+C para salir.\n")

    while True:
        try:
            input("> ")
            audio = grabar()
            if audio.size < SAMPLE_RATE // 2:
                print("  [muy corto]")
                continue

            segmentos, _ = whisper.transcribe(audio, language="es", beam_size=1)
            texto = " ".join(s.text for s in segmentos).strip()
            if not texto:
                continue
            print(f"  TU: {texto}")

            historial.append({"role": "user", "content": texto})
            herramientas = enrutar(texto)

            respuesta = preguntar(historial, herramientas)
            historial.append(respuesta)

            # Una sola ronda de herramientas. Encadenar varias rompe a los modelos
            # chicos, asi que por ahora no lo intentamos.
            if respuesta.get("tool_calls"):
                historial.extend(ejecutar_herramientas(respuesta))
                respuesta = preguntar(historial, [])
                historial.append(respuesta)

            contenido = respuesta.get("content", "").strip()
            if contenido:
                hablar(contenido)

            # Ventana corta: 8K de contexto se llena rapido.
            if len(historial) > 21:
                historial = [historial[0]] + historial[-20:]

        except KeyboardInterrupt:
            print("\nChao.")
            sys.exit(0)
        except Exception as e:
            print(f"  [error: {e}]")


if __name__ == "__main__":
    main()
