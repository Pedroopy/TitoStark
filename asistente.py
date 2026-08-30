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
import pathlib
import queue
import re
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

# --- ajuste del VAD (Silero) ---
VAD_CHUNK = 512        # muestras por ventana; Silero exige 512 a 16 kHz
UMBRAL_VOZ = 0.5       # mas alto = menos falsos positivos, mas te ignora
SILENCIO_FINAL = 0.8   # segundos de silencio para dar la frase por terminada
ESPERA_INICIAL = 5.0   # segundos esperando a que empieces a hablar
MAX_DURACION = 30.0    # tope duro por si el ruido nunca deja ver silencio

NOMBRE = "JARVIS"

# Voces en espanol de Kokoro: em_alex y em_santa (masculinas),
# ef_dora (femenina). Las pf_/pm_ son portuguesas.
VOZ = "em_alex"

# Se dice sola despues de cada herramienta ejecutada con exito. Va en codigo
# y no en el prompt a proposito: un modelo de 3B se olvida, el codigo no.
CONFIRMACION = "Tarea hecha, señor."

SISTEMA = """Te llamas Jarvis. Eres el asistente personal de tu usuario y
te diriges a el como "señor".

Respondes en espanol, en frases cortas, como hablaria una persona. Nada de
listas ni markdown: esto se lee en voz alta. Si necesitas una herramienta,
usala sin anunciarlo."""


# ------------------------------------------------------------------ herramientas
# Cada herramienta es una funcion normal mas su descripcion en formato OpenAI.
# El enrutador de abajo decide cual subconjunto se le muestra al modelo.

# ------------------------------------------------------------- vault (Obsidian)

# Un vault de Obsidian es solo una carpeta con archivos .md. No hace falta
# plugin ni API: Jarvis escribe archivos, Obsidian los muestra al instante.
# Vive FUERA del repositorio: las conversaciones son personales.
VAULT = pathlib.Path(r"C:\Users\Administrator\Documents\Jarvis")
CONVERSACIONES = VAULT / "Conversaciones"
NOTAS = VAULT / "Notas"

# Cuanto texto recuperado se le pasa al modelo. Un 3B se degrada si le
# inundas el contexto, asi que conviene poco y bueno.
MAX_CONTEXTO = 900
MAX_FRAGMENTOS = 4

# Palabras que no sirven para buscar: aparecen en todas las notas.
VACIAS = {
    "que", "como", "cuando", "donde", "quien", "cual", "para", "por", "con",
    "sin", "los", "las", "del", "una", "uno", "unos", "unas", "este", "esta",
    "esto", "eso", "ese", "esa", "mas", "muy", "sobre", "algo", "todo", "toda",
    "hay", "fue", "era", "son", "estoy", "tengo", "tiene", "hacer", "dijo",
    "dije", "anote", "anota", "acuerdas", "recuerdas", "habia", "senor",
    "jarvis", "yo", "mi", "me", "tu", "te", "se", "lo", "la", "el", "un",
}


def _asegurar_vault():
    CONVERSACIONES.mkdir(parents=True, exist_ok=True)
    NOTAS.mkdir(parents=True, exist_ok=True)


def _hoy() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def registrar_turno(usuario: str, jarvis: str):
    """Guarda cada intercambio en la nota del dia.

    No es una herramienta: corre solo, en cada turno. Si el modelo tuviera
    que acordarse de llamarla, la memoria tendria agujeros.
    """
    from datetime import datetime

    if not usuario.strip():
        return
    try:
        _asegurar_vault()
        archivo = CONVERSACIONES / f"{_hoy()}.md"
        nuevo = not archivo.exists()
        with archivo.open("a", encoding="utf-8") as f:
            if nuevo:
                f.write(f"# Conversaciones del {_hoy()}\n\n#conversacion\n")
            f.write(f"\n## {datetime.now().strftime('%H:%M')}\n")
            f.write(f"**Yo:** {usuario.strip()}\n\n")
            if jarvis.strip():
                f.write(f"**Jarvis:** {jarvis.strip()}\n")
    except OSError as e:
        print(f"  [no pude escribir en el vault: {e}]")


def _normalizar(texto: str) -> str:
    """Minusculas y sin acentos: "cumpleanos" tiene que empatar con "anos"."""
    import unicodedata

    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def _coincide(clave: str, bloque: str) -> bool:
    """Empata la palabra entera o su raiz, para que "cumpleanos" encuentre
    "cumple anos", que es como lo dijo el usuario en su momento."""
    if clave in bloque:
        return True
    return len(clave) >= 6 and clave[:6] in bloque


def _claves(texto: str) -> list:
    palabras = re.findall(r"\w{4,}", _normalizar(texto), re.UNICODE)
    return [p for p in palabras if p not in VACIAS]


def _bloques(texto: str):
    """Parte una nota en unidades que tenga sentido devolver enteras:
    cada turno de conversacion, y cada vinieta de una lista por separado."""
    for trozo in re.split(r"\n(?=## )|\n\n+", texto):
        trozo = trozo.strip()
        if not trozo or trozo.startswith("# "):
            continue
        # Una lista de notas son varias cosas distintas, no un solo bloque.
        if trozo.lstrip().startswith("- "):
            for linea in trozo.splitlines():
                linea = linea.strip()
                if len(linea) > 10:
                    yield linea
        elif len(trozo) >= 15:
            yield trozo


def recordar(texto: str) -> str | None:
    """Busca en el vault y devuelve contexto relevante, o None.

    Esto NO es una herramienta. Corre solo, antes de cada pregunta, y el
    resultado se inyecta en el contexto. La razon: un modelo de 3B no
    decide de forma fiable cuando le hace falta recordar algo, y reforzar
    el prompt para que lo haga degrada el resto del tool calling. Buscar
    siempre y dejarle el material servido funciona; pedirle que lo pida, no.
    """
    claves = _claves(texto)
    if not claves:
        return None
    # Una sola coincidencia basta: exigir dos dejaba fuera casos obvios,
    # porque nadie repite las mismas palabras al preguntar que al contar.
    minimo = 1

    candidatos = []
    for archivo in sorted(VAULT.rglob("*.md")) if VAULT.exists() else []:
        try:
            contenido = archivo.read_text(encoding="utf-8")
        except OSError:
            continue
        for bloque in _bloques(contenido):
            bajo = _normalizar(bloque)
            distintas = sum(1 for c in claves if _coincide(c, bajo))
            if distintas < minimo:
                continue
            repeticiones = sum(bajo.count(c) for c in claves)
            candidatos.append(
                (distintas * 10 + min(repeticiones, 9), archivo.stem, bloque)
            )

    if not candidatos:
        return None
    candidatos.sort(key=lambda c: (c[0], c[1]), reverse=True)

    partes, total = [], 0
    for _, fecha, bloque in candidatos[:MAX_FRAGMENTOS]:
        linea = f"({fecha}) {bloque[:300].replace(chr(10), ' ').strip()}"
        if total + len(linea) > MAX_CONTEXTO:
            break
        partes.append(linea)
        total += len(linea)

    if not partes:
        return None
    return (
        "Notas anteriores del usuario que pueden venir al caso:\n"
        + "\n".join(partes)
    )


# ----------------------------------------------------------------- herramientas

def obtener_hora() -> str:
    from datetime import datetime
    return datetime.now().strftime("Son las %H:%M del %d de %B")


def tomar_nota(texto: str) -> str:
    """Guarda una nota en el vault, en la nota del dia.

    Solo aniade al final. Nunca modifica ni borra lo que ya existe: un
    modelo de 3B confundido no puede estropear notas viejas.
    """
    from datetime import datetime

    texto = texto.strip()
    if not texto:
        return "No entendi que anotar"
    try:
        _asegurar_vault()
        archivo = NOTAS / f"{_hoy()}.md"
        nuevo = not archivo.exists()
        with archivo.open("a", encoding="utf-8") as f:
            if nuevo:
                f.write(f"# Notas del {_hoy()}\n\n#nota\n\n")
            f.write(f"- {datetime.now().strftime('%H:%M')} — {texto}\n")
    except OSError as e:
        return f"No pude guardar la nota: {e}"
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


NAVEGADOR = r"C:\Program Files\Mozilla Firefox\firefox.exe"


def _primer_video(consulta: str) -> str | None:
    """Saca el ID del primer resultado de YouTube. Sin API key ni scraping
    fragil: el ID aparece en el HTML de la pagina de resultados."""
    import re
    from urllib.parse import quote_plus

    url = f"https://www.youtube.com/results?search_query={quote_plus(consulta)}"
    try:
        html = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "es"},
        ).text
    except requests.RequestException:
        return None
    m = re.search(r'"videoId":"([\w-]{11})"', html)
    return m.group(1) if m else None


def ver_en_youtube(consulta: str) -> str:
    """Reproduce el primer resultado de YouTube para lo que le pidas.

    El modelo solo aporta el texto a buscar. La URL y el ejecutable los
    arma este codigo: la linea del diseno se mantiene, el modelo nunca
    escribe el comando.
    """
    import subprocess
    from urllib.parse import quote_plus

    consulta = consulta.strip()
    if not consulta:
        return "No entendi que quieres ver"

    video = _primer_video(consulta)
    if video:
        destino = f"https://www.youtube.com/watch?v={video}"
        aviso = f"Reproduciendo {consulta}"
    else:
        # Sin conexion o YouTube cambio el HTML: al menos deja la busqueda.
        destino = f"https://www.youtube.com/results?search_query={quote_plus(consulta)}"
        aviso = f"No pude elegir el video, te dejo la busqueda de {consulta}"

    try:
        subprocess.Popen([NAVEGADOR, destino])
    except OSError as e:
        return f"No pude abrir el navegador: {e}"
    return aviso


HERRAMIENTAS = {
    "ver_en_youtube": {
        "fn": ver_en_youtube,
        "categoria": "pc",
        "schema": {
            "type": "function",
            "function": {
                "name": "ver_en_youtube",
                "description": (
                    "Busca en YouTube y reproduce el primer video que "
                    "encuentre. Usala cuando pidan ver, poner o buscar "
                    "un video, una cancion o un canal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "consulta": {
                            "type": "string",
                            "description": (
                                "Que buscar en YouTube, por ejemplo "
                                "'elrubius ultimo video' o 'musica para "
                                "concentrarse'"
                            ),
                        }
                    },
                    "required": ["consulta"],
                },
            },
        },
    },
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
    if any(p in t for p in ["nota", "anota", "apunta"]):
        categorias.add("notas")
    if any(
        p in t
        for p in [
            "abre", "abrir", "ejecuta", "lanza",
            "video", "youtube", "pon", "poner", "reproduce",
            "busca", "cancion", "musica", "ver",
        ]
    ):
        categorias.add("pc")
    return [h["schema"] for h in HERRAMIENTAS.values() if h["categoria"] in categorias]


# ------------------------------------------------------------------------ audio

def _cargar_vad():
    """Carga Silero una sola vez. Corre en CPU: no le quita VRAM al modelo."""
    global _vad
    if "_vad" not in globals():
        import torch

        from silero_vad import load_silero_vad

        torch.set_num_threads(1)  # sin esto pelea con Whisper por los nucleos
        _vad = load_silero_vad()
    return _vad


def grabar() -> np.ndarray:
    """Graba y corta sola cuando detecta que dejaste de hablar.

    Espera a que empieces a hablar, y desde ahi corta tras SILENCIO_FINAL
    segundos seguidos sin voz. Los tres limites de abajo son el dial entre
    cortarte a mitad de frase y quedarse esperando para siempre.
    """
    import torch

    vad = _cargar_vad()
    vad.reset_states()  # sin esto el estado de la frase anterior contamina

    buffer = queue.Queue()

    def callback(indata, frames, time_info, status):
        buffer.put(indata.copy())

    trozos = []
    hablando = False
    chunks_silencio = 0
    chunks_totales = 0

    # Silero trabaja con ventanas de 512 muestras a 16 kHz = 32 ms.
    por_segundo = SAMPLE_RATE / VAD_CHUNK
    limite_silencio = int(SILENCIO_FINAL * por_segundo)
    limite_espera = int(ESPERA_INICIAL * por_segundo)
    limite_total = int(MAX_DURACION * por_segundo)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=VAD_CHUNK,
        callback=callback,
    )
    with stream:
        print("  [escuchando... corta solo al terminar de hablar]")
        while True:
            try:
                chunk = buffer.get(timeout=1.0)
            except queue.Empty:
                break

            trozos.append(chunk)
            chunks_totales += 1

            plano = chunk.flatten()
            if len(plano) != VAD_CHUNK:
                continue

            with torch.no_grad():
                prob = vad(torch.from_numpy(plano), SAMPLE_RATE).item()

            if prob >= UMBRAL_VOZ:
                hablando = True
                chunks_silencio = 0
            elif hablando:
                chunks_silencio += 1
                if chunks_silencio >= limite_silencio:
                    break

            # Nunca empezaste a hablar: no te quedes colgado.
            if not hablando and chunks_totales >= limite_espera:
                print("  [no escuche nada]")
                return np.array([], dtype=np.float32)

            # Tope duro, por si el ruido de fondo nunca deja ver silencio.
            if chunks_totales >= limite_total:
                print("  [corte por limite de duracion]")
                break

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
        audio, sr = _kokoro.create(texto, voice=VOZ, lang="es")
        sd.play(audio, sr)
        sd.wait()
    except Exception as e:
        print(f"  [TTS no disponible: {e}]")
        print(f"  {NOMBRE}: {texto}")


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


def ejecutar_herramientas(mensaje: dict) -> tuple[list, bool]:
    """Devuelve los resultados y si todas salieron bien."""
    resultados = []
    todo_ok = True
    for llamada in mensaje.get("tool_calls", []):
        nombre = llamada["function"]["name"]
        args = llamada["function"].get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)

        print(f"  [herramienta: {nombre}({args})]")
        if nombre not in HERRAMIENTAS:
            salida = f"Error: no existe la herramienta {nombre}"
            todo_ok = False
        else:
            try:
                salida = HERRAMIENTAS[nombre]["fn"](**args)
            except Exception as e:
                salida = f"Error ejecutando {nombre}: {e}"
                todo_ok = False
        resultados.append({"role": "tool", "content": str(salida), "name": nombre})
    return resultados, todo_ok


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

            # El recuerdo se inyecta solo para esta pregunta: no se guarda en
            # el historial, que ya va justo de contexto.
            recuerdo = recordar(texto)
            contexto = historial
            if recuerdo:
                print("  [memoria: encontre algo relacionado]")
                contexto = (
                    historial[:-1]
                    + [{"role": "system", "content": recuerdo}]
                    + historial[-1:]
                )

            respuesta = preguntar(contexto, herramientas)
            historial.append(respuesta)

            # Una sola ronda de herramientas. Encadenar varias rompe a los modelos
            # chicos, asi que por ahora no lo intentamos.
            hubo_tarea = False
            if respuesta.get("tool_calls"):
                resultados, hubo_tarea = ejecutar_herramientas(respuesta)
                historial.extend(resultados)
                respuesta = preguntar(historial, [])
                historial.append(respuesta)

            contenido = respuesta.get("content", "").strip()
            if hubo_tarea:
                contenido = f"{contenido} {CONFIRMACION}".strip()
            if contenido:
                hablar(contenido)

            # La memoria se escribe sola en cada turno. Si dependiera de que
            # el modelo llame a una herramienta, tendria agujeros.
            registrar_turno(texto, contenido)

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
