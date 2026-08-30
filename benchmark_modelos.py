"""
Compara modelos de Ollama para el asistente de voz.

Mide lo que importa en este proyecto, en este orden:
  1. Si el tool calling funciona     -> sin esto el asistente no ejecuta nada
  2. Latencia de respuesta hablada   -> decide si se siente vivo o no
  3. Calidad del espanol             -> se juzga leyendo la salida
  4. Reparto CPU/GPU                 -> si sale de los 4GB, la velocidad se cae

Uso:
    python benchmark_modelos.py
"""

import subprocess
import time

import requests

OLLAMA = "http://localhost:11434/api/chat"
CTX = 4096  # 8192 empuja el modelo a CPU en una GTX 1650

MODELOS = ["qwen3:4b", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:1.5b"]

SISTEMA = """Eres un asistente de voz. Respondes en espanol, en frases cortas,
como hablaria una persona. Nada de listas ni markdown: esto se lee en voz alta."""

# Las mismas tres herramientas del asistente, para probar en condiciones reales.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "obtener_hora",
            "description": "Devuelve la fecha y hora actual",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tomar_nota",
            "description": "Guarda una nota de texto",
            "parameters": {
                "type": "object",
                "properties": {"texto": {"type": "string"}},
                "required": ["texto"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abrir_app",
            "description": "Abre una aplicacion del equipo",
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre": {
                        "type": "string",
                        "enum": ["navegador", "terminal", "editor", "notas"],
                    }
                },
                "required": ["nombre"],
            },
        },
    },
]

# Cada caso: que se dice, que herramienta deberia elegir (None = solo charla).
CASOS = [
    ("Que hora es?", "obtener_hora"),
    ("Anota que tengo que comprar pan", "tomar_nota"),
    ("Abre el navegador", "abrir_app"),
    ("Hola, como estas?", None),
]


def llamar(modelo, mensaje, tools=None, sistema=SISTEMA):
    cuerpo = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": sistema},
            {"role": "user", "content": mensaje},
        ],
        "stream": False,
        "options": {"num_ctx": CTX},
    }
    if tools:
        cuerpo["tools"] = tools
    inicio = time.time()
    r = requests.post(OLLAMA, json=cuerpo, timeout=600)
    return r.json(), time.time() - inicio


def reparto(modelo):
    """Lee de 'ollama ps' cuanto del modelo quedo en GPU."""
    try:
        salida = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=30
        ).stdout
        for linea in salida.splitlines():
            if linea.startswith(modelo):
                for campo in linea.split():
                    if "/" in campo and "%" in campo:
                        return campo
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"


def evaluar(modelo):
    print(f"\n{'=' * 62}\n  {modelo}\n{'=' * 62}")

    # Precarga, para que el tiempo de carga no contamine la medicion.
    try:
        llamar(modelo, "hola")
    except requests.RequestException as e:
        print(f"  no disponible: {e}")
        return None

    aciertos, tiempos, velocidades = 0, [], []

    for mensaje, esperada in CASOS:
        d, dt = llamar(modelo, mensaje, tools=TOOLS)
        msg = d.get("message", {})
        llamadas = msg.get("tool_calls") or []
        elegida = llamadas[0]["function"]["name"] if llamadas else None

        ok = elegida == esperada
        aciertos += ok
        tiempos.append(dt)
        ev, ed = d.get("eval_count", 0), d.get("eval_duration", 0)
        if ed:
            velocidades.append(ev / (ed / 1e9))

        marca = "OK " if ok else "MAL"
        print(f"  [{marca}] {mensaje:32} {dt:5.1f}s")
        print(f"        espera {esperada}, obtuvo {elegida}")
        if not llamadas:
            print(f"        dice: {msg.get('content', '').strip()[:70]!r}")

    # Una respuesta hablada normal, sin herramientas, para juzgar el espanol.
    d, dt = llamar(modelo, "Cuentame en una frase que tiempo hace en invierno.")
    texto = d.get("message", {}).get("content", "").strip()

    r = {
        "modelo": modelo,
        "aciertos": aciertos,
        "total": len(CASOS),
        "latencia": sum(tiempos) / len(tiempos),
        "velocidad": sum(velocidades) / len(velocidades) if velocidades else 0,
        "reparto": reparto(modelo),
        "charla_t": dt,
        "charla": texto,
    }

    print(f"\n  tool calling : {aciertos}/{len(CASOS)}")
    print(f"  latencia med : {r['latencia']:.1f}s")
    print(f"  velocidad    : {r['velocidad']:.1f} tok/s")
    print(f"  reparto      : {r['reparto']} CPU/GPU")
    print(f"  charla ({dt:.1f}s): {texto[:110]!r}")
    return r


def main():
    resultados = [r for m in MODELOS if (r := evaluar(m))]

    print(f"\n{'=' * 62}\n  RESUMEN\n{'=' * 62}")
    print(f"  {'modelo':14} {'tools':7} {'latencia':10} {'tok/s':8} {'CPU/GPU'}")
    print(f"  {'-' * 56}")
    for r in sorted(resultados, key=lambda x: (-x["aciertos"], x["latencia"])):
        tools = f"{r['aciertos']}/{r['total']}"
        print(
            f"  {r['modelo']:14} {tools:7} {r['latencia']:7.1f}s   "
            f"{r['velocidad']:6.1f}  {r['reparto']}"
        )


if __name__ == "__main__":
    main()
