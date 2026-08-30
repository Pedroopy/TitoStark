# Asistente de voz local tipo JARVIS

Documento de contexto del proyecto. Pensado para leerse al inicio de una sesión
de Claude Code, para no tener que reexplicar las decisiones cada vez.

---

## Qué se quiere construir

Un asistente de voz que corre **completamente local**, sin que ningún dato salga
del equipo. Siempre escuchando, despierta con palabra clave, entiende contexto,
ejecuta acciones reales y responde hablando.

### Alcance decidido

Cuatro áreas de capacidades, en este orden de dificultad:

1. **Domótica** (luces, enchufes, sensores) — la más fácil, Home Assistant ya lo resuelve
2. **Correo, calendario, notas, tareas** — plomería directa vía IMAP / CalDAV / Notion
3. **Controlar el PC y abrir apps** — fácil si se acota a acciones concretas, horrible si se busca control universal
4. **Programar y correr scripts** — **fuera de alcance con el hardware actual** (ver sección de hardware)

### Qué es realista y qué no

Se puede lograr: asistente siempre encendido, wake word, comprensión de
contexto, ejecución de acciones reales, memoria entre sesiones, respuesta hablada
en 2 a 4 segundos.

No se puede lograr hoy: iniciativa propia (que interrumpa al usuario porque
notó algo por su cuenta), autonomía de días sin supervisión, latencia de
conversación humana real.

---

## Hardware

- **GPU: GTX 1650, 4GB VRAM, sin tensor cores**
- Equipo de escritorio

Esta restricción define todo el diseño. Consecuencias directas:

- El LLM debe caber en 4GB → modelos de 3B a 4B cuantizados
- Todo lo que no sea el LLM va a CPU, para no competir por VRAM
- Contexto máximo 8K. Más que eso y el caché de atención empuja el modelo a RAM,
  con lo que se pasa de ~30 tokens/segundo a ~6 y el asistente deja de sentirse vivo
- El agente de programación no es viable: un modelo que entra en 4GB no sostiene
  refactors ni cadenas largas de herramientas
- Las bibliotecas que aceleran inferencia en FP16 no dan el salto habitual,
  porque la 1650 no tiene tensor cores

**Ruta de upgrade** (no urgente, decidir después de tener la versión de 4GB
andando): una RTX 3060 de 12GB usada habilita modelos de 12B a 14B, que es donde
el tool calling empieza a ser confiable y donde la parte de programación deja de
ser imposible.

**Hardware pendiente:** micrófono con array. El del notebook no capta desde el
otro lado de la pieza y da la falsa impresión de que el problema es de software.

---

## Stack

| Pieza | Dónde corre | Elección |
|---|---|---|
| Wake word | CPU | openWakeWord (permite entrenar palabra propia) |
| Corte de voz (VAD) | CPU | Silero VAD |
| Transcripción | CPU (o GPU si el CPU sufre) | faster-whisper `small`, int8 |
| Cerebro | GPU | **qwen2.5:3b vía Ollama** (qwen3:4b descartado, ver comparativa) |
| Voz | CPU | Kokoro-82M, voz `em_alex` |

### Por qué cada uno

**faster-whisper `small` en int8** es el punto dulce para español. `base` se come
los nombres propios, `medium` no cabe junto al LLM. Corriendo en CPU se demora
1 a 2 segundos, que es el principal costo de latencia del sistema. Alternativa si
molesta: mover a GPU con `device="cuda", compute_type="int8_float16"` y bajar a
`base`, aceptando perder precisión en nombres propios.

**Ollama** expone un endpoint compatible con OpenAI, así que el código no queda
casado con ninguna implementación. Se puede cambiar a llama.cpp o vLLM después sin
reescribir.

**Kokoro-82M**: 82M de parámetros, pesos abiertos, licencia Apache, tiene voces
en español y corre en CPU dejando los 4GB enteros para el LLM. Si en algún momento
se quiere clonar una voz específica, el paso siguiente es Qwen3-TTS (Apache 2.0),
al costo de ~medio segundo más por respuesta.

### Atajo arquitectónico a evaluar

**Home Assistant** ya tiene todo el pipeline armado y funcionando en local: wake
word, STT, LLM vía Ollama, TTS. Se llama Assist. Como igual se va a necesitar Home
Assistant para la parte de domótica, tiene sentido usarlo como columna vertebral
en vez de escribir el orquestador desde cero, y conectarle herramientas propias
por encima.

---

## Decisiones de diseño

### 1. El enrutador de herramientas

Un modelo de 4B se marea si se le declaran treinta funciones: elige mal o inventa
parámetros. La solución es filtrar primero por intención (casa, agenda, PC, charla)
y mostrarle solo cinco o seis herramientas de esa categoría.

La versión inicial filtra por palabras clave, tosca a propósito. Cuando el catálogo
crezca, se reemplaza por una llamada corta al LLM que solo clasifique la intención.

### 2. Una sola ronda de herramientas

El modelo llama una función, ve el resultado, responde, se acabó. Encadenar dos o
tres llamadas seguidas es exactamente donde un 4B se cae. No habilitar cadenas
hasta tener el resto sólido.

### 3. El modelo elige de un menú, no escribe el comando

`abrir_app` recibe un nombre de una lista blanca cerrada, nunca un comando armado
por el modelo. **Esta es la línea que no se cruza.**

Para cuando llegue la ejecución de scripts (si se llega): contenedor aparte,
usuario sin permisos sobre archivos personales, lista blanca de comandos, y
confirmación por voz obligatoria para cualquier cosa que escriba o borre. Un
modelo local alucina más que uno de nube y se le está dando una shell.

### 4. Si el tool calling en JSON falla mucho

Aider no usa tool calling en JSON en absoluto: parsea formatos de edición en texto
plano, lo que esquiva el modo de falla que rompe a la mayoría de los agentes con
modelos locales. Si el modelo empieza a inventar parámetros o llamar funciones
inexistentes, esa es la salida: menos JSON, más formato simple parseado a mano.

---

## Roadmap

### Semana 1 — el hito que importa
Script en Python que escuche por micrófono, transcriba, llame al modelo con tres
herramientas de juguete y responda por parlante. Feo y lento, pero vivo. Todo lo
demás cuelga de esto.

**Estado: COMPLETA.** El ciclo entero funciona verificado con voz real.
Ver "Estado al cerrar" al final del documento.

### Semanas 2 y 3 — wake word y streaming
Que deje de ser "aprieto Enter y hablo" y pase a estar siempre escuchando.
Aquí aparecen los problemas reales:

- Se dispara solo viendo videos o música
- Corta al usuario a mitad de frase
- Se queda esperando eternamente cuando no hay corte claro
- Manejar interrupciones del usuario es más difícil de lo que parece

### Mes 2 — herramientas de verdad
Home Assistant conectado. Cada herramienta agregada de a una, con su prueba.
Memoria persistente entre sesiones, que es lo que hace la diferencia entre un
asistente y un chat con amnesia.

### Mes 3 en adelante
Bajar latencia, personalidad, refinamiento del manejo de interrupciones.

---

## Problemas conocidos que van a aparecer

- **La latencia se acumula por etapa.** Cada eslabón suma, y el total es lo que
  decide si se siente vivo o no
- **El wake word se dispara solo**, sobre todo con audio de fondo
- **El tool calling local es más frágil** que el de nube, notoriamente
- **El micrófono integrado no sirve** para uso a distancia

---

## Estructura del código actual

`asistente.py`, ya adaptado a Windows.

```
configuración      OLLAMA, MODELO, SAMPLE_RATE, NOMBRE, VOZ, CONFIRMACION,
                   SISTEMA, y los cuatro límites del VAD
herramientas       obtener_hora, tomar_nota, abrir_app, ver_en_youtube
                   cada una con su función + schema formato OpenAI + categoría
enrutar()          filtra herramientas por palabras clave según intención
_cargar_vad()      carga Silero una sola vez, en CPU
grabar()           un Enter para empezar, corta sola por silencio (Silero VAD)
hablar()           Kokoro con voz VOZ, fallback a imprimir en consola
preguntar()        POST a Ollama con num_ctx 8192
ejecutar_...()     despacha tool_calls, devuelve (resultados, todo_ok)
main()             carga Whisper, bucle, añade CONFIRMACION tras cada tarea
```

`benchmark_modelos.py` compara modelos de Ollama midiendo tool calling,
latencia, tokens por segundo y reparto CPU/GPU. Volver a correrlo al cambiar
de GPU.

### Instalación

```bash
pip install sounddevice numpy faster-whisper requests silero-vad
pip install kokoro-onnx        # mejor voz
# o
pip install piper-tts          # más rápido, más robótico

# Ollama aparte, desde ollama.com
ollama pull qwen3:4b
```

---

## Próximos pasos concretos

1. ~~Levantar `asistente.py` y confirmar que el ciclo completo funciona~~ **hecho**
2. ~~Medir la latencia real de cada etapa por separado~~ **hecho**
3. Decidir si se adopta Home Assistant como columna vertebral o se sigue con
   orquestador propio — **pendiente, no bloquea nada**
4. Agregar openWakeWord y Silero VAD — **Silero hecho, falta openWakeWord**

---

## Mediciones reales del equipo (30-08-2026)

Entorno montado y verificado: Python 3.12.10, venv en `.venv`, Ollama 0.33.2,
`qwen3:4b` descargado (2.5 GB), GTX 1650 con 4096 MiB confirmados.

### El modelo no entra completo en la GPU

| Contexto | Reparto CPU/GPU | Tamaño en memoria | Velocidad |
|---|---|---|---|
| 8192 | 45% / 55% | 4.1 GB | ~11 tok/s |
| 4096 | 33% / 67% | 3.5 GB | ~16 tok/s |

El supuesto de "contexto 8192" del diseño original no se sostiene: a 8K casi la
mitad del modelo queda en CPU. Bajar a 4096 recupera velocidad, pero ni así entra
entero. La estimación de ~30 tok/s no se cumple en ninguna configuración.

### Qwen3 razona antes de responder, y eso domina la latencia

Es un modelo de razonamiento: genera un bloque de pensamiento antes de contestar.
Para "¿qué hora es si son las tres de la tarde?" gastó 659 tokens en producir una
respuesta de seis palabras.

- Sin mitigar, contexto 8192: **155 a 171 segundos**
- Con `/no_think` al inicio del prompt de sistema y contexto 4096: **37 segundos**

El parámetro `think: false` de la API de Ollama **no funciona** con este modelo en
la versión 0.33.2: el razonamiento se filtra al contenido de la respuesta, en
inglés. La directiva `/no_think` dentro del mensaje de sistema sí limpia la salida.

### Conclusión

37 segundos contra el objetivo de 2 a 4 segundos del documento. La brecha no se
cierra optimizando: hay que cambiar de modelo. Un modelo sin razonamiento y más
chico (rango 1.5B a 3B) es el camino para la semana 1, dejando `qwen3:4b` para
cuando exista la RTX 3060 de 12 GB de la ruta de upgrade.

### Estado del entorno

- `abrir_app` adaptado a Windows con rutas absolutas: Firefox, PowerShell,
  VS Code, Obsidian y el explorador. La lista blanca cerrada se mantiene.
- Micrófono detectado: headset HyperX. Sirve para push-to-talk de cerca; el
  micrófono con array del documento sigue pendiente para uso a distancia.
- `torch` quedó en versión CPU, que es lo deseado: no compite por VRAM.

---

## Comparativa de modelos (30-08-2026)

Medido con `benchmark_modelos.py`, contexto 4096, las tres herramientas reales
del asistente y cuatro escenarios: pedir la hora, tomar una nota, abrir una app,
y un saludo que **no** debe disparar ninguna herramienta.

| Modelo | Tool calling | Latencia media | tok/s | Reparto |
|---|---|---|---|---|
| **qwen2.5:3b** | **4/4** | **2.8 s** | **63.8** | **100% GPU** |
| qwen3:4b | 4/4 | 18.7 s | 19.0 | 33% / 67% |
| llama3.2:3b | 3/4 | 2.9 s | 45.7 | 20% / 80% |
| qwen2.5:1.5b | 2/4 | 2.5 s | 99.1 | 100% GPU |

### Decisión: qwen2.5:3b

Único modelo que acierta las cuatro herramientas y entra completo en la GPU.
Ocupa 2.3 GB, deja ~1.8 GB libres y sostiene **contexto 8192 sin salir de la
GPU**: el supuesto de 8K del diseño original se recupera, y a 63 tok/s.

Latencia de respuesta hablada: **2.8 segundos**, dentro del objetivo de 2 a 4.

### Por qué se descartaron los otros

**qwen3:4b** acierta igual las cuatro herramientas, pero su modo de razonamiento
y el desborde a CPU lo dejan seis veces más lento. Es el modelo correcto para la
RTX 3060 de la ruta de upgrade, no para la 1650.

**llama3.2:3b** falla el caso más importante: ante un simple "hola, ¿cómo estás?"
llama a `obtener_hora`. Inventar una herramienta donde no hace falta es peor que
ser lento, porque el asistente ejecuta acciones que nadie pidió.

**qwen2.5:1.5b** es el más rápido y el más inútil: 2 de 4. Ante "¿qué hora es?"
se inventó "las 14:23" en vez de llamar a la herramienta, y ante la petición de
tomar nota se puso a preguntar qué anotar. Confirma que 1.5B es demasiado poco
para tool calling confiable.

### Lo que esto valida del diseño original

La sección "Si el tool calling en JSON falla mucho" preveía tener que abandonar
JSON por formatos de texto plano. Con qwen2.5:3b **no hace falta**: 4 de 4 en
JSON, incluyendo el caso negativo. Ese plan B queda archivado, no descartado.

---

## Pesos de Kokoro (no van en el repositorio)

`pip install kokoro-onnx` instala la librería pero **no** los pesos. Hay que
bajarlos a la raíz del proyecto, donde `hablar()` los busca por nombre relativo:

```bash
curl -sSL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -sSL -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Son 311 MB y 27 MB. El `.gitignore` los excluye. Si al clonar el proyecto en otra
máquina el asistente responde por consola en vez de hablar, es que faltan estos
dos archivos.

Voces en español: `ef_dora` (femenina), `em_alex` y `em_santa` (masculinas). Las
que empiezan con `pf_` o `pm_` son portuguesas.

### Latencia por etapa, medida

| Etapa | Tiempo |
|---|---|
| Transcripción (Whisper `small`, CPU) | 1 a 2 s |
| Modelo (qwen2.5:3b, 100% GPU) | ~2.8 s |
| Síntesis (Kokoro, CPU) | ~0.7 s |
| **Total por turno** | **~4 a 5 s** |

Kokoro carga en 1.3 s la primera vez y queda en memoria. El cuello de botella es
el modelo, y después Whisper: son los dos lugares donde vale la pena optimizar.

## Estado: la semana 1 está cerrada

El ciclo completo funciona de punta a punta, verificado con voz real: micrófono →
Whisper transcribe → qwen2.5:3b elige la herramienta → se ejecuta → Kokoro
responde hablando. El hito que sostenía todo lo demás ya está.

Siguiente: openWakeWord y Silero VAD, para dejar el push-to-talk.

---

## `ver_en_youtube`: parámetro sí, comando no

La regla del diseño es que el modelo **no arma el comando**. Esta herramienta la
respeta: el modelo solo aporta el texto a buscar, igual que ya aportaba el texto
de una nota. La URL, el escapado y la ruta del ejecutable los construye el código.

Funciona en dos pasos. Primero pide la página de resultados de YouTube y extrae
el ID del primer video con una expresión regular sobre el campo `"videoId"` del
HTML — sin API key. Después abre Firefox directamente en `watch?v=ID`, así el
video se reproduce en vez de dejar una lista de resultados.

Si no hay conexión o YouTube cambia el HTML, cae en abrir la página de búsqueda.
Degrada, no falla.

### Verificado

| Se dijo | Herramienta elegida | Parámetro |
|---|---|---|
| "pon un video del rubius" | `ver_en_youtube` | `rubius ultimo video` |
| "quiero ver el ultimo video de elrubius" | `ver_en_youtube` | `elrubius ultimo video` |
| "busca musica para concentrarse en youtube" | `ver_en_youtube` | `musica para concentrarse` |
| "abre el navegador" | `abrir_app` | `navegador` |
| "hola que tal" | ninguna | — |

Cinco de cinco. El enrutador suma las palabras `video`, `youtube`, `pon`,
`reproduce`, `busca`, `cancion`, `musica` y `ver` a la categoría `pc`.

### Lo que esto amplía

`abrir_app` acepta cinco palabras fijas. Esta acepta texto libre, así que el
modelo puede hacer abrir cualquier búsqueda de YouTube. Comparado con darle una
shell es un riesgo menor, pero no es cero, y conviene tenerlo presente al agregar
la siguiente herramienta con parámetro libre.

---

## VAD ajustado y validado

`SILENCIO_FINAL = 0.8` probado con voz real y el headset HyperX: no corta a
mitad de frase ni se siente lento. El resto de los valores por defecto también
quedan (`UMBRAL_VOZ = 0.5`, `ESPERA_INICIAL = 5.0`, `MAX_DURACION = 30.0`).

Este ajuste está atado al micrófono. Cuando llegue el de array habrá que
revisarlo: capta más ambiente, así que probablemente haya que **subir**
`UMBRAL_VOZ` para que el ruido de la pieza no cuente como voz.

El push-to-talk se redujo a un solo Enter. Falta el wake word para eliminarlo.

---

# Estado al cerrar — 30 de agosto de 2026

**Empezar leyendo esto.** El resto del documento es el razonamiento; esta sección
es dónde quedó todo.

## Funciona, verificado con voz real

El ciclo completo: aprietas Enter, hablas, y Jarvis transcribe, decide, ejecuta y
te responde hablando. Probado en vivo, no solo en pruebas sintéticas.

| Pieza | Estado |
|---|---|
| Repositorio | `Pedroopy/TitoStark` en GitHub, privado |
| Ubicación | `C:\Users\Administrator\Proyectos\TitoStark` |
| Entorno | `.venv` con todo instalado, Python 3.12.10 |
| Modelo | `qwen2.5:3b`, 100% en GPU, contexto 8192 |
| Transcripción | faster-whisper `small` int8, en CPU |
| Voz | Kokoro, `em_alex` (masculina) |
| Corte de voz | Silero VAD, ajustado y validado |
| Herramientas | `obtener_hora`, `tomar_nota`, `abrir_app`, `ver_en_youtube` |
| Latencia | ~4 a 5 segundos por turno |

## Cómo levantarlo

```
cd C:\Users\Administrator\Proyectos\TitoStark
.\.venv\Scripts\python.exe asistente.py
```

Si responde por consola en vez de hablar, faltan los pesos de Kokoro: ver la
sección correspondiente más arriba.

## Personalidad

Se llama **Jarvis**, trata al usuario de "señor", y dice **"Tarea hecha, señor."**
después de cada herramienta ejecutada con éxito. Esa frase está en código
(constante `CONFIRMACION`), no en el prompt: un modelo de 3B se olvida de decirla
y queda inconsistente. Solo se dice cuando una herramienta corrió de verdad y sin
error, para no sonar tras una charla cualquiera ni tapar un fallo.

## Lo que cambió respecto al plan original

1. **El modelo.** `qwen3:4b` era la elección del diseño, pero razona antes de
   responder y no entra en 4 GB: 18 a 37 segundos por respuesta. `qwen2.5:3b`
   hace lo mismo en 2.8 s y entra entero en la GPU. El contexto 8192 del diseño
   original **sí se sostiene**, con el modelo correcto.
2. **El plan B del tool calling queda archivado.** No hizo falta abandonar JSON
   por formatos de texto plano: 4 de 4 en JSON, incluido el caso negativo.
3. **La regla de la lista blanca sigue intacta**, pero ahora hay una herramienta
   con parámetro libre (`ver_en_youtube`). El modelo aporta el texto a buscar;
   la URL y el ejecutable los arma el código.

## Pendientes, en orden

1. **openWakeWord.** Es lo único que separa esto de un asistente siempre
   encendido. Conviene hacerlo *después* de tener el micrófono con array: el
   ajuste del umbral depende del micrófono y no vale la pena afinarlo dos veces.
2. **Micrófono con array.** El headset HyperX sirve para push-to-talk de cerca,
   pero el wake word solo tiene sentido si escucha desde el otro lado de la pieza.
3. **Voz tipo JARVIS.** Kokoro solo tiene dos voces masculinas en español y
   ninguna se parece al mayordomo británico. El camino es Qwen3-TTS con clonación
   de voz, a costa de ~0.5 s más por respuesta. Ojo: se acerca al timbre, no al
   acento inglés, que viene del idioma del modelo.
4. **Home Assistant.** No bloquea nada. Recomendación: instalarlo cuando llegue
   la domótica y usarlo *como herramienta* desde el orquestador propio, en vez de
   migrar a su Assist. Lo que hay funciona; migrar sería cambiar algo probado por
   algo desconocido.

## Detalles menores anotados

- A la 1:47 de la madrugada dice "de la noche". El modelo interpreta el formato
  de 24 horas de forma discutible. Se arregla con una línea en el prompt de
  sistema, si llega a molestar.
- Los ajustes del VAD están atados al headset. Con el micrófono de array habrá
  que **subir** `UMBRAL_VOZ`, porque captará más ambiente.
