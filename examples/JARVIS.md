# JARVIS — Asistente Personal por Voz

> Asistente personal de voz construido sobre Open Interpreter.
> Interfaz web minimalista con escucha continua, TTS por ElevenLabs y pipeline LLM streaming.

---

## Indice

1. [Vision general](#vision-general)
2. [Arquitectura](#arquitectura)
3. [Requisitos e instalacion](#requisitos-e-instalacion)
4. [Configuracion](#configuracion)
5. [Como ejecutarlo](#como-ejecutarlo)
6. [Funcionalidades implementadas](#funcionalidades-implementadas)
7. [Endpoints de la API](#endpoints-de-la-api)
8. [Pipeline de procesamiento](#pipeline-de-procesamiento)
9. [Frontend — estados y UI](#frontend-estados-y-ui)
10. [Ficheros clave](#ficheros-clave)

---

## Vision general

JARVIS es una capa de interfaz de voz construida sobre Open Interpreter que expone todas
las capacidades de ejecucion de codigo del framework a traves de un asistente conversacional
de voz accesible desde el navegador.

El usuario habla → Whisper transcribe → el LLM (configurable) responde → ElevenLabs sintetiza
la voz → el frontend reproduce el audio mientras muestra la respuesta en texto en tiempo real.

### Capacidades heredadas de Open Interpreter

JARVIS tiene acceso completo a todas las capacidades de Open Interpreter:
- Ejecucion de codigo Python, JavaScript y Shell en la maquina del usuario
- `auto_run = True`: ejecuta el codigo sin pedir confirmacion
- Computer API: control de pantalla, raton, teclado, portapapeles
- Instalacion de paquetes en tiempo de ejecucion
- Acceso a internet
- Lectura y escritura de ficheros del sistema

---

## Arquitectura

```
Navegador (HTML/CSS/JS)
        |
        | WebSocket  (audio binario / JSON)
        v
FastAPI (run_jarvis.py)
   |          |          |
   v          v          v
Whisper    LLM via    ElevenLabs
(STT)    LiteLLM      TTS API
           |
     Open Interpreter
     (ejecucion de codigo)
```

### Componentes

| Componente | Tecnologia | Rol |
|---|---|---|
| Backend | FastAPI + uvicorn | Servidor HTTP + WebSocket |
| STT | OpenAI Whisper (base, local) | Transcripcion de audio |
| LLM | Cualquier modelo via LiteLLM | Generacion de respuesta |
| Code execution | Open Interpreter | Ejecucion de codigo en el sistema |
| TTS | ElevenLabs API | Sintesis de voz |
| Frontend | HTML/CSS/JS nativo | Interfaz de usuario |
| Config | JSON (`jarvis_config.json`) | Persistencia de configuracion |

---

## Requisitos e instalacion

### Dependencias Python

```
open-interpreter     (instalado en editable mode: pip install -e .)
fastapi
uvicorn
openai-whisper
elevenlabs           (>= 2.x, API client ElevenLabs())
pydub
pyyaml
```

### Variables de entorno

Se configuran en `~/.config/open-interpreter/profiles/default.yaml`
bajo la clave `api_keys:` (ver seccion Configuracion):

```yaml
api_keys:
  openai:          sk-...
  elevenlabs:      ...
  elevenlabs_voice_id: ...
  claude:          ...
  gemini:          ...
```

### Ejecucion

```bash
cd open-interpreter
.venv/Scripts/python.exe examples/run_jarvis.py
# o en Linux/macOS:
.venv/bin/python examples/run_jarvis.py
```

Abre `http://127.0.0.1:7860` en el navegador.
La pagina de configuracion esta en `http://127.0.0.1:7860/config`.

---

## Configuracion

### Fichero de perfil: `default.yaml`

Ubicacion: `~/.config/open-interpreter/profiles/default.yaml`
(o `interpreter/terminal_interface/profiles/defaults/default.yaml` como fallback)

```yaml
llm:
  model: gpt-4o
  temperature: 0.7

auto_run: true

api_keys:
  openai: sk-...
  elevenlabs: ...
  elevenlabs_voice_id: onwK4e9ZLuTAKqWW03F9
```

### Fichero de configuracion JARVIS: `jarvis_config.json`

Ubicacion: `~/.config/open-interpreter/jarvis_config.json`

```json
{
  "assistant_name": "JARVIS",
  "model": "gpt-4o",
  "tts_provider": "elevenlabs",
  "el_model": "eleven_turbo_v2_5",
  "el_voice_id": "onwK4e9ZLuTAKqWW03F9",
  "el_stability": 0.35,
  "el_similarity_boost": 0.75,
  "el_style": 0.35,
  "el_speaker_boost": true,
  "custom_instructions": "Responde siempre en espanol de Espana. Se conciso."
}
```

Este fichero se crea y actualiza desde la pagina `/config` del frontend.
Al guardar configuracion se resetea el historial de conversacion para que
la nueva personalidad/instrucciones entren en vigor inmediatamente.

### Modelos ElevenLabs disponibles

| Model ID | Latencia | Calidad | Uso recomendado |
|---|---|---|---|
| `eleven_flash_v2_5` | ~75ms | Buena | Maxima velocidad |
| `eleven_turbo_v2_5` | ~250ms | Muy buena | Conversacion (default) |
| `eleven_multilingual_v2` | ~500ms | Maxima | Alta calidad, menos urgencia |

> ElevenLabs v3 (mayor expresividad) esta en alpha sin API publica aun (Feb 2026).
> Recomiendan turbo v2.5 para casos conversacionales.

### Voces ElevenLabs recomendadas para espanol

**Masculinas** (con `eleven_turbo_v2_5` o `eleven_multilingual_v2`):

| Nombre | Voice ID | Descripcion |
|---|---|---|
| George | `JBFqnCBsd6RMkjVDRZzb` | Grave, narrador |
| Daniel | `onwK4e9ZLuTAKqWW03F9` | Claro, locutor |
| Brian | `nPczCjzI2devNBz1zQrb` | Resonante, serio |
| Eric | `cjVigY5qzO86Huf0OWal` | Suave, calido |
| Charlie | `IKne3meq5aSn9XLyUdCD` | Natural, amigable |
| Bill | `pqHfZKP75CvOlQylNhV4` | Profundo, formal |

**Femeninas:**

| Nombre | Voice ID | Descripcion |
|---|---|---|
| Sarah | `EXAVITQu4vr4xnSDxMaL` | Clara, segura |
| Matilda | `XrExE9yKIg1WjnnlVkGX` | Profesional |
| Laura | `FGY2WhTYpPnrIDTdsKH5` | Calida, expresiva |
| Alice | `Xb7hH8MSUJpSbSDYk0k2` | Energica, directa |
| Jessica | `cgSgspJ2msm6clMCkdW9` | Vivaz, moderna |

---

## Funcionalidades implementadas

### 1. Escucha continua con VAD (Voice Activity Detection)

- Un boton activa el modo de escucha continua
- Deteccion de voz basada en energia RMS del microfono
- Parametros:
  - `VAD_THRESH = 0.020`: umbral de energia para detectar voz
  - `SILENCE_MS = 1400`: ms de silencio para considerar que el usuario termino de hablar
  - `IDLE_OFF_MS = 10000`: 10s de silencio total desactiva el modo escucha
- Proteccion anti-feedback: el VAD se suspende mientras ElevenLabs esta reproduciendo audio
- Post-TTS cooldown de 900ms: evita que el eco del altavoz dispare el VAD al terminar la respuesta
- Interrupcion: si el usuario habla mientras JARVIS responde, se interrumpe el TTS y el LLM

### 2. Transcripcion de voz (STT)

- Motor: OpenAI Whisper, modelo `base` (cargado localmente)
- El audio grabado en el navegador (formato webm/opus) se envia por WebSocket como binario
- Se guarda en un fichero temporal y se transcribe con `whisper.decode()`
- El texto transcrito se muestra en el chat (primero como placeholder "Transcribiendo...")

### 3. Pipeline LLM con streaming

- Se llama a `interpreter.chat(user_text, stream=True, display=False)`
- Los tokens llegan en chunks de tipo `message`, `code`, `console`
- Los tokens de mensaje se acumulan en un buffer y se dividen por frases completas (regex `(?<=[.!?])\s+`)
- Cada frase completa se envia a la cola TTS para generar audio mientras el LLM sigue generando
- El texto se muestra en el frontend en tiempo real via mensajes JSON `token`

### 4. Sintesis de voz TTS (ElevenLabs)

- Proveedor unico: ElevenLabs
- Se genera audio por frases para reducir la latencia percibida
- El audio (mp3, 44100Hz, 128kbps) se envia al cliente como bytes binarios por WebSocket
- El frontend encola los chunks de audio y los reproduce secuencialmente sin cortes

### 5. Reproduccion de audio en el frontend

- Web Audio API: `AudioContext` + `AudioBufferSourceNode`
- Cola de reproduccion (`aQueue`): los chunks llegan del servidor y se reproducen en orden
- Un `AnalyserNode` conectado a la salida captura los datos para la animacion de onda
- Al terminar cada chunk se llama a `playNext()` para reproducir el siguiente sin pausa

### 6. Visualizacion de onda de audio (waveform)

- Canvas HTML5 con animacion continua via `requestAnimationFrame`
- Estados visuales:
  - **Reposo**: onda senoidal doble suave (gris oscuro)
  - **Escuchando**: forma de onda del microfono (verde)
  - **Capturando (hablando)**: forma de onda del microfono (rojo)
  - **JARVIS hablando**: forma de onda de la salida TTS (azul)

### 7. Personalidad e identidad del asistente

- `interpreter.system_message` se construye dinamicamente desde la config
- Incluye: nombre del asistente, descripcion de identidad, instrucciones de comportamiento, capacidades tecnicas
- El asistente NO se identifica como "Open Interpreter" sino con el nombre configurado
- Al guardar config se reinicia `interpreter.messages = []` para que el nuevo perfil entre en vigor

### 8. Cancelacion de sesion (session-based cancellation)

- Cada mensaje entrante incrementa un `session["id"]` (entero)
- Todas las tareas async (llm_thread, tts_runner) comprueban `session["id"] == sid` en cada iteracion
- Si el usuario habla mientras JARVIS responde, el sid incrementa y las tareas en curso se detienen
- Esto permite interrupcion limpia sin corrupcion de estado

### 9. Manejo de desconexion WebSocket

- Se captura `WebSocketDisconnect` (desconexion limpia)
- Se captura `RuntimeError` (Starlette lanza esto cuando se llama `receive()` en un WS ya desconectado)
- Ambos casos se tratan como desconexion normal del cliente

### 10. Configuracion desde el navegador (/config)

Parametros configurables via UI:
- **Nombre del asistente**: define la identidad del asistente
- **Modelo LLM**: selector con OpenAI (gpt-4o, gpt-4o-mini, gpt-4.1, gpt-5.1, gpt-5.2),
  Anthropic (Claude Sonnet/Opus 4.6), Google (Gemini 2.5 Flash/Pro), o personalizado
- **Instrucciones de comportamiento**: custom instructions embebidas en el system prompt
- **Modelo ElevenLabs**: Flash v2.5 / Turbo v2.5 / Multilingual v2
- **Voice ID**: campo libre + chips de voces recomendadas para espanol (masculinas y femeninas)
- **Parametros de voz**: stability, similarity boost, style exaggeration, speaker boost

---

## Endpoints de la API

| Metodo | Ruta | Descripcion |
|---|---|---|
| `GET` | `/` | Frontend principal (HTML embebido) |
| `GET` | `/config` | Pagina de configuracion (HTML embebido) |
| `GET` | `/api/config` | Devuelve la config actual como JSON |
| `POST` | `/api/config` | Actualiza config, aplica al interprete, reinicia conversacion |
| `WS` | `/ws` | WebSocket: recibe audio binario, envia JSON + audio binario |

### Protocolo WebSocket

**Cliente → Servidor:**
- `bytes`: audio grabado (webm/opus) para transcribir
- `JSON { type: "cancel" }`: cancela el pipeline en curso

**Servidor → Cliente:**
- `bytes`: chunk de audio MP3 (TTS)
- `JSON { type: "user_text", text }`: texto transcrito del usuario
- `JSON { type: "token", text }`: chunk de texto de la respuesta LLM
- `JSON { type: "code", text, format }`: fragmento de codigo generado
- `JSON { type: "console", text }`: salida de consola del codigo ejecutado
- `JSON { type: "tts_start" }`: senal de inicio de audio (cambio de estado visual)
- `JSON { type: "error", text }`: mensaje de error
- `JSON { type: "done" }`: fin del pipeline para este mensaje

---

## Pipeline de procesamiento

```
[Microfono]
    |
    | WebM/Opus bytes via WebSocket
    v
_transcribe_bytes() -- Whisper base local
    |
    | texto transcrito
    v
interpreter.chat(stream=True)
    |
    | chunks streaming
    +---> { type: "message" } --> buffer de frases
    |                                    |
    +---> { type: "code" }              | frase completa
    |                                    v
    +---> { type: "console" }    _generate_tts(sentence)
                                         |
                                         | ElevenLabs API
                                         | mp3 bytes
                                         v
                                  ws.send_bytes(audio)
                                         |
                                         | WebSocket
                                         v
                                  [AudioContext.decodeAudioData]
                                         |
                                         v
                                  [BufferSource.start()]
                                         |
                                  [AnalyserNode → Canvas]
```

---

## Frontend — estados y UI

### Maquina de estados

| Estado | Color | Descripcion |
|---|---|---|
| `idle` | Gris | Sin actividad |
| `listening` | Verde | VAD activo, esperando voz |
| `capturing` | Rojo | Detectando/grabando voz del usuario |
| `processing` | Amarillo | Transcribiendo + LLM procesando |
| `speaking` | Azul | JARVIS reproduciendo audio |

### Diseno

- Dark mode: fondo `#080c12`, acentos azul `#60a5fa`, verde `#4ade80`, rojo `#f87171`
- Tipografia: Inter (texto) + JetBrains Mono (titulo, codigo)
- Fuentes: Google Fonts
- Animaciones: pulsaciones en el boton de microfono segun estado, onda de audio en canvas
- Responsive: ancho maximo 660px, funciona en movil

---

## Ficheros clave

| Fichero | Descripcion |
|---|---|
| `examples/run_jarvis.py` | Todo el codigo: backend FastAPI + HTML/CSS/JS embebido |
| `interpreter/terminal_interface/profiles/defaults/default.yaml` | Perfil base con API keys y modelo |
| `~/.config/open-interpreter/jarvis_config.json` | Config especifica de JARVIS (generada en runtime) |

---

## Notas tecnicas

### Por que interpreter.system_message y no custom_instructions

Open Interpreter construye el system message como:
```
{interpreter.system_message}
{interpreter.custom_instructions}
```
El `system_message` por defecto dice "You are Open Interpreter...". Para que el asistente
adopte la identidad de JARVIS sin que el texto original interfiera, se sobrescribe
`interpreter.system_message` directamente con la plantilla de personalidad. `custom_instructions`
se deja vacio para evitar duplicacion.

### Por que el VAD falla sin la supresion durante TTS

El `echoCancellation` del navegador funciona bien para llamadas WebRTC pero NO cancela
audio que proviene de un `AudioContext` (que es lo que usa JARVIS para reproducir TTS).
Por eso el microfono capta el audio de ElevenLabs y el VAD lo interpreta como voz del usuario.
La solucion es suprimir el procesamiento VAD mientras `aSrc !== null` o `aQueue.length > 0`.

### Modelo Whisper

Se usa el modelo `base` (74M parametros) por balance velocidad/precision.
Para mejorar precision en espanol se puede cambiar a `small` o `medium`.
El modelo se carga una vez al arrancar el servidor.
