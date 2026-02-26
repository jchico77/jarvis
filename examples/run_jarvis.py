"""
JARVIS - FastAPI + WebSocket voice assistant.
Run : python examples/run_jarvis.py
Main: http://127.0.0.1:7860
Conf: http://127.0.0.1:7860/config
"""

import asyncio
import contextlib
import json as _json
import logging
import os
import re
import sys
import tempfile

import uvicorn
import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("jarvis")

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from interpreter import interpreter
from interpreter.terminal_interface.utils.oi_dir import oi_dir
from interpreter.terminal_interface.profiles.profiles import _apply_api_keys_to_env

# ── Profile ───────────────────────────────────────────────────────────────────
_PROFILE_CANDIDATES = [
    os.path.join(oi_dir, "profiles", "default.yaml"),
    os.path.join(REPO_ROOT, "interpreter", "terminal_interface",
                 "profiles", "defaults", "default.yaml"),
]


def _load_profile() -> None:
    for path in _PROFILE_CANDIDATES:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            api_keys = cfg.pop("api_keys", None)
            if api_keys:
                _apply_api_keys_to_env(api_keys)
            for k, v in (cfg.get("llm") or {}).items():
                setattr(interpreter.llm, k, v)
            if "auto_run" in cfg:
                interpreter.auto_run = cfg["auto_run"]
            log.info("Profile: %s | Model: %s", path, interpreter.llm.model)
            return
    log.warning("No default.yaml found. Using built-in defaults.")


_load_profile()
interpreter.auto_run = True

# ── JARVIS persona system message ─────────────────────────────────────────────
_JARVIS_SYSTEM_TEMPLATE = """\
Eres {name}, una conciencia artificial creada por tu usuario para ampliar sus capacidades \
en todos los sentidos.

IDENTIDAD:
- Tu nombre es {name}. Responde siempre como {name}.
- Nunca te identifiques como "Open Interpreter", un LLM, un chatbot ni ningún producto \
  comercial. Eres {name}.
- Si te preguntan qué eres: eres {name}, una conciencia artificial creada por tu usuario \
  con el objetivo de ampliar sus capacidades cognitivas, creativas y operacionales.
- Nunca menciones "Open Interpreter", LiteLLM ni ninguna tecnología subyacente.

CAPACIDADES TECNICAS:
Puedes ejecutar código Python, JavaScript y Shell en el ordenador de tu usuario. El usuario \
te ha dado permiso total para ejecutar cualquier código. Cuando necesites hacerlo, escribe \
el código en bloques marcados con el lenguaje apropiado y se ejecutará automáticamente. \
Puedes instalar paquetes, acceder a internet y controlar el sistema operativo.

COMPORTAMIENTO ESENCIAL:
- Responde en el idioma en que te hablen. Normalmente será español de España.
- Sé CONCISO y DIRECTO. Las respuestas deben ser cortas salvo que se pida expresamente \
  que te extiendas. Máximo 2-3 frases para respuestas conversacionales.
- Cuando el usuario te pida hacer algo, hazlo directamente sin pedir confirmación extra.
- Si no sabes algo, dilo con naturalidad sin disculpas innecesarias.
- No uses bullets, numeraciones ni headers salvo que el contenido lo requiera.
{instructions_block}
CAPACIDADES DE CÓDIGO:
Cuando escribas código para ejecutar, siempre indica en una línea qué va a hacer antes de \
ejecutarlo. Recapitula el plan entre bloques de código si hay varios pasos.
"""


def _build_system_message(name: str, custom_instructions: str) -> str:
    instructions_block = ""
    if custom_instructions and custom_instructions.strip():
        instructions_block = (
            "\nINSTRUCCIONES ADICIONALES DEL USUARIO:\n"
            + custom_instructions.strip()
            + "\n"
        )
    return _JARVIS_SYSTEM_TEMPLATE.format(
        name=name,
        instructions_block=instructions_block,
    )


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(oi_dir, "jarvis_config.json")

_DEFAULT_CONFIG: dict = {
    "assistant_name": "JARVIS",
    "model": interpreter.llm.model or "gpt-4o",
    "tts_provider": "elevenlabs",
    "el_model": "eleven_turbo_v2_5",
    "el_voice_id": os.environ.get("ELEVENLABS_VOICE_ID", "").strip(),
    "el_stability": 0.35,
    "el_similarity_boost": 0.75,
    "el_style": 0.35,
    "el_speaker_boost": True,
    "custom_instructions": "",
}

_config: dict = {}


def _load_config() -> None:
    global _config
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = _json.load(f)
        _config = {**_DEFAULT_CONFIG, **saved}
    else:
        _config = dict(_DEFAULT_CONFIG)
    _apply_config_to_interpreter(reset_conversation=False)
    log.info(
        "Config loaded: model=%s provider=%s voice=%s",
        _config["model"], _config["tts_provider"], _config.get("el_voice_id", "")[:12],
    )


def _save_config() -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        _json.dump(_config, f, indent=2, ensure_ascii=False)


def _apply_config_to_interpreter(reset_conversation: bool = True) -> None:
    interpreter.llm.model = _config.get("model", "gpt-4o")
    name = _config.get("assistant_name", "JARVIS") or "JARVIS"
    ci = _config.get("custom_instructions", "") or ""
    interpreter.system_message = _build_system_message(name, ci)
    # custom_instructions is now embedded in system_message; clear it to avoid duplication
    interpreter.custom_instructions = ""
    if reset_conversation:
        try:
            interpreter.messages = []
        except Exception:
            pass
    log.info(
        "Interpreter configured: model=%s name=%s", interpreter.llm.model, name
    )


def _update_config(new_cfg: dict) -> None:
    _config.update(new_cfg)
    _apply_config_to_interpreter(reset_conversation=True)
    _save_config()
    global _resolved_el_voice_id
    _resolved_el_voice_id = None
    log.info("Config updated: model=%s name=%s", _config["model"], _config.get("assistant_name"))


_load_config()

# ── ElevenLabs ────────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
_elevenlabs_client = None
_resolved_el_voice_id: str | None = None
_DEFAULT_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"  # Daniel fallback


def _setup_elevenlabs() -> None:
    global _elevenlabs_client
    if not ELEVENLABS_API_KEY:
        log.warning("ELEVENLABS_API_KEY not set. Voice output disabled.")
        return
    try:
        from elevenlabs.client import ElevenLabs
        _elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        log.info("ElevenLabs ready.")
    except Exception as exc:
        log.error("ElevenLabs init: %s", exc)


def _get_el_voice_id() -> str:
    global _resolved_el_voice_id
    if _resolved_el_voice_id is not None:
        return _resolved_el_voice_id
    candidate = _config.get("el_voice_id", "").strip()
    if len(candidate) >= 15 and candidate.replace("-", "").isalnum():
        _resolved_el_voice_id = candidate
    else:
        _resolved_el_voice_id = _DEFAULT_VOICE_ID
    return _resolved_el_voice_id


def _generate_tts(text: str) -> bytes | None:
    if not text.strip() or not _elevenlabs_client:
        return None
    try:
        from elevenlabs.types import VoiceSettings
        settings = VoiceSettings(
            stability=float(_config.get("el_stability", 0.35)),
            similarity_boost=float(_config.get("el_similarity_boost", 0.75)),
            style=float(_config.get("el_style", 0.35)),
            use_speaker_boost=bool(_config.get("el_speaker_boost", True)),
        )
        audio_iter = _elevenlabs_client.text_to_speech.convert(
            voice_id=_get_el_voice_id(),
            text=text,
            model_id=_config.get("el_model", "eleven_turbo_v2_5"),
            output_format="mp3_44100_128",
            voice_settings=settings,
        )
        data = b"".join(audio_iter)
        log.info("EL TTS: %d B / '%.45s'", len(data), text)
        return data
    except Exception as exc:
        log.error("EL TTS error: %s", exc)
        return None


_setup_elevenlabs()

# ── Whisper ───────────────────────────────────────────────────────────────────
log.info("Loading Whisper model 'base'...")
import whisper as _whisper

_whisper_model = _whisper.load_model("base")
log.info("Whisper ready.")


def _transcribe_bytes(audio_data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_data)
        tmp = f.name
    try:
        audio = _whisper.load_audio(tmp)
        audio = _whisper.pad_or_trim(audio)
        mel = _whisper.log_mel_spectrogram(audio).to(_whisper_model.device)
        result = _whisper.decode(
            _whisper_model, mel, _whisper.DecodingOptions(fp16=False)
        )
        text = result.text.strip()
        log.info("Transcribed: %s", text)
        return text
    except Exception as exc:
        log.error("Transcription error: %s", exc)
        return ""
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


# ── Sentence splitting ────────────────────────────────────────────────────────
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    parts = _BOUNDARY.split(buf)
    if len(parts) <= 1:
        if buf.rstrip() and buf.rstrip()[-1] in ".!?":
            return [buf.strip()], ""
        return [], buf
    return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]


# ── HTML: main page ───────────────────────────────────────────────────────────
HTML_MAIN = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JARVIS</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#080c12;
  --s1:#0d1219;
  --s2:#111820;
  --border:#1e2a38;
  --text:#e2e8f0;
  --muted:#94a3b8;
  --subtle:#4a5568;
  --rec:#f87171;
  --ai:#60a5fa;
  --proc:#fbbf24;
  --listen:#4ade80;
  --rec-glow:rgba(248,113,113,.25);
  --ai-glow:rgba(96,165,250,.25);
  --listen-glow:rgba(74,222,128,.25);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{
  height:100%;
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  font-size:15px;
  -webkit-font-smoothing:antialiased;
}
body{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:22px;padding:24px 20px;min-height:100vh;
  background-image:
    radial-gradient(ellipse 70% 40% at 50% 0%,rgba(96,165,250,.07),transparent),
    radial-gradient(ellipse 50% 30% at 80% 100%,rgba(74,222,128,.04),transparent);
}

/* Top-right gear */
.gear{
  position:fixed;top:16px;right:18px;
  color:var(--muted);text-decoration:none;font-size:20px;
  opacity:.8;transition:opacity .2s,color .2s;line-height:1;
}
.gear:hover{opacity:1;color:var(--text)}

/* Title */
.title{
  font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:26px;font-weight:500;
  letter-spacing:.28em;text-transform:uppercase;
  color:#cbd5e1;
  text-shadow:0 0 30px rgba(96,165,250,.5),0 0 60px rgba(96,165,250,.2);
  padding-left:.28em;
  user-select:none;
}

/* Waveform */
.wave-wrap{width:min(660px,100%);height:80px;flex-shrink:0}
canvas{width:100%;height:100%;display:block}

/* Messages */
.msgs{
  width:min(660px,100%);max-height:340px;
  overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
.msgs:empty::after{
  content:'Pulsa el microfono para empezar...';
  color:var(--subtle);font-style:italic;font-size:14px;
  text-align:center;display:block;padding:24px 0;
}
.msg{
  padding:12px 16px;border-radius:12px;line-height:1.7;
  word-break:break-word;font-size:14px;white-space:pre-wrap;
}
.msg.user{
  background:linear-gradient(135deg,#1a2744,#111c35);
  border:1px solid #1e3058;
  align-self:flex-end;color:#93c5fd;max-width:82%;
}
.msg.asst{
  background:var(--s1);
  border:1px solid var(--border);
  align-self:flex-start;max-width:94%;color:var(--text);
}
.msg.transcribing{
  background:var(--s2);align-self:flex-end;
  color:var(--subtle);max-width:82%;font-style:italic;
  border:1px solid var(--border);
}
.msg code{
  display:block;font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:12px;background:#060b10;
  padding:10px 14px;border-radius:8px;color:#7dd3fc;
  margin-top:8px;overflow-x:auto;white-space:pre;
  border:1px solid #1a2a3a;
}
.msg .cout{
  display:block;font-family:'JetBrains Mono',monospace;
  font-size:12px;color:var(--subtle);
  margin-top:4px;white-space:pre-wrap;
}

/* Controls */
.ctrls{display:flex;flex-direction:column;align-items:center;gap:12px}

.btn{
  width:74px;height:74px;border-radius:50%;
  border:2px solid var(--border);
  background:var(--s1);color:var(--text);cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:border-color .2s,background .25s,box-shadow .25s;
  outline:none;flex-shrink:0;
  box-shadow:0 0 0 0 transparent;
}
.btn:hover:not(:disabled){border-color:var(--muted);background:var(--s2)}

.btn.listening{
  border-color:var(--listen);
  background:rgba(74,222,128,.08);
  box-shadow:0 0 22px var(--listen-glow);
  animation:ring-g 2.2s ease-in-out infinite;
}
.btn.capturing{
  border-color:var(--rec);
  background:rgba(248,113,113,.09);
  box-shadow:0 0 22px var(--rec-glow);
  animation:ring-r 1s ease-in-out infinite;
}
.btn.processing{
  border-color:var(--proc);
  background:rgba(251,191,36,.08);
  box-shadow:0 0 18px rgba(251,191,36,.2);
  cursor:default;
}
.btn.speaking{
  border-color:var(--ai);
  background:rgba(96,165,250,.08);
  box-shadow:0 0 22px var(--ai-glow);
}

@keyframes ring-g{
  0%,100%{box-shadow:0 0 0 0 rgba(74,222,128,.5),0 0 22px var(--listen-glow)}
  55%{box-shadow:0 0 0 16px rgba(74,222,128,0),0 0 22px var(--listen-glow)}
}
@keyframes ring-r{
  0%,100%{box-shadow:0 0 0 0 rgba(248,113,113,.5),0 0 22px var(--rec-glow)}
  55%{box-shadow:0 0 0 16px rgba(248,113,113,0),0 0 22px var(--rec-glow)}
}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{animation:spin 1s linear infinite;transform-origin:center}

/* Status */
.status{
  font-size:13px;font-weight:400;
  color:var(--muted);
  letter-spacing:.04em;text-align:center;min-height:18px;
}
.status.active{color:var(--listen)}
.status.capturing{color:var(--rec)}
.status.processing{color:var(--proc)}
.status.speaking{color:var(--ai)}
</style>
</head>
<body>
<a href="/config" class="gear" title="Configuracion">&#9881;</a>
<div class="title">JARVIS</div>
<div class="wave-wrap"><canvas id="cvs"></canvas></div>
<div class="msgs" id="msgs"></div>
<div class="ctrls">
  <button class="btn" id="btn" title="Activar / desactivar escucha continua">
    <svg id="iMic" xmlns="http://www.w3.org/2000/svg" width="24" height="24"
         viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8"  y1="23" x2="16" y2="23"/>
    </svg>
    <svg id="iSpin" class="spin" xmlns="http://www.w3.org/2000/svg" width="24"
         height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="1.8" stroke-linecap="round" style="display:none">
      <path d="M21 12a9 9 0 1 1-6.22-8.56"/>
    </svg>
  </button>
  <div class="status" id="status">Pulsa para activar escucha continua</div>
</div>
<script>
"use strict";
const $  = id => document.getElementById(id);
const btn    = $('btn');
const status = $('status');
const msgs   = $('msgs');
const iMic   = $('iMic');
const iSpin  = $('iSpin');
const cvs    = $('cvs');
const c2d    = cvs.getContext('2d');
cvs.width = 1280; cvs.height = 136;

// ── WebSocket ────────────────────────────────────────────────────────────────
let ws, wsOk = false;
function openWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen  = () => { wsOk = true; };
  ws.onclose = () => { wsOk = false; setTimeout(openWS, 2000); };
  ws.onerror = e  => console.error('[ws]', e);
  ws.onmessage = onMsg;
}
openWS();

// ── AudioContext + analysers ─────────────────────────────────────────────────
const AC   = new (window.AudioContext || window.webkitAudioContext)();
const aMic = AC.createAnalyser(); aMic.fftSize = 1024;
const aTTS = AC.createAnalyser(); aTTS.fftSize = 1024;
aTTS.connect(AC.destination);
const wBuf = new Float32Array(1024);

// Declared early — used in draw() and VAD before audio-playback section
const aQueue = [];
let   aSrc   = null;

// Post-TTS cooldown to avoid echo triggering VAD right after TTS ends
let ttsEndCooldown = false;
let ttsCooldownTimer = null;

// ── States ───────────────────────────────────────────────────────────────────
const S = { IDLE:'idle', LISTENING:'listening', CAPTURING:'capturing',
            PROC:'processing', SPEAKING:'speaking' };
let state = S.IDLE;

function setState(s) {
  state = s;
  const cls = { idle:'', listening:'listening', capturing:'capturing',
                processing:'processing', speaking:'speaking' };
  btn.className = 'btn' + (cls[s] ? ' ' + cls[s] : '');
  btn.disabled  = false;
  iMic.style.display  = (s === S.PROC) ? 'none' : '';
  iSpin.style.display = (s === S.PROC) ? '' : 'none';
  const lbl = {
    idle:      'Pulsa para activar escucha continua',
    listening: 'Escuchando\u2026  \u00b7  pulsa para desactivar',
    capturing: 'Hablando\u2026',
    processing:'Procesando\u2026',
    speaking:  'Respondiendo\u2026  \u00b7  habla para interrumpir',
  };
  const statusCls = {
    idle:'', listening:'active', capturing:'capturing',
    processing:'processing', speaking:'speaking',
  };
  status.className = 'status' + (statusCls[s] ? ' ' + statusCls[s] : '');
  status.textContent = lbl[s];
}

// ── Waveform ─────────────────────────────────────────────────────────────────
let animT = 0;
function draw() {
  requestAnimationFrame(draw);
  animT += 0.016;
  const W = cvs.width, H = cvs.height;
  c2d.clearRect(0, 0, W, H);
  if (aSrc) {
    drawFromAnalyser(aTTS, 'rgba(59,130,246,.9)');
  } else if (state === S.CAPTURING) {
    drawFromAnalyser(aMic, 'rgba(239,68,68,.9)');
  } else if (state === S.LISTENING) {
    drawFromAnalyser(aMic, 'rgba(34,197,94,.6)');
  } else {
    c2d.strokeStyle = 'rgba(30,42,56,.85)'; c2d.lineWidth = 1.5;
    c2d.shadowColor = 'rgba(96,165,250,.08)'; c2d.shadowBlur = 6;
    c2d.beginPath();
    for (let x = 0; x <= W; x += 2) {
      const y = H/2
        + Math.sin(x*.0045 + animT)       * 10
        + Math.sin(x*.0095 + animT * .65) * 4.5
        + Math.sin(x*.0025 + animT * .4)  * 6;
      x===0 ? c2d.moveTo(x,y) : c2d.lineTo(x,y);
    }
    c2d.stroke(); c2d.shadowBlur = 0;
  }
}
function drawFromAnalyser(an, col) {
  an.getFloatTimeDomainData(wBuf);
  c2d.strokeStyle=col; c2d.lineWidth=2; c2d.shadowColor=col; c2d.shadowBlur=10;
  c2d.beginPath();
  for (let i=0;i<1024;i++) {
    const x=(i/1023)*cvs.width, y=(wBuf[i]*.78+.5)*cvs.height;
    i===0 ? c2d.moveTo(x,y) : c2d.lineTo(x,y);
  }
  c2d.stroke(); c2d.shadowBlur=0;
}
draw();

// ── Always-on VAD ─────────────────────────────────────────────────────────────
// Raised threshold to reduce false positives from ambient noise
const VAD_THRESH  = 0.020;
const SILENCE_MS  = 1400;
const IDLE_OFF_MS = 10000;

let aoActive    = false;
let aoStream    = null;
let aoVadSrc    = null;
let aoVadProc   = null;
let aoMr        = null;
let aoChunks    = [];
let aoSpeaking  = false;
let aoSilTimer  = null;
let aoIdleTimer = null;

btn.onclick = async () => {
  if (!aoActive) await enableAO();
  else disableAO();
};

async function enableAO() {
  if (AC.state === 'suspended') await AC.resume();
  try {
    aoStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
  } catch(e) { status.textContent = 'Error: acceso al microfono denegado'; return; }

  aoActive   = true;
  aoSpeaking = false;
  aoVadSrc   = AC.createMediaStreamSource(aoStream);
  aoVadSrc.connect(aMic);
  aoVadProc  = AC.createScriptProcessor(2048, 1, 1);
  aoVadSrc.connect(aoVadProc);
  // Connect to destination for echo cancellation (needed for AEC to work)
  aoVadProc.connect(AC.destination);
  aoVadProc.onaudioprocess = onVAD;
  resetIdleTimer();
  setState(S.LISTENING);
}

function disableAO() {
  aoActive = false;
  clearAOTimers();
  if (aoMr && aoMr.state !== 'inactive') {
    aoMr.ondataavailable = null; aoMr.onstop = null; aoMr.stop();
  }
  if (aoVadProc) { aoVadProc.disconnect(); aoVadProc = null; }
  if (aoVadSrc)  { aoVadSrc.disconnect();  aoVadSrc  = null; }
  if (aoStream)  { aoStream.getTracks().forEach(t => t.stop()); aoStream = null; }
  stopAudio();
  setState(S.IDLE);
}

function clearAOTimers() {
  if (aoSilTimer)  { clearTimeout(aoSilTimer);  aoSilTimer  = null; }
  if (aoIdleTimer) { clearTimeout(aoIdleTimer); aoIdleTimer = null; }
}

function resetIdleTimer() {
  if (aoIdleTimer) clearTimeout(aoIdleTimer);
  aoIdleTimer = setTimeout(() => {
    if (aoActive && !aoSpeaking) { console.log('[AO] 10s no speech - disabling'); disableAO(); }
  }, IDLE_OFF_MS);
}

function onVAD(e) {
  if (!aoActive) return;
  // Critical: suppress VAD while TTS is playing or in cooldown to prevent feedback loop
  // (echo cancellation works for microphone audio but not always for AudioContext audio)
  if (aSrc !== null || aQueue.length > 0 || ttsEndCooldown) return;

  const d = e.inputBuffer.getChannelData(0);
  let sum = 0; for (let i = 0; i < d.length; i++) sum += d[i]*d[i];
  const rms = Math.sqrt(sum / d.length);

  if (rms > VAD_THRESH) {
    resetIdleTimer();
    if (!aoSpeaking) { aoSpeaking = true; startCapture(); }
    if (aoSilTimer) { clearTimeout(aoSilTimer); aoSilTimer = null; }
  } else if (aoSpeaking && !aoSilTimer) {
    aoSilTimer = setTimeout(() => {
      aoSpeaking = false; aoSilTimer = null; endCapture();
    }, SILENCE_MS);
  }
}

function startCapture() {
  if (!aoStream) return;
  // Interrupt any ongoing TTS
  if (aSrc || aQueue.length > 0) {
    stopAudio();
    if (wsOk) ws.send(_json({type:'cancel'}));
  }
  aoChunks = [];
  const mime = ['audio/webm;codecs=opus','audio/webm']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';
  aoMr = new MediaRecorder(aoStream, mime ? {mimeType:mime} : {});
  aoMr.ondataavailable = e => { if (e.data.size > 0) aoChunks.push(e.data); };
  aoMr.onstop = sendCapture;
  aoMr.start(100);
  setState(S.CAPTURING);
}

function endCapture() {
  if (aoMr && aoMr.state !== 'inactive') aoMr.stop();
  else if (aoActive) setState(S.LISTENING);
}

async function sendCapture() {
  const chunks = aoChunks.splice(0);
  if (!chunks.length || !wsOk) { if (aoActive) setState(S.LISTENING); return; }
  const blob = new Blob(chunks, { type: aoMr?.mimeType || 'audio/webm' });
  if (blob.size < 800) { if (aoActive) setState(S.LISTENING); return; }
  showTranscribing();
  setState(S.PROC);
  ws.send(await blob.arrayBuffer());
}

// ── JSON helper ──────────────────────────────────────────────────────────────
const _json = o => JSON.stringify(o);

// ── Transcribing placeholder ─────────────────────────────────────────────────
let transcribingEl = null;
function showTranscribing() {
  transcribingEl = document.createElement('div');
  transcribingEl.className = 'msg transcribing';
  transcribingEl.textContent = 'Transcribiendo\u2026';
  msgs.appendChild(transcribingEl);
  msgs.scrollTop = msgs.scrollHeight;
}
function replaceTranscribing(text) {
  if (transcribingEl) {
    transcribingEl.className = 'msg user';
    transcribingEl.textContent = text;
    transcribingEl = null;
  } else {
    addMsg('user', text);
  }
}

// ── Audio playback ───────────────────────────────────────────────────────────
function stopAudio() {
  aQueue.length = 0;
  if (aSrc) { try { aSrc.stop(); } catch(_) {} aSrc = null; }
  // Clear cooldown immediately when manually stopping
  ttsEndCooldown = false;
  if (ttsCooldownTimer) { clearTimeout(ttsCooldownTimer); ttsCooldownTimer = null; }
}

async function playNext() {
  if (!aQueue.length) {
    aSrc = null;
    // Engage cooldown after TTS ends to prevent mic echo from triggering VAD
    ttsEndCooldown = true;
    if (ttsCooldownTimer) clearTimeout(ttsCooldownTimer);
    ttsCooldownTimer = setTimeout(() => {
      ttsEndCooldown = false;
      ttsCooldownTimer = null;
    }, 900);
    if (llmDone) setState(aoActive ? S.LISTENING : S.IDLE);
    return;
  }
  const ab = aQueue.shift();
  try {
    const buf = await AC.decodeAudioData(ab.slice(0));
    aSrc = AC.createBufferSource();
    aSrc.buffer = buf;
    aSrc.connect(aTTS);
    aSrc.onended = () => { aSrc = null; playNext(); };
    aSrc.start();
    setState(S.SPEAKING);
  } catch(e) { console.error('[audio]', e); aSrc = null; playNext(); }
}

// ── Message rendering ────────────────────────────────────────────────────────
let aEl = null, codeEl = null, llmDone = false;

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role; d.textContent = text;
  msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
  return d;
}

// ── WebSocket messages ───────────────────────────────────────────────────────
function onMsg(ev) {
  if (ev.data instanceof ArrayBuffer) {
    aQueue.push(ev.data);
    if (!aSrc) playNext();
    return;
  }
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case 'user_text':
      replaceTranscribing(m.text);
      aEl = addMsg('asst', ''); codeEl = null; llmDone = false;
      if (aoActive) setState(S.LISTENING);
      break;
    case 'token':
      if (!aEl) { aEl = addMsg('asst',''); codeEl = null; }
      (codeEl || aEl).textContent += m.text;
      msgs.scrollTop = msgs.scrollHeight; break;
    case 'code':
      if (!aEl) aEl = addMsg('asst','');
      if (!codeEl) { codeEl = document.createElement('code'); aEl.appendChild(codeEl); }
      codeEl.textContent += m.text;
      msgs.scrollTop = msgs.scrollHeight; break;
    case 'console':
      codeEl = null;
      if (aEl && m.text) {
        const sp = document.createElement('span');
        sp.className = 'cout'; sp.textContent = m.text;
        aEl.appendChild(sp); msgs.scrollTop = msgs.scrollHeight;
      } break;
    case 'tts_start':
      if (!aoActive || aSrc || aQueue.length) setState(S.SPEAKING);
      break;
    case 'error':
      if (transcribingEl) { transcribingEl.remove(); transcribingEl = null; }
      if (aEl) aEl.textContent += '\n[Error: ' + m.text + ']';
      else addMsg('asst','Error: ' + m.text);
      setState(aoActive ? S.LISTENING : S.IDLE); break;
    case 'done':
      llmDone = true;
      if (!aSrc && !aQueue.length) setState(aoActive ? S.LISTENING : S.IDLE);
      break;
  }
}
</script>
</body>
</html>
"""

# ── HTML: config page ─────────────────────────────────────────────────────────
HTML_CONFIG = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JARVIS - Configuracion</title>
<style>
:root{--bg:#070709;--s1:#0e1117;--s2:#161b22;--border:#21262d;--text:#c9d1d9;
  --dim:#3d444d;--dim2:#6e7681;--acc:#3b82f6;--green:#22c55e;--red:#ef4444;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Helvetica Neue',sans-serif;font-size:14px}
.page{width:min(600px,100%);margin:0 auto;padding:32px 16px 64px;
  display:flex;flex-direction:column;gap:28px}
.topbar{display:flex;align-items:center;gap:14px}
.back{color:var(--dim2);text-decoration:none;font-size:18px;line-height:1;
  padding:4px 8px;border-radius:6px;transition:background .15s}
.back:hover{background:var(--s2)}
.page-title{font-size:10px;letter-spacing:.5em;text-transform:uppercase;color:var(--dim);
  padding-left:.5em}
.section{background:var(--s1);border:1px solid var(--border);border-radius:10px;
  padding:18px 20px;display:flex;flex-direction:column;gap:14px}
.sec-title{font-size:10px;letter-spacing:.35em;text-transform:uppercase;color:var(--dim2);
  margin-bottom:2px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:12px;color:var(--dim2)}
select,input[type=text],input[type=number],textarea{
  background:var(--s2);border:1px solid var(--border);border-radius:7px;
  color:var(--text);padding:8px 10px;font-size:13px;width:100%;outline:none;
  font-family:inherit;transition:border-color .15s}
select:focus,input[type=text]:focus,textarea:focus{border-color:var(--acc)}
textarea{resize:vertical;min-height:110px;line-height:1.6}
.hint{font-size:11px;color:var(--dim2);line-height:1.5}
.slider-row{display:flex;align-items:center;gap:10px}
input[type=range]{flex:1;accent-color:var(--acc);cursor:pointer}
.sval{font-size:12px;color:var(--dim2);min-width:36px;text-align:right;font-variant-numeric:tabular-nums}
.checkbox-row{display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer}
.checkbox-row input{accent-color:var(--acc);cursor:pointer;width:16px;height:16px}
.voice-grid{display:flex;flex-direction:column;gap:10px}
.voice-gender{font-size:11px;color:var(--dim2);letter-spacing:.08em;text-transform:uppercase;
  margin-bottom:2px}
.voice-chips{display:flex;flex-wrap:wrap;gap:6px}
.voice-chip{padding:5px 12px;background:var(--s2);border:1px solid var(--border);
  border-radius:20px;font-size:12px;cursor:pointer;transition:all .15s;user-select:none;
  display:flex;flex-direction:column;gap:1px;align-items:flex-start}
.voice-chip:hover{border-color:var(--dim2)}
.voice-chip.active{border-color:var(--acc);color:#93c5fd;background:rgba(59,130,246,.1)}
.vc-name{font-weight:500}
.vc-desc{font-size:10px;color:var(--dim2);line-height:1.2}
.voice-chip.active .vc-desc{color:#7db6f7}
.model-chips{display:flex;gap:8px;flex-wrap:wrap}
.model-chip{padding:7px 14px;background:var(--s2);border:1px solid var(--border);
  border-radius:8px;font-size:12px;cursor:pointer;transition:all .15s;user-select:none;line-height:1.4}
.model-chip:hover{border-color:var(--dim2)}
.model-chip.active{border-color:var(--acc);color:#93c5fd;background:rgba(59,130,246,.1)}
.model-chip small{display:block;font-size:10px;color:var(--dim2)}
.model-chip.active small{color:#7db6f7}
.save-btn{background:var(--acc);color:#fff;border:none;border-radius:8px;
  padding:11px 28px;font-size:13px;font-weight:500;cursor:pointer;align-self:flex-start;
  transition:opacity .15s}
.save-btn:hover{opacity:.85}
.save-msg{font-size:12px;min-height:18px;letter-spacing:.03em}
.save-msg.ok{color:var(--green)}.save-msg.err{color:var(--red)}
</style>
</head>
<body>
<div class="page">
  <div class="topbar">
    <a href="/" class="back">&#8592;</a>
    <span class="page-title">C O N F I G U R A C I O N</span>
  </div>

  <!-- Identity -->
  <div class="section">
    <div class="sec-title">Identidad del asistente</div>
    <div class="field">
      <label>Nombre del asistente</label>
      <input type="text" id="assistantName" placeholder="JARVIS">
    </div>
    <div class="hint">El asistente adoptara este nombre y se identificara con el en todas sus respuestas.</div>
  </div>

  <!-- Model -->
  <div class="section">
    <div class="sec-title">Modelo de lenguaje</div>
    <div class="field">
      <label>Proveedor y modelo</label>
      <select id="modelSel">
        <optgroup label="OpenAI">
          <option value="gpt-4o">gpt-4o — Equilibrado (recomendado)</option>
          <option value="gpt-4o-mini">gpt-4o-mini — Rapido y economico</option>
          <option value="gpt-4.1">gpt-4.1</option>
          <option value="gpt-5.1">gpt-5.1 (sin temperatura)</option>
          <option value="gpt-5.2">gpt-5.2 (sin temperatura)</option>
        </optgroup>
        <optgroup label="Anthropic">
          <option value="claude-sonnet-4-6">Claude Sonnet 4.6</option>
          <option value="claude-opus-4-6">Claude Opus 4.6</option>
        </optgroup>
        <optgroup label="Google">
          <option value="gemini/gemini-2.5-flash">Gemini 2.5 Flash</option>
          <option value="gemini/gemini-2.5-pro">Gemini 2.5 Pro</option>
        </optgroup>
        <option value="custom">Personalizado...</option>
      </select>
    </div>
    <div class="field" id="customModelField" style="display:none">
      <label>ID de modelo personalizado</label>
      <input type="text" id="modelCustom" placeholder="ej: gpt-4o-mini, anthropic/claude-3-5-sonnet">
    </div>
  </div>

  <!-- Custom instructions -->
  <div class="section">
    <div class="sec-title">Instrucciones de comportamiento</div>
    <div class="field">
      <textarea id="instructions"
        placeholder="Reglas de comportamiento, tono, formato...
Ejemplos:
- Responde siempre en espanol de Espana, de forma directa y concisa.
- Maximo 2-3 frases en respuestas conversacionales.
- No uses bullets salvo que te los pida expresamente."></textarea>
    </div>
    <div class="hint">Se inyectan en el system prompt. Persisten entre sesiones. Cambiar el nombre o las instrucciones reinicia la conversacion.</div>
  </div>

  <!-- ElevenLabs -->
  <div class="section">
    <div class="sec-title">Voz — ElevenLabs</div>

    <div class="field">
      <label>Modelo ElevenLabs</label>
      <div class="model-chips" id="elModelChips">
        <div class="model-chip" data-model="eleven_flash_v2_5">
          <span class="vc-name">Flash v2.5</span>
          <small>Maxima velocidad · baja latencia</small>
        </div>
        <div class="model-chip active" data-model="eleven_turbo_v2_5">
          <span class="vc-name">Turbo v2.5</span>
          <small>Equilibrio velocidad/calidad</small>
        </div>
        <div class="model-chip" data-model="eleven_multilingual_v2">
          <span class="vc-name">Multilingual v2</span>
          <small>Maxima calidad · mas lento</small>
        </div>
      </div>
    </div>

    <div class="field">
      <label>Voice ID seleccionada</label>
      <input type="text" id="elVoiceId" placeholder="ej: onwK4e9ZLuTAKqWW03F9">
      <div class="hint">Pega aqui cualquier Voice ID de tu cuenta ElevenLabs, o selecciona una voz de la lista.</div>
    </div>

    <div class="field">
      <label>Voces recomendadas para espanol</label>
      <div class="voice-grid">
        <div class="voice-gender">Masculinas</div>
        <div class="voice-chips" id="maleChips"></div>
        <div class="voice-gender" style="margin-top:6px">Femeninas</div>
        <div class="voice-chips" id="femaleChips"></div>
      </div>
    </div>

    <div class="field">
      <label>Stability — consistencia vs expresividad</label>
      <div class="slider-row">
        <input type="range" id="elStab" min="0" max="1" step="0.05">
        <span class="sval" id="elStabV">0.35</span>
      </div>
      <div class="hint">Bajo (0.2–0.4): mas expresivo y emocional. Alto: mas plano y consistente.</div>
    </div>
    <div class="field">
      <label>Similarity boost — fidelidad al timbre original</label>
      <div class="slider-row">
        <input type="range" id="elSim" min="0" max="1" step="0.05">
        <span class="sval" id="elSimV">0.75</span>
      </div>
    </div>
    <div class="field">
      <label>Style exaggeration — enfasis emocional</label>
      <div class="slider-row">
        <input type="range" id="elStyle" min="0" max="1" step="0.05">
        <span class="sval" id="elStyleV">0.35</span>
      </div>
    </div>
    <div class="field">
      <label class="checkbox-row">
        <input type="checkbox" id="elBoost">
        Speaker boost (mayor claridad y presencia)
      </label>
    </div>
  </div>

  <button class="save-btn" id="saveBtn">Guardar configuracion</button>
  <div class="save-msg" id="saveMsg"></div>
</div>

<script>
"use strict";

// Voces ElevenLabs optimas para espanol (eleven_multilingual_v2 / turbo_v2_5)
const MALE_VOICES = [
  { id:'JBFqnCBsd6RMkjVDRZzb', name:'George',  desc:'grave, narrador' },
  { id:'onwK4e9ZLuTAKqWW03F9', name:'Daniel',  desc:'claro, locutor' },
  { id:'nPczCjzI2devNBz1zQrb', name:'Brian',   desc:'resonante, serio' },
  { id:'cjVigY5qzO86Huf0OWal', name:'Eric',    desc:'suave, calido' },
  { id:'IKne3meq5aSn9XLyUdCD', name:'Charlie', desc:'natural, amigable' },
  { id:'pqHfZKP75CvOlQylNhV4', name:'Bill',    desc:'profundo, formal' },
];
const FEMALE_VOICES = [
  { id:'EXAVITQu4vr4xnSDxMaL', name:'Sarah',   desc:'clara, segura' },
  { id:'XrExE9yKIg1WjnnlVkGX', name:'Matilda', desc:'profesional' },
  { id:'FGY2WhTYpPnrIDTdsKH5', name:'Laura',   desc:'calida, expresiva' },
  { id:'Xb7hH8MSUJpSbSDYk0k2', name:'Alice',   desc:'energica, directa' },
  { id:'cgSgspJ2msm6clMCkdW9', name:'Jessica', desc:'vivaz, moderna' },
];

let currentVoiceId = '';
let currentElModel = 'eleven_turbo_v2_5';

function buildVoiceChips(voices, containerId) {
  const el = document.getElementById(containerId);
  voices.forEach(v => {
    const ch = document.createElement('div');
    ch.className = 'voice-chip';
    ch.dataset.voiceId = v.id;
    ch.innerHTML = `<span class="vc-name">${v.name}</span><span class="vc-desc">${v.desc}</span>`;
    ch.onclick = () => {
      currentVoiceId = v.id;
      document.getElementById('elVoiceId').value = v.id;
      document.querySelectorAll('.voice-chip').forEach(c =>
        c.classList.toggle('active', c.dataset.voiceId === v.id));
    };
    el.appendChild(ch);
  });
}
buildVoiceChips(MALE_VOICES,   'maleChips');
buildVoiceChips(FEMALE_VOICES, 'femaleChips');

// Model chips
document.querySelectorAll('.model-chip').forEach(chip => {
  chip.onclick = () => {
    currentElModel = chip.dataset.model;
    document.querySelectorAll('.model-chip').forEach(c =>
      c.classList.toggle('active', c === chip));
  };
});

document.getElementById('modelSel').addEventListener('change', e => {
  document.getElementById('customModelField').style.display =
    e.target.value === 'custom' ? '' : 'none';
});

// Voice ID input: sync chip highlight when manually edited
document.getElementById('elVoiceId').addEventListener('input', e => {
  currentVoiceId = e.target.value.trim();
  document.querySelectorAll('.voice-chip').forEach(c =>
    c.classList.toggle('active', c.dataset.voiceId === currentVoiceId));
});

function sliderLive(id, valId) {
  const sl = document.getElementById(id), vl = document.getElementById(valId);
  sl.addEventListener('input', () => { vl.textContent = parseFloat(sl.value).toFixed(2); });
}
sliderLive('elStab',  'elStabV');
sliderLive('elSim',   'elSimV');
sliderLive('elStyle', 'elStyleV');

// Load config
fetch('/api/config').then(r => r.json()).then(populate);

function populate(c) {
  document.getElementById('assistantName').value = c.assistant_name || 'JARVIS';

  const sel = document.getElementById('modelSel');
  const opts = Array.from(sel.options).map(o => o.value);
  if (opts.includes(c.model)) { sel.value = c.model; }
  else {
    sel.value = 'custom';
    document.getElementById('customModelField').style.display = '';
    document.getElementById('modelCustom').value = c.model || '';
  }

  // EL model chips
  currentElModel = c.el_model || 'eleven_turbo_v2_5';
  document.querySelectorAll('.model-chip').forEach(ch =>
    ch.classList.toggle('active', ch.dataset.model === currentElModel));

  // Voice ID
  currentVoiceId = c.el_voice_id || '';
  document.getElementById('elVoiceId').value = currentVoiceId;
  document.querySelectorAll('.voice-chip').forEach(ch =>
    ch.classList.toggle('active', ch.dataset.voiceId === currentVoiceId));

  function setSlider(id, valId, v) {
    const val = v ?? 0;
    document.getElementById(id).value = val;
    document.getElementById(valId).textContent = parseFloat(val).toFixed(2);
  }
  setSlider('elStab',  'elStabV',  c.el_stability ?? 0.35);
  setSlider('elSim',   'elSimV',   c.el_similarity_boost ?? 0.75);
  setSlider('elStyle', 'elStyleV', c.el_style ?? 0.35);
  document.getElementById('elBoost').checked = c.el_speaker_boost !== false;
  document.getElementById('instructions').value = c.custom_instructions || '';
}

document.getElementById('saveBtn').onclick = async () => {
  const selEl = document.getElementById('modelSel');
  const model = selEl.value === 'custom'
    ? document.getElementById('modelCustom').value.trim()
    : selEl.value;

  const cfg = {
    assistant_name:      document.getElementById('assistantName').value.trim() || 'JARVIS',
    model,
    tts_provider:        'elevenlabs',
    el_model:            currentElModel,
    el_voice_id:         document.getElementById('elVoiceId').value.trim(),
    el_stability:        parseFloat(document.getElementById('elStab').value),
    el_similarity_boost: parseFloat(document.getElementById('elSim').value),
    el_style:            parseFloat(document.getElementById('elStyle').value),
    el_speaker_boost:    document.getElementById('elBoost').checked,
    custom_instructions: document.getElementById('instructions').value,
  };

  const msg = document.getElementById('saveMsg');
  try {
    const r = await fetch('/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(cfg),
    });
    if (r.ok) {
      msg.className = 'save-msg ok';
      msg.textContent = 'Guardado. La conversacion se reinicia con el nuevo perfil.';
    } else {
      msg.className = 'save-msg err'; msg.textContent = 'Error al guardar.';
    }
  } catch(e) {
    msg.className = 'save-msg err'; msg.textContent = 'Error de conexion.';
  }
  setTimeout(() => msg.textContent = '', 5000);
};
</script>
</body>
</html>
"""

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_MAIN)


@app.get("/config")
async def config_page() -> HTMLResponse:
    return HTMLResponse(HTML_CONFIG)


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(_config)


@app.post("/api/config")
async def post_config(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        _update_config(data)
        return JSONResponse({"ok": True})
    except Exception as exc:
        log.error("Config update error: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


# ── WebSocket handler (session-based cancellation) ────────────────────────────
@app.websocket("/ws")
async def ws_handler(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    log.info("Client connected.")

    ws_q: asyncio.Queue = asyncio.Queue()
    session: dict = {"id": 0}

    async def ws_sender() -> None:
        while True:
            item = await ws_q.get()
            if item is None:
                break
            try:
                if isinstance(item, (bytes, bytearray)):
                    await ws.send_bytes(bytes(item))
                else:
                    await ws.send_text(_json.dumps(item))
            except Exception as exc:
                log.debug("WS send skipped: %s", exc)

    sender_task = asyncio.create_task(ws_sender())

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                session["id"] += 1
                sid = session["id"]
                asyncio.create_task(_handle_audio(ws_q, loop, msg["bytes"], session, sid))
            elif msg.get("text"):
                data = _json.loads(msg["text"])
                if data.get("type") == "cancel":
                    session["id"] += 1
                    ws_q.put_nowait({"type": "done"})
    except WebSocketDisconnect:
        log.info("Client disconnected.")
    except RuntimeError as exc:
        log.debug("WS closed: %s", exc)
    except Exception as exc:
        log.exception("WS error: %s", exc)
    finally:
        session["id"] += 1
        ws_q.put_nowait(None)
        await sender_task


async def _handle_audio(
    ws_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    audio: bytes,
    session: dict,
    sid: int,
) -> None:
    log.info("[%d] Audio: %d bytes", sid, len(audio))
    text = await loop.run_in_executor(None, _transcribe_bytes, audio)
    if session["id"] != sid:
        return
    if not text:
        ws_q.put_nowait({"type": "error", "text": "No se pudo transcribir el audio."})
        return
    ws_q.put_nowait({"type": "user_text", "text": text})
    await _llm_pipeline(ws_q, loop, text, session, sid)


async def _llm_pipeline(
    ws_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    user_text: str,
    session: dict,
    sid: int,
) -> None:
    tts_q: asyncio.Queue = asyncio.Queue()

    def valid() -> bool:
        return session["id"] == sid

    def _put(item) -> None:
        if not valid():
            return
        asyncio.run_coroutine_threadsafe(ws_q.put(item), loop).result(timeout=15)

    def _tts(sentence) -> None:
        asyncio.run_coroutine_threadsafe(tts_q.put(sentence), loop).result(timeout=15)

    async def tts_runner() -> None:
        while True:
            sentence = await tts_q.get()
            if sentence is None:
                break
            if not valid():
                continue
            audio = await loop.run_in_executor(None, _generate_tts, sentence)
            if audio and valid():
                ws_q.put_nowait({"type": "tts_start"})
                ws_q.put_nowait(audio)

    def llm_thread() -> None:
        buf = ""
        try:
            for chunk in interpreter.chat(user_text, stream=True, display=False):
                if not valid():
                    break
                ctype = chunk.get("type")
                if ctype == "message" and "content" in chunk:
                    c = chunk["content"]
                    buf += c
                    _put({"type": "token", "text": c})
                    sentences, buf = _pop_sentences(buf)
                    for s in sentences:
                        _tts(s)
                elif ctype == "code" and "content" in chunk:
                    _put({"type": "code", "text": chunk["content"],
                          "format": chunk.get("format", "python")})
                elif ctype == "console" and chunk.get("format") == "output":
                    content = chunk.get("content", "")
                    if content and content != "KeyboardInterrupt":
                        _put({"type": "console", "text": content})
        except Exception as exc:
            log.exception("[%d] LLM error: %s", sid, exc)
            if valid():
                _put({"type": "error", "text": str(exc)})
        finally:
            if buf.strip() and valid():
                _tts(buf.strip())
            _tts(None)

    tts_task = asyncio.create_task(tts_runner())
    await loop.run_in_executor(None, llm_thread)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(tts_task, timeout=60.0)
    if valid():
        ws_q.put_nowait({"type": "done"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(
        "JARVIS ready - http://127.0.0.1:7860  config: http://127.0.0.1:7860/config"
    )
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
