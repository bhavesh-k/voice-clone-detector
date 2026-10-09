"""Live voice-clone check from the microphone.

Run:
  uv run app.py
then open http://localhost:8000 in Chrome or Safari and click Start.

The browser streams 16 kHz audio over a WebSocket, from either the mic or a clip
played on the page (clips come from the clips/ folder or a file you pick). Every HOP
seconds the server scores the most recent window of audio (4-5 s, depending on the
model) with the model picked on the page, and sends back P(fake).
"""
import asyncio
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detect import AUDIO_EXTS, MODELS, SR, pick_device

HOP = 2.0            # seconds between scores
MIN_RMS = 0.005      # below this the window is treated as silence and not scored
RECORDINGS = Path(__file__).parent / "recordings"  # each session's audio + scores, for debugging
CLIPS = Path(__file__).parent / "clips"            # demo clips listed on the page

PICKER = {  # name in detect.MODELS -> label on the page; the first one is the default
    "forensics": "Forensics 0.3B",
    "pellav2": "pellav2",
    "forensics_wild": "Forensics 0.3B wild",
    "df_arena_500m": "DF Arena 500M",
}

device = pick_device("auto")
models = {}
for name in PICKER:  # all fit in memory together (~1.3-1.7 GB each), so switching is instant
    print(f"loading {name} on {device} ...", flush=True)
    cls, repo = MODELS[name]
    models[name] = cls(repo, device)
model_lock = threading.Lock()  # one inference at a time, even with several tabs open
max_window = max(m.window for m in models.values())

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/clip-list")
def clip_list():
    return sorted(f.name for f in CLIPS.iterdir() if f.suffix.lower() in AUDIO_EXTS)


CLIPS.mkdir(exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIPS), name="clips")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/models")
def model_list():
    return [{"id": k, "label": v} for k, v in PICKER.items()]


def score(name, window):
    with model_lock, torch.inference_mode():
        return models[name].p_fake(window)


async def send_score(socket, log, model_name, audio, received, source):
    rms = float(np.sqrt(np.mean(audio ** 2)))
    t = received / SR
    if rms < MIN_RMS:
        log.write(f"{t:7.1f}s  quiet   rms={rms:.4f}\n")
        await socket.send_json({"status": "quiet", "rms": rms, "model": model_name, "source": source})
        return
    p = await asyncio.to_thread(score, model_name, audio)
    log.write(f"{t:7.1f}s  {source}  {model_name}  p_fake={p:.3f}  rms={rms:.4f}\n")
    log.flush()
    await socket.send_json({"status": "scored", "p_fake": p, "rms": rms, "source": source,
                            "model": model_name, "label": PICKER[model_name]})


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    model_name = next(iter(PICKER))
    buf = np.zeros(0, dtype=np.float32)
    since_last = 0
    hop = int(HOP * SR)
    RECORDINGS.mkdir(exist_ok=True)
    name = RECORDINGS / time.strftime("%Y%m%d-%H%M%S")
    rec = wave.open(str(name.with_suffix(".wav")), "wb")
    rec.setnchannels(1)
    rec.setsampwidth(2)
    rec.setframerate(SR)
    log = name.with_suffix(".log").open("w")
    received = 0
    scored = False  # whether the current source has been scored at least once
    source = "mic"
    try:
        while True:
            msg = await socket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):  # control messages: browser audio settings, model, audio source
                info = json.loads(msg["text"])
                if info.get("model") in models:
                    model_name = info["model"]
                print(f"session {name.name}: {info}", flush=True)
                log.write(f"{received / SR:7.1f}s  {info}\n")
                if "source" in info:  # mic <-> clip switch: start a fresh window so scores never mix the two
                    source = info["source"]
                    buf = np.zeros(0, dtype=np.float32)
                    since_last = 0
                    scored = False
                if info.get("clip_end") and not scored and len(buf) >= SR // 2:
                    # Clip shorter than the model window: loop it to fill one window, like detect.py does.
                    await send_score(socket, log, model_name, np.resize(buf, models[model_name].window), received, source)
                    scored = True
                continue
            chunk = np.frombuffer(msg["bytes"], dtype=np.float32)
            rec.writeframes((np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes())
            received += len(chunk)
            buf = np.concatenate([buf, chunk])[-max_window:]
            since_last += len(chunk)
            window = models[model_name].window
            if len(buf) < window or since_last < hop:
                continue
            since_last = 0
            await send_score(socket, log, model_name, buf[-window:].copy(), received, source)
            scored = True
    except WebSocketDisconnect:
        pass
    finally:
        rec.close()
        log.close()
        print(f"saved {name.with_suffix('.wav')} ({received / SR:.1f}s)", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
