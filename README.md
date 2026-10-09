# Voice Clone Detector

A hackathon prototype that flags AI-cloned voices on a phone call. A web page that looks like an incoming call on a Pixel phone listens to your microphone, or plays an audio clip, and shows a live **REAL** / **FAKE** verdict from an open-weights speech deepfake detector.

Everything runs locally. On an Apple Silicon Mac, each model scores a 4–5 second window in well under a second.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and `ffmpeg` (`brew install ffmpeg`).

```bash
uv run app.py
```

Open http://localhost:8000 in Chrome or Safari. The first run downloads about 5.5 GB of model weights from Hugging Face. To speed that up, set `HF_HUB_DISABLE_XET=1` if downloads stall.

- **Answer the call** (green button) to start the mic, or use **Start mic** in the demo controls.
- **Play a clip** to send audio straight into the model while it also plays on the speakers. The mic is paused while a clip plays.
- **Model** switches the detector live.

### Your own caller photo and clips

- Put a photo at `static/caller.jpg`. Without one, the call screen shows a plain gradient.
- Put `.wav`/`.mp3`/`.m4a` files in `clips/` to list them in the clip picker, or use **Choose file…**.

Both are git-ignored, as is `recordings/`, where each session's audio and scores are saved for debugging.

## Score files offline

```bash
uv run detect.py real.m4a clone.mp3            # files or folders
uv run detect.py clips/ --models forensics pellav2
```

Each clip is scored as-is (`clean`), after a 16 kbps Opus round trip (`hd_call`, roughly an HD Voice or app call) and after 8 kHz G.711 μ-law (`old_phone`, a landline or 2G call). The output is P(fake): 0 means real, 1 means fake.

## Models

| Picker name | Hugging Face repo | Notes from testing |
|---|---|---|
| Forensics 0.3B (default) | [eliya/forensics_0.3B_base_deepfake_classifier](https://huggingface.co/eliya/forensics_0.3B_base_deepfake_classifier) | Best balance: real mic audio stays real, longer Fish S2 Pro clones score ~0.85. Fake scores top out around 0.86. |
| pellav2 | [Sadanie/pellav2-audio-deepfake-detector](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector) | Catches clones very well but flags noisy or muffled real audio (e.g. a laptop mic) as fake. |
| Forensics 0.3B wild | [eliya/forensics_0.3B_xlsr_wild_deepfake_classifier](https://huggingface.co/eliya/forensics_0.3B_xlsr_wild_deepfake_classifier) | Robust on real audio, weaker on clones. |
| DF Arena 500M | [Speech-Arena-2025/DF_Arena_500M_V_1](https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1) | Unstable: small shifts in the window flip the verdict. |

These are early observations on a handful of clips, not a benchmark. Check each model's license before using it beyond research. Several are non-commercial.

## How it works

```
browser mic / clip ──► AudioContext @ 16 kHz ──► WebSocket (float32) ──► app.py
                                                                            │ every 2 s: last 4–5 s window
UI (REAL / FAKE, %, sparkline) ◄──────────── JSON {p_fake} ◄──── detector model
```

- `app.py`: FastAPI server. Loads all models at startup and scores the stream every 2 seconds. It skips silence and starts a fresh window when the source switches between mic and clip.
- `detect.py`: model wrappers, matching each model's own preprocessing, plus the offline CLI.
- `static/index.html`: the phone-call UI and demo controls.

Echo cancellation, noise suppression and auto gain are turned off in the browser, so the model hears the raw audio. Echo cancellation in particular would erase a clone played through the same laptop's speakers.
