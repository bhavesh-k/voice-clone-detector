"""Score audio clips with several open-weights voice-clone (speech deepfake) detectors.

Each clip is scored under three conditions:
  clean      the file as-is (resampled to 16 kHz mono)
  hd_call    16 kHz Opus at 16 kbps, a stand-in for HD Voice / app calls
  old_phone  8 kHz G.711 mu-law with 300-3400 Hz band limit, like a landline/2G call

Usage:
  uv run detect.py my_voice.m4a cloned_voice.mp3
  uv run detect.py real/ fake/            # folders work too
  uv run detect.py clip.wav --models forensics

Output is the probability each model thinks the clip is FAKE (0 = real, 1 = fake).
Clips longer than the model's window are split into windows and the scores averaged.
Needs ffmpeg on PATH. First run downloads ~4 GB of weights into the Hugging Face cache.
"""
import argparse
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR = 16000
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".mp4", ".aiff", ".aif", ".caf"}

CONDITIONS = {
    "clean": None,
    "hd_call": ["-ar", "16000", "-c:a", "libopus", "-b:a", "16k", "-application", "voip", "-f", "ogg"],
    "old_phone": ["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"],
}


# ---------- audio ----------

def ffmpeg(args, stdin=None):
    out = subprocess.run(["ffmpeg", "-v", "error", *args], input=stdin, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode().strip())
    return out.stdout


def load(path, condition):
    """Decode any audio file to 16 kHz mono float32, optionally passing it through a phone codec first."""
    codec = CONDITIONS[condition]
    if codec is None:
        raw = ffmpeg(["-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"])
    else:
        encoded = ffmpeg(["-i", str(path), "-ac", "1", *codec, "-"])
        raw = ffmpeg(["-i", "-", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], stdin=encoded)
    return np.frombuffer(raw, dtype=np.float32).copy()


def windows(wav, size):
    """Split into non-overlapping windows; short audio (or a short tail) is looped to fill a window."""
    if len(wav) <= size:
        return [np.resize(wav, size)]
    out = [wav[i:i + size] for i in range(0, len(wav) - size + 1, size)]
    tail = wav[len(out) * size:]
    if len(tail) >= SR:  # keep a tail of at least 1 s
        out.append(np.resize(tail, size))
    return out


# ---------- models ----------
# Each model mirrors the preprocessing from its own Hugging Face inference code.

class DFArena:
    """Idiap DF_Arena (500M or 1B). XLS-R + conformer. Logit index 0 = spoof."""

    def __init__(self, repo, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
        self.device = device
        self.window = 64600  # ~4 s, from the repo's feature extractor

    def p_fake(self, wav):
        logits = self.model(input_values=torch.from_numpy(wav).to(self.device))["logits"]
        return torch.softmax(logits, dim=-1)[0, 0].item()


class Forensics:
    """eliya/forensics_0.3B (WavLM-large + AASIST). sigmoid(logit) = P(real)."""

    def __init__(self, repo, device):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Model, WavLMModel

        sys.path.insert(0, str(Path(hf_hub_download(repo, "model.py")).parent))
        import model as forensics_model

        # Build the backbone (WavLM or XLS-R, depending on the variant) from its config only;
        # the checkpoint holds all the weights, so this skips downloading a separate 1.2 GB base model.
        backbones = (WavLMModel, Wav2Vec2Model)
        originals = [b.from_pretrained for b in backbones]
        for b in backbones:
            b.from_pretrained = classmethod(lambda cls, name, *a, **k: cls(cls.config_class.from_pretrained(name)))
        try:
            self.model = forensics_model.DeepfakeDetector()
        finally:
            for b, orig in zip(backbones, originals):
                b.from_pretrained = orig
        sys.path.pop(0)
        sys.modules.pop("model", None)

        weights = next(f for f in ("checkpoint_epoch_5.safetensors", "checkpoint_epoch_4.safetensors")
                       if _repo_has(repo, f))
        # The checkpoint also holds a training-only "projection" head; ignore it, but no weights may be missing.
        missing, _ = self.model.load_state_dict(load_file(hf_hub_download(repo, weights)), strict=False)
        assert not missing, f"checkpoint is missing weights: {missing[:5]}"
        self.model.to(device).eval()
        self.device = device
        self.window = 5 * SR

    def p_fake(self, wav):
        wav = wav / (np.abs(wav).max() + 1e-8)
        logit = self.model(torch.from_numpy(wav)[None].to(self.device))
        return 1.0 - torch.sigmoid(logit.float()).item()


class Pellav2(nn.Module):
    """Sadanie/pellav2 (XLS-R 300M + learned layer weights). sigmoid(logit) = P(fake)."""

    def __init__(self, repo, device):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        # Config-only backbone, as above: the checkpoint contains the backbone weights too.
        self.backbone = Wav2Vec2Model(Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m"))
        self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        state = torch.load(hf_hub_download(repo, "pellav2_detector.pt"), map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
        self.to(device).eval()
        self.device = device
        self.window = 4 * SR

    def p_fake(self, wav):
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        hs = self.backbone(torch.from_numpy(wav)[None].to(self.device), output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        h = torch.stack(hs).mul(w[:, None, None, None]).sum(0).mean(dim=1)
        return torch.sigmoid(self.head(h).squeeze(-1)).item()


def _repo_has(repo, filename):
    from huggingface_hub import file_exists
    return file_exists(repo, filename)


MODELS = {
    "df_arena_500m": (DFArena, "Speech-Arena-2025/DF_Arena_500M_V_1"),
    "df_arena_1b": (DFArena, "Speech-Arena-2025/DF_Arena_1B_V_1"),
    "forensics": (Forensics, "eliya/forensics_0.3B_base_deepfake_classifier"),
    "forensics_wild": (Forensics, "eliya/forensics_0.3B_xlsr_wild_deepfake_classifier"),
    "pellav2": (Pellav2, "Sadanie/pellav2-audio-deepfake-detector"),
}
DEFAULT_MODELS = ["df_arena_500m", "forensics", "pellav2"]


# ---------- main ----------

def find_clips(paths):
    clips = []
    for p in map(Path, paths):
        if p.is_dir():
            clips += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.exists():
            clips.append(p)
        else:
            sys.exit(f"not found: {p}")
    if not clips:
        sys.exit("no audio files found")
    return clips


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="audio files or folders")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    args = ap.parse_args()

    clips = find_clips(args.paths)
    device = pick_device(args.device)
    audio = {(c, cond): load(c, cond) for c in clips for cond in args.conditions}

    scores = {}  # (clip, condition, model) -> p_fake
    for name in args.models:
        cls, repo = MODELS[name]
        print(f"loading {name} on {device} ...", file=sys.stderr, flush=True)
        model = cls(repo, device)
        for (clip, cond), wav in audio.items():
            ps = [model.p_fake(w) for w in windows(wav, model.window)]
            scores[clip, cond, name] = float(np.mean(ps))
        del model  # free memory before loading the next one
        gc.collect()

    clip_w = max(len(c.name) for c in clips)
    print(f"\nP(fake): 0.00 = sure it's real, 1.00 = sure it's fake, 0.50 = default cutoff\n")
    print(f"{'clip':<{clip_w}}  {'condition':<10}" + "".join(f"{m:>18}" for m in args.models))
    for clip in clips:
        secs = len(audio[clip, args.conditions[0]]) / SR
        for i, cond in enumerate(args.conditions):
            label = [clip.name, f"({secs:.1f}s)"][i] if i < 2 else ""
            row = "".join(f"{scores[clip, cond, m]:>13.2f} {'FAKE' if scores[clip, cond, m] >= 0.5 else 'real'}"
                          for m in args.models)
            print(f"{label:<{clip_w}}  {cond:<10}{row}")
        print()


if __name__ == "__main__":
    main()
