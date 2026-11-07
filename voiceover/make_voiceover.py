#!/usr/bin/env python3
"""
Stat Lab voiceover: turn a timed script into a commentary track, and optionally put it on a video.

Script format (one line per cue; anything else is ignored):
    0:08  Twenty sixteen. He's nineteen years old, and it is pouring.
    0:12  [1] He starts fourth behind the safety car.

Each cue starts at its timestamp. A cue that would run into the next one is sped up
(up to --max-speed); if it still runs long, the next cue waits for it, so lines never overlap.

Run (after `bash voiceover/setup.sh`):
    python voiceover/make_voiceover.py voiceover/scripts/max_interlagos_2m15.txt
    python voiceover/make_voiceover.py script.txt --voice am_michael --video reel.mp4
    python voiceover/make_voiceover.py --list-voices

Voices: Kokoro v1.0 (Apache 2.0). British: bm_george, bm_lewis, bm_daniel, bm_fable, bf_emma ...
American: am_michael, am_fenrir, am_puck, af_heart ... Names starting with "b" use British pronunciation.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile as wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
CACHE = os.path.join(HERE, "cache")
SR = 24000

# Spellings that help the voice say F1 names; add to this as you find mispronunciations.
RESPELL = {
    "Räikkönen": "Raikkonen", "Hülkenberg": "Hulkenberg", "Ocon": "Oh-kon",
    "Verstappen": "Ver-stappen", "Pérez": "Perez", "Leclerc": "Le-clair",
}


def find_espeak():
    for p in [shutil.which("espeak-ng"), os.path.join(HERE, "espeak", "bin", "espeak-ng")]:
        if p and os.path.exists(p):
            return p
    sys.exit("espeak-ng not found. Run: bash voiceover/setup.sh")


class Voice:
    def __init__(self, voice):
        model = os.path.join(MODELS, "kokoro-v1.0.int8.onnx")
        voices = os.path.join(MODELS, "voices-v1.0.bin")
        if not (os.path.exists(model) and os.path.exists(voices)):
            sys.exit("Voice model missing. Run: bash voiceover/setup.sh")
        self.vocab = json.load(open(os.path.join(HERE, "vocab.json"), encoding="utf-8"))["vocab"]
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i.type for i in self.sess.get_inputs()}
        self.tok_name = "input_ids" if "input_ids" in self.inputs else "tokens"
        self.styles = np.load(voices)
        if voice not in self.styles:
            sys.exit(f"Unknown voice {voice}. Try --list-voices")
        self.voice, self.lang = voice, ("en-gb" if voice.startswith("b") else "en-us")
        self.espeak = find_espeak()
        env = dict(os.environ)
        data = os.path.join(HERE, "espeak", "share", "espeak-ng-data")
        if os.path.isdir(data):
            env["ESPEAK_DATA_PATH"] = data
        self.env = env

    def phonemes(self, text):
        out = []
        for part in re.findall(r"[^,.!?;:—]+|[,.!?;:—]", text):
            if re.fullmatch(r"[,.!?;:—]", part):
                out.append("," if part == "—" else part)
            elif part.strip():
                r = subprocess.run([self.espeak, "-q", "--ipa", "-v", self.lang, part],
                                   capture_output=True, text=True, env=self.env)
                out.append(" " + " ".join(r.stdout.split()))
        return "".join(c for c in "".join(out).strip() if c in self.vocab)

    def say(self, text, speed):
        """Audio for one line, cached so re-runs only redo lines you changed."""
        key = hashlib.md5(f"{self.voice}|{text}|{speed}".encode()).hexdigest()
        path = os.path.join(CACHE, key + ".npy")
        if os.path.exists(path):
            return np.load(path)
        toks = [self.vocab[c] for c in self.phonemes(text)][:509]
        speed_t = np.float32 if "float" in self.inputs.get("speed", "float") else np.int32
        audio = self.sess.run(None, {
            self.tok_name: np.array([[0, *toks, 0]], dtype=np.int64),
            "style": self.styles[self.voice][len(toks)].astype(np.float32),
            "speed": np.array([speed], dtype=speed_t),
        })[0].ravel().astype(np.float32)
        loud = np.where(np.abs(audio) > 0.01)[0]              # trim silence at both ends
        if len(loud):
            audio = audio[max(0, loud[0] - 240): loud[-1] + 2400]
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, audio)
        return audio


def read_script(path):
    cues = []
    for m in re.finditer(r"^(\d+):(\d\d)\s+(?:\[\d+\]\s*)?(.+)$", open(path, encoding="utf-8").read(), re.M):
        text = m.group(3).strip()
        for a, b in RESPELL.items():
            text = text.replace(a, b)
        cues.append((int(m.group(1)) * 60 + int(m.group(2)), text))
    if not cues:
        sys.exit("No timed lines found. Lines must look like:  0:08  Your sentence here.")
    return cues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", nargs="?")
    ap.add_argument("--voice", default="bm_george")
    ap.add_argument("--speed", type=float, default=1.12, help="normal speaking speed (1.0 = model default)")
    ap.add_argument("--max-speed", type=float, default=1.25, help="fastest a line may be sped up to fit")
    ap.add_argument("--duration", type=float, help="track length in seconds (default: video length, or last line + 3s)")
    ap.add_argument("--video", help="MP4 to put the voice on")
    ap.add_argument("--out", help="output path (default: next to the script)")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()

    if a.list_voices:
        print(" ".join(sorted(np.load(os.path.join(MODELS, "voices-v1.0.bin")).keys())))
        return
    if not a.script:
        ap.error("give a script file")

    cues = read_script(a.script)
    total = a.duration
    if total is None and a.video:
        total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                                     capture_output=True, text=True).stdout or 0) or None
    voice = Voice(a.voice)
    starts = [t for t, _ in cues]
    ends = starts[1:] + [total or starts[-1] + 10]

    mix, prev_end = np.zeros(int((ends[-1] + 10) * SR), np.float32), 0.0
    for (t, text), nxt in zip(cues, ends):
        slot, speed = nxt - t - 0.2, a.speed
        audio = voice.say(text, speed)
        if len(audio) / SR > slot:
            speed = min(a.max_speed, round(speed * len(audio) / SR / slot + 0.02, 2))
            audio = voice.say(text, speed)
        start = max(t + 0.1, prev_end + 0.12)
        i = int(start * SR)
        if i + len(audio) > len(mix):
            mix = np.concatenate([mix, np.zeros(i + len(audio) - len(mix), np.float32)])
        mix[i:i + len(audio)] += audio
        prev_end = start + len(audio) / SR
        late = start - t - 0.1
        print(f"{t // 60}:{t % 60:02d}  {len(audio) / SR:4.1f}s  x{speed:<4}{'  +%.1fs late' % late if late > 0.05 else ''}  {text[:48]}")

    length = total or prev_end + 3
    mix = mix[:int(length * SR)]
    mix = mix / (np.abs(mix).max() or 1) * 0.89
    base = a.out or os.path.splitext(a.script)[0] + f"_{a.voice}"
    wav = base if base.endswith(".wav") else base + ".wav"
    wavfile.write(wav, SR, (mix * 32767).astype(np.int16))
    print(f"\nVoice track: {wav}")

    if a.video:
        mp4 = os.path.splitext(wav)[0] + "_video.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video, "-i", wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-af", "highpass=f=70,acompressor=threshold=-18dB:ratio=3:attack=5:release=80,"
                        "loudnorm=I=-14:TP=-1.5:LRA=7", "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", mp4], check=True)
        print(f"Video with voice: {mp4}")


if __name__ == "__main__":
    main()
