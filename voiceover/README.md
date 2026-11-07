# Voiceover

Makes a commentary voice track from a timed script, and can put it straight onto a Reel.
Everything runs on your own machine or in GitHub Actions; nothing is sent to a paid service.

## Run it in GitHub (no setup)

1. Put your script in `voiceover/scripts/` (see the format below) and, if you want the voice put
   on a video, the silent MP4 in `voiceover/videos/`.
2. Actions → **Voiceover** → Run workflow. Fill in the script path, voice and (optionally) video path.
3. Download **voiceover** from the run page under Artifacts. It holds the voice track (WAV) and,
   if you gave a video, the finished MP4.

## Run it on your computer or in Colab

```bash
bash voiceover/setup.sh                       # once: packages, espeak-ng, ~120 MB voice model
python voiceover/make_voiceover.py voiceover/scripts/max_interlagos_2m15.txt
python voiceover/make_voiceover.py my_script.txt --voice am_michael --video reel.mp4
python voiceover/make_voiceover.py --list-voices
```

## Script format

One cue per line: a timestamp, then what to say. `[1]`-style marker numbers are optional and are not read out.

```
0:08  Twenty sixteen. He's nineteen years old, and it is pouring.
0:12  [1] He starts fourth behind the safety car.
```

Each line starts at its time. If it would run into the next line it is sped up a little
(up to `--max-speed`, default 1.25×); if it is still too long the next line waits, so lines never overlap.
The run prints how late any line started, so you can trim the words if needed.

## Voices

Kokoro v1.0. British: `bm_george` (default), `bm_lewis`, `bm_daniel`, `bm_fable`, `bf_emma`, `bf_isabella`.
American: `am_michael`, `am_fenrir`, `am_puck`, `af_heart`, `af_bella`. Names starting with `b` use British pronunciation.

If a driver's name sounds wrong, add a sound-alike spelling to `RESPELL` at the top of
`make_voiceover.py` (for example `"Ocon": "Oh-kon"`). Only lines you changed are regenerated.

## Credits and licences

- Voice model: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, Apache 2.0. Commercial use is fine.
- ONNX model files and phoneme vocabulary: [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), MIT.
- Pronunciation: [espeak-ng](https://github.com/espeak-ng/espeak-ng), GPL-3.0. It is installed separately and not stored in this repo.
