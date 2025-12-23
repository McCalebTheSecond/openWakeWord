# Custom wakeword models (for `scripts/voice_hotkeys.py`)

This folder is for **your local wakeword model files** (not included with openWakeWord).

## What you need for the “whisper / hush” hotkey demo

Put these two files here:

- `wakewords/whisper.onnx`
- `wakewords/hush.onnx`

Then run:

```powershell
python scripts/voice_hotkeys.py
```

## How to get the models

openWakeWord does **not** ship with models for the words “whisper” or “hush”, so you must train them.

- **Easiest path**: use the Google Colab notebook linked from the main README (`README.md`) under *Training New Models*.
  - Train a model for the phrase `whisper`
  - Train a model for the phrase `hush`
  - Download the exported `.onnx` files and rename/copy them into this folder as `whisper.onnx` and `hush.onnx`

## One-time download for openWakeWord’s feature models

The runtime also needs the shared preprocessing models (melspectrogram + embedding). Download them once:

```powershell
python -c "import openwakeword; openwakeword.utils.download_models()"
```

## Quick test without training (optional)

You can temporarily use built-in wakewords just to test mic + hotkeys:

```powershell
python scripts/voice_hotkeys.py --whisper-model "hey jarvis" --hush-model "alexa"
```


