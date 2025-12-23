"""
Voice-controlled hotkeys using openWakeWord (Windows-friendly).

Goal (default behavior):
- Say "whisper"  -> hold Ctrl+Alt down
- Say "hush"     -> release Ctrl+Alt

You must provide wakeword models for "whisper" and "hush" (custom models), OR temporarily
use built-in model names like "hey jarvis" to verify your microphone setup.

Recommended on Windows: use ONNX models (.onnx) and onnxruntime.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hold/release keys based on openWakeWord wakeword detections.")

    p.add_argument(
        "--whisper-model",
        default=str(Path("wakewords") / "whisper.onnx"),
        help='Path to the "whisper" wakeword model (.onnx recommended), or a built-in model name like "hey jarvis".',
    )
    p.add_argument(
        "--hush-model",
        default=str(Path("wakewords") / "hush.onnx"),
        help='Path to the "hush" wakeword model (.onnx recommended), or a built-in model name like "alexa".',
    )

    p.add_argument(
        "--inference-framework",
        choices=["onnx", "tflite"],
        default="onnx",
        help="Inference backend. On Windows, 'onnx' is recommended.",
    )

    p.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate (openWakeWord expects 16000).")
    p.add_argument("--blocksize", type=int, default=1280, help="Samples per audio block (1280 = 80ms @ 16kHz).")
    p.add_argument("--device", type=str, default=None, help="Input device name or index (sounddevice).")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")

    p.add_argument("--threshold-whisper", type=float, default=0.5, help='Detection threshold for "whisper".')
    p.add_argument("--threshold-hush", type=float, default=0.5, help='Detection threshold for "hush".')
    p.add_argument("--patience", type=int, default=2, help="Consecutive frames above threshold required to trigger.")

    p.add_argument(
        "--hold-keys",
        type=str,
        default="ctrl+alt",
        help="Keys to hold while active. Supported: ctrl, alt, shift. Format: ctrl+alt",
    )

    p.add_argument("--status-interval", type=float, default=2.0, help="Seconds between printing status lines. 0 disables.")
    p.add_argument("--queue-size", type=int, default=32, help="Audio block queue size (higher = more buffering).")

    return p.parse_args()


def _looks_like_path(s: str) -> bool:
    if any(sep in s for sep in ("/", "\\")):
        return True
    suffix = Path(s).suffix.lower()
    return suffix in {".onnx", ".tflite", ".pb", ".pt", ".pth"}


def _resolve_model_arg(s: str) -> str:
    """
    If the arg is an existing file, return its path.
    If it *looks* like a file path but doesn't exist, raise a helpful error.
    Otherwise treat it as a pre-trained model name (e.g. 'hey jarvis').
    """
    p = Path(s)
    if p.exists():
        return str(p)
    if _looks_like_path(s):
        raise FileNotFoundError(
            f"Model file not found: '{s}'.\n"
            "If you haven't trained/downloaded your custom wakeword models yet, see wakewords/README.md.\n"
            'Tip: you can temporarily test with built-in names, e.g. --whisper-model "hey jarvis".'
        )
    return s


def _ensure_feature_models_present(inference_framework: str) -> None:
    """
    openWakeWord needs feature extraction models (melspectrogram + embedding) in
    openwakeword/resources/models. The easiest way to fetch them is:
        python -c "import openwakeword; openwakeword.utils.download_models()"
    """
    import openwakeword

    resources_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    if inference_framework == "onnx":
        required = ["melspectrogram.onnx", "embedding_model.onnx"]
    else:
        required = ["melspectrogram.tflite", "embedding_model.tflite"]

    missing = [name for name in required if not (resources_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "openWakeWord feature models are missing (required for audio preprocessing): "
            f"{missing_str}\n"
            "Run this once to download the required models:\n"
            '  python -c "import openwakeword; openwakeword.utils.download_models()"\n'
            f"Expected directory: {resources_dir}"
        )


def _parse_hold_keys(spec: str):
    try:
        from pynput.keyboard import Key
    except ImportError as e:
        raise RuntimeError('Missing dependency "pynput". Install it with: pip install pynput') from e

    normalized = [part.strip().lower() for part in spec.replace(",", "+").split("+") if part.strip()]
    if not normalized:
        raise ValueError("--hold-keys must not be empty")

    mapping = {
        "ctrl": Key.ctrl_l,
        "control": Key.ctrl_l,
        "alt": Key.alt_l,
        "shift": Key.shift,
    }

    keys = []
    for name in normalized:
        if name not in mapping:
            raise ValueError(f"Unsupported key '{name}' in --hold-keys. Supported: {', '.join(mapping.keys())}")
        keys.append(mapping[name])

    # Remove duplicates but keep order
    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            deduped.append(k)
            seen.add(k)
    return tuple(deduped)


@dataclass
class HotkeyState:
    held: bool = False
    last_status_print_s: float = 0.0
    whisper_streak: int = 0
    hush_streak: int = 0


def _get_trigger_labels(oww, parent_model_name: str) -> tuple[str, ...]:
    """
    Returns the prediction labels to watch for a given loaded model.

    - For single-output wakewords, the prediction label is the model name (e.g. "hey jarvis")
    - For multi-class models (like "timer"), prediction labels are the class mapping values
      (e.g. "10_minute_timer", "1_hour_timer", ...)
    """
    # Single-output models
    if getattr(oww, "model_outputs", {}).get(parent_model_name, 1) == 1:
        return (parent_model_name,)

    # Multi-class models
    mapping = getattr(oww, "class_mapping", {}).get(parent_model_name, None)
    if mapping:
        return tuple(mapping.values())

    # Fallback (shouldn't happen, but avoids crashing)
    return (parent_model_name,)


def _press_keys(keys) -> None:
    from pynput.keyboard import Controller

    kb = Controller()
    for k in keys:
        kb.press(k)


def _release_keys(keys) -> None:
    from pynput.keyboard import Controller

    kb = Controller()
    for k in reversed(keys):
        kb.release(k)


def _run() -> int:
    args = _parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError('Missing dependency "sounddevice". Install it with: pip install sounddevice') from e
        print(sd.query_devices())
        return 0

    # Third-party imports (kept after --list-devices for nicer UX)
    try:
        import sounddevice as sd
    except ImportError as e:
        raise RuntimeError('Missing dependency "sounddevice". Install it with: pip install sounddevice') from e

    from openwakeword.model import Model

    whisper_model = _resolve_model_arg(args.whisper_model)
    hush_model = _resolve_model_arg(args.hush_model)
    _ensure_feature_models_present(args.inference_framework)

    hold_keys = _parse_hold_keys(args.hold_keys)
    state = HotkeyState()

    # openWakeWord uses model "names" for prediction keys:
    # - if you pass a filepath, name == basename without extension (e.g. wakewords/whisper.onnx -> "whisper")
    # - if you pass a built-in model name, name == the provided string (e.g. "hey jarvis")
    whisper_label = Path(whisper_model).stem if Path(whisper_model).exists() else whisper_model
    hush_label = Path(hush_model).stem if Path(hush_model).exists() else hush_model

    # Initialize model
    print("Loading models:")
    print(f'  whisper -> {whisper_model!r}')
    print(f'  hush    -> {hush_model!r}')
    oww = Model(
        wakeword_models=[whisper_model, hush_model],
        inference_framework=args.inference_framework,
    )

    whisper_trigger_labels = _get_trigger_labels(oww, whisper_label)
    hush_trigger_labels = _get_trigger_labels(oww, hush_label)

    threshold_whisper = float(args.threshold_whisper)
    threshold_hush = float(args.threshold_hush)
    patience = max(1, int(args.patience))

    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, int(args.queue_size)))
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
        # NOTE: Keep this callback lightweight; do inference in the worker thread.
        if status:
            # Print once in a while; too much printing can cause dropouts.
            if args.status_interval and (time.monotonic() - state.last_status_print_s) > args.status_interval:
                print(f"[audio] {status}", file=sys.stderr)
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            # Drop audio if the main thread can't keep up.
            pass

    def worker():
        nonlocal state
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Flatten to mono int16
            if block.ndim == 2:
                block_mono = block[:, 0]
            else:
                block_mono = block

            if block_mono.dtype != np.int16:
                block_mono = block_mono.astype(np.int16, copy=False)

            preds = oww.predict(block_mono)

            now = time.monotonic()
            if args.status_interval and (now - state.last_status_print_s) > args.status_interval:
                w_best = max(((preds.get(lbl, 0.0), lbl) for lbl in whisper_trigger_labels), default=(0.0, whisper_label))
                h_best = max(((preds.get(lbl, 0.0), lbl) for lbl in hush_trigger_labels), default=(0.0, hush_label))
                print(f"[status] held={state.held} whisper={w_best[1]}:{w_best[0]:.3f} hush={h_best[1]}:{h_best[0]:.3f}")
                state.last_status_print_s = now

            # Transition logic
            w_score, w_lbl = max(((preds.get(lbl, 0.0), lbl) for lbl in whisper_trigger_labels), default=(0.0, whisper_label))
            if w_score >= threshold_whisper:
                state.whisper_streak += 1
            else:
                state.whisper_streak = 0

            if not state.held and state.whisper_streak >= patience:
                print(f'[trigger] "{w_lbl}" ({w_score:.3f}) -> HOLD {args.hold_keys}')
                _press_keys(hold_keys)
                state.held = True
                state.whisper_streak = 0
                state.hush_streak = 0
                continue

            h_score, h_lbl = max(((preds.get(lbl, 0.0), lbl) for lbl in hush_trigger_labels), default=(0.0, hush_label))
            if h_score >= threshold_hush:
                state.hush_streak += 1
            else:
                state.hush_streak = 0

            if state.held and state.hush_streak >= patience:
                print(f'[trigger] "{h_lbl}" ({h_score:.3f}) -> RELEASE {args.hold_keys}')
                _release_keys(hold_keys)
                state.held = False
                state.whisper_streak = 0
                state.hush_streak = 0

    t = threading.Thread(target=worker, name="oww-worker", daemon=True)
    t.start()

    print(
        "\nListening... (Ctrl+C to quit)\n"
        "Tip: If your mic won't open at 16kHz, set the Windows device 'Default format' to 16000 Hz.\n"
    )

    try:
        with sd.InputStream(
            samplerate=int(args.sample_rate),
            channels=1,
            dtype="int16",
            blocksize=int(args.blocksize),
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    finally:
        stop_event.set()
        try:
            t.join(timeout=1.0)
        except RuntimeError:
            pass
        # Always release keys on shutdown, in case we were holding them.
        try:
            _release_keys(hold_keys)
        except Exception:
            pass


def main() -> None:
    try:
        raise SystemExit(_run())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


