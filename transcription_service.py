"""Core transcription library.

This module exposes functions that perform the heavy lifting of
transcribing and optionally translating audio/video files using
faster‑whisper and Demucs.  It is separated from any command-line or
web interface to make the logic easy to reuse.

Functions:
- separate_vocals(media_path, demucs_model) -> Path
- transcribe(...) -> dict

You can import and call `transcribe()` from both the CLI and the FastAPI app.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys
from typing import Dict, Optional

from faster_whisper import WhisperModel


# transcription_service.py
import sys
import subprocess
from pathlib import Path

def separate_vocals(media_path: Path, demucs_model: str = "htdemucs") -> Path:
    cmd = [
        sys.executable,  # use the active venv's Python
        "-m", "demucs",
        "-n", demucs_model,
        "--two-stems", "vocals",
        str(media_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Demcus is running")
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs failed (exit {result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    out_dir = Path("separated") / demucs_model / media_path.stem
    vocals_path = out_dir / "vocals.wav"

    if not vocals_path.exists():
        raise RuntimeError(f"Expected vocals file not found: {vocals_path}")
    return vocals_path



def transcribe(
    media_path: Path,
    model: str = "small",
    lang: Optional[str] = None,
    temp: float = 0.0,
    chunk: int = 30,
    timestamps: bool = False,
    srt: bool = False,
    translate: bool = False,
    no_vad: bool = False,
    beam: int = 5,
    noprev: bool = False,
    prompt: Optional[str] = None,
    song: bool = False,
    demucs_model: str = "htdemucs",
    device: str = "cpu",
    compute_type: Optional[str] = None,
) -> Dict[str, Path]:
    """Transcribe or translate an audio/video file and return output paths."""
    if song:
        out_dir = Path("transcription/lyrics")
        base_name = f"{media_path.stem}_lyrics"
    else:
        out_dir = Path("transcription")
        base_name = media_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    output_txt = out_dir / f"{base_name}.txt"
    output_srt = out_dir / f"{base_name}.srt"

    # Optionally extract vocals
    media_for_asr: Path = media_path
    demucs_used = False
    if song:
        try:
            media_for_asr = separate_vocals(media_path, demucs_model)
            demucs_used = True
        except RuntimeError as e:
            print(f"[WARN] Demucs failed: {e}\nFalling back to original mix.")
            demucs_used = False

    # Determine default compute_type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    model_obj = WhisperModel(model, device=device, compute_type=compute_type)
    seg_iter, info = model_obj.transcribe(
        str(media_for_asr),
        language=lang,
        beam_size=beam,
        temperature=temp,
        chunk_length=chunk,
        vad_filter=not no_vad,
        condition_on_previous_text=not noprev,
        no_speech_threshold=0.3,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        initial_prompt=prompt,
        task="translate" if translate else "transcribe",
    )
    segments = list(seg_iter)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Detected language '{info.language}' (p={info.language_probability:.2f})\n")
        f.write(f"Audio duration: {info.duration:.1f} sec\n")
        f.write(f"Audio duration_after_vad: {info.duration_after_vad:.1f} sec\n")
        if song:
            f.write("Source: vocals stem (Demucs)\n" if demucs_used else "Source: original mix (Demucs fallback)\n")
        f.write("Transcript:\n\n")
        if timestamps:
            for seg in segments:
                f.write(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}\n")
        else:
            text = "".join(seg.text for seg in segments).strip().replace(",", "\n")
            f.write(text)

    result: Dict[str, Path] = {"txt_path": output_txt}
    if srt:
        def to_srt_time(seconds: float) -> str:
            total_ms = int(round(seconds * 1000.0))
            hh, rem = divmod(total_ms, 3_600_000)
            mm, rem = divmod(rem, 60_000)
            ss, ms = divmod(rem, 1_000)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        with open(output_srt, "w", encoding="utf-8") as srt_file:
            for i, seg in enumerate(segments, 1):
                start = to_srt_time(seg.start)
                end = to_srt_time(seg.end)
                srt_file.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")
        result["srt_path"] = output_srt

    return result
