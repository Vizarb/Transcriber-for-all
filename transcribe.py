import sys
import argparse
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel


def separate_vocals_with_demucs(media_path: Path, demucs_model: str) -> Path:
    """
    Separate vocals from a mixed audio track using Demucs and return the vocals path.

    Parameters
    ----------
    media_path : Path
        Path to the input media file (e.g., .mp3, .wav, .m4a).
    demucs_model : str
        Demucs model name to use (e.g., "htdemucs", "htdemucs_ft").

    Returns
    -------
    Path
        Path to the generated vocals stem: separated/<model>/<input_stem>/vocals.wav

    Raises
    ------
    RuntimeError
        If Demucs invocation fails or the expected vocals file is not found.

    Notes
    -----
    - Uses the CLI: `python -m demucs -n <model> --two-stems vocals <file>`
    - Output layout:
        separated/{model}/{input_stem}/vocals.wav
        separated/{model}/{input_stem}/no_vocals.wav
    - Only the vocals stem is used downstream; `no_vocals.wav` is ignored.
    """
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        demucs_model,
        "--two-stems",
        "vocals",
        str(media_path),
    ]
    try:
        print(f"[demucs] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Demucs invocation failed. Make sure Demucs is installed "
            "(pip install demucs) and ffmpeg is available."
        ) from e

    out_dir = Path("separated") / demucs_model / media_path.stem
    vocals_path = out_dir / "vocals.wav"
    if not vocals_path.exists():
        raise RuntimeError(f"Expected vocals file not found: {vocals_path}")
    return vocals_path




def srt_ts(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp 'HH:MM:SS,mmm'.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem,    60_000)
    ss, ms  = divmod(rem,     1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"



def write_srt(segments, srt_path: Path) -> None:
    """
    Write segments to an SRT file at srt_path.
    """
    with open(srt_path, "w", encoding="utf-8") as srt:
        for i, seg in enumerate(segments, 1):
            start = srt_ts(seg.start)
            end = srt_ts(seg.end)
            text = seg.text.strip()
            srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def main():
    """
    Transcribe or translate audio/video files using faster-whisper, with optional
    Demucs vocal separation for songs.

    Behavior
    --------
    - If `--song` is passed, Demucs isolates vocals first and the transcription
        runs on `vocals.wav`. Output goes to: `transcription/lyrics/<name>_lyrics.txt`
    - Without `--song`, the original audio is transcribed. Output goes to:
        `transcription/<name>.txt`
    - If `--srt` is passed, an additional `.srt` file is written alongside the `.txt`.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe or translate audio using faster-whisper, with optional Demucs vocal separation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py audio/podcast.mp3\n"
            "  python transcribe.py audio/interview.m4a --lang he --timestamps\n"
            "  python transcribe.py audio/lecture.mp3 --lang he --translate --model medium --timestamps\n"
            "  python transcribe.py audio/song.mp3 --song --timestamps --srt\n"
            "  python transcribe.py audio/clip.wav --device cuda --compute-type float16\n"
        ),
    )

    parser.add_argument("media", help="Path to audio/video (mp3, wav, m4a, mp4, mkv, ...)")
    parser.add_argument("--model", default="small", help="Whisper model: tiny|base|small|medium|large-v3 (default: small)")
    parser.add_argument("--lang", default=None, help="Force language code (e.g., en, he). Otherwise auto-detect.")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--chunk", type=int, default=30, help="Chunk length in seconds (default: 30)")
    parser.add_argument("--timestamps", action="store_true", help="Include segment timestamps in the .txt output")
    parser.add_argument("--srt", action="store_true", help="Also write an .srt subtitle file")
    parser.add_argument("--translate", action="store_true", help="Translate to English instead of transcribing")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD (voice activity detection)")
    parser.add_argument("--beam", type=int, default=5, help="Beam size for decoding (default: 5)")
    parser.add_argument("--noprev", action="store_true", help="Do not condition on previous text")
    parser.add_argument("--prompt", default=None, help="Initial prompt to bias decoding")

    # Song / Demucs options
    parser.add_argument("--song", action="store_true", help="Use Demucs to extract vocals before transcription")
    parser.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)")

    # Device/compute knobs
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for faster-whisper (default: cpu)")
    parser.add_argument(
        "--compute-type",
        default=None,
        help=(
            "Override compute type for faster-whisper (e.g., int8, int8_float16, float16, float32). "
            "Default: int8 on CPU, float16 on CUDA."
        ),
    )

    args = parser.parse_args()

    # Decide output directory + filename by --song
    media_path = Path(args.media)
    if args.song:
        out_dir = Path("transcription/lyrics")
        base_name = f"{media_path.stem}_lyrics"
    else:
        out_dir = Path("transcription")
        base_name = f"{media_path.stem}"

    out_dir.mkdir(parents=True, exist_ok=True)
    output_txt = out_dir / f"{base_name}.txt"
    output_srt = out_dir / f"{base_name}.srt"

    # If song mode: separate vocals with Demucs and point Whisper at the vocals stem
    media_for_asr = media_path
    if args.song:
        try:
            vocals_path = separate_vocals_with_demucs(media_path, args.demucs_model)
            print(f"[demucs] Using vocals stem: {vocals_path}")
            media_for_asr = vocals_path
        except RuntimeError as e:
            print(f"[demucs] Warning: {e} — proceeding with original audio.")

    # Choose compute_type default based on device if not explicitly set
    compute_type = args.compute_type if args.compute_type is not None else ("float16" if args.device == "cuda" else "int8")

    # Load model
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    # Transcribe
    seg_iter, info = model.transcribe(
        str(media_for_asr),
        language=args.lang,
        beam_size=args.beam,
        temperature=args.temp,
        chunk_length=args.chunk,
        vad_filter=not args.no_vad,
        condition_on_previous_text=not args.noprev,
        no_speech_threshold=0.3,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        initial_prompt=args.prompt,
        task="translate" if args.translate else "transcribe",
    )
    # Materialize generator so we can reuse for .txt and .srt
    segments = list(seg_iter)

    # Write .txt
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Detected language '{info.language}' (p={info.language_probability:.2f})\n")
        f.write(f"Audio duration: {info.duration:.1f} sec\n")
        f.write(f"Audio duration_after_vad: {info.duration_after_vad:.1f} sec\n")
        if args.song:
            f.write("Source: vocals stem (Demucs)\n")
        f.write("Transcript:\n\n")

        if args.timestamps:
            for s in segments:
                f.write(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}\n")
        else:
            text = "".join(s.text for s in segments).strip()
            text = text.replace(",", "\n")
            f.write(text)

    print(f"Transcript saved to {output_txt}")

    # Write .srt if requested
    if args.srt:
        write_srt(segments, output_srt)
        print(f"SRT saved to {output_srt}")


if __name__ == "__main__":
    main()
