"""Command-line interface for the transcription service.

This script is a thin wrapper around the core functions defined in
`transcription_service.py`.  It parses command-line arguments,
invokes the `transcribe` function, and prints where the result files
were saved.
"""

import argparse
from pathlib import Path
from transcription_service import transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe or translate audio/video files using faster-whisper, "
            "with optional Demucs vocal separation."
        ),
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
    parser.add_argument("media", help="Path to the audio or video file to process")
    parser.add_argument("--model", default="small", help="Whisper model: tiny|base|small|medium|large-v3 (default: small)")
    parser.add_argument("--lang", default=None, help="Force language code instead of auto-detection")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--chunk", type=int, default=30, help="Chunk length in seconds (default: 30)")
    parser.add_argument("--timestamps", action="store_true", help="Include segment timestamps in the .txt output")
    parser.add_argument("--srt", action="store_true", help="Also write an .srt subtitle file")
    parser.add_argument("--translate", action="store_true", help="Translate to English instead of transcribing")
    parser.add_argument("--no-vad", action="store_true", help="Disable voice activity detection")
    parser.add_argument("--beam", type=int, default=5, help="Beam size for decoding (default: 5)")
    parser.add_argument("--noprev", action="store_true", help="Do not condition on previous text")
    parser.add_argument("--prompt", default=None, help="Initial prompt to bias decoding")
    parser.add_argument("--song", action="store_true", help="Use Demucs to separate vocals before transcription")
    parser.add_argument("--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for faster-whisper (default: cpu)")
    parser.add_argument(
        "--compute-type",
        default=None,
        help=(
            "Override compute type for faster-whisper (e.g., int8, int8_float16, float16, float32). "
            "Defaults to int8 on CPU and float16 on CUDA."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    media_path = Path(args.media)
    result = transcribe(
        media_path=media_path,
        model=args.model,
        lang=args.lang,
        temp=args.temp,
        chunk=args.chunk,
        timestamps=args.timestamps,
        srt=args.srt,
        translate=args.translate,
        no_vad=args.no_vad,
        beam=args.beam,
        noprev=args.noprev,
        prompt=args.prompt,
        song=args.song,
        demucs_model=args.demucs_model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(f"Transcript saved to {result['txt_path']}")
    if args.srt:
        print(f"SRT saved to {result['srt_path']}")


if __name__ == "__main__":
    main()
