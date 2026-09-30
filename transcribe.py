#!/usr/bin/env python3
"""Convert audio (or video) files to text using Whisper, running locally.

Requires:  pip install faster-whisper
Accepts any format ffmpeg/PyAV can decode: mp3, mp4, m4a, ogg/opus, wav, ...

Examples:
  python transcribe.py voice.mp4
  python transcribe.py voice.mp4 --language hi --model large-v3-turbo
  python transcribe.py voice.mp4 --translate            # output in English
  python transcribe.py a.mp3 b.ogg --format srt -o out/
"""
import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]


def fmt_time(seconds, srt=False):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe(model, path, language, translate, beam_size):
    segments, info = model.transcribe(
        str(path),
        language=language,
        task="translate" if translate else "transcribe",
        beam_size=beam_size,
        vad_filter=True,  # skip silence; reduces hallucinated text
        condition_on_previous_text=False,  # avoids repetition loops on long audio
    )
    print(f"[{path.name}] language={info.language} "
          f"({info.language_probability:.0%}), duration={fmt_time(info.duration)}",
          file=sys.stderr)
    result = []
    for seg in segments:
        # Show progress live, since long files can take a while on CPU.
        print(f"  [{fmt_time(seg.start)}] {seg.text.strip()}", file=sys.stderr)
        result.append(seg)
    return result


def render(segments, fmt):
    if fmt == "txt":
        return "\n".join(s.text.strip() for s in segments) + "\n"
    if fmt == "timestamps":
        return "\n".join(f"[{fmt_time(s.start)}] {s.text.strip()}" for s in segments) + "\n"
    # srt
    blocks = [
        f"{i}\n{fmt_time(s.start, True)} --> {fmt_time(s.end, True)}\n{s.text.strip()}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def main():
    p = argparse.ArgumentParser(description="Convert audio files to text (local Whisper).")
    p.add_argument("files", nargs="+", type=Path, help="audio/video files to transcribe")
    p.add_argument("-m", "--model", default="large-v3-turbo", choices=MODELS,
                   help="bigger = more accurate but slower (default: large-v3-turbo; "
                        "small is faster but unreliable on Hindi/Urdu or noisy audio)")
    p.add_argument("-l", "--language", default=None,
                   help="language code, e.g. en, hi, ur (default: auto-detect)")
    p.add_argument("-t", "--translate", action="store_true",
                   help="translate the speech into English instead of transcribing")
    p.add_argument("-f", "--format", default="timestamps", choices=["txt", "timestamps", "srt"],
                   help="output format (default: timestamps)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="where to write output files (default: next to each input)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--beam-size", type=int, default=5,
                   help="higher is slightly more accurate but slower (default: 5)")
    args = p.parse_args()

    missing = [f for f in args.files if not f.is_file()]
    if missing:
        p.error("file(s) not found: " + ", ".join(map(str, missing)))

    compute_type = "int8" if args.device == "cpu" else "default"
    print(f"Loading model '{args.model}' (downloads on first use)...", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    ext = "srt" if args.format == "srt" else "txt"
    for path in args.files:
        segments = transcribe(model, path, args.language, args.translate, args.beam_size)
        out_dir = args.output_dir or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.{ext}"
        out_path.write_text(render(segments, args.format), encoding="utf-8")
        print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
