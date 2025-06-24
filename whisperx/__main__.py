from .whisperx_api import WhisperXTranscriber
import argparse
import os
import subprocess  # For running m2.py

DEFAULT_AUDIO_DIR = "audio"
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac')

def get_audio_files(audio_dir, specific_files=None):
    if specific_files:
        return specific_files
    return [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", nargs="*", help="Specific audio file(s) to process")
    parser.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR, help="Directory to scan for audio files")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"), help="HuggingFace token (optional if set in env)")
    parser.add_argument("--model", default="large-v2", help="Model size (tiny, base, small, medium, large-v1, large-v2)")
    
    args = parser.parse_args()
    
    audio_files = get_audio_files(args.audio_dir, args.audio)
    if not audio_files:
        raise ValueError(f"No supported audio files found in {args.audio_dir}")
    
    transcriber = WhisperXTranscriber(
        model_name=args.model,
        hf_token=args.hf_token
    )
    
    for audio_path in audio_files:
        transcriber.transcribe(
            audio_path=audio_path,
            language="en",
            diarize=True if args.hf_token else False
        )
    
    transcriber.cleanup()

    # ✅ Run m2.py after transcription is complete
    subprocess.run(["python", "/teamspace/studios/this_studio/whisperx/latexGen.py"])

if __name__ == "__main__":
    cli()
