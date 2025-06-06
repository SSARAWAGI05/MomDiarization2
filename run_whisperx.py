#!/usr/bin/env python3
"""
Enhanced WhisperX runner with additional functionality.
"""

import os
import sys
import argparse
from pathlib import Path
from whisperx import cli as whisperx_cli

def parse_args():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Enhanced WhisperX Transcription Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add all original whisperx arguments
    parser.add_argument("--audio", nargs="*", help="Audio file(s) to process")
    parser.add_argument("--audio_dir", default="./audio", help="Directory containing audio files")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    
    # Add new enhanced arguments
    parser.add_argument("--config", help="Path to config file with default settings")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()

def validate_paths(args):
    """Validate input/output paths."""
    if args.audio:
        for file in args.audio:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Audio file not found: {file}")
    
    if args.audio_dir and not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)

def main():
    args = parse_args()
    
    try:
        validate_paths(args)
        
        # Set up environment for whisperx
        sys.argv = [sys.argv[0]]  # Reset arguments
        
        # Add original whisperx arguments back
        if args.audio:
            sys.argv.extend(["--audio"] + args.audio)
        else:
            sys.argv.extend(["--audio_dir", args.audio_dir])
        
        sys.argv.extend(["--output_dir", args.output_dir])
        
        if args.debug:
            sys.argv.append("--verbose")
        
        # Execute whisperx
        whisperx_cli()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure proper path resolution
    sys.path.insert(0, str(Path(__file__).parent))
    main()