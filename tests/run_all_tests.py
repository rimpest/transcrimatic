#!/usr/bin/env python3
"""Simple test runner for TranscriMatic transcription tests."""

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    else:
        print(f"\n✅ Success: {description}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="TranscriMatic Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/run_all_tests.py --all
  
  # Test with specific audio file
  python tests/run_all_tests.py --audio personal_samples/sample_audio.mp3
  
  # Generate test audio and run basic test
  python tests/run_all_tests.py --generate --basic
  
  # Run unit tests only
  python tests/run_all_tests.py --unit
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests")
    parser.add_argument("--unit", action="store_true",
                        help="Run unit tests")
    parser.add_argument("--basic", action="store_true",
                        help="Run basic transcription test")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run end-to-end pipeline test")
    parser.add_argument("--generate", action="store_true",
                        help="Generate test audio files first")
    parser.add_argument("--audio", type=Path,
                        help="Audio file to test with")
    parser.add_argument("--config", type=Path, default=Path("config.test.yaml"),
                        help="Configuration file to use")
    
    args = parser.parse_args()
    
    # If no specific test selected, show help
    if not any([args.all, args.unit, args.basic, args.pipeline, args.audio]):
        parser.print_help()
        return 0
    
    success = True
    
    # Generate test audio if requested
    if args.generate:
        success &= run_command(
            [sys.executable, "tests/generate_test_audio.py"],
            "Generate test audio files"
        )
    
    # Determine audio file to use
    audio_file = args.audio
    if not audio_file and (args.basic or args.pipeline or args.all):
        # Use default test audio
        test_audio = Path("test_audio/test_spanish_greeting.wav")
        if test_audio.exists():
            audio_file = test_audio
        else:
            print("\n⚠️  No test audio found. Generate with --generate flag")
            # Try to use sample audio
            sample_audio = Path("personal_samples/sample_audio.mp3")
            if sample_audio.exists():
                audio_file = sample_audio
                print(f"Using sample audio: {audio_file}")
    
    # Run unit tests
    if args.unit or args.all:
        # Check if pytest is installed
        try:
            import pytest
            success &= run_command(
                [sys.executable, "-m", "pytest", "tests/test_transcription_engine.py", "-v"],
                "Unit tests"
            )
        except ImportError:
            print("\n⚠️  pytest not installed. Skipping unit tests.")
            print("Install with: pip install pytest")
    
    # Run basic transcription test
    if args.basic or args.all or args.audio:
        if audio_file and audio_file.exists():
            success &= run_command(
                [sys.executable, "tests/test_transcription_integration.py", 
                 "--audio", str(audio_file),
                 "--config", str(args.config)],
                f"Basic transcription test with {audio_file.name}"
            )
        else:
            print(f"\n⚠️  Audio file not found: {audio_file}")
    
    # Run pipeline test
    if args.pipeline or args.all:
        if audio_file and audio_file.exists():
            success &= run_command(
                [sys.executable, "tests/test_end_to_end.py",
                 "--audio", str(audio_file),
                 "--config", str(args.config)],
                f"End-to-end pipeline test with {audio_file.name}"
            )
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check output above.")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())