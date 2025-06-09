#!/usr/bin/env python3
"""Integration test script for TranscriMatic transcription engine."""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager
from src.transcription.transcription_engine_with_fallback import TranscriptionEngineWithFallback
from src.transcription.batch_processor import BatchTranscriptionProcessor
from src.audio.data_classes import ProcessedAudio, AudioChunk, AudioMetadata, AudioFile


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_transcription.log')
        ]
    )


def create_test_audio_object(audio_path: Path) -> ProcessedAudio:
    """Create a ProcessedAudio object from a test audio file."""
    # Create AudioFile object
    audio_file = AudioFile(
        source_path=audio_path,
        destination_path=audio_path,
        filename=audio_path.name,
        size=audio_path.stat().st_size,
        checksum="test_checksum",
        timestamp=datetime.now(),
        duration=None,  # Will be determined during processing
        format=audio_path.suffix[1:]
    )
    
    # For testing, create a single chunk with the whole file
    chunks = [AudioChunk(
        chunk_id=0,
        file_path=audio_path,
        start_time=0.0,
        end_time=60.0,  # Assume 60 seconds for testing
        duration=60.0
    )]
    
    # Create mock metadata
    metadata = AudioMetadata(
        duration=60.0,
        sample_rate=16000,
        channels=1,
        bitrate=256000,
        format=audio_path.suffix[1:],
        avg_volume=-20.0,
        peak_volume=-10.0,
        silence_ratio=0.1
    )
    
    return ProcessedAudio(
        original_file=audio_file,
        processed_path=audio_path,
        chunks=chunks,
        metadata=metadata,
        processing_time=1.0
    )


def test_single_file_transcription(config_path: Path, audio_path: Path):
    """Test transcription of a single audio file."""
    print(f"\n=== Testing Single File Transcription ===")
    print(f"Config: {config_path}")
    print(f"Audio: {audio_path}")
    
    # Load configuration
    config = ConfigManager(str(config_path))
    
    # Create transcription engine
    engine = TranscriptionEngineWithFallback(config)
    
    # Create test audio object
    processed_audio = create_test_audio_object(audio_path)
    
    # Validate audio quality
    print("\nValidating audio quality...")
    if not engine.validate_audio_quality(processed_audio):
        print("WARNING: Audio quality validation failed")
    
    # Perform transcription
    print("\nStarting transcription...")
    start_time = datetime.now()
    
    try:
        transcription = engine.transcribe_with_retry(processed_audio)
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nTranscription completed in {duration:.2f} seconds")
        
        # Display results
        print(f"\n=== Transcription Results ===")
        print(f"Language: {transcription.language}")
        print(f"Confidence: {transcription.confidence:.2%}")
        print(f"Audio Duration: {transcription.duration:.2f}s")
        print(f"Processing Time: {transcription.processing_time:.2f}s")
        print(f"Real-time Factor: {transcription.processing_time/transcription.duration:.2f}x")
        print(f"Segments: {len(transcription.segments)}")
        print(f"Speakers: {transcription.speaker_count}")
        
        print(f"\n=== Full Text ===")
        print(transcription.full_text[:500] + "..." if len(transcription.full_text) > 500 else transcription.full_text)
        
        if transcription.speaker_count > 1:
            print(f"\n=== Speaker Transcript ===")
            print(transcription.speaker_transcript[:500] + "..." if len(transcription.speaker_transcript) > 500 else transcription.speaker_transcript)
        
        # Save results
        output_file = Path(f"transcription_result_{audio_path.stem}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transcription.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return transcription
        
    except Exception as e:
        print(f"\nERROR: Transcription failed - {e}")
        logging.exception("Transcription error")
        return None


def test_batch_transcription(config_path: Path, audio_dir: Path):
    """Test batch transcription of multiple files."""
    print(f"\n=== Testing Batch Transcription ===")
    print(f"Config: {config_path}")
    print(f"Audio directory: {audio_dir}")
    
    # Find audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.m4a']:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Load configuration
    config = ConfigManager(str(config_path))
    
    # Create batch processor
    batch_processor = BatchTranscriptionProcessor(config)
    
    # Create ProcessedAudio objects
    processed_audios = [create_test_audio_object(f) for f in audio_files]
    
    # Process batch
    print("\nStarting batch processing...")
    start_time = datetime.now()
    
    results = batch_processor.process_batch(processed_audios)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nBatch processing completed in {duration:.2f} seconds")
    
    # Generate report
    report = batch_processor.generate_batch_report(results)
    
    print(f"\n=== Batch Report ===")
    print(f"Total files: {report['total_files']}")
    print(f"Total duration: {report['total_duration']:.2f}s")
    print(f"Total segments: {report['total_segments']}")
    print(f"Average confidence: {report['average_confidence']:.2%}")
    print(f"Language stats: {report['language_stats']}")
    print(f"Speaker stats: {report['speaker_stats']}")
    
    # Save report
    report_file = Path("batch_transcription_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_file}")


def test_model_loading():
    """Test model loading and basic functionality."""
    print("\n=== Testing Model Loading ===")
    
    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper imported successfully")
        
        # Try to load small model
        print("\nLoading 'base' model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ Model loaded successfully")
        
        # Check CUDA availability
        import torch
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    try:
        from pyannote.audio import Pipeline
        print("\n✓ pyannote.audio imported successfully")
    except Exception as e:
        print(f"\n✗ pyannote.audio import failed: {e}")
        print("  Speaker diarization will not be available")
    
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test TranscriMatic transcription engine")
    parser.add_argument("--config", type=Path, default=Path("config.test.yaml"),
                        help="Path to configuration file")
    parser.add_argument("--audio", type=Path, help="Path to audio file for single file test")
    parser.add_argument("--audio-dir", type=Path, help="Path to directory for batch test")
    parser.add_argument("--test-loading", action="store_true", help="Test model loading only")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("TranscriMatic Transcription Engine Test")
    print("=" * 50)
    
    # Test model loading
    if args.test_loading or not (args.audio or args.audio_dir):
        if not test_model_loading():
            print("\nModel loading test failed. Please check dependencies.")
            return 1
    
    # Test single file
    if args.audio:
        if not args.audio.exists():
            print(f"Error: Audio file not found: {args.audio}")
            return 1
        test_single_file_transcription(args.config, args.audio)
    
    # Test batch processing
    if args.audio_dir:
        if not args.audio_dir.is_dir():
            print(f"Error: Directory not found: {args.audio_dir}")
            return 1
        test_batch_transcription(args.config, args.audio_dir)
    
    if not (args.audio or args.audio_dir or args.test_loading):
        print("\nNo test specified. Use --help for options.")
        print("\nExamples:")
        print("  Test model loading:")
        print("    python test_transcription_integration.py --test-loading")
        print("\n  Test single file:")
        print("    python test_transcription_integration.py --audio test.wav")
        print("\n  Test batch processing:")
        print("    python test_transcription_integration.py --audio-dir ./audio_files/")
    
    print("\nTest completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())