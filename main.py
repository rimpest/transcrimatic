#!/usr/bin/env python3
"""
TranscriMatic - Audio Transcription & Analysis System
Main entry point with CLI interface, file processing, and USB device detection
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# Import TranscriMatic components
from src.config.config_manager import ConfigManager, ConfigurationError
from src.transcription.transcription_engine_with_fallback import TranscriptionEngineWithFallback
from src.audio.audio_processor import AudioProcessor
from src.conversation.conversation_segmenter import ConversationSegmenter
from src.llm.analyzer import LLMAnalyzer


class TranscriMaticCLI:
    """Main CLI interface for TranscriMatic."""
    
    def __init__(self):
        self.config = None
        self.logger = None
        
    def setup_logging(self, level: str = "INFO"):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_interactive_config(self, config_path: Path) -> bool:
        """Interactive configuration creation."""
        print("\n" + "="*60)
        print("TRANSCRIMATIC CONFIGURATION WIZARD")
        print("="*60)
        print("Let's create your configuration file step by step.")
        
        config = {
            "device": {},
            "paths": {},
            "processing": {
                "whisper_model": "base",
                "whisper_device": "auto",
                "language": "es",
                "faster_whisper": {
                    "beam_size": 5,
                    "best_of": 5,
                    "temperature": 0.0,
                    "vad_filter": True
                },
                "diarization": {
                    "enabled": False,
                    "min_speakers": 1,
                    "max_speakers": 10
                }
            },
            "llm": {
                "provider": "ollama",
                "model": "llama2"
            },
            "output": {
                "format": "json",
                "transcriptions_dir": "output/transcriptions"
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        try:
            # Device configuration
            print("\n📱 DEVICE CONFIGURATION")
            print("-" * 30)
            device_id = input("Device identifier (e.g., 'my_recorder'): ").strip() or "default_device"
            config["device"]["identifier"] = device_id
            
            # Paths configuration
            print("\n📁 PATHS CONFIGURATION")
            print("-" * 30)
            audio_dest = input("Audio destination directory [./audio]: ").strip() or "./audio"
            transcriptions = input("Transcriptions directory [./output/transcriptions]: ").strip() or "./output/transcriptions"
            summaries = input("Summaries directory [./output/summaries]: ").strip() or "./output/summaries"
            
            config["paths"] = {
                "audio_destination": audio_dest,
                "transcriptions": transcriptions,
                "summaries": summaries
            }
            
            # Processing configuration
            print("\n🎙️ TRANSCRIPTION CONFIGURATION")
            print("-" * 30)
            print("Whisper model options: tiny, base, small, medium, large-v2, large-v3")
            model = input("Whisper model [base]: ").strip() or "base"
            
            print("Device options: auto, cpu, cuda")
            device = input("Processing device [auto]: ").strip() or "auto"
            
            print("Language code (e.g., 'es' for Spanish, 'en' for English)")
            language = input("Language [es]: ").strip() or "es"
            
            config["processing"]["whisper_model"] = model
            config["processing"]["whisper_device"] = device
            config["processing"]["language"] = language
            
            # Speaker diarization
            print("\n👥 SPEAKER DIARIZATION")
            print("-" * 30)
            print("Speaker diarization identifies different speakers in conversations.")
            enable_diar = input("Enable speaker diarization? (y/N): ").strip().lower()
            
            if enable_diar == 'y':
                config["processing"]["diarization"]["enabled"] = True
                print("\\n⚠️  Speaker diarization requires a HuggingFace token.")
                print("   1. Create account at https://huggingface.co")
                print("   2. Get token from https://huggingface.co/settings/tokens")
                print("   3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
                
                hf_token = input("HuggingFace token (or press Enter to configure later): ").strip()
                if hf_token:
                    config["pyannote"] = {"auth_token": hf_token}
            
            # Output configuration
            print("\n📄 OUTPUT CONFIGURATION")
            print("-" * 30)
            print("Output format options: json, txt, md")
            output_format = input("Output format [json]: ").strip() or "json"
            config["output"]["format"] = output_format
            
            # LLM configuration
            print("\n🤖 LLM ANALYSIS (Optional)")
            print("-" * 30)
            print("LLM providers: ollama, openai, gemini")
            llm_provider = input("LLM provider [ollama]: ").strip() or "ollama"
            config["llm"]["provider"] = llm_provider
            
            # Save configuration
            print(f"\n💾 Saving configuration to {config_path}...")
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Configuration created successfully!")
            print(f"\nYou can edit {config_path} manually to fine-tune settings.")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n❌ Configuration cancelled.")
            return False
        except Exception as e:
            print(f"\n❌ Error creating configuration: {e}")
            return False
    
    def detect_usb_devices(self) -> List[Path]:
        """Detect mounted USB devices."""
        usb_paths = []
        
        # Common mount points for USB devices
        mount_points = [
            Path("/media"),      # Linux
            Path("/mnt"),        # Linux
            Path("/Volumes"),    # macOS
        ]
        
        for mount_point in mount_points:
            if mount_point.exists():
                try:
                    for path in mount_point.iterdir():
                        if path.is_dir() and not path.name.startswith('.'):
                            usb_paths.append(path)
                except PermissionError:
                    continue
        
        # Windows drive detection
        if sys.platform == "win32":
            import string
            import psutil
            for drive in string.ascii_uppercase:
                drive_path = Path(f"{drive}:/")
                if drive_path.exists():
                    try:
                        # Check if it's a removable drive
                        for partition in psutil.disk_partitions():
                            if partition.mountpoint == str(drive_path) and 'removable' in partition.opts:
                                usb_paths.append(drive_path)
                    except:
                        continue
        
        return usb_paths
    
    def find_audio_files(self, directory: Path, extensions: List[str] = None) -> List[Path]:
        """Find audio files in directory."""
        if extensions is None:
            extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        
        audio_files = []
        for ext in extensions:
            audio_files.extend(directory.glob(f"**/*{ext}"))
            audio_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        return sorted(audio_files)
    
    def process_single_file(self, audio_path: Path, config: ConfigManager) -> bool:
        """Process a single audio file."""
        print(f"\n🎵 Processing: {audio_path.name}")
        print("-" * 50)
        
        try:
            # Initialize components
            self.logger.info("Initializing transcription pipeline...")
            
            audio_processor = AudioProcessor(config)
            transcription_engine = TranscriptionEngineWithFallback(config)
            segmenter = ConversationSegmenter(config)
            llm_analyzer = LLMAnalyzer(config)
            
            # Step 1: Process audio
            self.logger.info("Step 1: Processing audio file...")
            start_time = time.time()
            
            if not audio_processor.validate_audio(audio_path):
                print(f"❌ Invalid audio file: {audio_path}")
                return False
            
            processed_audio = audio_processor.process_audio_file(audio_path)
            audio_time = time.time() - start_time
            print(f"✓ Audio processed in {audio_time:.2f}s")
            
            # Step 2: Transcription
            self.logger.info("Step 2: Transcribing audio...")
            transcription_start = time.time()
            
            transcription = transcription_engine.transcribe_with_retry(processed_audio)
            transcription_time = time.time() - transcription_start
            
            print(f"✓ Transcribed in {transcription_time:.2f}s")
            print(f"  - Language: {transcription.language}")
            print(f"  - Confidence: {transcription.confidence:.1%}")
            print(f"  - Segments: {len(transcription.segments)}")
            print(f"  - Speakers: {transcription.speaker_count}")
            print(f"  - Real-time factor: {transcription.processing_time/transcription.duration:.2f}x")
            
            # Step 3: Conversation segmentation
            self.logger.info("Step 3: Segmenting conversation...")
            segments = segmenter.segment_conversation(transcription)
            print(f"✓ Segmented into {len(segments)} conversation parts")
            
            # Step 4: LLM Analysis
            if config.get("llm.provider"):
                self.logger.info("Step 4: Analyzing with LLM...")
                analyses = []
                for segment in segments:
                    analysis = llm_analyzer.analyze_conversation(
                        segment["transcript"], segment
                    )
                    analyses.append(analysis)
                print(f"✓ LLM analysis completed for {len(analyses)} segments")
            
            # Step 5: Save results
            output_dir = Path(config.get("output.transcriptions_dir", "output/transcriptions"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{audio_path.stem}_transcription.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(transcription.to_dict(), f, ensure_ascii=False, indent=2)
            
            print(f"✅ Results saved to: {output_file}")
            
            # Display preview
            print(f"\n📝 Transcription Preview:")
            preview = transcription.full_text[:300]
            print(f"{preview}..." if len(transcription.full_text) > 300 else preview)
            
            if transcription.speaker_count > 1:
                print(f"\n👥 Speaker Transcript Preview:")
                speaker_preview = transcription.speaker_transcript[:300]
                print(f"{speaker_preview}..." if len(transcription.speaker_transcript) > 300 else speaker_preview)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            print(f"❌ Error processing {audio_path}: {e}")
            return False
    
    def process_usb_files(self, config: ConfigManager) -> bool:
        """Process audio files from USB devices."""
        print("\n🔍 Scanning for USB devices...")
        
        usb_devices = self.detect_usb_devices()
        if not usb_devices:
            print("❌ No USB devices found.")
            return False
        
        print(f"📱 Found {len(usb_devices)} USB device(s):")
        for i, device in enumerate(usb_devices):
            print(f"  {i+1}. {device}")
        
        # Let user select device or process all
        if len(usb_devices) == 1:
            selected_device = usb_devices[0]
            print(f"\\n🎯 Using device: {selected_device}")
        else:
            try:
                choice = input(f"\\nSelect device (1-{len(usb_devices)}) or 'a' for all: ").strip()
                if choice.lower() == 'a':
                    selected_device = None  # Process all
                else:
                    idx = int(choice) - 1
                    selected_device = usb_devices[idx]
            except (ValueError, IndexError):
                print("❌ Invalid selection.")
                return False
        
        # Find audio files
        total_processed = 0
        total_successful = 0
        
        devices_to_process = usb_devices if selected_device is None else [selected_device]
        
        for device in devices_to_process:
            print(f"\\n🔍 Scanning {device} for audio files...")
            audio_files = self.find_audio_files(device)
            
            if not audio_files:
                print(f"📭 No audio files found in {device}")
                continue
            
            print(f"🎵 Found {len(audio_files)} audio file(s)")
            
            for audio_file in audio_files:
                total_processed += 1
                print(f"\\n[{total_processed}] Processing: {audio_file.relative_to(device)}")
                
                if self.process_single_file(audio_file, config):
                    total_successful += 1
                    
                    # Optional: Move processed files
                    if config.get("transfer.delete_after_transfer", False):
                        try:
                            audio_file.unlink()
                            print(f"🗑️  Deleted: {audio_file}")
                        except Exception as e:
                            print(f"⚠️  Could not delete {audio_file}: {e}")
        
        print(f"\\n📊 Summary: {total_successful}/{total_processed} files processed successfully")
        return total_successful > 0
    
    def run(self, args):
        """Main run method."""
        # Setup logging
        self.setup_logging(args.log_level)
        
        print("🎙️  TranscriMatic - Audio Transcription & Analysis System")
        print("=" * 60)
        
        # Handle configuration
        if args.setup_config or not args.config.exists():
            if not args.config.exists():
                print(f"⚠️  Configuration file not found: {args.config}")
                create_config = input("Create configuration interactively? (Y/n): ").strip().lower()
                if create_config in ['', 'y', 'yes']:
                    if not self.create_interactive_config(args.config):
                        return 1
                else:
                    print("❌ Cannot proceed without configuration.")
                    return 1
            else:
                if not self.create_interactive_config(args.config):
                    return 1
        
        # Load configuration
        try:
            self.config = ConfigManager(str(args.config))
            print(f"✅ Configuration loaded: {args.config}")
        except ConfigurationError as e:
            print(f"❌ Configuration error: {e}")
            print("\\nRun with --setup-config to create a new configuration.")
            return 1
        
        # Test mode - just validate setup
        if args.test:
            print("\\n🧪 Testing configuration and dependencies...")
            
            # Test transcription engine
            try:
                engine = TranscriptionEngineWithFallback(self.config)
                print("✅ Transcription engine initialized")
            except Exception as e:
                print(f"❌ Transcription engine error: {e}")
                return 1
            
            print("✅ All tests passed!")
            return 0
        
        # Process specific file
        if args.file:
            if not args.file.exists():
                print(f"❌ File not found: {args.file}")
                return 1
            
            success = self.process_single_file(args.file, self.config)
            return 0 if success else 1
        
        # Process USB files
        if args.usb:
            success = self.process_usb_files(self.config)
            return 0 if success else 1
        
        # Daemon mode (future implementation)
        if args.daemon:
            print("🔄 Starting daemon mode...")
            print("⚠️  Daemon mode not yet implemented. Use --usb for now.")
            return 1
        
        # No specific action - show help
        print("\\n❓ No action specified. Use --help for options.")
        print("\\nQuick start:")
        print("  --file audio.mp3        Process a specific file")
        print("  --usb                   Process files from USB devices")
        print("  --setup-config          Create/update configuration")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TranscriMatic - Audio Transcription & Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific audio file
  python main.py --file audio.mp3
  
  # Process all audio files from USB devices
  python main.py --usb
  
  # Create/update configuration interactively
  python main.py --setup-config
  
  # Test configuration and dependencies
  python main.py --test
  
  # Run in daemon mode (monitor USB devices)
  python main.py --daemon
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("config.json"),
        help="Configuration file path (default: config.json)"
    )
    parser.add_argument(
        "--setup-config",
        action="store_true",
        help="Run interactive configuration setup"
    )
    
    # Processing options
    parser.add_argument(
        "--file", 
        type=Path,
        help="Process a specific audio file"
    )
    parser.add_argument(
        "--usb",
        action="store_true",
        help="Process audio files from USB devices"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode (monitor for new files)"
    )
    
    # Other options
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test configuration and dependencies"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="TranscriMatic v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = TranscriMaticCLI()
    try:
        return cli.run(args)
    except KeyboardInterrupt:
        print("\\n\\n🛑 Interrupted by user")
        return 130
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        logging.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())