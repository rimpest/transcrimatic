"""Mock audio processor for testing transcription engine."""

import logging
from pathlib import Path
from typing import List
from datetime import datetime
import hashlib
import wave
import contextlib

# Import from main project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.data_classes import (
    AudioFile, AudioMetadata, AudioChunk, ProcessedAudio
)
from src.config.config_manager import ConfigManager


class MockAudioProcessor:
    """Mock audio processor that creates ProcessedAudio objects for testing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
    
    def process_audio_file(self, audio_path: Path) -> ProcessedAudio:
        """Create a ProcessedAudio object from an audio file."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create AudioFile object
        audio_file = AudioFile(
            source_path=audio_path,
            destination_path=audio_path,
            filename=audio_path.name,
            size=audio_path.stat().st_size,
            checksum=self._calculate_checksum(audio_path),
            timestamp=datetime.now(),
            duration=None,
            format=audio_path.suffix[1:]
        )
        
        # Extract metadata
        metadata = self._extract_metadata(audio_path)
        audio_file.duration = metadata.duration
        
        # Create chunks (for testing, we'll create based on duration)
        chunks = self._create_chunks(audio_path, metadata.duration)
        
        return ProcessedAudio(
            original_file=audio_file,
            processed_path=audio_path,
            chunks=chunks,
            metadata=metadata,
            processing_time=0.5  # Mock processing time
        )
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def _extract_metadata(self, audio_path: Path) -> AudioMetadata:
        """Extract or mock audio metadata."""
        # Try to read actual WAV metadata
        if audio_path.suffix.lower() == '.wav':
            try:
                with contextlib.closing(wave.open(str(audio_path), 'r')) as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    channels = wav_file.getnchannels()
                    
                    return AudioMetadata(
                        duration=duration,
                        sample_rate=rate,
                        channels=channels,
                        bitrate=rate * channels * 16,  # Assuming 16-bit
                        format='wav',
                        avg_volume=-20.0,  # Mock values
                        peak_volume=-10.0,
                        silence_ratio=0.1
                    )
            except Exception as e:
                self.logger.warning(f"Failed to read WAV metadata: {e}")
        
        # Return mock metadata for other formats or on error
        return AudioMetadata(
            duration=60.0,  # Mock 1 minute
            sample_rate=16000,
            channels=1,
            bitrate=256000,
            format=audio_path.suffix[1:],
            avg_volume=-20.0,
            peak_volume=-10.0,
            silence_ratio=0.1
        )
    
    def _create_chunks(self, audio_path: Path, duration: float) -> List[AudioChunk]:
        """Create audio chunks based on configuration."""
        chunk_duration = self.config.get("processing.audio.chunking.max_duration", 300)
        chunks = []
        
        if duration <= chunk_duration:
            # Single chunk for short audio
            chunks.append(AudioChunk(
                chunk_id=0,
                file_path=audio_path,
                start_time=0.0,
                end_time=duration,
                duration=duration
            ))
        else:
            # Multiple chunks for long audio
            chunk_id = 0
            start_time = 0.0
            
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                
                chunks.append(AudioChunk(
                    chunk_id=chunk_id,
                    file_path=audio_path,  # In reality, each chunk would be a separate file
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time
                ))
                
                chunk_id += 1
                start_time = end_time
        
        return chunks
    
    def validate_audio(self, audio_path: Path) -> bool:
        """Validate if audio file is suitable for processing."""
        if not audio_path.exists():
            return False
        
        # Check file size
        size = audio_path.stat().st_size
        min_size = self.config.get("transfer.min_file_size", 1024)
        max_size = self.config.get("transfer.max_file_size", 2147483648)
        
        if size < min_size or size > max_size:
            self.logger.warning(f"File size {size} outside valid range")
            return False
        
        # Check format
        valid_formats = self.config.get("transfer.source_extensions", [".wav", ".mp3", ".m4a"])
        if audio_path.suffix.lower() not in valid_formats:
            self.logger.warning(f"Invalid format: {audio_path.suffix}")
            return False
        
        return True