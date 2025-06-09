"""Data classes for audio processing module."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

# Import AudioFile from transfer module
from ..transfer.data_classes import AudioFile


@dataclass
class AudioMetadata:
    """Metadata extracted from audio file."""
    duration: float  # seconds
    sample_rate: int  # Hz
    channels: int
    bitrate: int  # bits per second
    format: str
    avg_volume: float  # dBFS
    peak_volume: float  # dBFS
    silence_ratio: float  # 0.0 to 1.0


@dataclass
class AudioChunk:
    """Represents a chunk of processed audio."""
    chunk_id: int
    file_path: Path
    start_time: float  # seconds from beginning
    end_time: float
    duration: float
    
    # Optional: include raw data for in-memory processing
    data: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None


@dataclass
class SilenceSegment:
    """Represents a silence segment in audio."""
    start_time: float
    end_time: float
    duration: float


@dataclass
class ProcessedAudio:
    """Result of audio processing."""
    original_file: AudioFile
    processed_path: Path
    chunks: List[AudioChunk]
    metadata: AudioMetadata
    processing_time: float  # seconds