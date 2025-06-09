"""Data classes for file transfer module."""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional


@dataclass
class AudioFile:
    """Represents an audio file transferred from device."""
    source_path: Path
    destination_path: Path
    filename: str
    size: int  # bytes
    checksum: str
    timestamp: datetime
    duration: Optional[float]  # seconds
    format: str  # mp3, wav, etc.