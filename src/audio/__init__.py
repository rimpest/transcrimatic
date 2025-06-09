"""
Audio processing module for TranscriMatic.
"""

from .audio_processor import AudioProcessor, AudioProcessingError
from .data_classes import (
    AudioChunk, AudioMetadata, ProcessedAudio, SilenceSegment
)

__all__ = [
    'AudioProcessor',
    'AudioProcessingError',
    'AudioChunk',
    'AudioMetadata', 
    'ProcessedAudio',
    'SilenceSegment'
]