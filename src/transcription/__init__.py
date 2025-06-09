"""Transcription engine module for Spanish audio transcription."""

from .transcription_engine import TranscriptionEngine
from .transcription_engine_with_fallback import TranscriptionEngineWithFallback
from .data_classes import (
    Transcription,
    TranscriptionSegment,
    WhisperSegment,
    SpeakerSegment,
    WordTiming
)

__all__ = [
    'TranscriptionEngine',
    'TranscriptionEngineWithFallback',
    'Transcription',
    'TranscriptionSegment',
    'WhisperSegment',
    'SpeakerSegment',
    'WordTiming'
]