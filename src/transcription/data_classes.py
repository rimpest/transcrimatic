"""Data classes for the transcription engine module."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from ..audio.data_classes import ProcessedAudio


@dataclass
class WordTiming:
    """Word-level timing information from Whisper."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class WhisperSegment:
    """Raw segment output from faster-whisper."""
    start: float
    end: float
    text: str
    words: Optional[List[WordTiming]]
    avg_logprob: float
    no_speech_prob: float


@dataclass
class SpeakerSegment:
    """Speaker diarization segment from pyannote."""
    start: float
    end: float
    speaker: str
    confidence: float


@dataclass
class TranscriptionSegment:
    """Merged transcription segment with speaker information."""
    id: int
    start_time: float
    end_time: float
    text: str
    speaker_id: str
    confidence: float
    

@dataclass
class Transcription:
    """Complete transcription result with all metadata."""
    audio_file: ProcessedAudio
    segments: List[TranscriptionSegment]
    full_text: str
    speaker_transcript: str  # Formatted for LLM with speaker labels
    language: str
    confidence: float
    duration: float
    speaker_count: int
    processing_time: Optional[float] = None  # Time taken to transcribe
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'audio_file': {
                'original_filename': self.audio_file.original_file.filename,
                'duration': self.duration,
                'chunks': len(self.audio_file.chunks)
            },
            'segments': [
                {
                    'id': seg.id,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'speaker_id': seg.speaker_id,
                    'confidence': seg.confidence
                }
                for seg in self.segments
            ],
            'full_text': self.full_text,
            'speaker_transcript': self.speaker_transcript,
            'language': self.language,
            'confidence': self.confidence,
            'duration': self.duration,
            'speaker_count': self.speaker_count,
            'processing_time': self.processing_time,
            'real_time_factor': self.processing_time / self.duration if self.processing_time and self.duration > 0 else None
        }