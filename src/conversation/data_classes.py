"""Data classes for conversation segmentation module."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np

from ..transcription.data_classes import TranscriptionSegment


@dataclass
class Conversation:
    """Represents a distinct conversation within a recording."""
    id: str
    segments: List[TranscriptionSegment]
    start_time: float
    end_time: float
    duration: float
    speakers: List[str]
    speaker_transcript: str  # Formatted transcript with speaker labels
    topic_summary: Optional[str] = None
    boundary_confidence: float = 1.0
    
    def __post_init__(self):
        """Validate conversation data."""
        if not self.segments:
            raise ValueError("Conversation must have at least one segment")
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")


@dataclass
class ConversationBoundary:
    """Represents a detected boundary between conversations."""
    index: int  # Segment index where boundary occurs
    timestamp: float  # Time when boundary occurs
    reason: str  # "time_gap", "topic_change", "speaker_pattern", "manual"
    confidence: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate boundary data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.reason not in ["time_gap", "topic_change", "speaker_pattern", "manual", "final_boundary"]:
            raise ValueError(f"Invalid boundary reason: {self.reason}")


@dataclass
class TopicBoundary:
    """Represents a detected topic change boundary."""
    segment_index: int
    similarity_score: float  # Similarity between before/after segments
    keywords_before: List[str]
    keywords_after: List[str]
    confidence: float = 0.0  # Calculated as 1.0 - similarity_score
    
    def __post_init__(self):
        """Calculate confidence from similarity score."""
        self.confidence = max(0.0, 1.0 - self.similarity_score)


@dataclass
class TimeGap:
    """Represents a time gap between segments."""
    start: float
    end: float
    duration: float
    segment_before_index: int
    segment_after_index: int
    
    def __post_init__(self):
        """Validate time gap data."""
        if self.start >= self.end:
            raise ValueError("Start time must be before end time")
        if self.duration != (self.end - self.start):
            self.duration = self.end - self.start


@dataclass
class SpeakerAnalysis:
    """Analysis of speaker patterns in a conversation."""
    speaker_turns: Dict[str, int]  # Number of speaking turns per speaker
    interaction_matrix: Optional[np.ndarray] = None  # Speaker interaction patterns
    dominant_speakers: List[str] = None  # Speakers ordered by participation
    speaker_changes: List[int] = None  # Segment indices where speaker changes
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.dominant_speakers is None:
            # Sort speakers by number of turns
            self.dominant_speakers = sorted(
                self.speaker_turns.keys(), 
                key=lambda s: self.speaker_turns[s], 
                reverse=True
            )


@dataclass
class SegmentationResult:
    """Result of conversation segmentation."""
    conversations: List[Conversation]
    boundaries: List[ConversationBoundary]
    processing_time: float
    method: str  # "full", "bypass", "fallback"
    confidence: float  # Overall confidence in segmentation
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {
                "total_conversations": len(self.conversations),
                "total_boundaries": len(self.boundaries),
                "average_conversation_duration": sum(c.duration for c in self.conversations) / len(self.conversations) if self.conversations else 0,
                "timestamp": datetime.now().isoformat()
            }


@dataclass
class SegmentationConfig:
    """Configuration for conversation segmentation."""
    # Bypass options
    bypass_segmentation: bool = False
    force_single_conversation: bool = False
    
    # Time-based segmentation
    min_conversation_gap: float = 30.0  # seconds
    min_conversation_length: float = 60.0  # seconds
    
    # Topic-based segmentation
    topic_similarity_threshold: float = 0.3
    topic_window_size: int = 10
    enable_topic_detection: bool = True
    
    # Speaker-based segmentation
    enable_speaker_analysis: bool = True
    speaker_change_weight: float = 0.5
    
    # Boundary processing
    combine_threshold: float = 60.0  # seconds - merge boundaries within this window
    confidence_threshold: float = 0.7
    
    # Post-processing
    merge_short_conversations: bool = True
    min_segments_per_conversation: int = 5
    
    # NLP settings
    language_model: str = "es_core_news_lg"
    nlp_method: str = "tfidf"  # "tfidf" or "embeddings"
    
    @classmethod
    def create_bypass_config(cls) -> 'SegmentationConfig':
        """Create configuration that bypasses segmentation."""
        return cls(
            bypass_segmentation=True,
            force_single_conversation=True,
            enable_topic_detection=False,
            enable_speaker_analysis=False,
            merge_short_conversations=False
        )
    
    @classmethod
    def create_minimal_config(cls) -> 'SegmentationConfig':
        """Create configuration with minimal segmentation (time gaps only)."""
        return cls(
            bypass_segmentation=False,
            enable_topic_detection=False,
            enable_speaker_analysis=False,
            min_conversation_gap=60.0,  # Longer gaps only
            confidence_threshold=0.8    # Higher confidence required
        )