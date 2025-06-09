"""
Conversation segmentation module for TranscriMatic.
Handles segmenting continuous transcriptions into distinct conversations.
"""

from .conversation_segmenter import ConversationSegmenter, ConversationSegmenterError, NLPInitializationError
from .data_classes import (
    Conversation, ConversationBoundary, TopicBoundary, TimeGap, 
    SpeakerAnalysis, SegmentationResult, SegmentationConfig
)

__all__ = [
    'ConversationSegmenter',
    'ConversationSegmenterError', 
    'NLPInitializationError',
    'Conversation',
    'ConversationBoundary',
    'TopicBoundary',
    'TimeGap',
    'SpeakerAnalysis',
    'SegmentationResult',
    'SegmentationConfig'
]