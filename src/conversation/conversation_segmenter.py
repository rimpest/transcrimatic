"""
Conversation Segmenter Module for TranscriMatic

Analyzes transcribed text with speaker labels to identify and separate 
distinct conversations within a continuous recording.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

# Core dependencies
from src.config import ConfigManager
from src.transcription.data_classes import Transcription, TranscriptionSegment

# Local imports
from .data_classes import (
    Conversation, ConversationBoundary, TopicBoundary, TimeGap, 
    SpeakerAnalysis, SegmentationResult, SegmentationConfig
)


logger = logging.getLogger(__name__)


class ConversationSegmenterError(Exception):
    """Base exception for conversation segmentation errors."""
    pass


class NLPInitializationError(ConversationSegmenterError):
    """Raised when NLP models fail to initialize."""
    pass


class ConversationSegmenter:
    """
    Segments continuous audio transcriptions into distinct conversations
    using speaker changes, topic analysis, and time gaps.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the conversation segmenter.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Load segmentation configuration
        self.segmentation_config = self._load_segmentation_config()
        
        # Initialize NLP components (lazy loading)
        self.nlp = None
        self.vectorizer = None
        self.stop_words = None
        self._nlp_initialized = False
        
        # Statistics
        self.stats = {
            "total_segmentations": 0,
            "bypassed_segmentations": 0,
            "failed_segmentations": 0,
            "average_processing_time": 0.0
        }
        
        logger.info("ConversationSegmenter initialized")
    
    def segment_conversations(self, transcription: Transcription, 
                            bypass: bool = None) -> SegmentationResult:
        """
        Segment transcription into distinct conversations.
        
        Args:
            transcription: Input transcription to segment
            bypass: Override config to bypass segmentation (optional)
            
        Returns:
            SegmentationResult with conversations and metadata
        """
        start_time = time.time()
        self.stats["total_segmentations"] += 1
        
        try:
            # Check bypass conditions
            if self._should_bypass_segmentation(bypass):
                result = self._create_bypass_result(transcription)
                self.stats["bypassed_segmentations"] += 1
                logger.info("Segmentation bypassed - treating as single conversation")
                return result
            
            # Validate input
            if not transcription.segments:
                return self._create_single_conversation_result(transcription, "empty_input")
            
            # Initialize NLP if needed and enabled
            if self.segmentation_config.enable_topic_detection and not self._nlp_initialized:
                self._initialize_nlp()
            
            # Perform segmentation
            conversations = self._perform_segmentation(transcription)
            
            # Create result
            processing_time = time.time() - start_time
            result = SegmentationResult(
                conversations=conversations,
                boundaries=self._extract_boundaries_from_conversations(conversations),
                processing_time=processing_time,
                method="full",
                confidence=self._calculate_overall_confidence(conversations)
            )
            
            # Update stats
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["total_segmentations"] - 1) + 
                 processing_time) / self.stats["total_segmentations"]
            )
            
            logger.info(f"Segmented into {len(conversations)} conversations in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats["failed_segmentations"] += 1
            logger.error(f"Segmentation failed: {e}")
            
            # Fallback to single conversation
            processing_time = time.time() - start_time
            return SegmentationResult(
                conversations=[self._create_single_conversation(transcription)],
                boundaries=[],
                processing_time=processing_time,
                method="fallback",
                confidence=0.5,
                metadata={"error": str(e)}
            )
    
    def _should_bypass_segmentation(self, bypass_override: Optional[bool]) -> bool:
        """Determine if segmentation should be bypassed."""
        if bypass_override is not None:
            return bypass_override
        
        return (self.segmentation_config.bypass_segmentation or 
                self.segmentation_config.force_single_conversation)
    
    def _perform_segmentation(self, transcription: Transcription) -> List[Conversation]:
        """Perform the actual conversation segmentation."""
        segments = transcription.segments
        
        # 1. Find different types of boundaries
        boundaries = []
        
        # Time-based boundaries (always enabled)
        time_boundaries = self._find_time_boundaries(segments)
        boundaries.extend(time_boundaries)
        logger.debug(f"Found {len(time_boundaries)} time-based boundaries")
        
        # Topic-based boundaries (optional)
        if self.segmentation_config.enable_topic_detection:
            topic_boundaries = self._find_topic_boundaries(segments)
            boundaries.extend(topic_boundaries)
            logger.debug(f"Found {len(topic_boundaries)} topic-based boundaries")
        
        # Speaker-based boundaries (optional)
        if self.segmentation_config.enable_speaker_analysis:
            speaker_boundaries = self._find_speaker_boundaries(segments)
            boundaries.extend(speaker_boundaries)
            logger.debug(f"Found {len(speaker_boundaries)} speaker-based boundaries")
        
        # 2. Combine and filter boundaries
        final_boundaries = self._process_boundaries(boundaries)
        logger.debug(f"Final boundaries after processing: {len(final_boundaries)}")
        
        # 3. Create conversations from boundaries
        conversations = self._create_conversations(segments, final_boundaries)
        
        # 4. Post-process conversations
        if self.segmentation_config.merge_short_conversations:
            conversations = self.merge_short_conversations(conversations)
        
        # 5. Add speaker transcripts and metadata
        for conv in conversations:
            conv.speaker_transcript = self._format_conversation_transcript(conv.segments)
            if self.segmentation_config.enable_topic_detection and self._nlp_initialized:
                conv.topic_summary = self._generate_topic_summary(conv.segments)
        
        return conversations
    
    def _find_time_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
        """Find conversation boundaries based on time gaps."""
        boundaries = []
        min_gap = self.segmentation_config.min_conversation_gap
        
        for i in range(1, len(segments)):
            gap = segments[i].start_time - segments[i-1].end_time
            
            if gap >= min_gap:
                # Scale confidence: min_gap = 0.5, 2*min_gap+ = 1.0
                confidence = min(1.0, 0.5 + (gap - min_gap) / min_gap)
                
                boundaries.append(ConversationBoundary(
                    index=i,
                    timestamp=segments[i-1].end_time,
                    reason="time_gap",
                    confidence=confidence
                ))
        
        return boundaries
    
    def _find_topic_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
        """Find conversation boundaries based on topic changes."""
        if not self._nlp_initialized:
            return []
        
        boundaries = []
        window_size = self.segmentation_config.topic_window_size
        threshold = self.segmentation_config.topic_similarity_threshold
        
        for i in range(window_size, len(segments) - window_size):
            # Extract text windows
            before_text = " ".join(s.text for s in segments[i-window_size:i])
            after_text = " ".join(s.text for s in segments[i:i+window_size])
            
            # Calculate similarity
            similarity = self._calculate_text_similarity(before_text, after_text)
            
            if similarity < threshold:
                boundaries.append(ConversationBoundary(
                    index=i,
                    timestamp=segments[i].start_time,
                    reason="topic_change",
                    confidence=1.0 - similarity
                ))
        
        return boundaries
    
    def _find_speaker_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
        """Find conversation boundaries based on speaker patterns."""
        boundaries = []
        
        # Simple speaker change detection
        for i in range(1, len(segments)):
            if segments[i].speaker_id != segments[i-1].speaker_id:
                # Look ahead to see if this is a sustained speaker change
                sustained_change = self._is_sustained_speaker_change(segments, i)
                
                if sustained_change:
                    confidence = self.segmentation_config.speaker_change_weight
                    boundaries.append(ConversationBoundary(
                        index=i,
                        timestamp=segments[i].start_time,
                        reason="speaker_pattern",
                        confidence=confidence
                    ))
        
        return boundaries
    
    def _is_sustained_speaker_change(self, segments: List[TranscriptionSegment], 
                                   change_index: int, lookahead: int = 5) -> bool:
        """Check if a speaker change is sustained (not just a brief interruption)."""
        if change_index + lookahead >= len(segments):
            return True  # End of recording
        
        new_speaker = segments[change_index].speaker_id
        
        # Count how many of the next segments have the new speaker
        same_speaker_count = 0
        for i in range(change_index, min(change_index + lookahead, len(segments))):
            if segments[i].speaker_id == new_speaker:
                same_speaker_count += 1
        
        # Consider it sustained if majority of lookahead segments have new speaker
        return same_speaker_count >= lookahead // 2
    
    def _process_boundaries(self, boundaries: List[ConversationBoundary]) -> List[ConversationBoundary]:
        """Process and filter boundaries based on confidence and proximity."""
        if not boundaries:
            return []
        
        # Sort by timestamp
        boundaries.sort(key=lambda b: b.timestamp)
        
        # Filter by confidence
        filtered = [b for b in boundaries if b.confidence >= self.segmentation_config.confidence_threshold]
        
        # Merge nearby boundaries
        merged = self._merge_nearby_boundaries(filtered)
        
        return merged
    
    def _merge_nearby_boundaries(self, boundaries: List[ConversationBoundary]) -> List[ConversationBoundary]:
        """Merge boundaries that are close together in time."""
        if len(boundaries) <= 1:
            return boundaries
        
        merged = []
        current = boundaries[0]
        combine_threshold = self.segmentation_config.combine_threshold
        
        for next_boundary in boundaries[1:]:
            time_diff = next_boundary.timestamp - current.timestamp
            
            if time_diff <= combine_threshold:
                # Merge boundaries - keep the one with higher confidence
                if next_boundary.confidence > current.confidence:
                    current = next_boundary
                # If same confidence, prefer time_gap over others
                elif (next_boundary.confidence == current.confidence and 
                      next_boundary.reason == "time_gap"):
                    current = next_boundary
            else:
                merged.append(current)
                current = next_boundary
        
        merged.append(current)
        return merged
    
    def _create_conversations(self, segments: List[TranscriptionSegment], 
                            boundaries: List[ConversationBoundary]) -> List[Conversation]:
        """Create conversation objects from segments and boundaries."""
        conversations = []
        
        # If no boundaries, create single conversation
        if not boundaries:
            return [self._create_single_conversation_from_segments(segments)]
        
        # Create conversations between boundaries
        start_idx = 0
        
        for boundary in boundaries:
            if boundary.index > start_idx:
                conv_segments = segments[start_idx:boundary.index]
                if conv_segments:
                    conversations.append(self._create_conversation_from_segments(conv_segments))
            start_idx = boundary.index
        
        # Create final conversation from last boundary to end
        if start_idx < len(segments):
            conv_segments = segments[start_idx:]
            if conv_segments:
                conversations.append(self._create_conversation_from_segments(conv_segments))
        
        return conversations
    
    def _create_conversation_from_segments(self, segments: List[TranscriptionSegment]) -> Conversation:
        """Create a conversation object from a list of segments."""
        if not segments:
            raise ValueError("Cannot create conversation from empty segments")
        
        # Extract unique speakers
        speakers = list(set(segment.speaker_id for segment in segments if segment.speaker_id))
        
        # Calculate times
        start_time = segments[0].start_time
        end_time = segments[-1].end_time
        duration = end_time - start_time
        
        # Generate conversation ID
        conv_id = str(uuid.uuid4())[:8]
        
        return Conversation(
            id=conv_id,
            segments=segments,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            speakers=speakers,
            speaker_transcript="",  # Will be filled later
            boundary_confidence=1.0  # Default confidence
        )
    
    def _create_single_conversation_from_segments(self, segments: List[TranscriptionSegment]) -> Conversation:
        """Create a single conversation containing all segments."""
        return self._create_conversation_from_segments(segments)
    
    def _create_single_conversation(self, transcription: Transcription) -> Conversation:
        """Create a single conversation from entire transcription."""
        return self._create_single_conversation_from_segments(transcription.segments)
    
    def merge_short_conversations(self, conversations: List[Conversation]) -> List[Conversation]:
        """Merge conversations that are too short."""
        if len(conversations) <= 1:
            return conversations
        
        min_length = self.segmentation_config.min_conversation_length
        min_segments = self.segmentation_config.min_segments_per_conversation
        merged = []
        current = conversations[0]
        
        for next_conv in conversations[1:]:
            # Check if current conversation is too short
            if (current.duration < min_length or 
                len(current.segments) < min_segments):
                # Merge with next conversation
                current = self._merge_conversations(current, next_conv)
            else:
                merged.append(current)
                current = next_conv
        
        merged.append(current)
        return merged
    
    def _merge_conversations(self, conv1: Conversation, conv2: Conversation) -> Conversation:
        """Merge two conversations into one."""
        # Combine segments
        combined_segments = conv1.segments + conv2.segments
        
        # Combine speakers
        combined_speakers = list(set(conv1.speakers + conv2.speakers))
        
        # Calculate new times
        start_time = min(conv1.start_time, conv2.start_time)
        end_time = max(conv1.end_time, conv2.end_time)
        duration = end_time - start_time
        
        # Use first conversation's ID
        return Conversation(
            id=conv1.id,
            segments=combined_segments,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            speakers=combined_speakers,
            speaker_transcript="",  # Will be regenerated
            boundary_confidence=min(conv1.boundary_confidence, conv2.boundary_confidence)
        )
    
    def _format_conversation_transcript(self, segments: List[TranscriptionSegment]) -> str:
        """Format segments into a readable transcript with speaker labels."""
        lines = []
        current_speaker = None
        
        for segment in segments:
            speaker = segment.speaker_id or "Speaker"
            
            if speaker != current_speaker:
                if lines:  # Add blank line between speakers
                    lines.append("")
                lines.append(f"{speaker}:")
                current_speaker = speaker
            
            # Add timestamp and text
            timestamp = f"[{segment.start_time:.1f}s]"
            lines.append(f"  {timestamp} {segment.text.strip()}")
        
        return "\\n".join(lines)
    
    def _generate_topic_summary(self, segments: List[TranscriptionSegment]) -> Optional[str]:
        """Generate a topic summary for conversation segments."""
        if not self._nlp_initialized or not segments:
            return None
        
        try:
            # Combine all text
            full_text = " ".join(segment.text for segment in segments)
            
            # Extract keywords
            keywords = self._extract_keywords(full_text)
            
            # Return top keywords as summary
            return ", ".join(keywords[:5]) if keywords else None
            
        except Exception as e:
            logger.warning(f"Failed to generate topic summary: {e}")
            return None
    
    def _initialize_nlp(self):
        """Initialize NLP components for topic detection."""
        try:
            import spacy
            import nltk
            from nltk.corpus import stopwords
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Download required NLTK data
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Load Spanish model
            try:
                self.nlp = spacy.load(self.segmentation_config.language_model)
            except IOError:
                logger.warning(f"Spanish model {self.segmentation_config.language_model} not found, using basic tokenizer")
                self.nlp = None
            
            # Load Spanish stop words
            try:
                self.stop_words = set(stopwords.words('spanish'))
            except LookupError:
                logger.warning("Spanish stopwords not found, using empty set")
                self.stop_words = set()
            
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                stop_words=list(self.stop_words) if self.stop_words else None,
                max_features=100,
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            self._nlp_initialized = True
            logger.info("NLP components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"NLP dependencies not available: {e}")
            self._nlp_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize NLP: {e}")
            self._nlp_initialized = False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments."""
        if not self._nlp_initialized:
            return 0.5  # Neutral similarity if NLP not available
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Vectorize texts
            vectors = self.vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Text similarity calculation failed: {e}")
            return 0.5  # Neutral similarity on error
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from Spanish text."""
        if not self._nlp_initialized or not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            keywords = []
            
            for token in doc:
                # Include nouns, verbs, and proper nouns
                if (not token.is_stop and 
                    token.pos_ in ["NOUN", "VERB", "PROPN"] and
                    len(token.text) > 2 and
                    token.is_alpha):
                    keywords.append(token.lemma_.lower())
            
            # Also extract named entities
            for ent in doc.ents:
                if len(ent.text) > 2:
                    keywords.append(ent.text.lower())
            
            return list(set(keywords))
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _load_segmentation_config(self) -> SegmentationConfig:
        """Load segmentation configuration from config manager."""
        # Check for bypass configuration first
        if self.config.get("segmentation.bypass_segmentation", False):
            return SegmentationConfig.create_bypass_config()
        
        return SegmentationConfig(
            bypass_segmentation=self.config.get("segmentation.bypass_segmentation", False),
            force_single_conversation=self.config.get("segmentation.force_single_conversation", False),
            min_conversation_gap=self.config.get("segmentation.min_conversation_gap", 30.0),
            min_conversation_length=self.config.get("segmentation.min_conversation_length", 60.0),
            topic_similarity_threshold=self.config.get("segmentation.topic_similarity_threshold", 0.3),
            topic_window_size=self.config.get("segmentation.topic_detection.window_size", 10),
            enable_topic_detection=self.config.get("segmentation.enable_topic_detection", True),
            enable_speaker_analysis=self.config.get("segmentation.enable_speaker_analysis", True),
            speaker_change_weight=self.config.get("segmentation.speaker_change_weight", 0.5),
            combine_threshold=self.config.get("segmentation.boundaries.combine_threshold", 60.0),
            confidence_threshold=self.config.get("segmentation.boundaries.confidence_threshold", 0.7),
            merge_short_conversations=self.config.get("segmentation.post_processing.merge_short", True),
            min_segments_per_conversation=self.config.get("segmentation.post_processing.min_segments", 5),
            language_model=self.config.get("segmentation.topic_detection.language_model", "es_core_news_lg"),
            nlp_method=self.config.get("segmentation.topic_detection.method", "tfidf")
        )
    
    def _create_bypass_result(self, transcription: Transcription) -> SegmentationResult:
        """Create a result that bypasses segmentation."""
        conversation = self._create_single_conversation(transcription)
        conversation.speaker_transcript = self._format_conversation_transcript(conversation.segments)
        
        return SegmentationResult(
            conversations=[conversation],
            boundaries=[],
            processing_time=0.0,
            method="bypass",
            confidence=1.0,
            metadata={"bypassed": True}
        )
    
    def _create_single_conversation_result(self, transcription: Transcription, 
                                         reason: str) -> SegmentationResult:
        """Create a single conversation result."""
        if transcription.segments:
            conversation = self._create_single_conversation(transcription)
            conversation.speaker_transcript = self._format_conversation_transcript(conversation.segments)
            conversations = [conversation]
        else:
            conversations = []
        
        return SegmentationResult(
            conversations=conversations,
            boundaries=[],
            processing_time=0.0,
            method="single",
            confidence=0.8,
            metadata={"reason": reason}
        )
    
    def _extract_boundaries_from_conversations(self, conversations: List[Conversation]) -> List[ConversationBoundary]:
        """Extract boundaries from final conversations for result metadata."""
        boundaries = []
        
        for i, conv in enumerate(conversations[1:], 1):
            boundaries.append(ConversationBoundary(
                index=0,  # Simplified for result
                timestamp=conv.start_time,
                reason="final_boundary",
                confidence=conv.boundary_confidence
            ))
        
        return boundaries
    
    def _calculate_overall_confidence(self, conversations: List[Conversation]) -> float:
        """Calculate overall confidence in the segmentation."""
        if not conversations:
            return 0.0
        
        if len(conversations) == 1:
            return 0.8  # Single conversation is moderately confident
        
        # Average boundary confidence
        total_confidence = sum(conv.boundary_confidence for conv in conversations)
        return total_confidence / len(conversations)
    
    def get_statistics(self) -> Dict:
        """Get segmentation statistics."""
        return self.stats.copy()
    
    def configure_bypass(self, bypass: bool = True):
        """Configure segmentation bypass at runtime."""
        self.segmentation_config.bypass_segmentation = bypass
        self.segmentation_config.force_single_conversation = bypass
        logger.info(f"Segmentation bypass {'enabled' if bypass else 'disabled'}")
    
    def configure_minimal_segmentation(self, enabled: bool = True):
        """Configure minimal segmentation (time gaps only)."""
        if enabled:
            self.segmentation_config.enable_topic_detection = False
            self.segmentation_config.enable_speaker_analysis = False
            self.segmentation_config.min_conversation_gap = 60.0  # Longer gaps
            self.segmentation_config.confidence_threshold = 0.8
        else:
            # Restore full segmentation
            self.segmentation_config.enable_topic_detection = True
            self.segmentation_config.enable_speaker_analysis = True
            self.segmentation_config.min_conversation_gap = 30.0
            self.segmentation_config.confidence_threshold = 0.7
        
        logger.info(f"Minimal segmentation {'enabled' if enabled else 'disabled'}")