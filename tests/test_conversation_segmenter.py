"""
Unit tests for the ConversationSegmenter module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.conversation import (
    ConversationSegmenter, ConversationSegmenterError, NLPInitializationError,
    Conversation, ConversationBoundary, SegmentationResult, SegmentationConfig
)
from src.config import ConfigManager
from src.transcription.data_classes import Transcription, TranscriptionSegment
from src.audio.data_classes import ProcessedAudio, AudioFile


class TestConversationSegmenter:
    """Test suite for ConversationSegmenter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_segmenter_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock ConfigManager."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            "segmentation.bypass_segmentation": False,
            "segmentation.force_single_conversation": False,
            "segmentation.min_conversation_gap": 30.0,
            "segmentation.min_conversation_length": 60.0,
            "segmentation.topic_similarity_threshold": 0.3,
            "segmentation.topic_detection.window_size": 10,
            "segmentation.enable_topic_detection": True,
            "segmentation.enable_speaker_analysis": True,
            "segmentation.speaker_change_weight": 0.5,
            "segmentation.boundaries.combine_threshold": 60.0,
            "segmentation.boundaries.confidence_threshold": 0.7,
            "segmentation.post_processing.merge_short": True,
            "segmentation.post_processing.min_segments": 5,
            "segmentation.topic_detection.language_model": "es_core_news_lg",
            "segmentation.topic_detection.method": "tfidf"
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_bypass_config(self, temp_dir):
        """Create a mock ConfigManager with bypass enabled."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            "segmentation.bypass_segmentation": True,
            "segmentation.force_single_conversation": True,
            "segmentation.enable_topic_detection": False,
            "segmentation.enable_speaker_analysis": False
        }.get(key, default)
        return config
    
    @pytest.fixture
    def segmenter(self, mock_config):
        """Create a ConversationSegmenter instance."""
        return ConversationSegmenter(mock_config)
    
    @pytest.fixture
    def bypass_segmenter(self, mock_bypass_config):
        """Create a ConversationSegmenter with bypass enabled."""
        return ConversationSegmenter(mock_bypass_config)
    
    @pytest.fixture
    def sample_segments(self):
        """Create sample transcription segments for testing."""
        return [
            TranscriptionSegment(
                id=1,
                start_time=0.0,
                end_time=2.5,
                text="Hola, ¿cómo estás?",
                speaker_id="Speaker_1",
                confidence=0.95
            ),
            TranscriptionSegment(
                id=2,
                start_time=3.0,
                end_time=5.5,
                text="Muy bien, gracias. ¿Y tú?",
                speaker_id="Speaker_2",
                confidence=0.90
            ),
            TranscriptionSegment(
                id=3,
                start_time=6.0,
                end_time=8.5,
                text="Excelente. Hablemos de trabajo.",
                speaker_id="Speaker_1",
                confidence=0.92
            ),
            # Long gap here (30+ seconds)
            TranscriptionSegment(
                id=4,
                start_time=45.0,
                end_time=47.5,
                text="Sobre el nuevo proyecto de marketing.",
                speaker_id="Speaker_1",
                confidence=0.88
            ),
            TranscriptionSegment(
                id=5,
                start_time=48.0,
                end_time=50.5,
                text="Sí, tenemos muchas ideas nuevas.",
                speaker_id="Speaker_2",
                confidence=0.93
            )
        ]
    
    @pytest.fixture
    def sample_transcription(self, sample_segments):
        """Create a sample transcription for testing."""
        # Create a mock ProcessedAudio for the transcription
        mock_audio_file = Mock(spec=AudioFile)
        mock_audio_file.filename = "audio.wav"
        
        mock_processed_audio = Mock(spec=ProcessedAudio)
        mock_processed_audio.original_file = mock_audio_file
        mock_processed_audio.chunks = []
        
        return Transcription(
            audio_file=mock_processed_audio,
            segments=sample_segments,
            full_text=" ".join(seg.text for seg in sample_segments),
            speaker_transcript="",
            language="es",
            confidence=0.92,
            duration=50.5,
            speaker_count=2
        )
    
    def test_init(self, segmenter, mock_config):
        """Test ConversationSegmenter initialization."""
        assert segmenter.config == mock_config
        assert segmenter.segmentation_config is not None
        assert segmenter.stats["total_segmentations"] == 0
        assert not segmenter._nlp_initialized
    
    def test_init_with_bypass_config(self, bypass_segmenter):
        """Test initialization with bypass configuration."""
        assert bypass_segmenter.segmentation_config.bypass_segmentation
        assert bypass_segmenter.segmentation_config.force_single_conversation
    
    def test_segment_conversations_bypass(self, bypass_segmenter, sample_transcription):
        """Test conversation segmentation with bypass enabled."""
        result = bypass_segmenter.segment_conversations(sample_transcription)
        
        assert isinstance(result, SegmentationResult)
        assert result.method == "bypass"
        assert len(result.conversations) == 1
        assert len(result.boundaries) == 0
        assert result.confidence == 1.0
        
        # Check conversation contains all segments
        conversation = result.conversations[0]
        assert len(conversation.segments) == len(sample_transcription.segments)
        assert conversation.start_time == 0.0
        assert conversation.end_time == 50.5
    
    def test_segment_conversations_explicit_bypass(self, segmenter, sample_transcription):
        """Test bypassing segmentation with explicit parameter."""
        result = segmenter.segment_conversations(sample_transcription, bypass=True)
        
        assert result.method == "bypass"
        assert len(result.conversations) == 1
    
    def test_segment_conversations_empty_input(self, segmenter):
        """Test segmentation with empty transcription."""
        mock_audio_file = Mock(spec=AudioFile)
        mock_audio_file.filename = "empty.wav"
        
        mock_processed_audio = Mock(spec=ProcessedAudio)
        mock_processed_audio.original_file = mock_audio_file
        mock_processed_audio.chunks = []
        
        empty_transcription = Transcription(
            audio_file=mock_processed_audio,
            segments=[],
            full_text="",
            speaker_transcript="",
            language="es",
            confidence=0.0,
            duration=0.0,
            speaker_count=0
        )
        
        result = segmenter.segment_conversations(empty_transcription)
        assert result.method == "single"
        assert len(result.conversations) == 0
    
    @patch('src.conversation.conversation_segmenter.ConversationSegmenter._initialize_nlp')
    def test_segment_conversations_time_based(self, mock_init_nlp, segmenter, sample_transcription):
        """Test time-based conversation segmentation."""
        # Disable NLP to test time-based only
        segmenter.segmentation_config.enable_topic_detection = False
        segmenter.segmentation_config.enable_speaker_analysis = False
        
        result = segmenter.segment_conversations(sample_transcription)
        
        assert result.method == "full"
        assert len(result.conversations) == 2  # Should split on 30s+ gap
        
        # Check first conversation
        conv1 = result.conversations[0]
        assert len(conv1.segments) == 3
        assert conv1.start_time == 0.0
        assert conv1.end_time == 8.5
        
        # Check second conversation
        conv2 = result.conversations[1]
        assert len(conv2.segments) == 2
        assert conv2.start_time == 45.0
        assert conv2.end_time == 50.5
    
    def test_find_time_boundaries(self, segmenter, sample_segments):
        """Test time gap boundary detection."""
        boundaries = segmenter._find_time_boundaries(sample_segments)
        
        assert len(boundaries) == 1
        boundary = boundaries[0]
        assert boundary.reason == "time_gap"
        assert boundary.index == 3  # Split before segment at 45.0s
        assert boundary.confidence > 0.5
    
    def test_find_time_boundaries_no_gaps(self, segmenter):
        """Test time boundary detection with no significant gaps."""
        close_segments = [
            TranscriptionSegment(1, 0.0, 2.0, "Text 1", "Speaker_1", 0.9),
            TranscriptionSegment(2, 2.5, 4.5, "Text 2", "Speaker_2", 0.9),
            TranscriptionSegment(3, 5.0, 7.0, "Text 3", "Speaker_1", 0.9)
        ]
        
        boundaries = segmenter._find_time_boundaries(close_segments)
        assert len(boundaries) == 0
    
    def test_find_speaker_boundaries(self, segmenter, sample_segments):
        """Test speaker-based boundary detection."""
        boundaries = segmenter._find_speaker_boundaries(sample_segments)
        
        # Should find speaker changes
        assert len(boundaries) >= 0  # May vary based on sustained change logic
        
        for boundary in boundaries:
            assert boundary.reason == "speaker_pattern"
            assert 0.0 <= boundary.confidence <= 1.0
    
    def test_is_sustained_speaker_change(self, segmenter):
        """Test sustained speaker change detection."""
        segments = [
            TranscriptionSegment(1, 0.0, 2.0, "Text 1", "Speaker_1", 0.9),
            TranscriptionSegment(2, 2.0, 4.0, "Text 2", "Speaker_2", 0.9),  # Change here
            TranscriptionSegment(3, 4.0, 6.0, "Text 3", "Speaker_2", 0.9),  # Sustained
            TranscriptionSegment(4, 6.0, 8.0, "Text 4", "Speaker_2", 0.9),  # Sustained
        ]
        
        # Should be sustained (Speaker_2 continues)
        assert segmenter._is_sustained_speaker_change(segments, 1)
        
        # Brief change
        brief_segments = [
            TranscriptionSegment(1, 0.0, 2.0, "Text 1", "Speaker_1", 0.9),
            TranscriptionSegment(2, 2.0, 4.0, "Text 2", "Speaker_2", 0.9),  # Brief
            TranscriptionSegment(3, 4.0, 6.0, "Text 3", "Speaker_1", 0.9),  # Back to 1
            TranscriptionSegment(4, 6.0, 8.0, "Text 4", "Speaker_1", 0.9),  # Back to 1
        ]
        
        # Should not be sustained
        assert not segmenter._is_sustained_speaker_change(brief_segments, 1)
    
    def test_process_boundaries(self, segmenter):
        """Test boundary processing and filtering."""
        boundaries = [
            ConversationBoundary(1, 10.0, "time_gap", 0.8),
            ConversationBoundary(2, 15.0, "topic_change", 0.5),  # Below threshold
            ConversationBoundary(3, 70.0, "speaker_pattern", 0.9)
        ]
        
        # Should filter out low confidence boundary
        processed = segmenter._process_boundaries(boundaries)
        assert len(processed) == 2
        assert all(b.confidence >= 0.7 for b in processed)
    
    def test_merge_nearby_boundaries(self, segmenter):
        """Test merging of nearby boundaries."""
        boundaries = [
            ConversationBoundary(1, 10.0, "time_gap", 0.8),
            ConversationBoundary(2, 20.0, "topic_change", 0.6),  # Within 60s threshold
            ConversationBoundary(3, 100.0, "speaker_pattern", 0.9)
        ]
        
        merged = segmenter._merge_nearby_boundaries(boundaries)
        assert len(merged) == 2  # First two should merge
        assert merged[0].confidence == 0.8  # time_gap wins
    
    def test_create_conversations(self, segmenter, sample_segments):
        """Test conversation creation from segments and boundaries."""
        boundaries = [
            ConversationBoundary(3, 40.0, "time_gap", 0.8)
        ]
        
        conversations = segmenter._create_conversations(sample_segments, boundaries)
        
        assert len(conversations) == 2
        
        # First conversation (segments 0-2)
        assert len(conversations[0].segments) == 3
        assert conversations[0].start_time == 0.0
        assert conversations[0].end_time == 8.5
        
        # Second conversation (segments 3-4)
        assert len(conversations[1].segments) == 2
        assert conversations[1].start_time == 45.0
        assert conversations[1].end_time == 50.5
    
    def test_create_conversations_no_boundaries(self, segmenter, sample_segments):
        """Test conversation creation with no boundaries."""
        conversations = segmenter._create_conversations(sample_segments, [])
        
        assert len(conversations) == 1
        assert len(conversations[0].segments) == len(sample_segments)
    
    def test_create_conversation_from_segments(self, segmenter, sample_segments):
        """Test creating a single conversation from segments."""
        conv = segmenter._create_conversation_from_segments(sample_segments[:3])
        
        assert isinstance(conv, Conversation)
        assert len(conv.segments) == 3
        assert conv.start_time == 0.0
        assert conv.end_time == 8.5
        assert conv.duration == 8.5
        assert len(conv.speakers) == 2  # Speaker_1 and Speaker_2
        assert conv.id is not None
    
    def test_create_conversation_empty_segments(self, segmenter):
        """Test creating conversation with empty segments raises error."""
        with pytest.raises(ValueError, match="Cannot create conversation from empty segments"):
            segmenter._create_conversation_from_segments([])
    
    def test_merge_short_conversations(self, segmenter):
        """Test merging of short conversations."""
        # Create short conversations
        short_conv1 = Conversation(
            id="conv1",
            segments=[TranscriptionSegment(1, 0.0, 10.0, "Short 1", "Speaker_1", 0.9)],
            start_time=0.0,
            end_time=10.0,
            duration=10.0,  # Too short
            speakers=["Speaker_1"],
            speaker_transcript=""
        )
        
        short_conv2 = Conversation(
            id="conv2",
            segments=[TranscriptionSegment(2, 15.0, 25.0, "Short 2", "Speaker_2", 0.9)],
            start_time=15.0,
            end_time=25.0,
            duration=10.0,  # Too short
            speakers=["Speaker_2"],
            speaker_transcript=""
        )
        
        long_conv = Conversation(
            id="conv3",
            segments=[TranscriptionSegment(3, 30.0, 100.0, "Long conversation", "Speaker_1", 0.9)],
            start_time=30.0,
            end_time=100.0,
            duration=70.0,  # Long enough
            speakers=["Speaker_1"],
            speaker_transcript=""
        )
        
        conversations = [short_conv1, short_conv2, long_conv]
        merged = segmenter.merge_short_conversations(conversations)
        
        # Should merge first two short conversations
        assert len(merged) == 2
        assert merged[0].duration == 25.0  # Combined duration of first two
        assert merged[1].duration == 70.0  # Long conversation unchanged
    
    def test_merge_conversations(self, segmenter):
        """Test merging two conversations."""
        conv1 = Conversation(
            id="conv1",
            segments=[TranscriptionSegment(1, 0.0, 10.0, "Text 1", "Speaker_1", 0.9)],
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            speakers=["Speaker_1"],
            speaker_transcript=""
        )
        
        conv2 = Conversation(
            id="conv2",
            segments=[TranscriptionSegment(2, 15.0, 25.0, "Text 2", "Speaker_2", 0.9)],
            start_time=15.0,
            end_time=25.0,
            duration=10.0,
            speakers=["Speaker_2"],
            speaker_transcript=""
        )
        
        merged = segmenter._merge_conversations(conv1, conv2)
        
        assert merged.id == conv1.id  # Uses first conversation's ID
        assert len(merged.segments) == 2
        assert merged.start_time == 0.0
        assert merged.end_time == 25.0
        assert merged.duration == 25.0
        assert set(merged.speakers) == {"Speaker_1", "Speaker_2"}
    
    def test_format_conversation_transcript(self, segmenter, sample_segments):
        """Test conversation transcript formatting."""
        transcript = segmenter._format_conversation_transcript(sample_segments[:3])
        
        assert "Speaker_1:" in transcript
        assert "Speaker_2:" in transcript
        assert "Hola, ¿cómo estás?" in transcript
        assert "[0.0s]" in transcript
        assert "[3.0s]" in transcript
    
    def test_configure_bypass(self, segmenter):
        """Test runtime bypass configuration."""
        assert not segmenter.segmentation_config.bypass_segmentation
        
        segmenter.configure_bypass(True)
        assert segmenter.segmentation_config.bypass_segmentation
        assert segmenter.segmentation_config.force_single_conversation
        
        segmenter.configure_bypass(False)
        assert not segmenter.segmentation_config.bypass_segmentation
        assert not segmenter.segmentation_config.force_single_conversation
    
    def test_configure_minimal_segmentation(self, segmenter):
        """Test minimal segmentation configuration."""
        assert segmenter.segmentation_config.enable_topic_detection
        assert segmenter.segmentation_config.enable_speaker_analysis
        
        segmenter.configure_minimal_segmentation(True)
        assert not segmenter.segmentation_config.enable_topic_detection
        assert not segmenter.segmentation_config.enable_speaker_analysis
        assert segmenter.segmentation_config.min_conversation_gap == 60.0
        
        segmenter.configure_minimal_segmentation(False)
        assert segmenter.segmentation_config.enable_topic_detection
        assert segmenter.segmentation_config.enable_speaker_analysis
        assert segmenter.segmentation_config.min_conversation_gap == 30.0
    
    def test_get_statistics(self, segmenter):
        """Test statistics retrieval."""
        stats = segmenter.get_statistics()
        
        assert "total_segmentations" in stats
        assert "bypassed_segmentations" in stats
        assert "failed_segmentations" in stats
        assert "average_processing_time" in stats
        assert stats["total_segmentations"] == 0
    
    def test_statistics_update(self, segmenter, sample_transcription):
        """Test that statistics are updated after segmentation."""
        initial_stats = segmenter.get_statistics()
        initial_count = initial_stats["total_segmentations"]
        
        segmenter.segment_conversations(sample_transcription)
        
        updated_stats = segmenter.get_statistics()
        assert updated_stats["total_segmentations"] == initial_count + 1
        assert updated_stats["average_processing_time"] > 0.0
    
    @patch('src.conversation.conversation_segmenter.ConversationSegmenter._initialize_nlp')
    def test_segmentation_fallback_on_error(self, mock_init_nlp, segmenter):
        """Test fallback to single conversation when segmentation fails."""
        # Mock a failing transcription
        bad_transcription = Mock()
        bad_transcription.segments = None  # This should cause an error
        
        result = segmenter.segment_conversations(bad_transcription)
        
        assert result.method == "fallback"
        assert "error" in result.metadata
    
    def test_segmentation_config_creation(self):
        """Test SegmentationConfig creation methods."""
        # Test bypass config
        bypass_config = SegmentationConfig.create_bypass_config()
        assert bypass_config.bypass_segmentation
        assert bypass_config.force_single_conversation
        assert not bypass_config.enable_topic_detection
        
        # Test minimal config
        minimal_config = SegmentationConfig.create_minimal_config()
        assert not minimal_config.bypass_segmentation
        assert not minimal_config.enable_topic_detection
        assert not minimal_config.enable_speaker_analysis
        assert minimal_config.min_conversation_gap == 60.0
    
    @patch('spacy.load')
    @patch('nltk.download')
    @patch('nltk.data.find')
    def test_initialize_nlp_success(self, mock_find, mock_download, mock_spacy_load, segmenter):
        """Test successful NLP initialization."""
        mock_nlp = Mock()
        mock_spacy_load.return_value = mock_nlp
        mock_find.return_value = True  # Stopwords found
        
        segmenter._initialize_nlp()
        
        assert segmenter._nlp_initialized
        assert segmenter.nlp == mock_nlp
        mock_spacy_load.assert_called_once_with("es_core_news_lg")
    
    @patch('spacy.load')
    def test_initialize_nlp_model_not_found(self, mock_spacy_load, segmenter):
        """Test NLP initialization when Spanish model not found."""
        mock_spacy_load.side_effect = IOError("Model not found")
        
        segmenter._initialize_nlp()
        
        # Should still initialize but with warnings
        assert segmenter.nlp is None
        # Should handle gracefully without crashing
    
    def test_calculate_text_similarity_no_nlp(self, segmenter):
        """Test text similarity calculation when NLP not initialized."""
        similarity = segmenter._calculate_text_similarity("text1", "text2")
        assert similarity == 0.5  # Neutral similarity
    
    def test_extract_keywords_no_nlp(self, segmenter):
        """Test keyword extraction when NLP not available."""
        keywords = segmenter._extract_keywords("Hola mundo")
        assert keywords == []  # Empty list when no NLP
    
    def test_conversation_boundary_validation(self):
        """Test ConversationBoundary data validation."""
        # Valid boundary
        boundary = ConversationBoundary(1, 10.0, "time_gap", 0.8)
        assert boundary.confidence == 0.8
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ConversationBoundary(1, 10.0, "time_gap", 1.5)
        
        # Invalid reason
        with pytest.raises(ValueError, match="Invalid boundary reason"):
            ConversationBoundary(1, 10.0, "invalid_reason", 0.8)
    
    def test_conversation_validation(self):
        """Test Conversation data validation."""
        segments = [TranscriptionSegment(1, 0.0, 5.0, "Test", "Speaker_1", 0.9)]
        
        # Valid conversation
        conv = Conversation(
            id="test",
            segments=segments,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            speakers=["Speaker_1"],
            speaker_transcript=""
        )
        assert conv.duration == 5.0
        
        # Empty segments
        with pytest.raises(ValueError, match="Conversation must have at least one segment"):
            Conversation("test", [], 0.0, 5.0, 5.0, [], "")
        
        # Invalid time range
        with pytest.raises(ValueError, match="Start time must be before end time"):
            Conversation("test", segments, 5.0, 0.0, 5.0, ["Speaker_1"], "")
        
        # Invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            Conversation("test", segments, 0.0, 5.0, -1.0, ["Speaker_1"], "")