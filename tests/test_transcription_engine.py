"""Unit tests for the transcription engine module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np

from src.transcription.transcription_engine import TranscriptionEngine
from src.transcription.transcription_engine_with_fallback import TranscriptionEngineWithFallback
from src.transcription.data_classes import (
    WhisperSegment, SpeakerSegment, TranscriptionSegment,
    Transcription, WordTiming
)
from src.audio.data_classes import ProcessedAudio, AudioChunk, AudioMetadata, AudioFile
from src.config.config_manager import ConfigManager


class TestTranscriptionEngine(unittest.TestCase):
    """Test cases for TranscriptionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config manager
        self.mock_config = Mock(spec=ConfigManager)
        self.mock_config.get.side_effect = self._mock_config_get
        
        # Create test audio data
        self.test_audio = self._create_test_audio()
        
    def _mock_config_get(self, key, default=None):
        """Mock configuration values."""
        config_map = {
            "processing.whisper_model": "base",
            "processing.whisper_device": "cpu",
            "processing.language": "es",
            "processing.diarization.enabled": True,
            "processing.diarization.min_speakers": 1,
            "processing.diarization.max_speakers": 10,
            "processing.diarization.min_duration": 0.5,
            "processing.faster_whisper": {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "vad_filter": True,
                "vad_options": {
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200
                }
            },
            "processing.post_processing.custom_vocabulary": ["TranscriMatic", "API"]
        }
        return config_map.get(key, default)
    
    def _create_test_audio(self):
        """Create test ProcessedAudio object."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            test_path = Path(tmp.name)
        
        audio_file = AudioFile(
            source_path=test_path,
            destination_path=test_path,
            filename="test.wav",
            size=1000,
            checksum="abc123",
            timestamp=None,
            duration=10.0,
            format="wav"
        )
        
        chunks = [
            AudioChunk(
                chunk_id=0,
                file_path=test_path,
                start_time=0.0,
                end_time=5.0,
                duration=5.0
            ),
            AudioChunk(
                chunk_id=1,
                file_path=test_path,
                start_time=5.0,
                end_time=10.0,
                duration=5.0
            )
        ]
        
        metadata = AudioMetadata(
            duration=10.0,
            sample_rate=16000,
            channels=1,
            bitrate=256000,
            format="wav",
            avg_volume=-20.0,
            peak_volume=-10.0,
            silence_ratio=0.1
        )
        
        return ProcessedAudio(
            original_file=audio_file,
            processed_path=test_path,
            chunks=chunks,
            metadata=metadata,
            processing_time=1.0
        )
    
    @patch('src.transcription.transcription_engine.WhisperModel')
    @patch('src.transcription.transcription_engine.Pipeline')
    def test_initialization(self, mock_pipeline, mock_whisper):
        """Test engine initialization."""
        engine = TranscriptionEngine(self.mock_config)
        
        self.assertIsNotNone(engine.model)
        self.assertEqual(engine.language, "es")
        mock_whisper.assert_called_once()
    
    def test_spanish_post_processing(self):
        """Test Spanish text post-processing."""
        with patch('src.transcription.transcription_engine.WhisperModel'):
            with patch('src.transcription.transcription_engine.Pipeline'):
                engine = TranscriptionEngine(self.mock_config)
        
        # Test common corrections
        test_text = "avia una vez aora ise algo"
        processed = engine.post_process_spanish(test_text)
        self.assertIn("había", processed)
        self.assertIn("ahora", processed)
        self.assertIn("hice", processed)
        
        # Test punctuation fixes
        test_text = "Hola?Como estas!Bien."
        processed = engine._fix_spanish_punctuation(test_text)
        self.assertEqual(processed, "Hola? Como estas! Bien.")
        
        # Test capitalization
        test_text = "hola. como estas? bien!"
        processed = engine._fix_capitalization(test_text)
        self.assertTrue(processed.startswith("Hola"))
    
    def test_speaker_mapping(self):
        """Test speaker label creation."""
        with patch('src.transcription.transcription_engine.WhisperModel'):
            with patch('src.transcription.transcription_engine.Pipeline'):
                engine = TranscriptionEngine(self.mock_config)
        
        segments = [
            TranscriptionSegment(0, 0, 1, "Hola", "SPEAKER_00", 0.9),
            TranscriptionSegment(1, 1, 2, "Buenos días", "SPEAKER_01", 0.8),
            TranscriptionSegment(2, 2, 3, "Como estas", "SPEAKER_00", 0.85),
        ]
        
        mapping = engine._create_speaker_mapping(segments)
        
        self.assertEqual(mapping["SPEAKER_00"], "Hablante 1")
        self.assertEqual(mapping["SPEAKER_01"], "Hablante 2")
    
    def test_format_speaker_transcript(self):
        """Test speaker-aware transcript formatting."""
        with patch('src.transcription.transcription_engine.WhisperModel'):
            with patch('src.transcription.transcription_engine.Pipeline'):
                engine = TranscriptionEngine(self.mock_config)
        
        segments = [
            TranscriptionSegment(0, 0, 1, "Hola", "SPEAKER_00", 0.9),
            TranscriptionSegment(1, 1, 2, "Como estas", "SPEAKER_00", 0.8),
            TranscriptionSegment(2, 2, 3, "Bien gracias", "SPEAKER_01", 0.85),
            TranscriptionSegment(3, 3, 4, "Y tu?", "SPEAKER_01", 0.87),
        ]
        
        transcript = engine.format_speaker_transcript(segments)
        
        self.assertIn("[Hablante 1]:", transcript)
        self.assertIn("[Hablante 2]:", transcript)
        self.assertIn("Hola\nComo estas", transcript)
        self.assertIn("Bien gracias\nY tu?", transcript)
    
    def test_merge_transcription_diarization(self):
        """Test merging of transcription and speaker diarization."""
        with patch('src.transcription.transcription_engine.WhisperModel'):
            with patch('src.transcription.transcription_engine.Pipeline'):
                engine = TranscriptionEngine(self.mock_config)
        
        whisper_segments = [
            WhisperSegment(0, 2, "Hola buenos días", None, -0.5, 0.1),
            WhisperSegment(2, 4, "Como está usted", None, -0.3, 0.05),
        ]
        
        speaker_segments = [
            SpeakerSegment(0, 1.5, "SPEAKER_00", 0.9),
            SpeakerSegment(1.5, 4, "SPEAKER_01", 0.85),
        ]
        
        merged = engine.merge_transcription_diarization(whisper_segments, speaker_segments)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].speaker_id, "SPEAKER_00")
        self.assertEqual(merged[1].speaker_id, "SPEAKER_01")
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        with patch('src.transcription.transcription_engine.WhisperModel'):
            with patch('src.transcription.transcription_engine.Pipeline'):
                engine = TranscriptionEngine(self.mock_config)
        
        segments = [
            TranscriptionSegment(0, 0, 2, "Segment 1", "SPEAKER_00", 0.9),
            TranscriptionSegment(1, 2, 3, "Segment 2", "SPEAKER_00", 0.7),
            TranscriptionSegment(2, 3, 5, "Segment 3", "SPEAKER_01", 0.8),
        ]
        
        confidence = engine._calculate_confidence(segments)
        
        # Weighted average: (0.9*2 + 0.7*1 + 0.8*2) / 5 = 0.82
        self.assertAlmostEqual(confidence, 0.82, places=2)
    
    @patch('src.transcription.transcription_engine.WhisperModel')
    def test_language_detection(self, mock_whisper):
        """Test language detection functionality."""
        with patch('src.transcription.transcription_engine.Pipeline'):
            engine = TranscriptionEngine(self.mock_config)
        
        # Mock model transcribe for language detection
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.95
        
        engine.model.transcribe.return_value = ([], mock_info)
        
        lang, prob = engine.detect_language(Path("test.wav"))
        
        self.assertEqual(lang, "es")
        self.assertEqual(prob, 0.95)


class TestTranscriptionEngineWithFallback(unittest.TestCase):
    """Test cases for fallback strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=ConfigManager)
        self.mock_config.get.return_value = True
        self.mock_config.set.return_value = None
        
        self.test_audio = self._create_test_audio()
    
    def _create_test_audio(self):
        """Create minimal test audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            test_path = Path(tmp.name)
        
        return ProcessedAudio(
            original_file=Mock(),
            processed_path=test_path,
            chunks=[],
            metadata=Mock(duration=10.0, silence_ratio=0.1, avg_volume=-20),
            processing_time=1.0
        )
    
    @patch('src.transcription.transcription_engine.WhisperModel')
    @patch('src.transcription.transcription_engine.Pipeline')
    def test_audio_quality_validation(self, mock_pipeline, mock_whisper):
        """Test audio quality validation."""
        engine = TranscriptionEngineWithFallback(self.mock_config)
        
        # Good audio
        self.assertTrue(engine.validate_audio_quality(self.test_audio))
        
        # Too silent
        self.test_audio.metadata.silence_ratio = 0.99
        self.assertFalse(engine.validate_audio_quality(self.test_audio))
        
        # Too quiet
        self.test_audio.metadata.silence_ratio = 0.1
        self.test_audio.metadata.avg_volume = -60
        self.assertFalse(engine.validate_audio_quality(self.test_audio))
    
    @patch('src.transcription.transcription_engine.WhisperModel')
    @patch('src.transcription.transcription_engine.Pipeline')
    def test_empty_transcription_creation(self, mock_pipeline, mock_whisper):
        """Test creation of empty transcription on failure."""
        engine = TranscriptionEngineWithFallback(self.mock_config)
        
        result = engine._create_empty_transcription(self.test_audio)
        
        self.assertIsInstance(result, Transcription)
        self.assertEqual(result.segments, [])
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("failed", result.full_text.lower())


if __name__ == '__main__':
    unittest.main()