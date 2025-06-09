"""
Unit tests for the AudioProcessor module.
"""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pytest

from pydub import AudioSegment
from pydub.generators import Sine

from src.audio import AudioProcessor, AudioProcessingError
from src.audio.data_classes import AudioChunk, AudioMetadata, ProcessedAudio, SilenceSegment
from src.transfer.data_classes import AudioFile
from src.config import ConfigManager


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock ConfigManager."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            "processing.audio.output_format": "wav",
            "processing.audio.sample_rate": 16000,
            "processing.audio.channels": 1,
            "processing.audio.normalization.enabled": True,
            "processing.audio.normalization.target_dBFS": -20.0,
            "processing.audio.noise_reduction.enabled": True,
            "processing.audio.noise_reduction.noise_floor": -40,
            "processing.audio.chunking.max_duration": 300,
            "processing.audio.chunking.min_silence": 0.5,
            "processing.audio.chunking.silence_thresh": -40,
            "processing.audio.enhancement.high_pass_filter": 80,
            "processing.audio.enhancement.low_pass_filter": 8000,
        }.get(key, default)
        return config
    
    @pytest.fixture
    def audio_processor(self, mock_config):
        """Create an AudioProcessor instance."""
        processor = AudioProcessor(mock_config)
        yield processor
        processor.cleanup()
    
    @pytest.fixture
    def sample_audio(self, temp_dir):
        """Create a sample audio file for testing."""
        # Generate a 3-second sine wave
        frequency = 440  # A4 note
        duration_ms = 3000
        audio = Sine(frequency).to_audio_segment(duration=duration_ms)
        
        # Add some silence
        silence = AudioSegment.silent(duration=500)
        audio = silence + audio + silence
        
        # Save to file
        file_path = temp_dir / "sample.wav"
        audio.export(file_path, format="wav")
        
        return AudioFile(
            source_path=file_path,
            destination_path=file_path,
            filename="sample.wav",
            size=file_path.stat().st_size,
            checksum="test_checksum",
            timestamp=datetime.now(),
            duration=4.0,
            format="wav"
        )
    
    @pytest.fixture
    def long_audio(self, temp_dir):
        """Create a long audio file for chunking tests."""
        # Generate multiple segments with silence between
        segments = []
        for i in range(3):
            # 2 minutes of tone
            tone = Sine(440 + i * 100).to_audio_segment(duration=120000)
            silence = AudioSegment.silent(duration=1000)
            segments.extend([tone, silence])
        
        audio = sum(segments[:-1])  # Remove last silence
        
        file_path = temp_dir / "long_sample.wav"
        audio.export(file_path, format="wav")
        
        return AudioFile(
            source_path=file_path,
            destination_path=file_path,
            filename="long_sample.wav",
            size=file_path.stat().st_size,
            checksum="test_checksum",
            timestamp=datetime.now(),
            duration=audio.duration_seconds,
            format="wav"
        )
    
    def test_init(self, audio_processor, mock_config):
        """Test AudioProcessor initialization."""
        assert audio_processor.config == mock_config
        assert audio_processor.sample_rate == 16000
        assert audio_processor.channels == 1
        assert audio_processor.output_format == "wav"
        assert audio_processor.normalization_enabled is True
        assert audio_processor.target_dbfs == -20.0
        assert audio_processor.temp_dir.exists()
    
    def test_process_audio_basic(self, audio_processor, sample_audio):
        """Test basic audio processing."""
        result = audio_processor.process_audio(sample_audio)
        
        assert isinstance(result, ProcessedAudio)
        assert result.original_file == sample_audio
        assert result.processed_path.exists()
        assert len(result.chunks) == 1
        assert result.metadata.duration > 0
        assert result.processing_time > 0
    
    def test_load_audio_file_not_found(self, audio_processor, temp_dir):
        """Test loading non-existent audio file."""
        missing_file = AudioFile(
            source_path=temp_dir / "missing.wav",
            destination_path=temp_dir / "missing.wav",
            filename="missing.wav",
            size=0,
            checksum="test_checksum",
            timestamp=datetime.now(),
            duration=0,
            format="wav"
        )
        
        with pytest.raises(AudioProcessingError) as exc_info:
            audio_processor.process_audio(missing_file)
        
        assert "Audio file not found" in str(exc_info.value)
    
    def test_load_audio_unsupported_format(self, audio_processor, temp_dir):
        """Test loading unsupported audio format."""
        # Create a fake file with unsupported extension
        fake_file = temp_dir / "audio.xyz"
        fake_file.write_text("fake audio data")
        
        unsupported_file = AudioFile(
            source_path=fake_file,
            destination_path=fake_file,
            filename="audio.xyz",
            size=fake_file.stat().st_size,
            checksum="test_checksum",
            timestamp=datetime.now(),
            duration=0,
            format="xyz"
        )
        
        with pytest.raises(AudioProcessingError) as exc_info:
            audio_processor.process_audio(unsupported_file)
        
        assert "Unsupported audio format" in str(exc_info.value)
    
    def test_normalize_audio(self, audio_processor):
        """Test audio normalization."""
        # Create audio with different volume
        quiet_audio = Sine(440).to_audio_segment(duration=1000) - 30  # -30 dB
        
        normalized = audio_processor.normalize_audio(quiet_audio)
        
        # Check that volume is close to target
        assert abs(normalized.dBFS - audio_processor.target_dbfs) < 1.0
    
    def test_normalize_audio_prevent_clipping(self, audio_processor):
        """Test normalization prevents clipping."""
        # Create very loud audio
        loud_audio = Sine(440).to_audio_segment(duration=1000) + 10  # +10 dB
        
        normalized = audio_processor.normalize_audio(loud_audio)
        
        # Check that max doesn't exceed 0 dBFS
        assert normalized.max_dBFS <= 0.0
    
    @patch('noisereduce.reduce_noise')
    def test_reduce_noise(self, mock_reduce_noise, audio_processor):
        """Test noise reduction."""
        audio = Sine(440).to_audio_segment(duration=1000)
        
        # Mock the noise reduction function
        mock_reduce_noise.return_value = np.zeros(len(audio.get_array_of_samples()))
        
        result = audio_processor.reduce_noise(audio)
        
        assert isinstance(result, AudioSegment)
        mock_reduce_noise.assert_called_once()
    
    def test_reduce_noise_failure(self, audio_processor):
        """Test noise reduction failure handling."""
        audio = Sine(440).to_audio_segment(duration=1000)
        
        with patch('noisereduce.reduce_noise', side_effect=Exception("Noise reduction failed")):
            result = audio_processor.reduce_noise(audio)
            
            # Should return original audio on failure
            assert result == audio
    
    def test_enhance_speech(self, audio_processor):
        """Test speech enhancement."""
        audio = Sine(440).to_audio_segment(duration=1000)
        
        enhanced = audio_processor.enhance_speech(audio)
        
        assert isinstance(enhanced, AudioSegment)
        assert enhanced.frame_rate == audio.frame_rate
        assert enhanced.channels == audio.channels
    
    def test_split_audio_short(self, audio_processor, sample_audio):
        """Test that short audio is not split."""
        audio = AudioSegment.from_file(sample_audio.destination_path)
        
        chunks = audio_processor.split_audio(audio, max_duration=300)
        
        assert len(chunks) == 1
        assert abs(chunks[0].duration_seconds - audio.duration_seconds) < 0.01  # Allow small difference
    
    def test_split_audio_long(self, audio_processor, long_audio):
        """Test splitting long audio."""
        audio = AudioSegment.from_file(long_audio.destination_path)
        
        # Split with 2-minute max duration
        chunks = audio_processor.split_audio(audio, max_duration=120)
        
        assert len(chunks) > 1
        assert all(chunk.duration_seconds <= 120 * 1.2 for chunk in chunks)  # Allow 20% overflow
    
    def test_detect_silence(self, audio_processor):
        """Test silence detection."""
        # Create audio with silence
        tone = Sine(440).to_audio_segment(duration=1000)
        silence = AudioSegment.silent(duration=1000)
        audio = tone + silence + tone
        
        silence_segments = audio_processor.detect_silence(audio)
        
        assert len(silence_segments) > 0
        assert all(isinstance(seg, SilenceSegment) for seg in silence_segments)
    
    def test_get_audio_info(self, audio_processor, sample_audio):
        """Test metadata extraction."""
        metadata = audio_processor.get_audio_info(sample_audio.destination_path)
        
        assert isinstance(metadata, AudioMetadata)
        assert metadata.duration > 0
        assert metadata.sample_rate > 0
        assert metadata.channels > 0
        assert metadata.format == "wav"
        assert -100 <= metadata.avg_volume <= 0
        assert -100 <= metadata.peak_volume <= 0
        assert 0 <= metadata.silence_ratio <= 1
    
    def test_standardize_format(self, audio_processor):
        """Test format standardization."""
        # Create stereo audio with different sample rate
        stereo_audio = Sine(440).to_audio_segment(duration=1000)
        stereo_audio = stereo_audio.set_channels(2).set_frame_rate(44100)
        
        standardized = audio_processor._standardize_format(stereo_audio)
        
        assert standardized.channels == 1
        assert standardized.frame_rate == 16000
        assert standardized.sample_width == 2
    
    def test_save_chunks(self, audio_processor):
        """Test saving audio chunks."""
        chunks = [
            Sine(440).to_audio_segment(duration=1000),
            Sine(880).to_audio_segment(duration=1000)
        ]
        
        saved_chunks = audio_processor._save_chunks(chunks, "test")
        
        assert len(saved_chunks) == 2
        assert all(chunk.file_path.exists() for chunk in saved_chunks)
        assert saved_chunks[0].chunk_id == 0
        assert saved_chunks[1].chunk_id == 1
        assert saved_chunks[1].start_time == saved_chunks[0].end_time
    
    def test_cleanup(self, audio_processor):
        """Test cleanup of temporary files."""
        temp_dir = audio_processor.temp_dir
        
        # Create a test file
        test_file = temp_dir / "test.wav"
        test_file.write_text("test")
        
        assert temp_dir.exists()
        assert test_file.exists()
        
        audio_processor.cleanup()
        
        assert not temp_dir.exists()
    
    def test_process_audio_with_error_handling(self, audio_processor, sample_audio):
        """Test error handling in process_audio."""
        # Mock a processing error
        with patch.object(audio_processor, 'normalize_audio', side_effect=Exception("Processing failed")):
            with pytest.raises(AudioProcessingError):
                audio_processor.process_audio(sample_audio)
    
    def test_process_multiple_formats(self, audio_processor, temp_dir):
        """Test processing different audio formats."""
        formats = ['mp3', 'wav', 'm4a']
        
        for fmt in formats:
            # Create test audio
            audio = Sine(440).to_audio_segment(duration=1000)
            file_path = temp_dir / f"test.{fmt}"
            
            # Skip formats that might not be supported by the test environment
            try:
                audio.export(file_path, format=fmt)
            except:
                continue
            
            audio_file = AudioFile(
                source_path=file_path,
                destination_path=file_path,
                filename=f"test.{fmt}",
                size=file_path.stat().st_size,
                checksum="test_checksum",
                timestamp=datetime.now(),
                duration=1.0,
                format=fmt
            )
            
            result = audio_processor.process_audio(audio_file)
            
            assert result.processed_path.exists()
            assert result.metadata.format == "wav"  # Output should be WAV
    
    def test_smart_split_on_silence(self, audio_processor):
        """Test smart splitting on silence boundaries."""
        # Create audio with clear silence points
        segments = []
        for i in range(3):
            tone = Sine(440 + i * 100).to_audio_segment(duration=4000)
            silence = AudioSegment.silent(duration=1000)
            segments.extend([tone, silence])
        
        audio = sum(segments[:-1])
        
        # Mock silence detection
        silence_segments = [
            (4000, 5000),   # First silence
            (9000, 10000),  # Second silence
        ]
        
        chunks = audio_processor._smart_split_on_silence(audio, silence_segments, 6000)
        
        assert len(chunks) >= 2
        # Verify chunks are split at silence points
    
    def test_time_based_split(self, audio_processor):
        """Test time-based splitting fallback."""
        audio = Sine(440).to_audio_segment(duration=10000)  # 10 seconds
        
        chunks = audio_processor._time_based_split(audio, 3000)  # 3-second chunks
        
        assert len(chunks) == 4  # 3 full chunks + 1 partial
        assert all(len(chunk) <= 3000 for chunk in chunks[:-1])
    
    def test_update_metadata(self, audio_processor):
        """Test metadata update after processing."""
        original = AudioMetadata(
            duration=10.0,
            sample_rate=44100,
            channels=2,
            bitrate=320000,
            format="mp3",
            avg_volume=-25.0,
            peak_volume=-20.0,
            silence_ratio=0.1
        )
        
        processed_audio = Sine(440).to_audio_segment(duration=10000)
        processed_audio = processed_audio.set_frame_rate(16000).set_channels(1)
        
        updated = audio_processor._update_metadata(original, processed_audio)
        
        assert updated.sample_rate == 16000
        assert updated.channels == 1
        assert updated.format == "wav"
        assert updated.duration == processed_audio.duration_seconds