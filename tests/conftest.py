"""
Pytest configuration and shared fixtures for TranscriMatic tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

# Add src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
from .test_utils import (
    get_test_output_dir, cleanup_test_outputs, create_temp_dir,
    create_mock_config, MockDataFactory, create_sample_config_file
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Clean up any existing test outputs
    cleanup_test_outputs()
    yield
    # Optional: cleanup after all tests (comment out for debugging)
    # cleanup_test_outputs()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_output_dir():
    """Get test output directory for current test"""
    return get_test_output_dir("unit")


@pytest.fixture
def integration_output_dir():
    """Get integration test output directory"""
    return get_test_output_dir("integration")


@pytest.fixture
def performance_output_dir():
    """Get performance test output directory"""
    return get_test_output_dir("performance")


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration for testing"""
    config_content = """
# Test configuration
paths:
  input_path: {temp_dir}/input
  output_path: {temp_dir}/output
  archive_path: {temp_dir}/archive

audio:
  formats: [".wav", ".mp3", ".m4a", ".flac"]
  noise_reduction: true
  normalize_volume: true
  target_sample_rate: 16000

transcription:
  model: "whisper"
  model_size: "base"
  language: "en"
  enable_diarization: true
  min_speakers: 2
  max_speakers: 5

conversation:
  min_duration: 30
  max_silence: 10
  overlap_threshold: 0.5

llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
  temperature: 0.7
  max_tokens: 1000

logging:
  file: {temp_dir}/logs/test.log
  level: "DEBUG"
  max_size: 1048576
  backup_count: 2

system:
  auto_start: true
  daemon_mode: false
  processing:
    max_concurrent: 2
    retry_attempts: 3
    retry_delay: 5
  monitoring:
    health_check_interval: 60
    status_file: {temp_dir}/status.json
  notifications:
    enabled: false
""".format(temp_dir=temp_dir)
    
    config_file = temp_dir / "config.yaml"
    config_file.write_text(config_content)
    
    # Create necessary directories
    (temp_dir / "input").mkdir()
    (temp_dir / "output").mkdir()
    (temp_dir / "archive").mkdir()
    (temp_dir / "logs").mkdir()
    
    return config_file


@pytest.fixture
def mock_audio_file():
    """Create a mock AudioFile object"""
    from src.audio import AudioFile
    return AudioFile(
        filename="test_audio.wav",
        path="/tmp/test_audio.wav",
        size=1024000,  # 1MB
        duration=60.5  # 60.5 seconds
    )


@pytest.fixture
def mock_device_event():
    """Create a mock DeviceEvent object"""
    from src.device import DeviceEvent
    return DeviceEvent(
        event_type="connected",
        device_name="TEST_RECORDER",
        mount_point="/mnt/test_device",
        timestamp=datetime.now()
    )


@pytest.fixture
def mock_transcription():
    """Create a mock Transcription object"""
    return Mock(
        text="Hello, this is a test transcription.",
        segments=[
            Mock(start=0.0, end=2.5, text="Hello,", speaker="SPEAKER_00"),
            Mock(start=2.5, end=5.0, text="this is a test transcription.", speaker="SPEAKER_00")
        ],
        speakers=["SPEAKER_00"],
        duration=5.0,
        language="en"
    )


@pytest.fixture
def mock_conversation():
    """Create a mock Conversation object"""
    return Mock(
        id="conv_001",
        start_time=0.0,
        end_time=60.0,
        duration=60.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            Mock(speaker="SPEAKER_00", text="Hello, how are you?", start=0.0, end=2.0),
            Mock(speaker="SPEAKER_01", text="I'm doing well, thank you.", start=2.5, end=5.0)
        ],
        transcript="SPEAKER_00: Hello, how are you?\nSPEAKER_01: I'm doing well, thank you."
    )


@pytest.fixture
def mock_analysis():
    """Create a mock ConversationAnalysis object"""
    return Mock(
        conversation_id="conv_001",
        topics=["greetings", "well-being"],
        summary="A brief greeting exchange between two speakers.",
        sentiment="positive",
        key_points=["Speaker 1 greets Speaker 2", "Speaker 2 responds positively"],
        action_items=[],
        insights="Polite and friendly interaction"
    )


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests"""
    import logging
    # Remove all handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    yield
    # Clean up after test
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


@pytest.fixture
def mock_config_manager(sample_config):
    """Create a mock ConfigManager with test configuration"""
    from unittest.mock import Mock
    
    mock = Mock()
    mock.config_path = sample_config
    mock.config = {
        "paths": {
            "input_path": str(sample_config.parent / "input"),
            "output_path": str(sample_config.parent / "output"),
            "archive_path": str(sample_config.parent / "archive")
        },
        "audio": {
            "formats": [".wav", ".mp3", ".m4a", ".flac"],
            "noise_reduction": True,
            "normalize_volume": True,
            "target_sample_rate": 16000
        },
        "transcription": {
            "model": "whisper",
            "model_size": "base",
            "language": "en",
            "enable_diarization": True,
            "min_speakers": 2,
            "max_speakers": 5
        },
        "logging": {
            "file": str(sample_config.parent / "logs" / "test.log"),
            "level": "DEBUG"
        }
    }
    
    def get_side_effect(key, default=None):
        # Navigate through nested keys
        keys = key.split('.')
        value = mock.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    mock.get.side_effect = get_side_effect
    mock.validate_paths.return_value = True
    
    return mock


@pytest.fixture
def sample_audio_files(temp_dir):
    """Create sample audio files for testing"""
    audio_files = []
    for i in range(3):
        file_path = temp_dir / f"test_audio_{i}.wav"
        file_path.write_bytes(b"RIFF" + b"\x00" * 1000)  # Minimal WAV header
        
        from src.audio import AudioFile
        audio_files.append(AudioFile(
            filename=f"test_audio_{i}.wav",
            path=str(file_path),
            size=1004,
            duration=30.0 + i * 10
        ))
    
    return audio_files


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )