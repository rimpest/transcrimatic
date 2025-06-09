"""
Test utilities and shared functionality for TranscriMatic tests
"""

import shutil
import tempfile
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

# Test paths
TEST_ROOT = Path(__file__).parent
TEST_OUTPUTS = TEST_ROOT / "test_outputs"
UNIT_OUTPUTS = TEST_OUTPUTS / "unit"
INTEGRATION_OUTPUTS = TEST_OUTPUTS / "integration"
PERFORMANCE_OUTPUTS = TEST_OUTPUTS / "performance"
TEMP_OUTPUTS = TEST_OUTPUTS / "temp"
FIXTURES_OUTPUTS = TEST_OUTPUTS / "fixtures"


def get_test_output_dir(test_type: str = "unit") -> Path:
    """
    Get test output directory for specific test type.
    
    Args:
        test_type: Type of test ("unit", "integration", "performance", "temp", "fixtures")
        
    Returns:
        Path to test output directory
    """
    dirs = {
        "unit": UNIT_OUTPUTS,
        "integration": INTEGRATION_OUTPUTS, 
        "performance": PERFORMANCE_OUTPUTS,
        "temp": TEMP_OUTPUTS,
        "fixtures": FIXTURES_OUTPUTS
    }
    
    output_dir = dirs.get(test_type, UNIT_OUTPUTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def cleanup_test_outputs(test_type: Optional[str] = None):
    """
    Clean up test output directories.
    
    Args:
        test_type: Specific test type to clean, or None for all
    """
    if test_type:
        output_dir = get_test_output_dir(test_type)
        if output_dir.exists():
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Clean all except .gitignore and README.md
        for item in TEST_OUTPUTS.iterdir():
            if item.name not in ['.gitignore', 'README.md']:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # Recreate subdirectories
        for subdir in ["unit", "integration", "performance", "temp", "fixtures"]:
            (TEST_OUTPUTS / subdir).mkdir(exist_ok=True)


def create_temp_dir(test_name: str) -> Path:
    """
    Create a temporary directory for a specific test.
    
    Args:
        test_name: Name of the test (used in directory name)
        
    Returns:
        Path to temporary directory
    """
    temp_dir = TEMP_OUTPUTS / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def create_mock_config(base_path: Optional[Path] = None, **overrides) -> Mock:
    """
    Create a mock configuration manager for tests.
    
    Args:
        base_path: Base path for test outputs
        **overrides: Configuration overrides
        
    Returns:
        Mock configuration manager
    """
    if base_path is None:
        base_path = get_test_output_dir("temp")
    
    default_config = {
        # Output configuration
        "output.base_path": str(base_path),
        "output.structure.transcriptions_dir": "transcriptions",
        "output.structure.summaries_dir": "summaries",
        "output.formatting.date_format": "%Y-%m-%d",
        "output.formatting.time_format": "%H:%M:%S",
        "output.formatting.include_timestamps": True,
        "output.markdown.include_metadata": True,
        "output.markdown.include_navigation": True,
        
        # Logging configuration
        "logging.file": str(base_path / "test.log"),
        "logging.level": "DEBUG",
        "logging.max_size": 1048576,
        "logging.backup_count": 2,
        
        # System configuration
        "system.monitoring.health_check_interval": 1,
        "system.monitoring.status_file": str(base_path / "status.json"),
        "system.notifications.enabled": False,
        
        # Audio configuration
        "audio.formats": [".wav", ".mp3", ".m4a", ".flac"],
        "audio.noise_reduction": True,
        "audio.normalize_volume": True,
        "audio.target_sample_rate": 16000,
        
        # Transcription configuration
        "transcription.model": "whisper",
        "transcription.model_size": "base",
        "transcription.language": "es",
        "transcription.enable_diarization": True,
        "transcription.min_speakers": 2,
        "transcription.max_speakers": 5,
        
        # LLM configuration
        "llm.provider": "anthropic",
        "llm.model": "claude-3-sonnet",
        "llm.temperature": 0.7,
        "llm.max_tokens": 1000,
        
        # Device configuration
        "device.identifier": "Test_Device",
        "device.vendor_id": "1234",
        "device.product_id": "5678"
    }
    
    # Apply overrides
    default_config.update(overrides)
    
    config = Mock()
    config.get.side_effect = lambda key, default=None: default_config.get(key, default)
    config.validate_paths.return_value = True
    
    return config


class MockDataFactory:
    """Factory for creating mock test data"""
    
    @staticmethod
    def create_audio_file(filename: str = "test.wav", duration: float = 60.0) -> Mock:
        """Create mock audio file"""
        audio_file = Mock()
        audio_file.filename = filename
        audio_file.path = f"/tmp/{filename}"
        audio_file.size = 1024000
        audio_file.duration = duration
        return audio_file
    
    @staticmethod
    def create_transcription_segment(
        speaker_id: str = "SPEAKER_00",
        text: str = "Test text",
        start_time: float = 0.0,
        end_time: float = 2.0
    ) -> Dict[str, Any]:
        """Create mock transcription segment"""
        return {
            "speaker_id": speaker_id,
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": 0.95
        }
    
    @staticmethod
    def create_conversation(
        conversation_id: str = "test_001",
        speakers: list = None,
        duration: float = 120.0,
        timestamp: datetime = None
    ) -> Mock:
        """Create mock conversation"""
        if speakers is None:
            speakers = ["SPEAKER_00", "SPEAKER_01"]
        if timestamp is None:
            timestamp = datetime(2024, 1, 15, 10, 0, 0)
        
        conversation = Mock()
        conversation.id = conversation_id
        conversation.speakers = speakers
        conversation.duration = duration
        conversation.timestamp = timestamp
        conversation.transcript = "Test conversation transcript"
        conversation.speaker_transcript = "[SPEAKER_00] Hello\n[SPEAKER_01] Hi there"
        conversation.segments = [
            MockDataFactory.create_transcription_segment("SPEAKER_00", "Hello", 0.0, 1.0),
            MockDataFactory.create_transcription_segment("SPEAKER_01", "Hi there", 1.5, 3.0)
        ]
        return conversation
    
    @staticmethod
    def create_task(
        description: str = "Test task",
        priority: str = "media",
        assignee: str = "Test User"
    ) -> Mock:
        """Create mock task"""
        task = Mock()
        task.description = description
        task.priority = priority
        task.assignee = assignee
        task.assigned_by = "Test Manager"
        task.due_date = "2024-01-25"
        task.context = "Test context"
        
        # Add to_dict method
        task.to_dict.return_value = {
            "description": description,
            "priority": priority,
            "assignee": assignee,
            "assigned_by": "Test Manager",
            "due_date": "2024-01-25",
            "context": "Test context"
        }
        return task
    
    @staticmethod
    def create_analysis(
        conversation_id: str = "test_001",
        tasks: list = None,
        timestamp: datetime = None
    ) -> Mock:
        """Create mock analysis"""
        if tasks is None:
            tasks = [MockDataFactory.create_task()]
        if timestamp is None:
            timestamp = datetime(2024, 1, 15, 10, 0, 0)
        
        analysis = Mock()
        analysis.conversation_id = conversation_id
        analysis.summary = "Test analysis summary"
        analysis.key_points = ["Point 1", "Point 2", "Point 3"]
        analysis.tasks = tasks
        analysis.todos = []
        analysis.followups = []
        analysis.participants = ["Test User 1", "Test User 2"]
        analysis.duration = 120.0
        analysis.timestamp = timestamp
        analysis.llm_provider = "test"
        
        # Add to_dict method
        analysis.to_dict.return_value = {
            "conversation_id": conversation_id,
            "summary": analysis.summary,
            "key_points": analysis.key_points,
            "tasks": [t.to_dict() for t in tasks],
            "todos": [],
            "followups": [],
            "participants": analysis.participants,
            "duration": analysis.duration,
            "timestamp": timestamp.isoformat(),
            "llm_provider": "test"
        }
        return analysis


def assert_file_exists(file_path: Path, message: str = ""):
    """Assert that a file exists with custom message"""
    assert file_path.exists(), f"File does not exist: {file_path}. {message}"


def assert_file_contains(file_path: Path, content: str, message: str = ""):
    """Assert that a file contains specific content"""
    assert_file_exists(file_path)
    file_content = file_path.read_text(encoding='utf-8')
    assert content in file_content, f"File {file_path} does not contain '{content}'. {message}"


def assert_directory_structure(base_path: Path, expected_structure: Dict[str, Any]):
    """
    Assert that a directory has the expected structure.
    
    Args:
        base_path: Base directory to check
        expected_structure: Dict representing expected structure
                           {"file.txt": None, "subdir": {"nested.txt": None}}
    """
    def check_structure(current_path: Path, structure: Dict[str, Any]):
        for name, sub_structure in structure.items():
            item_path = current_path / name
            assert item_path.exists(), f"Expected {item_path} to exist"
            
            if sub_structure is None:
                # It's a file
                assert item_path.is_file(), f"Expected {item_path} to be a file"
            else:
                # It's a directory
                assert item_path.is_dir(), f"Expected {item_path} to be a directory"
                check_structure(item_path, sub_structure)
    
    check_structure(base_path, expected_structure)


def create_sample_config_file(output_dir: Path, filename: str = "test_config.yaml") -> Path:
    """Create a sample configuration file for testing"""
    config_content = f"""
# Test Configuration
paths:
  input_path: {output_dir}/input
  output_path: {output_dir}/output
  archive_path: {output_dir}/archive

audio:
  formats: [".wav", ".mp3", ".m4a", ".flac"]
  noise_reduction: true
  normalize_volume: true
  target_sample_rate: 16000

transcription:
  model: "whisper"
  model_size: "base"
  language: "es"
  enable_diarization: true
  min_speakers: 2
  max_speakers: 5

llm:
  provider: "anthropic"
  model: "claude-3-sonnet"
  temperature: 0.7
  max_tokens: 1000

logging:
  file: {output_dir}/logs/test.log
  level: "DEBUG"
  max_size: 1048576
  backup_count: 2

output:
  base_path: {output_dir}/output
  structure:
    transcriptions_dir: "transcriptions"
    summaries_dir: "summaries"
  formatting:
    date_format: "%Y-%m-%d"
    time_format: "%H:%M:%S"
    include_timestamps: true
  markdown:
    include_metadata: true
    include_navigation: true

system:
  monitoring:
    health_check_interval: 60
    status_file: {output_dir}/status.json
  notifications:
    enabled: false
"""
    
    config_file = output_dir / filename
    config_file.write_text(config_content)
    
    # Create necessary directories
    for subdir in ["input", "output", "archive", "logs"]:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    return config_file


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.4f} seconds")


def measure_memory_usage(func):
    """Decorator to measure memory usage of a function"""
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            print(f"{func.__name__} memory usage: {mem_diff:.2f} MB")
            return result
        except ImportError:
            # psutil not available, just run the function
            return func(*args, **kwargs)
    
    return wrapper