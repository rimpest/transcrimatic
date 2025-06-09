# Transcription Engine Testing Requirements

## Required Dependencies

### 1. Python Packages
Install the following packages in your virtual environment:

```bash
# Core transcription dependencies
pip install faster-whisper==1.0.3
pip install pyannote.audio==3.1.1
pip install torch>=2.0.0
pip install torchaudio>=2.0.0
pip install numpy
pip install onnxruntime

# For testing
pip install pytest
pip install pytest-mock
```

### 2. Models to Download

#### Faster-Whisper Models
The models will auto-download on first use, but you can pre-download them:

```python
from faster_whisper import download_model
# Download the Spanish-optimized model
download_model("large-v3")  # ~1.5GB
# Or for testing with smaller model
download_model("base")      # ~140MB
```

#### PyAnnote Speaker Diarization
You need a Hugging Face token for the speaker diarization model:

1. Create account at https://huggingface.co
2. Get access token from https://huggingface.co/settings/tokens
3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Add token to config or environment:
   ```bash
   export HUGGINGFACE_TOKEN="your_token_here"
   ```

## Test Data Requirements

### Audio Files
You need Spanish audio files in these formats:
- WAV (preferred) - 16kHz, mono
- MP3 
- Any format supported by ffmpeg

### Test Audio Characteristics
- Duration: 30 seconds to 5 minutes (for quick tests)
- Clear Spanish speech
- Multiple speakers (for diarization testing)
- Various audio qualities (clean, noisy, telephone)

## Mock Components Needed

Since the transcription engine depends on other modules, you'll need to mock:

### 1. ConfigManager
```python
from unittest.mock import Mock
from src.config.config_manager import ConfigManager

mock_config = Mock(spec=ConfigManager)
mock_config.get.side_effect = lambda key, default=None: {
    "processing.whisper_model": "base",
    "processing.whisper_device": "cpu",
    "processing.language": "es",
    "processing.diarization.enabled": True,
    "pyannote.auth_token": "your_token_here"
}.get(key, default)
```

### 2. ProcessedAudio Objects
```python
from src.audio.data_classes import ProcessedAudio, AudioChunk, AudioMetadata, AudioFile
from pathlib import Path

# Create test audio object
audio_file = AudioFile(
    source_path=Path("test.wav"),
    destination_path=Path("test.wav"),
    filename="test.wav",
    size=1000000,
    checksum="abc123",
    timestamp=datetime.now(),
    duration=60.0,
    format="wav"
)

chunks = [AudioChunk(
    chunk_id=0,
    file_path=Path("test.wav"),
    start_time=0.0,
    end_time=60.0,
    duration=60.0
)]

metadata = AudioMetadata(
    duration=60.0,
    sample_rate=16000,
    channels=1,
    bitrate=256000,
    format="wav",
    avg_volume=-20.0,
    peak_volume=-10.0,
    silence_ratio=0.1
)

processed_audio = ProcessedAudio(
    original_file=audio_file,
    processed_path=Path("test.wav"),
    chunks=chunks,
    metadata=metadata,
    processing_time=1.0
)
```

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended
  - NVIDIA GPU with CUDA support
  - 4GB+ VRAM for large-v3 model

### Software
- **Python**: 3.8 or higher
- **FFmpeg**: Required by faster-whisper
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Windows
  # Download from https://ffmpeg.org/download.html
  ```

## Environment Setup

1. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

2. **CUDA (Optional for GPU)**
   - Install CUDA Toolkit 11.8 or 12.1
   - Install cuDNN
   - Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## Quick Test Script

Create a simple test script to verify installation:

```python
# test_transcription_setup.py
import torch
from faster_whisper import WhisperModel

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    model = WhisperModel("base", device="cpu")
    print("✓ Faster-whisper loaded successfully")
except Exception as e:
    print(f"✗ Faster-whisper error: {e}")

try:
    from pyannote.audio import Pipeline
    print("✓ PyAnnote imported successfully")
except Exception as e:
    print(f"✗ PyAnnote error: {e}")
```

## Common Issues and Solutions

### 1. CUDA Out of Memory
- Use smaller model (base or small)
- Set device to "cpu"
- Reduce batch size

### 2. PyAnnote Access Denied
- Ensure HuggingFace token is set
- Accept model license agreement
- Check token permissions

### 3. FFmpeg Not Found
- Install FFmpeg system-wide
- Add to PATH environment variable

### 4. Slow CPU Performance
- Use smaller model (base)
- Enable int8 quantization
- Process shorter audio segments