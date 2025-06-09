# TranscriMatic Testing Guide

This guide covers all testing procedures for the TranscriMatic transcription engine and complete pipeline.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-transcription.txt

# For TTS (optional, for generating test audio)
pip install gtts
```

### 2. Generate Test Audio

```bash
# Generate all test audio files
python tests/generate_test_audio.py

# Generate specific types
python tests/generate_test_audio.py --types speech
python tests/generate_test_audio.py --types tones edge
```

This creates test audio files in the `test_audio/` directory.

### 3. Run Tests

#### Unit Tests
```bash
# Run transcription engine unit tests
python -m pytest tests/test_transcription_engine.py -v

# Run with coverage
python -m pytest tests/test_transcription_engine.py --cov=src.transcription
```

#### Integration Tests
```bash
# Test model loading
python test_transcription_integration.py --test-loading

# Test single file transcription
python test_transcription_integration.py --audio test_audio/test_spanish_greeting.wav

# Test batch processing
python test_transcription_integration.py --audio-dir test_audio/
```

#### End-to-End Pipeline Test
```bash
# Run complete pipeline test
python test_end_to_end.py --audio test_audio/test_spanish_greeting.wav

# Generate audio and test in one command
python test_end_to_end.py --generate-audio --audio test_audio/test_spanish_greeting.wav

# Use custom configuration
python test_end_to_end.py --audio test.wav --config my_config.yaml
```

## Test Components

### 1. Audio Generation (`tests/generate_test_audio.py`)
Generates various test audio files:
- **Tones**: Pure sine waves, multiple tones, tones with noise
- **Spanish Speech**: Greetings, numbers, conversations, technical vocabulary
- **Edge Cases**: Very short, very quiet, mostly silence, clipped audio

### 2. Mock Components
- **`tests/mock_audio_processor.py`**: Simulates audio processing
- **`tests/mock_llm_analyzer.py`**: Simulates LLM analysis and conversation segmentation

### 3. Test Scripts
- **`test_transcription_integration.py`**: Tests transcription engine specifically
- **`test_end_to_end.py`**: Tests complete pipeline from audio to analysis

## Configuration

### Test Configuration (`config.test.yaml`)
Optimized for testing with minimal resources:
- Uses `base` Whisper model (faster)
- CPU processing by default
- Diarization disabled by default
- Single-threaded processing

### Production Configuration
For production testing, modify:
```yaml
processing:
  whisper_model: "large-v3"  # Better accuracy
  whisper_device: "cuda"      # GPU acceleration
  diarization:
    enabled: true             # Speaker identification
  batch:
    max_workers: 4            # Parallel processing
```

## Testing Scenarios

### 1. Basic Functionality Test
```bash
# Generate simple test audio
python tests/generate_test_audio.py --types speech

# Run transcription
python test_transcription_integration.py --audio test_audio/test_spanish_greeting.wav
```

### 2. Performance Test
```bash
# Test with longer audio
python test_end_to_end.py --audio test_audio/test_spanish_long.wav

# Check real-time factor in results
```

### 3. Speaker Diarization Test
Enable diarization in config:
```yaml
processing:
  diarization:
    enabled: true
```

Then test with multi-speaker audio:
```bash
python test_transcription_integration.py --audio test_audio/test_spanish_conversation.wav
```

### 4. Error Handling Test
Test with edge cases:
```bash
# Very short audio
python test_end_to_end.py --audio test_audio/test_very_short.wav

# Mostly silence
python test_end_to_end.py --audio test_audio/test_mostly_silence.wav

# Very quiet
python test_end_to_end.py --audio test_audio/test_very_quiet.wav
```

### 5. Batch Processing Test
```bash
# Process multiple files
python test_transcription_integration.py --audio-dir test_audio/

# Check batch report in output
```

## Interpreting Results

### Transcription Quality Metrics
- **Confidence Score**: Should be > 0.7 for good audio
- **Real-time Factor**: < 1.0 means faster than real-time
- **Speaker Count**: Should match actual speakers

### Performance Benchmarks
On typical hardware:
- **CPU (base model)**: 0.3-0.5x real-time
- **CPU (large-v3)**: 2-5x real-time  
- **GPU (large-v3)**: 0.1-0.3x real-time

### Common Issues

#### 1. Model Download
Models auto-download on first use. Ensure internet connection and sufficient disk space:
- `base`: ~140MB
- `large-v3`: ~1.5GB

#### 2. PyAnnote Authentication
For speaker diarization:
1. Create HuggingFace account
2. Get token from https://huggingface.co/settings/tokens
3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Set token in config or environment

#### 3. Memory Issues
If GPU runs out of memory:
- The fallback will automatically switch to CPU
- Or manually set `whisper_device: "cpu"` in config
- Use smaller model (`base` or `small`)

#### 4. Audio Format Issues
Ensure audio files are:
- Supported format (WAV, MP3, M4A)
- Not corrupted
- Reasonable duration (0.5s - 2 hours)

## Advanced Testing

### Custom Audio Files
Requirements for custom test audio:
- **Language**: Spanish preferred
- **Quality**: Clear speech, minimal background noise
- **Format**: 16kHz mono WAV recommended
- **Duration**: 30 seconds to 5 minutes for quick tests

### Integration with Real Components
To test with actual components instead of mocks:

1. Implement the audio processor module
2. Implement the LLM analyzer module
3. Update imports in `test_end_to_end.py`
4. Ensure all dependencies are configured

### Continuous Integration
Example GitHub Actions workflow:
```yaml
name: Test Transcription
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements-transcription.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          python tests/generate_test_audio.py --types tones
          pytest tests/
```

## Troubleshooting

### Enable Debug Logging
Set in config:
```yaml
logging:
  level: "DEBUG"
```

### Check Component Status
```python
# In Python
from src.transcription import TranscriptionEngine
engine = TranscriptionEngine(config)
print(f"Model loaded: {engine.model is not None}")
print(f"Device: {engine.device}")
```

### Validate Configuration
```python
from src.config.config_manager import ConfigManager
config = ConfigManager("config.test.yaml")
config.validate_config(config.to_dict())
```

## Next Steps

1. Run basic tests to ensure setup works
2. Test with your own Spanish audio files
3. Tune configuration for your use case
4. Implement missing components (audio processor, LLM analyzer)
5. Set up automated testing in CI/CD pipeline