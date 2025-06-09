# TranscriMatic Testing Framework

This directory contains the comprehensive testing suite for TranscriMatic, providing unit tests, integration tests, performance benchmarks, and testing utilities organized for maintainability and scalability.

## Quick Start

### Modern Test Runner (Recommended)

```bash
# Run all tests
python tests/run_tests.py

# Run specific test suites
python tests/run_tests.py unit          # Unit tests only
python tests/run_tests.py integration   # Integration tests only
python tests/run_tests.py performance   # Performance benchmarks
python tests/run_tests.py coverage --html  # Generate coverage report

# Clean test outputs
python tests/run_tests.py --clean

# Check test dependencies
python tests/run_tests.py --check-deps
```

### Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Pytest configuration and shared fixtures
├── test_utils.py                # Shared testing utilities and mock factories
├── run_tests.py                 # Comprehensive test runner script
├── test_outputs/                # Directory for test output files
│   ├── unit/                    # Unit test outputs
│   ├── integration/             # Integration test outputs  
│   ├── performance/             # Performance test outputs
│   ├── temp/                    # Temporary test files
│   └── fixtures/                # Generated test fixtures
├── test_*.py                    # Individual test modules
└── mock_*.py                    # Mock implementations for testing
```

### Legacy Test Methods

For backwards compatibility, you can still use the original test methods:

```bash
# Navigate to project root
cd /path/to/TranscriMatic

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-transcription.txt
```

### 2. Generate Test Audio (Optional)

```bash
# Generate sample test audio files
python tests/generate_test_audio.py

# This creates various test files in test_audio/ directory:
# - test_spanish_greeting.wav
# - test_spanish_numbers.wav
# - test_spanish_conversation.wav
# - test_tone_440hz.wav
# - test_very_quiet.wav
# - etc.
```

### 3. Run Tests

#### Basic Transcription Test
```bash
# Test with a single audio file
python tests/test_transcription_integration.py --audio path/to/audio.mp3

# Example with sample audio
python tests/test_transcription_integration.py --audio personal_samples/sample_audio.mp3
```

#### End-to-End Pipeline Test
```bash
# Test complete pipeline (audio → transcription → segmentation → analysis)
python tests/test_end_to_end.py --audio personal_samples/sample_audio.mp3

# Generate test audio and run pipeline
python tests/test_end_to_end.py --generate-audio --audio test_audio/test_spanish_greeting.wav
```

#### Unit Tests
```bash
# Run unit tests with pytest
python -m pytest tests/test_transcription_engine.py -v

# Run with coverage report
python -m pytest tests/test_transcription_engine.py --cov=src.transcription --cov-report=html
```

## Test Scripts Overview

### 1. `test_transcription_integration.py`
Tests the transcription engine specifically. Features:
- Single file transcription
- Batch transcription (directory of files)
- Model loading verification
- Performance metrics (processing time, real-time factor)
- Confidence scores and speaker identification

**Usage Examples:**
```bash
# Test model loading only
python tests/test_transcription_integration.py --test-loading

# Test single file with custom config
python tests/test_transcription_integration.py \
    --audio audio.mp3 \
    --config config.test.yaml \
    --log-level DEBUG

# Test batch processing
python tests/test_transcription_integration.py --audio-dir ./audio_files/
```

### 2. `test_end_to_end.py`
Tests the complete TranscriMatic pipeline. Features:
- Audio processing (mock)
- Transcription
- Conversation segmentation (mock)
- LLM analysis (mock)
- Performance breakdown by component
- Results saved to JSON

**Usage Examples:**
```bash
# Basic pipeline test
python tests/test_end_to_end.py --audio audio.mp3

# Save results to custom location
python tests/test_end_to_end.py \
    --audio audio.mp3 \
    --output my_results.json

# Use production config
python tests/test_end_to_end.py \
    --audio audio.mp3 \
    --config config.yaml
```

### 3. `test_transcription_engine.py`
Unit tests for the transcription engine. Tests:
- Model initialization
- Spanish post-processing
- Speaker mapping
- Confidence calculation
- Error handling and fallbacks

**Run with:**
```bash
python -m pytest tests/test_transcription_engine.py -v
```

### 4. `generate_test_audio.py`
Generates various test audio files:
- **Tones**: Pure sine waves, multiple frequencies
- **Spanish Speech**: Using gTTS (Google Text-to-Speech)
- **Edge Cases**: Very short, quiet, or mostly silent audio

**Usage:**
```bash
# Generate all test types
python tests/generate_test_audio.py

# Generate specific types
python tests/generate_test_audio.py --types speech
python tests/generate_test_audio.py --types tones edge

# Custom output directory
python tests/generate_test_audio.py --output-dir my_test_audio/
```

## Understanding Test Results

### Processing Time Metrics

When running transcription tests, you'll see detailed timing information:

```
=== Transcription Results ===
Language: es
Confidence: 72.81%
Audio Duration: 301.46s      # Actual length of audio
Processing Time: 45.23s      # Time taken to transcribe
Real-time Factor: 0.15x      # Processing time / Audio duration
Segments: 52
Speakers: 1
```

**Real-time Factor Interpretation:**
- `< 1.0x`: Faster than real-time (good!)
- `= 1.0x`: Real-time processing
- `> 1.0x`: Slower than real-time

**Typical Performance:**
- CPU (base model): 0.3-0.5x
- CPU (large-v3): 2-5x
- GPU (large-v3): 0.1-0.3x

### Output Files

Tests generate several output files:

1. **Transcription Results** (`transcription_result_*.json`):
   ```json
   {
     "audio_file": {...},
     "segments": [...],
     "full_text": "...",
     "speaker_transcript": "...",
     "processing_time": 45.23,
     "real_time_factor": 0.15
   }
   ```

2. **Pipeline Results** (`test_results.json`):
   ```json
   {
     "audio_processing": {...},
     "transcription": {...},
     "segmentation": {...},
     "analysis": {...},
     "performance": {...}
   }
   ```

3. **Test Logs** (`test_pipeline.log`, `test_transcription.log`)

## Mock Components

The test suite includes mock implementations for components not yet developed:

### `mock_audio_processor.py`
- Simulates audio file processing
- Creates ProcessedAudio objects
- Validates audio files
- Extracts metadata

### `mock_llm_analyzer.py`
- Simulates LLM analysis
- Generates mock summaries and key points
- Creates sentiment analysis
- Simulates conversation segmentation

## Configuration

### Test Configuration (`config.test.yaml`)
Optimized for testing:
```yaml
processing:
  whisper_model: "base"      # Faster model for testing
  whisper_device: "cpu"      # No GPU required
  diarization:
    enabled: false           # Disabled by default (requires auth)
```

### Production Testing
For production-like testing:
```yaml
processing:
  whisper_model: "large-v3"  # Best accuracy
  whisper_device: "cuda"     # GPU acceleration
  diarization:
    enabled: true            # Speaker identification
```

## Common Test Scenarios

### 1. Quick Functionality Test
```bash
# Generate simple test audio and transcribe
python tests/generate_test_audio.py --types speech
python tests/test_transcription_integration.py \
    --audio test_audio/test_spanish_greeting.wav
```

### 2. Performance Benchmark
```bash
# Test with longer audio
python tests/test_end_to_end.py \
    --audio test_audio/test_spanish_long.wav
# Check "Real-time factor" in output
```

### 3. Accuracy Test
```bash
# Use larger model for better accuracy
# Edit config.test.yaml: whisper_model: "large-v3"
python tests/test_transcription_integration.py \
    --audio personal_samples/sample_audio.mp3
```

### 4. Error Handling Test
```bash
# Test with problematic audio
python tests/test_transcription_integration.py \
    --audio test_audio/test_very_quiet.wav

python tests/test_transcription_integration.py \
    --audio test_audio/test_mostly_silence.wav
```

### 5. Memory/GPU Test
```bash
# Force GPU usage (if available)
# Edit config: whisper_device: "cuda"
python tests/test_transcription_integration.py \
    --audio audio.mp3 --log-level DEBUG
# Watch for automatic CPU fallback on OOM
```

## Troubleshooting

### Issue: Model Download Fails
```bash
# Models are auto-downloaded on first use
# Ensure internet connection and disk space:
# - base: ~140MB
# - large-v3: ~1.5GB
```

### Issue: Import Errors
```bash
# Ensure you're in project root
cd /path/to/TranscriMatic

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements-transcription.txt
```

### Issue: PyAnnote Authentication
For speaker diarization:
1. Get HuggingFace token
2. Set in config: `pyannote.auth_token: "your_token"`
3. Or set environment: `export HUGGINGFACE_TOKEN="your_token"`

### Issue: Slow Performance
- Use smaller model (`base` instead of `large-v3`)
- Enable GPU if available
- Disable diarization for testing
- Process shorter audio segments

## Continuous Integration

Example GitHub Actions workflow:
```yaml
name: Test Transcription
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements-transcription.txt
          pip install pytest pytest-cov
      - name: Generate test audio
        run: python tests/generate_test_audio.py --types tones
      - name: Run tests
        run: |
          pytest tests/test_transcription_engine.py
          python tests/test_transcription_integration.py --test-loading
```

## Next Steps

1. **Basic Testing**: Start with the quick functionality test
2. **Your Audio**: Test with your own Spanish audio files
3. **Performance Tuning**: Adjust config for your hardware
4. **Integration**: Replace mocks with real implementations
5. **CI/CD**: Set up automated testing

For more information, see the main [Testing Guide](../docs/testing_guide.md).