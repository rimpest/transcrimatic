# TranscriMatic Main CLI Usage Guide

The main.py file provides a comprehensive command-line interface for TranscriMatic with support for processing individual files, USB device scanning, and interactive configuration.

## Quick Start

### 1. First Run - Interactive Setup
```bash
python main.py --setup-config
```
This launches an interactive wizard to create your configuration file.

### 2. Process a Single File
```bash
python main.py --file audio.mp3
```

### 3. Process Files from USB Devices
```bash
python main.py --usb
```

### 4. Test Your Setup
```bash
python main.py --test
```

## Command Reference

### Configuration Commands

#### `--setup-config`
Launch interactive configuration wizard.
```bash
python main.py --setup-config
```

**What it does:**
- Guides you through device setup
- Configures transcription settings (model, language, etc.)
- Sets up speaker diarization (optional)
- Configures output directories
- Saves configuration to `config.json`

#### `--config <file>`
Specify custom configuration file (default: `config.json`).
```bash
python main.py --config my_config.json --file audio.mp3
```

### Processing Commands

#### `--file <audio_file>`
Process a single audio file.
```bash
python main.py --file personal_samples/sample_audio.mp3
```

**Output:**
- Shows real-time processing progress
- Displays transcription metrics (confidence, speakers, timing)
- Saves results to configured output directory
- Shows preview of transcribed text

#### `--usb`
Scan and process files from USB devices.
```bash
python main.py --usb
```

**What happens:**
1. Scans for mounted USB devices (`/media`, `/mnt`, `/Volumes`)
2. Shows detected devices
3. Lets you select device(s) to process
4. Searches for audio files (.mp3, .wav, .m4a, .flac, .ogg)
5. Processes each file through the complete pipeline
6. Optionally deletes processed files if configured

**Interactive Selection:**
```
📱 Found 2 USB device(s):
  1. /media/usb-recorder
  2. /media/backup-drive

Select device (1-2) or 'a' for all: 1
```

#### `--daemon`
Monitor USB devices continuously (planned feature).
```bash
python main.py --daemon
```
*Currently shows warning that daemon mode is not yet implemented.*

### Utility Commands

#### `--test`
Test configuration and dependencies.
```bash
python main.py --test
```

**Validates:**
- Configuration file is valid
- All required dependencies are installed
- Transcription engine can initialize
- Models can be loaded

#### `--log-level <level>`
Set logging verbosity.
```bash
python main.py --file audio.mp3 --log-level DEBUG
```

**Options:** DEBUG, INFO, WARNING, ERROR

#### `--version`
Show version information.
```bash
python main.py --version
```

## Configuration Wizard

The interactive configuration wizard (`--setup-config`) guides you through:

### 📱 Device Configuration
```
Device identifier (e.g., 'my_recorder'): my_audio_recorder
```

### 📁 Paths Configuration
```
Audio destination directory [./audio]: ./recordings
Transcriptions directory [./output/transcriptions]: ./output/transcripts
Summaries directory [./output/summaries]: ./output/summaries
```

### 🎙️ Transcription Configuration
```
Whisper model options: tiny, base, small, medium, large-v2, large-v3
Whisper model [base]: large-v3

Device options: auto, cpu, cuda
Processing device [auto]: cuda

Language code (e.g., 'es' for Spanish, 'en' for English)
Language [es]: es
```

### 👥 Speaker Diarization
```
Speaker diarization identifies different speakers in conversations.
Enable speaker diarization? (y/N): y

⚠️  Speaker diarization requires a HuggingFace token.
   1. Create account at https://huggingface.co
   2. Get token from https://huggingface.co/settings/tokens
   3. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1

HuggingFace token (or press Enter to configure later): hf_xxx...
```

### 📄 Output Configuration
```
Output format options: json, txt, md
Output format [json]: json
```

### 🤖 LLM Analysis
```
LLM providers: ollama, openai, gemini
LLM provider [ollama]: ollama
```

## Example Workflows

### Workflow 1: First-Time Setup
```bash
# 1. Run setup wizard
python main.py --setup-config

# 2. Test the configuration
python main.py --test

# 3. Process a test file
python main.py --file test_audio.mp3
```

### Workflow 2: USB Recording Device
```bash
# 1. Connect your USB audio recorder
# 2. Scan and process files
python main.py --usb

# 3. Select the device when prompted
# 4. Files are automatically processed and saved
```

### Workflow 3: Batch Processing
```bash
# Process multiple files from a directory
python main.py --usb  # Select device with audio files

# Or process individual files
python main.py --file recording1.mp3
python main.py --file recording2.mp3
```

### Workflow 4: Development/Testing
```bash
# Use debug logging
python main.py --file audio.mp3 --log-level DEBUG

# Test with different configurations
python main.py --config test_config.json --file audio.mp3

# Validate setup
python main.py --test
```

## Output Examples

### Single File Processing
```
🎙️  TranscriMatic - Audio Transcription & Analysis System
============================================================
✅ Configuration loaded: config.json

🎵 Processing: sample_audio.mp3
--------------------------------------------------
✓ Audio processed in 0.05s
✓ Transcribed in 45.23s
  - Language: es
  - Confidence: 72.8%
  - Segments: 52
  - Speakers: 1
  - Real-time factor: 1.23x
✓ Segmented into 1 conversation parts
✓ LLM analysis completed for 1 segments
✅ Results saved to: output/transcriptions/sample_audio_transcription.json

📝 Transcription Preview:
Hola, y bienvenido a la familia de Provedores, ABA de Learn Behavioral...
```

### USB Device Processing
```
🔍 Scanning for USB devices...
📱 Found 1 USB device(s):
  1. /media/my-recorder

🎯 Using device: /media/my-recorder

🔍 Scanning /media/my-recorder for audio files...
🎵 Found 3 audio file(s)

[1] Processing: meeting_001.mp3
✓ Audio processed in 0.02s
✓ Transcribed in 89.45s
✅ Results saved to: output/transcriptions/meeting_001_transcription.json

[2] Processing: interview_002.mp3
...

📊 Summary: 3/3 files processed successfully
```

## Configuration File Format

The wizard creates a `config.json` file:

```json
{
  "device": {
    "identifier": "my_recorder"
  },
  "paths": {
    "audio_destination": "./audio",
    "transcriptions": "./output/transcriptions",
    "summaries": "./output/summaries"
  },
  "processing": {
    "whisper_model": "base",
    "whisper_device": "auto",
    "language": "es",
    "faster_whisper": {
      "beam_size": 5,
      "best_of": 5,
      "temperature": 0.0,
      "vad_filter": true
    },
    "diarization": {
      "enabled": false,
      "min_speakers": 1,
      "max_speakers": 10
    }
  },
  "llm": {
    "provider": "ollama",
    "model": "llama2"
  },
  "output": {
    "format": "json",
    "transcriptions_dir": "output/transcriptions"
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Troubleshooting

### Config File Issues
```bash
# Recreate configuration
python main.py --setup-config

# Test configuration
python main.py --test
```

### Missing Dependencies
```bash
# Install transcription dependencies
pip install -r requirements-transcription.txt

# Test installation
python main.py --test
```

### USB Device Not Detected
- Check if device is mounted
- Try different USB port
- On Linux: check `/media/` and `/mnt/`
- On macOS: check `/Volumes/`
- On Windows: check for removable drives

### Processing Errors
```bash
# Use debug logging to see details
python main.py --file audio.mp3 --log-level DEBUG

# Check audio file format
file audio.mp3  # Should show audio format

# Try with different audio file
python main.py --test  # Validate setup first
```

## Advanced Usage

### Custom Configuration
```bash
# Use production settings
python main.py --config config.production.json --usb

# Use test settings with diarization
python main.py --config config.diarization.yaml --file interview.mp3
```

### Integration with Scripts
```bash
#!/bin/bash
# Process all files in a directory
for file in recordings/*.mp3; do
    python main.py --file "$file"
done
```

### Monitoring Processing
```bash
# Watch log file
tail -f transcrimatic.log

# Monitor with timestamps
python main.py --file audio.mp3 --log-level INFO | ts
```

The main.py interface provides a complete, user-friendly way to run TranscriMatic for both individual files and batch processing from USB devices.