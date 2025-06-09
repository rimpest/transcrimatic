# Audio Recorder Auto-Import & Transcription System

## Project Description

An automated Python system that detects Sony ICD-TX660 voice recorders connected via USB, transfers audio files, transcribes Spanish conversations with speaker identification, and generates AI-powered summaries with actionable insights. The system operates without user intervention once configured.

## Key Features

- **Automatic USB Detection**: Monitors and identifies specific Sony recorder models
- **Smart File Transfer**: Transfers only new audio files to avoid duplicates
- **Local Processing**: All transcription and analysis happens locally for privacy
- **Speaker Diarization**: Identifies and differentiates between speakers
- **Conversation Segmentation**: Automatically splits recordings into separate conversations
- **AI-Powered Analysis**: Generates summaries, tasks, and follow-ups using local LLMs
- **Cross-Platform**: Supports both macOS and Linux

## System Modules

1. **[[ config_manager ]]**: Manages system configuration and validates settings
2. **[[ device_monitor ]]**: Detects and identifies USB devices
3. **[[ file_transfer ]]**: Handles audio file transfers and duplicate detection
4. **[[ audio_processor ]]**: Processes audio files and prepares them for transcription
5. **[[ transcription_engine ]]**: Transcribes audio to text with speaker identification
6. **[[ conversation_segmenter ]]**: Splits transcriptions into separate conversations
7. **[[ llm_analyzer ]]**: Analyzes conversations and generates summaries
8. **[[ output_formatter ]]**: Formats and saves all outputs in organized structure
9. **[[ main_controller ]]**: Orchestrates the entire workflow

## Technology Stack

- **Language**: Python 3.8+
- **USB Detection**: pyudev (Linux), pyobjc (macOS)
- **Audio Processing**: pydub, librosa
- **Transcription**: faster-whisper (local, optimized)
- **Speaker Diarization**: pyannote.audio
- **LLM Providers**: 
  - Local: Ollama
  - Cloud: Google Gemini, OpenAI
- **Configuration**: YAML/JSON
- **Logging**: Python logging module

## Output Structure

```
project_root/
├── audio_files/          # Transferred audio files
├── transcriptions/       # Raw transcription outputs
└── summaries/           # Processed summaries and tasks
    └── YYYY-MM-DD/
        ├── conversation_*.md
        ├── daily_summary.md
        └── tasks_followups.md
```

## Configuration Requirements

The system requires a configuration file with:
- USB device identifier
- Audio files destination folder
- Transcriptions output folder
- Summaries output folder
- LLM model selection
- Language settings (Spanish)

## Related Documentation

- **[[ project_architecture ]]**: Detailed system architecture and module relationships
- **Module Documentation**: Individual files for each module (see System Modules section)