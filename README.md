# TranscriMatic

An automated Python system that detects Sony ICD-TX660 voice recorders connected via USB, transfers audio files, transcribes Spanish conversations with speaker identification, and generates AI-powered summaries with actionable insights.

## Features

- **Automatic USB Detection**: Monitors and identifies specific Sony recorder models
- **Smart File Transfer**: Transfers only new audio files to avoid duplicates
- **Local Processing**: All transcription and analysis happens locally for privacy
- **Speaker Diarization**: Identifies and differentiates between speakers
- **Conversation Segmentation**: Automatically splits recordings into separate conversations
- **AI-Powered Analysis**: Generates summaries, tasks, and follow-ups using local LLMs
- **Cross-Platform**: Supports both macOS and Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TranscriMatic.git
cd TranscriMatic
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Spanish language model for spaCy:
```bash
python -m spacy download es_core_news_lg
```

5. Configure the system:
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## Configuration

Edit `config.yaml` to set:
- USB device identifier
- File paths for audio, transcriptions, and summaries
- Whisper model size (larger = more accurate but slower)
- LLM provider settings (Ollama, Gemini, or OpenAI)
- Processing parameters

## Usage

Run the system:
```bash
python main.py
```

Run with custom config:
```bash
python main.py --config /path/to/config.yaml
```

Run in daemon mode:
```bash
python main.py --daemon
```

## Output Structure

```
~/transcrimatic/
├── audio_recordings/     # Transferred audio files
├── transcriptions/       # Raw transcription outputs
└── summaries/           # Processed summaries and tasks
    └── 2024-01-15/
        ├── conversation_001.md
        ├── conversation_002.md
        ├── daily_summary.md
        └── tasks_and_followups.md
```

## System Requirements

- Python 3.8+
- 8GB+ RAM (for LLM models)
- GPU recommended for faster transcription (CUDA-compatible)
- Ollama installed for local LLM (recommended)

## Module Development

See the `module-instructions/` folder for detailed development instructions for each module:
- Config Manager
- Device Monitor
- File Transfer
- Audio Processor
- Transcription Engine
- Conversation Segmenter
- LLM Analyzer
- Output Formatter
- Main Controller

## License

[Your License Here]

## Contributing

Contributions are welcome! Please read the development instructions in the `module-instructions/` folder before contributing.