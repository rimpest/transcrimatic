# Project Architecture

## System Overview

The system follows a modular pipeline architecture where each module has specific responsibilities and well-defined interfaces. The [[ main_controller ]] orchestrates the workflow, and modules communicate through standardized data structures.

## Module Relationships Diagram

```
┌─────────────────┐
│ main_controller │
└────────┬────────┘
         │ Orchestrates
         ▼
┌─────────────────┐     ┌──────────────┐
│ config_manager  │────▶│ All Modules  │
└─────────────────┘     └──────────────┘
         │ Provides Config
         ▼
┌─────────────────┐
│ device_monitor  │
└────────┬────────┘
         │ Triggers
         ▼
┌─────────────────┐
│ file_transfer   │
└────────┬────────┘
         │ Audio Files
         ▼
┌─────────────────┐
│ audio_processor │
└────────┬────────┘
         │ Processed Audio
         ▼
┌──────────────────────┐
│ transcription_engine │
└────────┬─────────────┘
         │ Transcribed Text
         ▼
┌─────────────────────────┐
│ conversation_segmenter  │
└────────┬────────────────┘
         │ Segmented Conversations
         ▼
┌─────────────────┐
│ llm_analyzer    │
└────────┬────────┘
         │ Analysis Results
         ▼
┌──────────────────┐
│ output_formatter │
└──────────────────┘
```

## Module Interfaces

### Common Data Structures

```python
@dataclass
class AudioFile:
    path: Path
    filename: str
    timestamp: datetime
    size: int
    duration: Optional[float]

@dataclass
class Transcription:
    audio_file: AudioFile
    text: str
    segments: List[TranscriptionSegment]
    language: str
    confidence: float

@dataclass
class TranscriptionSegment:
    start_time: float
    end_time: float
    text: str
    speaker_id: Optional[str]

@dataclass
class Conversation:
    id: str
    transcription_segments: List[TranscriptionSegment]
    start_time: float
    end_time: float
    participants: List[str]

@dataclass
class Analysis:
    conversation: Conversation
    summary: str
    tasks: List[str]
    todos: List[str]
    followups: List[str]
```

## Module Dependencies

### Direct Dependencies
- **[[ config_manager ]]** → Used by all modules
- **[[ device_monitor ]]** → Triggers [[ file_transfer ]]
- **[[ file_transfer ]]** → Provides files to [[ audio_processor ]]
- **[[ audio_processor ]]** → Feeds [[ transcription_engine ]]
- **[[ transcription_engine ]]** → Outputs to [[ conversation_segmenter ]]
- **[[ conversation_segmenter ]]** → Provides segments to [[ llm_analyzer ]]
- **[[ llm_analyzer ]]** → Sends results to [[ output_formatter ]]
- **[[ main_controller ]]** → Orchestrates all modules

### Configuration Flow
All modules receive configuration from [[ config_manager ]], which loads and validates settings from the configuration file.

## Error Handling Strategy

Each module implements:
1. **Input validation**: Verify incoming data meets interface requirements
2. **Exception handling**: Catch and log module-specific errors
3. **Graceful degradation**: Continue processing other files if one fails
4. **Error propagation**: Return error status to [[ main_controller ]]

## Communication Patterns

### Synchronous Pipeline
- Each module processes data and passes results to the next
- Blocking operations ensure sequential processing
- Suitable for the linear workflow nature

### Event System
- [[ device_monitor ]] emits device connection events
- [[ main_controller ]] subscribes to events and initiates pipeline
- Allows for future expansion (e.g., progress notifications)

## Configuration Schema

```yaml
device:
  identifier: "Sony_ICD-TX660"
  vendor_id: "054c"
  product_id: "xxxx"

paths:
  audio_destination: "/path/to/audio/files"
  transcriptions: "/path/to/transcriptions"
  summaries: "/path/to/summaries"

processing:
  language: "es"
  whisper_model: "large-v3"
  llm_model: "llama3:8b"
  
features:
  speaker_diarization: true
  conversation_segmentation: true
  auto_cleanup: false
```

## Logging Architecture

- Centralized logging configuration in [[ main_controller ]]
- Module-specific loggers with hierarchical naming
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Rotating file handler with configurable retention

## Testing Strategy

- Unit tests for each module's core functionality
- Integration tests for module interactions
- Mock objects for external dependencies (USB, LLM)
- Test fixtures for audio files and transcriptions