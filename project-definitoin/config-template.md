# Configuration Template

## Complete Configuration File Example

```yaml
# Device Configuration
device:
  identifier: "Sony_ICD-TX660"
  vendor_id: "054c"  # Sony vendor ID
  product_id: "0xxx"  # Replace with actual product ID
  serial_pattern: "^[A-Z0-9]{8}$"  # Optional serial validation

# Path Configuration
paths:
  audio_destination: "/home/user/audio_recordings"
  transcriptions: "/home/user/audio_transcriptions"
  summaries: "/home/user/audio_summaries"

# Audio Processing Configuration
processing:
  # Faster-whisper settings
  whisper_model: "large-v3"  # tiny, base, small, medium, large-v2, large-v3
  whisper_device: "auto"     # auto, cuda, cpu
  language: "es"
  
  faster_whisper:
    beam_size: 5
    best_of: 5
    temperature: 0.0
    vad_filter: true
    vad_options:
      min_silence_duration_ms: 500
      speech_pad_ms: 200
  
  # Audio preprocessing
  audio:
    output_format: "wav"
    sample_rate: 16000
    channels: 1
    normalization:
      target_dBFS: -20
      enabled: true
    noise_reduction:
      enabled: true
      noise_floor: -40
    chunking:
      max_duration: 300  # seconds
      min_silence: 0.5
      silence_thresh: -40
  
  # Speaker diarization
  diarization:
    enabled: true
    min_speakers: 1
    max_speakers: 10
    min_duration: 0.5

# LLM Configuration
llm:
  provider: "ollama"  # Primary: ollama, gemini, openai
  fallback_order: ["ollama", "gemini", "openai"]
  
  # Local Ollama
  ollama:
    enabled: true
    host: "localhost"
    port: 11434
    model: "llama3:8b-instruct"
    timeout: 120
    temperature: 0.3
    
  # Google Gemini
  gemini:
    enabled: false
    api_key: "${GEMINI_API_KEY}"  # Set environment variable
    model: "gemini-1.5-pro"
    temperature: 0.3
    max_tokens: 2000
    
  # OpenAI
  openai:
    enabled: false
    api_key: "${OPENAI_API_KEY}"  # Set environment variable
    organization: "${OPENAI_ORG}"
    model: "gpt-4-turbo-preview"
    temperature: 0.3
    max_tokens: 2000
  
  # Analysis settings
  extraction:
    task_keywords: ["hacer", "necesito", "hay que", "debes", "tarea", "acción"]
    todo_keywords: ["pendiente", "recordar", "no olvidar", "importante"]
    followup_keywords: ["seguimiento", "revisar", "volver a", "próxima vez"]

# Conversation Segmentation
segmentation:
  min_conversation_gap: 30  # seconds
  min_conversation_length: 60
  topic_similarity_threshold: 0.3
  boundaries:
    combine_threshold: 60
    confidence_threshold: 0.7

# File Transfer
transfer:
  supported_formats: ["mp3", "wav", "m4a", "opus", "flac"]
  min_file_size: 1024  # bytes
  max_file_size: 2147483648  # 2GB
  cleanup_device: false
  verify_transfer: true
  
history:
  database_path: "~/.audio_transfer/history.db"
  retention_days: 365

# Output Configuration
output:
  structure:
    transcriptions_dir: "transcriptions"
    summaries_dir: "summaries"
  formatting:
    date_format: "%Y-%m-%d"
    time_format: "%H:%M:%S"
  markdown:
    include_metadata: true
    include_navigation: true
    include_speaker_labels: true

# System Configuration
system:
  auto_start: true
  daemon_mode: false
  processing:
    max_concurrent: 2
    retry_attempts: 3
    retry_delay: 5
  monitoring:
    health_check_interval: 60
    status_file: "/tmp/audio_system_status.json"
  notifications:
    enabled: false
    email: "user@example.com"

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "audio_system.log"
  max_size: 10485760  # 10MB
  backup_count: 5
```

## Environment Variables

Create a `.env` file or export these variables:

```bash
# For Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG="your-org-id"  # Optional
```

## Minimal Configuration

For a basic setup with only local processing:

```yaml
device:
  identifier: "Sony_ICD-TX660"

paths:
  audio_destination: "~/audio_recordings"
  transcriptions: "~/audio_transcriptions"
  summaries: "~/audio_summaries"

processing:
  whisper_model: "base"  # Smaller model for testing
  language: "es"

llm:
  provider: "ollama"
  ollama:
    model: "llama3:8b"
```

## Notes

1. **API Keys**: Never commit API keys to version control. Use environment variables.
2. **Model Selection**: 
   - Whisper: Larger models are more accurate but slower
   - LLM: Ensure your selected model supports Spanish
3. **Hardware Requirements**:
   - GPU recommended for faster-whisper large models
   - Sufficient RAM for LLM models (8GB+ for Ollama)
4. **File Paths**: Use absolute paths or expand `~` properly in your code