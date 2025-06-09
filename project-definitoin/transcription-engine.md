# Transcription Engine Module

## Purpose
Transcribes processed audio files to text using faster-whisper for efficient Spanish language processing, with support for speaker diarization. Provides accurate timestamps, speaker labels, and confidence scores for each segment.

## Dependencies
- **External**: `faster-whisper`, `pyannote.audio`, `torch`, `numpy`
- **Internal**: [[ config_manager ]], [[ audio_processor ]]

## Interface

### Input
- Processed audio files from [[ audio_processor ]]
- Model configuration from [[ config_manager ]]

### Output
- Transcribed text with timestamps and speaker labels
- Confidence scores and metadata

### Public Methods

```python
class TranscriptionEngine:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def transcribe(self, processed_audio: ProcessedAudio) -> Transcription:
        """Transcribe audio file with speaker diarization"""
        
    def load_model(self, model_size: str, device: str = "auto"):
        """Load faster-whisper model"""
        
    def diarize_speakers(self, audio_path: Path) -> List[SpeakerSegment]:
        """Identify different speakers in audio"""
        
    def merge_transcription_diarization(
        self, 
        whisper_segments: List[WhisperSegment],
        speaker_segments: List[SpeakerSegment]
    ) -> List[TranscriptionSegment]:
        """Merge transcription with speaker information"""
        
    def format_speaker_transcript(self, segments: List[TranscriptionSegment]) -> str:
        """Format transcript with clear speaker labels for LLM processing"""
        
    def detect_language(self, audio_path: Path) -> tuple[str, float]:
        """Detect audio language and confidence (verify Spanish)"""
        
    def post_process_spanish(self, text: str) -> str:
        """Apply Spanish-specific post-processing"""
```

## Data Structures

```python
@dataclass
class Transcription:
    audio_file: ProcessedAudio
    segments: List[TranscriptionSegment]
    full_text: str
    speaker_transcript: str  # Formatted for LLM with speaker labels
    language: str
    confidence: float
    duration: float
    speaker_count: int

@dataclass
class TranscriptionSegment:
    id: int
    start_time: float
    end_time: float
    text: str
    speaker_id: str
    confidence: float
    
@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    words: Optional[List[WordTiming]]
    avg_logprob: float
    no_speech_prob: float

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: float

@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: float
```

## Transcription Pipeline

```python
def transcribe(self, processed_audio: ProcessedAudio) -> Transcription:
    all_segments = []
    
    for chunk in processed_audio.chunks:
        # 1. Run faster-whisper transcription
        segments, info = self.model.transcribe(
            str(chunk.file_path),
            language="es",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            ),
            initial_prompt="Transcripción de una conversación en español entre múltiples participantes."
        )
        
        # Convert to list for processing
        whisper_segments = list(segments)
        
        # 2. Run speaker diarization
        speaker_segments = self.diarize_speakers(chunk.file_path)
        
        # 3. Merge results
        chunk_segments = self.merge_transcription_diarization(
            whisper_segments,
            speaker_segments
        )
        
        # 4. Adjust timestamps for chunk offset
        for segment in chunk_segments:
            segment.start_time += chunk.start_time
            segment.end_time += chunk.start_time
            
        all_segments.extend(chunk_segments)
    
    # 5. Post-process Spanish text
    full_text = self._combine_segments(all_segments)
    full_text = self.post_process_spanish(full_text)
    
    # 6. Create speaker-aware transcript for LLM
    speaker_transcript = self.format_speaker_transcript(all_segments)
    
    return Transcription(
        audio_file=processed_audio,
        segments=all_segments,
        full_text=full_text,
        speaker_transcript=speaker_transcript,
        language="es",
        confidence=self._calculate_confidence(all_segments),
        duration=processed_audio.metadata.duration,
        speaker_count=len(set(s.speaker_id for s in all_segments))
    )
```

## Faster-Whisper Integration

```python
def load_model(self, model_size: str = "large-v3", device: str = "auto"):
    """Load faster-whisper model with optimizations"""
    from faster_whisper import WhisperModel
    
    # Determine device and compute type
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    self.logger.info(f"Loading faster-whisper model: {model_size} on {device}")
    
    self.model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=2
    )
```

## Speaker Diarization

```python
def diarize_speakers(self, audio_path: Path) -> List[SpeakerSegment]:
    """Use pyannote.audio for speaker diarization"""
    # Load pre-trained pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    
    # Run diarization
    diarization = pipeline(str(audio_path))
    
    # Convert to segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
            confidence=0.0  # pyannote doesn't provide confidence
        ))
    
    return segments
```

## Speaker-Aware Formatting

```python
def format_speaker_transcript(self, segments: List[TranscriptionSegment]) -> str:
    """Format transcript with clear speaker labels for LLM processing"""
    transcript_lines = []
    current_speaker = None
    speaker_mapping = self._create_speaker_mapping(segments)
    
    for segment in segments:
        speaker_label = speaker_mapping.get(segment.speaker_id, segment.speaker_id)
        
        # Add speaker change marker
        if speaker_label != current_speaker:
            current_speaker = speaker_label
            transcript_lines.append(f"\n[{speaker_label}]:")
        
        # Add timestamped text
        timestamp = self._format_timestamp(segment.start_time)
        transcript_lines.append(f"{segment.text}")
    
    return "\n".join(transcript_lines)

def _create_speaker_mapping(self, segments: List[TranscriptionSegment]) -> Dict[str, str]:
    """Create human-readable speaker labels"""
    unique_speakers = sorted(set(s.speaker_id for s in segments))
    
    # Map to Hablante 1, Hablante 2, etc.
    return {
        speaker_id: f"Hablante {i+1}"
        for i, speaker_id in enumerate(unique_speakers)
    }
```

### Post-processing
```python
def post_process_spanish(self, text: str) -> str:
    """Spanish-specific corrections"""
    # Fix common Whisper Spanish errors
    corrections = {
        " avia ": " había ",
        " alla ": " haya ",
        " aller ": " ayer ",
        # Add more corrections
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Fix punctuation
    text = self._fix_spanish_punctuation(text)
    
    return text
```

## Configuration

```yaml
processing:
  whisper_model: "large-v3"  # Options: tiny, base, small, medium, large-v2, large-v3
  whisper_device: "auto"     # Options: auto, cuda, cpu
  language: "es"
  
  faster_whisper:
    beam_size: 5
    best_of: 5
    patience: 1.0
    length_penalty: 1.0
    temperature: 0.0
    compression_ratio_threshold: 2.4
    log_prob_threshold: -1.0
    no_speech_threshold: 0.6
    condition_on_previous_text: true
    vad_filter: true
    vad_options:
      min_silence_duration_ms: 500
      speech_pad_ms: 200
    
  diarization:
    enabled: true
    min_speakers: 1
    max_speakers: 10
    min_duration: 0.5  # Minimum speaker segment duration
    
  post_processing:
    fix_punctuation: true
    spell_check: false
    custom_vocabulary: ["list", "of", "technical", "terms"]
```

## Error Handling

### Common Issues
- **Model Loading**: Handle missing model files
- **GPU Memory**: Fall back to CPU if needed
- **Audio Quality**: Handle poor quality gracefully
- **Language Mismatch**: Verify Spanish content

### Fallback Strategies
```python
try:
    # Try GPU transcription
    result = self._transcribe_gpu(audio)
except torch.cuda.OutOfMemoryError:
    # Fall back to CPU
    result = self._transcribe_cpu(audio)
except Exception as e:
    # Use simpler model
    self.load_model("base")
    result = self._transcribe_basic(audio)
```

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Receives from: [[ audio_processor ]]
- Outputs to: [[ conversation_segmenter ]]

## Performance Optimization

1. **Batch Processing**: Process multiple chunks simultaneously
2. **GPU Utilization**: Use CUDA when available
3. **Model Caching**: Keep model in memory between files
4. **VAD Pre-filtering**: Skip silent segments
5. **Streaming Mode**: For real-time applications

## Quality Metrics

- Word Error Rate (WER)
- Character Error Rate (CER)
- Speaker diarization accuracy
- Processing speed (real-time factor)
- Confidence score distribution

## Testing Considerations

- Test with various Spanish accents
- Test with technical vocabulary
- Test speaker diarization accuracy
- Verify timestamp precision
- Test with background noise

## Future Enhancements

- Custom vocabulary support
- Real-time transcription
- Multi-language support
- Emotion detection
- Custom Spanish model fine-tuning