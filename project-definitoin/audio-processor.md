# Audio Processor Module

## Purpose
Processes audio files to prepare them for transcription. Handles format conversion, audio enhancement, noise reduction, and splitting long recordings into manageable chunks.

## Dependencies
- **External**: `pydub`, `librosa`, `numpy`, `scipy`, `soundfile`
- **Internal**: [[ config_manager ]]

## Interface

### Input
- Audio files from [[ file_transfer ]]
- Processing configuration from [[ config_manager ]]

### Output
- Processed audio files ready for transcription
- Audio metadata and characteristics

### Public Methods

```python
class AudioProcessor:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def process_audio(self, audio_file: AudioFile) -> ProcessedAudio:
        """Process single audio file for transcription"""
        
    def normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels"""
        
    def reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise reduction"""
        
    def split_audio(self, audio: AudioSegment, max_duration: int) -> List[AudioChunk]:
        """Split long audio into chunks"""
        
    def detect_silence(self, audio: AudioSegment) -> List[SilenceSegment]:
        """Detect silence periods for segmentation"""
        
    def enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """Enhance speech clarity"""
        
    def get_audio_info(self, file_path: Path) -> AudioMetadata:
        """Extract audio file metadata"""
```

## Data Structures

```python
@dataclass
class ProcessedAudio:
    original_file: AudioFile
    processed_path: Path
    chunks: List[AudioChunk]
    metadata: AudioMetadata
    processing_time: float

@dataclass
class AudioChunk:
    chunk_id: int
    file_path: Path
    start_time: float
    end_time: float
    duration: float
    
@dataclass
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int
    bitrate: int
    format: str
    avg_volume: float
    peak_volume: float
    silence_ratio: float

@dataclass
class SilenceSegment:
    start: float
    end: float
    duration: float
```

## Processing Pipeline

```python
def process_audio(self, audio_file: AudioFile) -> ProcessedAudio:
    # 1. Load audio
    audio = AudioSegment.from_file(audio_file.path)
    
    # 2. Convert to standard format
    audio = self._standardize_format(audio)
    
    # 3. Apply enhancements
    audio = self.normalize_audio(audio)
    audio = self.reduce_noise(audio)
    audio = self.enhance_speech(audio)
    
    # 4. Split if necessary
    chunks = self.split_audio(audio, max_duration=300)  # 5 minutes
    
    # 5. Save processed audio
    processed_paths = self._save_chunks(chunks)
    
    return ProcessedAudio(
        original_file=audio_file,
        processed_path=processed_paths[0],
        chunks=chunks,
        metadata=self.get_audio_info(audio_file.path)
    )
```

## Audio Processing Techniques

### Normalization
```python
def normalize_audio(self, audio: AudioSegment) -> AudioSegment:
    """Normalize to -20 dBFS"""
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)
```

### Noise Reduction
```python
def reduce_noise(self, audio: AudioSegment) -> AudioSegment:
    """Use spectral gating for noise reduction"""
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Apply noise reduction
    # 1. Estimate noise profile from quiet sections
    # 2. Apply spectral subtraction
    # 3. Smooth result
    
    return audio._spawn(samples.tobytes())
```

### Smart Chunking
```python
def split_audio(self, audio: AudioSegment, max_duration: int) -> List[AudioChunk]:
    """Split at silence points near max_duration"""
    silence_segments = self.detect_silence(audio)
    
    chunks = []
    current_start = 0
    
    for silence in silence_segments:
        if silence.start - current_start >= max_duration * 0.9:
            # Split at this silence
            chunks.append(self._create_chunk(
                audio[current_start:silence.start],
                current_start
            ))
            current_start = silence.end
    
    return chunks
```

## Configuration Options

```yaml
processing:
  output_format: "wav"  # Format for processed files
  sample_rate: 16000   # Hz, optimal for speech
  channels: 1          # Mono for speech
  
  normalization:
    target_dBFS: -20
    enabled: true
    
  noise_reduction:
    enabled: true
    noise_floor: -40  # dB
    
  chunking:
    max_duration: 300  # seconds
    min_silence: 0.5   # seconds
    silence_thresh: -40  # dB
    
  enhancement:
    high_pass_filter: 80  # Hz
    low_pass_filter: 8000  # Hz
    compressor: true
```

## Error Handling

### Common Issues
- **Corrupted Audio**: Gracefully handle unreadable files
- **Memory Overflow**: Process large files in streams
- **Format Issues**: Support various codecs
- **Processing Failures**: Log and skip problematic segments

### Fallback Strategies
```python
try:
    processed = self.process_audio(audio_file)
except AudioProcessingError:
    # Fall back to minimal processing
    processed = self._minimal_processing(audio_file)
except MemoryError:
    # Process in smaller chunks
    processed = self._stream_process(audio_file)
```

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Receives from: [[ file_transfer ]]
- Outputs to: [[ transcription_engine ]]

## Performance Optimization

1. **Streaming Processing**: For large files
2. **Parallel Chunk Processing**: Use multiprocessing
3. **GPU Acceleration**: For noise reduction algorithms
4. **Caching**: Store processed audio for reuse

## Quality Metrics

Track and log:
- Signal-to-noise ratio improvement
- Dynamic range compression
- Processing time per minute of audio
- File size reduction

## Testing Considerations

- Test with various audio qualities
- Test with different recording environments
- Verify chunk boundaries preserve sentences
- Test memory usage with large files
- Validate output format compatibility

## Future Enhancements

- AI-based noise reduction
- Speaker separation preprocessing
- Automatic gain control
- Echo cancellation
- Real-time processing mode