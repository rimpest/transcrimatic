# Speaker Diarization Guide for TranscriMatic

## Overview

Speaker diarization is the process of identifying "who spoke when" in an audio recording. TranscriMatic uses **pyannote.audio** for speaker diarization, which can identify different speakers in conversations, interviews, meetings, etc.

## Why Speaker Diarization Was Not Working

In the test results, you noticed only 1 speaker was detected because:

1. **Diarization was disabled** in the test configuration (`config.test.yaml`):
   ```yaml
   processing:
     diarization:
       enabled: false  # This was the issue
   ```

2. **HuggingFace token not configured** - Speaker diarization requires authentication

## How Speaker Diarization Works in TranscriMatic

### 1. **Two-Stage Process**

The system uses a two-stage approach:

```
Audio → Whisper (transcription) → Text segments with timestamps
  ↓
Audio → PyAnnote (diarization) → Speaker segments with timestamps
  ↓
Merge by temporal overlap → Text + Speaker identification
```

### 2. **Implementation Details**

#### When Enabled:
```python
# 1. Transcribe with Whisper
whisper_segments = self._transcribe_chunk(chunk)

# 2. Identify speakers with PyAnnote
speaker_segments = self.diarize_speakers(chunk.file_path)

# 3. Merge results
chunk_segments = self.merge_transcription_diarization(
    whisper_segments,
    speaker_segments
)
```

#### When Disabled:
```python
# All text assigned to default "SPEAKER_00"
chunk_segments = self._segments_without_diarization(whisper_segments)
```

### 3. **Speaker Merging Algorithm**

For each text segment, the system:
1. Calculates temporal overlap with all speaker segments
2. Assigns the speaker with maximum overlap
3. Creates human-readable labels: "Hablante 1", "Hablante 2", etc.

## Enabling Speaker Diarization

### Step 1: Get HuggingFace Token

1. Create account at https://huggingface.co
2. Generate token: https://huggingface.co/settings/tokens
3. Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1

### Step 2: Configure Token

Option A - In configuration file:
```yaml
pyannote:
  auth_token: "hf_xxxxxxxxxxxxxxxxxxxx"
```

Option B - Environment variable:
```bash
export HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### Step 3: Enable in Configuration

```yaml
processing:
  diarization:
    enabled: true
    min_speakers: 2      # Expected minimum speakers
    max_speakers: 10     # Maximum to detect
    min_duration: 0.5    # Ignore segments < 0.5s
```

### Step 4: Install Dependencies

```bash
pip install pyannote.audio
```

## Testing Speaker Diarization

### Test with Diarization Enabled:

```bash
# Using the diarization config
python tests/test_transcription_integration.py \
    --audio personal_samples/sample_audio_long.mp3 \
    --config config.diarization.yaml
```

### Expected Results with Diarization:

```
=== Transcription Results ===
Speakers: 2  # Should show 2+ speakers for interview
...
=== Speaker Transcript ===
[Hablante 1]:
Hola a todos, ¿qué tal? Bienvenidos a un episodio más...

[Hablante 2]:
Bien y tú? ¿Qué gusto también estar?

[Hablante 1]:
Bien, también, todo en orden...
```

## Performance Considerations

### With Diarization Enabled:
- **Processing time**: Increases by 20-50%
- **Memory usage**: Additional ~1-2GB
- **GPU benefit**: Can accelerate diarization if available

### Typical Processing Times:
- Transcription only: ~0.15x real-time
- With diarization: ~0.25x real-time

## Configuration Options

```yaml
processing:
  diarization:
    enabled: true
    min_speakers: 1      # Minimum speakers to detect
    max_speakers: 10     # Maximum speakers to detect
    min_duration: 0.5    # Minimum segment duration (seconds)
    
    # Advanced options (future enhancement)
    # overlap_detection: true
    # clustering_threshold: 0.5
```

## Troubleshooting

### Issue: "No module named 'pyannote'"
```bash
pip install pyannote.audio
```

### Issue: "Authentication required"
- Ensure HuggingFace token is set
- Verify token has read permissions
- Check license acceptance

### Issue: Still shows 1 speaker
- Audio might be mono with mixed speakers
- Try adjusting min_speakers parameter
- Check audio quality

### Issue: Too many speakers detected
- Reduce max_speakers
- Increase min_duration to filter short segments

## Quality Tips

1. **Audio Quality**: Better audio = better diarization
   - Clear speech
   - Minimal overlapping speech
   - Good speaker separation

2. **Parameter Tuning**:
   - For interviews: `min_speakers: 2, max_speakers: 3`
   - For meetings: `min_speakers: 2, max_speakers: 10`
   - For podcasts: `min_speakers: 1, max_speakers: 5`

3. **Post-Processing**: 
   - Review speaker changes
   - Merge very short segments
   - Handle overlapping speech

## Example Output Comparison

### Without Diarization:
```
[Hablante 1]:
Hola, ¿cómo estás?
Bien, gracias. ¿Y tú?
Todo bien, gracias por preguntar.
```

### With Diarization:
```
[Hablante 1]:
Hola, ¿cómo estás?

[Hablante 2]:
Bien, gracias. ¿Y tú?

[Hablante 1]:
Todo bien, gracias por preguntar.
```

## Future Enhancements

The current implementation provides basic speaker diarization. Future improvements could include:

1. **Speaker embedding**: Save voice profiles for consistent speaker identification
2. **Overlap handling**: Better handling of simultaneous speech
3. **Gender detection**: Identify speaker gender
4. **Speaker naming**: Allow custom speaker names
5. **Confidence scores**: Per-speaker confidence metrics