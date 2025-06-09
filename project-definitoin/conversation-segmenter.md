# Conversation Segmenter Module

## Purpose
Analyzes transcribed text to identify and separate distinct conversations within a continuous recording. Uses speaker changes, topic shifts, time gaps, and contextual cues to determine conversation boundaries.

## Dependencies
- **External**: `numpy`, `scikit-learn`, `nltk`, `spacy`
- **Internal**: [[ config_manager ]], [[ transcription_engine ]]

## Interface

### Input
- Transcription with speaker labels from [[ transcription_engine ]]
- Segmentation parameters from [[ config_manager ]]

### Output
- List of separate conversations with metadata
- Conversation boundaries and confidence scores

### Public Methods

```python
class ConversationSegmenter:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def segment_conversations(self, transcription: Transcription) -> List[Conversation]:
        """Segment transcription into separate conversations"""
        
    def detect_topic_changes(self, segments: List[TranscriptionSegment]) -> List[TopicBoundary]:
        """Detect topic shifts using NLP"""
        
    def find_time_gaps(self, segments: List[TranscriptionSegment]) -> List[TimeGap]:
        """Find significant silence periods"""
        
    def analyze_speaker_patterns(self, segments: List[TranscriptionSegment]) -> SpeakerAnalysis:
        """Analyze speaker interaction patterns"""
        
    def merge_short_conversations(self, conversations: List[Conversation]) -> List[Conversation]:
        """Merge very short adjacent conversations"""
        
    def score_boundaries(self, boundaries: List[ConversationBoundary]) -> List[float]:
        """Score confidence of each boundary"""
```

## Data Structures

```python
@dataclass
class Conversation:
    id: str
    segments: List[TranscriptionSegment]
    start_time: float
    end_time: float
    duration: float
    speakers: List[str]
    speaker_transcript: str  # Formatted transcript with speaker labels
    topic_summary: Optional[str]
    boundary_confidence: float

@dataclass
class ConversationBoundary:
    index: int
    timestamp: float
    reason: str  # "time_gap", "topic_change", "speaker_pattern"
    confidence: float

@dataclass
class TopicBoundary:
    segment_index: int
    similarity_score: float
    keywords_before: List[str]
    keywords_after: List[str]

@dataclass
class TimeGap:
    start: float
    end: float
    duration: float

@dataclass
class SpeakerAnalysis:
    speaker_turns: Dict[str, int]
    interaction_matrix: np.ndarray
    dominant_speakers: List[str]
```

## Segmentation Algorithm

```python
def segment_conversations(self, transcription: Transcription) -> List[Conversation]:
    # 1. Find potential boundaries
    time_boundaries = self._find_time_boundaries(transcription.segments)
    topic_boundaries = self._find_topic_boundaries(transcription.segments)
    speaker_boundaries = self._find_speaker_boundaries(transcription.segments)
    
    # 2. Combine and score boundaries
    all_boundaries = self._combine_boundaries(
        time_boundaries, 
        topic_boundaries, 
        speaker_boundaries
    )
    
    # 3. Filter by confidence threshold
    final_boundaries = [b for b in all_boundaries if b.confidence > 0.7]
    
    # 4. Create conversations from boundaries
    conversations = self._create_conversations(
        transcription.segments, 
        final_boundaries
    )
    
    # 5. Post-process
    conversations = self.merge_short_conversations(conversations)
    
    return conversations
```

## Boundary Detection Methods

### Time-based Segmentation
```python
def _find_time_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
    boundaries = []
    
    for i in range(1, len(segments)):
        gap = segments[i].start_time - segments[i-1].end_time
        
        if gap > self.config.min_conversation_gap:  # e.g., 30 seconds
            boundaries.append(ConversationBoundary(
                index=i,
                timestamp=segments[i-1].end_time,
                reason="time_gap",
                confidence=min(1.0, gap / 60.0)  # Higher confidence for longer gaps
            ))
    
    return boundaries
```

### Topic-based Segmentation
```python
def _find_topic_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
    # Use sliding window for topic coherence
    window_size = 10  # segments
    boundaries = []
    
    for i in range(window_size, len(segments) - window_size):
        before_text = " ".join(s.text for s in segments[i-window_size:i])
        after_text = " ".join(s.text for s in segments[i:i+window_size])
        
        # Calculate semantic similarity
        similarity = self._calculate_similarity(before_text, after_text)
        
        if similarity < self.config.topic_similarity_threshold:
            boundaries.append(ConversationBoundary(
                index=i,
                timestamp=segments[i].start_time,
                reason="topic_change",
                confidence=1.0 - similarity
            ))
    
    return boundaries
```

### Speaker Pattern Analysis
```python
def _find_speaker_boundaries(self, segments: List[TranscriptionSegment]) -> List[ConversationBoundary]:
    # Detect changes in speaker participation
    boundaries = []
    window = 20  # segments
    
    for i in range(window, len(segments) - window):
        speakers_before = set(s.speaker_id for s in segments[i-window:i])
        speakers_after = set(s.speaker_id for s in segments[i:i+window])
        
        # Significant change in participants
        if len(speakers_before ^ speakers_after) > len(speakers_before) * 0.5:
            boundaries.append(ConversationBoundary(
                index=i,
                timestamp=segments[i].start_time,
                reason="speaker_pattern",
                confidence=0.8
            ))
    
    return boundaries
```

## Configuration

```yaml
segmentation:
  min_conversation_gap: 30  # seconds
  min_conversation_length: 60  # seconds
  topic_similarity_threshold: 0.3
  
  boundaries:
    combine_threshold: 60  # seconds - merge boundaries within this window
    confidence_threshold: 0.7
    
  topic_detection:
    method: "tfidf"  # or "embeddings"
    language_model: "es_core_news_lg"  # spaCy model
    
  post_processing:
    merge_short: true
    min_segments: 5
```

## Natural Language Processing

### Spanish Language Support
```python
def _setup_nlp(self):
    """Initialize Spanish NLP models"""
    self.nlp = spacy.load("es_core_news_lg")
    self.stop_words = set(stopwords.words('spanish'))
    
def _extract_keywords(self, text: str) -> List[str]:
    """Extract Spanish keywords"""
    doc = self.nlp(text)
    keywords = [
        token.lemma_ for token in doc
        if not token.is_stop and token.pos_ in ["NOUN", "VERB", "PROPN"]
    ]
    return keywords
```

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Receives from: [[ transcription_engine ]]
- Outputs to: [[ llm_analyzer ]]

## Quality Metrics

- Boundary precision and recall
- Average conversation length
- Topic coherence scores
- Speaker distribution per conversation

## Testing Considerations

- Test with various conversation types
- Test with overlapping speakers
- Verify handling of monologues
- Test boundary detection accuracy
- Validate Spanish NLP processing

## Future Enhancements

- Machine learning boundary detection
- Contextual understanding improvement
- Real-time segmentation
- Multi-language support
- Interactive boundary adjustment