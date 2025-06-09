"""Transcription engine module for Spanish audio transcription with speaker diarization."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import re

import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from ..config.config_manager import ConfigManager
from ..audio.data_classes import ProcessedAudio, AudioChunk
from .data_classes import (
    Transcription, TranscriptionSegment, WhisperSegment,
    SpeakerSegment, WordTiming
)


class TranscriptionEngine:
    """Transcribes Spanish audio using faster-whisper with speaker diarization."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize transcription engine with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Model containers
        self.model: Optional[WhisperModel] = None
        self.diarization_pipeline: Optional[Pipeline] = None
        
        # Load configuration
        self.model_size = self.config.get("processing.whisper_model", "large-v3")
        self.device = self.config.get("processing.whisper_device", "auto")
        self.language = self.config.get("processing.language", "es")
        
        # Spanish-specific corrections
        self.spanish_corrections = {
            " avia ": " había ",
            " alla ": " haya ",
            " aller ": " ayer ",
            " aora ": " ahora ",
            " ise ": " hice ",
            " aber ": " haber ",
            " bamos ": " vamos ",
            " asta ": " hasta ",
            " acer ": " hacer ",
            " ablo ": " hablo ",
            " ablar ": " hablar ",
            " ay ": " hay ",
            " oi ": " hoy ",
        }
        
        # Initialize models
        self.load_model(self.model_size, self.device)
        if self.config.get("processing.diarization.enabled", True):
            self._load_diarization_pipeline()
    
    def load_model(self, model_size: str, device: str = "auto"):
        """Load faster-whisper model with optimizations."""
        try:
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
            
            self.logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_diarization_pipeline(self):
        """Load pyannote speaker diarization pipeline."""
        try:
            self.logger.info("Loading speaker diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.get("pyannote.auth_token", None)
            )
            
            # Configure pipeline
            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
            
            self.logger.info("Diarization pipeline loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load diarization pipeline: {e}")
            self.logger.warning("Continuing without speaker diarization")
            self.diarization_pipeline = None
    
    def transcribe(self, processed_audio: ProcessedAudio) -> Transcription:
        """Transcribe audio file with speaker diarization."""
        start_time = time.time()
        all_segments = []
        segment_id = 0
        
        try:
            for chunk in processed_audio.chunks:
                self.logger.info(f"Transcribing chunk {chunk.chunk_id + 1}/{len(processed_audio.chunks)}")
                
                # 1. Transcribe with faster-whisper
                whisper_segments = self._transcribe_chunk(chunk)
                
                # 2. Perform speaker diarization if enabled
                if self.config.get("processing.diarization.enabled", True) and self.diarization_pipeline:
                    speaker_segments = self.diarize_speakers(chunk.file_path)
                    
                    # 3. Merge transcription with speakers
                    chunk_segments = self.merge_transcription_diarization(
                        whisper_segments,
                        speaker_segments
                    )
                else:
                    # No diarization - assign all to single speaker
                    chunk_segments = self._segments_without_diarization(whisper_segments)
                
                # 4. Adjust timestamps for chunk offset and assign IDs
                for segment in chunk_segments:
                    segment.start_time += chunk.start_time
                    segment.end_time += chunk.start_time
                    segment.id = segment_id
                    segment_id += 1
                    
                all_segments.extend(chunk_segments)
            
            # 5. Create full text and apply post-processing
            full_text = " ".join(s.text for s in all_segments)
            full_text = self.post_process_spanish(full_text)
            
            # 6. Format speaker-aware transcript
            speaker_transcript = self.format_speaker_transcript(all_segments)
            
            # 7. Calculate overall confidence
            confidence = self._calculate_confidence(all_segments)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return Transcription(
                audio_file=processed_audio,
                segments=all_segments,
                full_text=full_text,
                speaker_transcript=speaker_transcript,
                language="es",
                confidence=confidence,
                duration=processed_audio.metadata.duration,
                speaker_count=len(set(s.speaker_id for s in all_segments)),
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_chunk(self, chunk: AudioChunk) -> List[WhisperSegment]:
        """Transcribe a single audio chunk with faster-whisper."""
        whisper_params = self.config.get("processing.faster_whisper", {})
        
        segments, info = self.model.transcribe(
            str(chunk.file_path),
            language=self.language,
            task="transcribe",
            beam_size=whisper_params.get("beam_size", 5),
            best_of=whisper_params.get("best_of", 5),
            patience=whisper_params.get("patience", 1.0),
            length_penalty=whisper_params.get("length_penalty", 1.0),
            temperature=whisper_params.get("temperature", 0.0),
            compression_ratio_threshold=whisper_params.get("compression_ratio_threshold", 2.4),
            log_prob_threshold=whisper_params.get("log_prob_threshold", -1.0),
            no_speech_threshold=whisper_params.get("no_speech_threshold", 0.6),
            condition_on_previous_text=whisper_params.get("condition_on_previous_text", True),
            vad_filter=whisper_params.get("vad_filter", True),
            vad_parameters=whisper_params.get("vad_options", {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            }),
            word_timestamps=True,
            initial_prompt="Transcripción de una conversación en español entre múltiples participantes."
        )
        
        # Convert generator to list
        whisper_segments = []
        for segment in segments:
            words = None
            if segment.words:
                words = [
                    WordTiming(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability
                    )
                    for w in segment.words
                ]
            
            whisper_segments.append(WhisperSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob
            ))
        
        self.logger.info(f"Transcribed {len(whisper_segments)} segments from chunk")
        return whisper_segments
    
    def diarize_speakers(self, audio_path: Path) -> List[SpeakerSegment]:
        """Identify different speakers in audio using pyannote."""
        if not self.diarization_pipeline:
            return []
        
        try:
            # Configure diarization parameters
            params = self.config.get("processing.diarization", {})
            
            diarization = self.diarization_pipeline(
                str(audio_path),
                min_speakers=params.get("min_speakers", 1),
                max_speakers=params.get("max_speakers", 10)
            )
            
            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.duration >= params.get("min_duration", 0.5):
                    segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                        confidence=1.0  # pyannote doesn't provide confidence
                    ))
            
            self.logger.info(f"Identified {len(set(s.speaker for s in segments))} speakers")
            return segments
            
        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            return []
    
    def merge_transcription_diarization(
        self,
        whisper_segments: List[WhisperSegment],
        speaker_segments: List[SpeakerSegment]
    ) -> List[TranscriptionSegment]:
        """Merge transcription with speaker information."""
        merged_segments = []
        
        for whisper_seg in whisper_segments:
            # Find the speaker segment with maximum overlap
            best_speaker = "SPEAKER_00"
            max_overlap = 0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(whisper_seg.start, speaker_seg.start)
                overlap_end = min(whisper_seg.end, speaker_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker_seg.speaker
            
            # Calculate confidence from whisper probabilities
            confidence = np.exp(whisper_seg.avg_logprob) * (1 - whisper_seg.no_speech_prob)
            
            merged_segments.append(TranscriptionSegment(
                id=0,  # Will be assigned later
                start_time=whisper_seg.start,
                end_time=whisper_seg.end,
                text=whisper_seg.text,
                speaker_id=best_speaker,
                confidence=float(confidence)
            ))
        
        return merged_segments
    
    def _segments_without_diarization(self, whisper_segments: List[WhisperSegment]) -> List[TranscriptionSegment]:
        """Convert whisper segments without speaker diarization."""
        segments = []
        
        for whisper_seg in whisper_segments:
            confidence = np.exp(whisper_seg.avg_logprob) * (1 - whisper_seg.no_speech_prob)
            
            segments.append(TranscriptionSegment(
                id=0,  # Will be assigned later
                start_time=whisper_seg.start,
                end_time=whisper_seg.end,
                text=whisper_seg.text,
                speaker_id="SPEAKER_00",
                confidence=float(confidence)
            ))
        
        return segments
    
    def format_speaker_transcript(self, segments: List[TranscriptionSegment]) -> str:
        """Format transcript with clear speaker labels for LLM processing."""
        transcript_lines = []
        current_speaker = None
        speaker_mapping = self._create_speaker_mapping(segments)
        
        for segment in segments:
            speaker_label = speaker_mapping.get(segment.speaker_id, segment.speaker_id)
            
            # Add speaker change marker
            if speaker_label != current_speaker:
                current_speaker = speaker_label
                transcript_lines.append(f"\n[{speaker_label}]:")
            
            # Add text
            transcript_lines.append(segment.text)
        
        return "\n".join(transcript_lines).strip()
    
    def _create_speaker_mapping(self, segments: List[TranscriptionSegment]) -> Dict[str, str]:
        """Create human-readable speaker labels in Spanish."""
        unique_speakers = sorted(set(s.speaker_id for s in segments))
        
        # Map to Hablante 1, Hablante 2, etc.
        return {
            speaker_id: f"Hablante {i+1}"
            for i, speaker_id in enumerate(unique_speakers)
        }
    
    def detect_language(self, audio_path: Path) -> Tuple[str, float]:
        """Detect audio language and confidence (verify Spanish)."""
        try:
            # Use a short sample for language detection
            _, info = self.model.transcribe(
                str(audio_path),
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                max_new_tokens=1  # Just detect language, don't transcribe
            )
            
            return info.language, info.language_probability
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "es", 0.0
    
    def post_process_spanish(self, text: str) -> str:
        """Apply Spanish-specific post-processing and corrections."""
        # Apply common corrections
        for wrong, correct in self.spanish_corrections.items():
            text = text.replace(wrong, correct)
        
        # Fix punctuation
        text = self._fix_spanish_punctuation(text)
        
        # Handle proper nouns and capitalization
        text = self._fix_capitalization(text)
        
        # Apply custom vocabulary if configured
        custom_vocab = self.config.get("processing.post_processing.custom_vocabulary", [])
        if custom_vocab:
            text = self._apply_custom_vocabulary(text, custom_vocab)
        
        return text.strip()
    
    def _fix_spanish_punctuation(self, text: str) -> str:
        """Fix Spanish-specific punctuation issues."""
        # Add space after punctuation if missing
        text = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ])', r'\1 \2', text)
        
        # Fix question marks (Spanish uses ¿?)
        text = re.sub(r'\s*\?\s*', '? ', text)
        
        # Fix exclamation marks (Spanish uses ¡!)
        text = re.sub(r'\s*!\s*', '! ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization for Spanish text."""
        # Capitalize first letter of sentences
        text = re.sub(r'(?:^|[.!?]\s+)([a-záéíóúñ])', lambda m: m.group(0).upper(), text)
        
        # Common Spanish proper nouns
        proper_nouns = ['España', 'Madrid', 'Barcelona', 'México', 'Argentina']
        for noun in proper_nouns:
            text = re.sub(rf'\b{noun.lower()}\b', noun, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_custom_vocabulary(self, text: str, custom_vocab: List[str]) -> str:
        """Apply custom vocabulary corrections."""
        for term in custom_vocab:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(term, text)
        
        return text
    
    def _calculate_confidence(self, segments: List[TranscriptionSegment]) -> float:
        """Calculate overall transcription confidence."""
        if not segments:
            return 0.0
        
        # Weighted average by segment duration
        total_duration = sum(s.end_time - s.start_time for s in segments)
        if total_duration == 0:
            return 0.0
        
        weighted_confidence = sum(
            s.confidence * (s.end_time - s.start_time)
            for s in segments
        )
        
        return weighted_confidence / total_duration