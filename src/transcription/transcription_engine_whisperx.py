"""
WhisperX-based Transcription Engine for TranscriMatic

This module provides transcription using WhisperX, which includes:
- Better word-level timestamps
- Integrated speaker diarization
- Phoneme-based voice activity detection
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

import torch
import whisperx
import numpy as np

from src.config import ConfigManager
from .data_classes import (
    Transcription, TranscriptionSegment, WordTiming,
    TranscriptionError, AudioTooShortError
)
from ..audio.data_classes import ProcessedAudio, AudioChunk


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class TranscriptionEngineWhisperX:
    """WhisperX-based transcription engine with integrated diarization."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the WhisperX transcription engine."""
        self.config = config_manager
        self.logger = logger
        
        # Model configuration
        self.model_size = self.config.get("processing.whisper_model", "large-v3")
        self.device = self.config.get("processing.whisper_device", "auto")
        self.language = self.config.get("processing.language", "es")
        
        # WhisperX specific settings
        self.batch_size = self.config.get("processing.whisperx.batch_size", 16)
        self.compute_type = self.config.get("processing.whisperx.compute_type", "int8")
        
        # Diarization settings
        self.diarization_enabled = self.config.get("processing.diarization.enabled", True)
        self.min_speakers = self.config.get("processing.diarization.min_speakers", 1)
        self.max_speakers = self.config.get("processing.diarization.max_speakers", 10)
        
        # Initialize model
        self.model = None
        self.diarize_model = None
        self.align_model = None
        self.load_models()
    
    def load_models(self):
        """Load WhisperX models."""
        try:
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading WhisperX model: {self.model_size} on {self.device}")
            
            # Load WhisperX model
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            
            # Load alignment model for better timestamps
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device
            )
            
            # Load diarization model if enabled
            if self.diarization_enabled:
                self.logger.info("Loading diarization model...")
                # Get Hugging Face token from environment
                hf_token = os.getenv("HF_TOKEN") or self.config.get("huggingface.token")
                if not hf_token:
                    self.logger.warning("No HuggingFace token found, diarization may not work")
                
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device
                )
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise TranscriptionError(f"Model loading failed: {e}")
    
    def transcribe(self, processed_audio: ProcessedAudio) -> Transcription:
        """
        Transcribe processed audio using WhisperX.
        
        Args:
            processed_audio: ProcessedAudio object with chunks
            
        Returns:
            Transcription object with results
        """
        start_time = time.time()
        all_segments = []
        
        try:
            # Process each chunk
            for i, chunk in enumerate(processed_audio.chunks):
                self.logger.info(f"Transcribing chunk {i+1}/{len(processed_audio.chunks)}")
                
                # Load audio
                audio = whisperx.load_audio(str(chunk.file_path))
                
                # Transcribe with WhisperX
                result = self.model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=self.language
                )
                
                # Align whisper output
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                
                # Apply diarization if enabled
                if self.diarization_enabled and self.diarize_model:
                    self.logger.info(f"Applying speaker diarization to chunk {i+1}")
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Convert to our format with time offset
                chunk_segments = self._convert_segments(
                    result["segments"],
                    time_offset=chunk.start_time
                )
                all_segments.extend(chunk_segments)
            
            # Create transcription object
            processing_time = time.time() - start_time
            
            transcription = Transcription(
                segments=all_segments,
                language=self.language,
                duration=processed_audio.metadata.duration,
                processing_time=processing_time,
                model_info={
                    "engine": "whisperx",
                    "model": self.model_size,
                    "device": self.device,
                    "diarization": self.diarization_enabled
                }
            )
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            return transcription
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Transcription failed: {e}")
    
    def _convert_segments(self, whisperx_segments: List[Dict], 
                         time_offset: float = 0.0) -> List[TranscriptionSegment]:
        """Convert WhisperX segments to our format."""
        segments = []
        
        for seg in whisperx_segments:
            # Extract word timings if available
            word_timings = []
            if "words" in seg:
                for word in seg["words"]:
                    word_timings.append(WordTiming(
                        word=word["word"],
                        start=word["start"] + time_offset,
                        end=word["end"] + time_offset,
                        confidence=word.get("score", 1.0)
                    ))
            
            # Determine speaker
            speaker = None
            if "speaker" in seg:
                speaker = seg["speaker"]
            elif "words" in seg and seg["words"] and "speaker" in seg["words"][0]:
                # Get speaker from first word
                speaker = seg["words"][0]["speaker"]
            
            segment = TranscriptionSegment(
                start=seg["start"] + time_offset,
                end=seg["end"] + time_offset,
                text=seg["text"].strip(),
                speaker=speaker,
                confidence=seg.get("score", 1.0),
                word_timings=word_timings
            )
            segments.append(segment)
        
        return segments
    
    def cleanup(self):
        """Clean up models and free memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'align_model'):
            del self.align_model
        if hasattr(self, 'diarize_model'):
            del self.diarize_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()