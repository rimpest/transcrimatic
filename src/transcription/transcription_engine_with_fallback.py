"""Enhanced transcription engine with robust error handling and fallback strategies."""

import gc
import time
from typing import Optional

import torch

from .transcription_engine import TranscriptionEngine
from .data_classes import Transcription
from ..audio.data_classes import ProcessedAudio


class TranscriptionEngineWithFallback(TranscriptionEngine):
    """Transcription engine with automatic fallback strategies for robustness."""
    
    def transcribe_with_fallback(self, processed_audio: ProcessedAudio) -> Transcription:
        """Transcribe with automatic fallback on errors."""
        try:
            # Try with configured settings
            return self.transcribe(processed_audio)
            
        except torch.cuda.OutOfMemoryError:
            self.logger.warning("GPU out of memory, falling back to CPU")
            return self._fallback_cpu_transcription(processed_audio)
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            self.logger.warning("Attempting fallback with smaller model")
            return self._fallback_basic_transcription(processed_audio)
    
    def _fallback_cpu_transcription(self, processed_audio: ProcessedAudio) -> Transcription:
        """Fallback to CPU transcription."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Reload model on CPU
            self.load_model(self.model_size, device="cpu")
            
            # Disable diarization to save memory
            diarization_enabled = self.config.get("processing.diarization.enabled", True)
            self.config.set("processing.diarization.enabled", False)
            
            result = self.transcribe(processed_audio)
            
            # Restore diarization setting
            self.config.set("processing.diarization.enabled", diarization_enabled)
            
            return result
            
        except Exception as e:
            self.logger.error(f"CPU fallback failed: {e}")
            return self._fallback_basic_transcription(processed_audio)
    
    def _fallback_basic_transcription(self, processed_audio: ProcessedAudio) -> Transcription:
        """Ultimate fallback with minimal model."""
        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Load smallest model
            self.load_model("base", device="cpu")
            
            # Disable all advanced features
            self.config.set("processing.diarization.enabled", False)
            self.config.set("processing.faster_whisper.beam_size", 1)
            self.config.set("processing.faster_whisper.best_of", 1)
            self.config.set("processing.faster_whisper.vad_filter", False)
            
            return self.transcribe(processed_audio)
            
        except Exception as e:
            self.logger.error(f"All transcription attempts failed: {e}")
            # Return empty transcription as last resort
            return self._create_empty_transcription(processed_audio)
    
    def _create_empty_transcription(self, processed_audio: ProcessedAudio) -> Transcription:
        """Create empty transcription when all else fails."""
        return Transcription(
            audio_file=processed_audio,
            segments=[],
            full_text="[Transcription failed - audio could not be processed]",
            speaker_transcript="[Transcription failed]",
            language="es",
            confidence=0.0,
            duration=processed_audio.metadata.duration,
            speaker_count=0,
            processing_time=0.0
        )
    
    def validate_audio_quality(self, processed_audio: ProcessedAudio) -> bool:
        """Validate audio quality before transcription."""
        metadata = processed_audio.metadata
        
        # Check silence ratio
        if metadata.silence_ratio > 0.95:
            self.logger.warning("Audio is mostly silent")
            return False
        
        # Check volume levels
        if metadata.avg_volume < -50:  # dBFS
            self.logger.warning("Audio volume is very low")
            return False
        
        # Check duration
        if metadata.duration < 0.5:
            self.logger.warning("Audio is too short")
            return False
        
        return True
    
    def transcribe_with_retry(self, processed_audio: ProcessedAudio, max_retries: int = 3) -> Transcription:
        """Transcribe with retry logic."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt}/{max_retries}")
                
                return self.transcribe_with_fallback(processed_audio)
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return self._create_empty_transcription(processed_audio)
                
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff