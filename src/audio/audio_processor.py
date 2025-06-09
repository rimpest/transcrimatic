"""
Audio Processor Module for TranscriMatic

Processes audio files to prepare them for transcription, including format conversion,
audio enhancement, noise reduction, and chunking for long recordings.
"""

import os
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.io import wavfile

# Audio processing libraries
from pydub import AudioSegment
from pydub.silence import detect_silence, split_on_silence
import librosa
import soundfile as sf
import noisereduce as nr

from src.config import ConfigManager
from src.transfer.data_classes import AudioFile
from .data_classes import (
    ProcessedAudio, AudioChunk, AudioMetadata, SilenceSegment
)


logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass


class AudioProcessor:
    """
    Processes audio files for optimal transcription quality.
    Handles format conversion, normalization, noise reduction, and chunking.
    """
    
    # Supported input formats
    SUPPORTED_FORMATS = ['mp3', 'wav', 'wma', 'm4a', 'flac', 'ogg', 'opus']
    
    # Default processing parameters
    DEFAULT_SAMPLE_RATE = 16000  # Optimal for speech recognition
    DEFAULT_CHANNELS = 1  # Mono
    DEFAULT_TARGET_DBFS = -20.0
    DEFAULT_SILENCE_THRESH = -40  # dB
    DEFAULT_MIN_SILENCE_LEN = 500  # ms
    DEFAULT_MAX_CHUNK_DURATION = 300  # seconds (5 minutes)
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the AudioProcessor with configuration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.temp_dir = Path(tempfile.mkdtemp(prefix="transcrimatic_audio_"))
        
        # Load configuration
        self.sample_rate = self.config.get("processing.audio.sample_rate", self.DEFAULT_SAMPLE_RATE)
        self.channels = self.config.get("processing.audio.channels", self.DEFAULT_CHANNELS)
        self.output_format = self.config.get("processing.audio.output_format", "wav")
        
        # Normalization settings
        self.normalization_enabled = self.config.get("processing.audio.normalization.enabled", True)
        self.target_dbfs = self.config.get("processing.audio.normalization.target_dBFS", self.DEFAULT_TARGET_DBFS)
        
        # Noise reduction settings
        self.noise_reduction_enabled = self.config.get("processing.audio.noise_reduction.enabled", True)
        self.noise_floor = self.config.get("processing.audio.noise_reduction.noise_floor", -40)
        
        # Chunking settings
        self.max_chunk_duration = self.config.get("processing.audio.chunking.max_duration", self.DEFAULT_MAX_CHUNK_DURATION)
        self.min_silence_len = int(self.config.get("processing.audio.chunking.min_silence", 0.5) * 1000)  # Convert to ms
        self.silence_thresh = self.config.get("processing.audio.chunking.silence_thresh", self.DEFAULT_SILENCE_THRESH)
        
        # Enhancement settings
        self.high_pass_freq = self.config.get("processing.audio.enhancement.high_pass_filter", 80)
        self.low_pass_freq = self.config.get("processing.audio.enhancement.low_pass_filter", 8000)
        
        logger.info(f"AudioProcessor initialized with temp dir: {self.temp_dir}")
    
    def process_audio(self, audio_file: AudioFile) -> ProcessedAudio:
        """
        Process audio file for transcription.
        
        Args:
            audio_file: AudioFile object to process
            
        Returns:
            ProcessedAudio object with processed audio and metadata
            
        Raises:
            AudioProcessingError: If processing fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing audio file: {audio_file.destination_path}")
            
            # 1. Load audio file
            audio = self._load_audio(audio_file.destination_path)
            logger.info(f"Loaded audio: {audio.duration_seconds:.1f}s, {audio.frame_rate}Hz, {audio.channels} channels")
            
            # 2. Extract metadata before processing
            original_metadata = self.get_audio_info(audio_file.destination_path, audio)
            
            # 3. Standardize format
            audio = self._standardize_format(audio)
            
            # 4. Apply enhancements
            if self.normalization_enabled:
                audio = self.normalize_audio(audio)
            
            if self.noise_reduction_enabled:
                audio = self.reduce_noise(audio)
            
            audio = self.enhance_speech(audio)
            
            # 5. Split if necessary
            chunks = []
            if audio.duration_seconds > self.max_chunk_duration:
                logger.info(f"Audio duration ({audio.duration_seconds:.1f}s) exceeds max chunk duration ({self.max_chunk_duration}s)")
                chunk_segments = self.split_audio(audio, self.max_chunk_duration)
                chunks = self._save_chunks(chunk_segments, audio_file.destination_path.stem)
            else:
                # Save as single chunk
                processed_path = self._save_audio(audio, f"{audio_file.destination_path.stem}_processed")
                chunks = [AudioChunk(
                    chunk_id=0,
                    file_path=processed_path,
                    start_time=0.0,
                    end_time=audio.duration_seconds,
                    duration=audio.duration_seconds
                )]
            
            # 6. Update metadata with processed audio info
            processed_metadata = self._update_metadata(original_metadata, audio)
            
            processing_time = time.time() - start_time
            logger.info(f"Audio processing completed in {processing_time:.2f}s")
            
            return ProcessedAudio(
                original_file=audio_file,
                processed_path=chunks[0].file_path if chunks else audio_file.destination_path,
                chunks=chunks,
                metadata=processed_metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise AudioProcessingError(f"Failed to process audio: {e}") from e
    
    def normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Normalize audio levels to target dBFS.
        
        Args:
            audio: Audio segment to normalize
            
        Returns:
            Normalized audio segment
        """
        try:
            current_dbfs = audio.dBFS
            change_in_dbfs = self.target_dbfs - current_dbfs
            
            # Prevent clipping
            if audio.max_dBFS + change_in_dbfs > 0:
                change_in_dbfs = -audio.max_dBFS - 0.1
            
            normalized = audio.apply_gain(change_in_dbfs)
            logger.debug(f"Normalized audio from {current_dbfs:.1f} dBFS to {normalized.dBFS:.1f} dBFS")
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return audio
    
    def reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply noise reduction using spectral subtraction.
        
        Args:
            audio: Audio segment to process
            
        Returns:
            Audio with reduced noise
        """
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            
            # Normalize samples to [-1, 1]
            if audio.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0
            
            # Apply noise reduction using noisereduce library
            reduced_noise = nr.reduce_noise(
                y=samples,
                sr=audio.frame_rate,
                stationary=True,
                prop_decrease=0.8  # Moderate noise reduction
            )
            
            # Convert back to int16
            reduced_noise = np.clip(reduced_noise * 32768, -32768, 32767).astype(np.int16)
            
            # Create new AudioSegment
            reduced_audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,  # 16-bit
                channels=audio.channels
            )
            
            logger.debug("Noise reduction applied successfully")
            return reduced_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """
        Enhance speech clarity with filters.
        
        Args:
            audio: Audio segment to enhance
            
        Returns:
            Enhanced audio segment
        """
        try:
            # Convert to numpy array for processing
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            sample_rate = audio.frame_rate
            
            # Design filters
            nyquist = sample_rate / 2
            
            # High-pass filter (remove low-frequency rumble)
            if self.high_pass_freq > 0:
                high_pass = self.high_pass_freq / nyquist
                if high_pass < 1:
                    b_high, a_high = signal.butter(4, high_pass, btype='high')
                    samples = signal.filtfilt(b_high, a_high, samples)
            
            # Low-pass filter (remove high-frequency noise)
            if self.low_pass_freq > 0:
                low_pass = self.low_pass_freq / nyquist
                if low_pass < 1:
                    b_low, a_low = signal.butter(4, low_pass, btype='low')
                    samples = signal.filtfilt(b_low, a_low, samples)
            
            # Convert back to int16
            enhanced = np.clip(samples, -32768, 32767).astype(np.int16)
            
            # Create new AudioSegment
            enhanced_audio = AudioSegment(
                enhanced.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=audio.channels
            )
            
            logger.debug(f"Speech enhanced with filters: HP={self.high_pass_freq}Hz, LP={self.low_pass_freq}Hz")
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Speech enhancement failed: {e}")
            return audio
    
    def split_audio(self, audio: AudioSegment, max_duration: int) -> List[AudioSegment]:
        """
        Split audio into chunks at silence points.
        
        Args:
            audio: Audio segment to split
            max_duration: Maximum chunk duration in seconds
            
        Returns:
            List of audio chunks
        """
        max_duration_ms = max_duration * 1000
        
        # Try to split on silence first
        try:
            # Detect silence segments
            silence_segments = detect_silence(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh,
                seek_step=100
            )
            
            if silence_segments:
                return self._smart_split_on_silence(audio, silence_segments, max_duration_ms)
            
        except Exception as e:
            logger.warning(f"Silence-based splitting failed: {e}")
        
        # Fallback to time-based splitting
        return self._time_based_split(audio, max_duration_ms)
    
    def detect_silence(self, audio: AudioSegment) -> List[SilenceSegment]:
        """
        Detect silence periods in audio.
        
        Args:
            audio: Audio segment to analyze
            
        Returns:
            List of silence segments
        """
        try:
            silence_ranges = detect_silence(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh,
                seek_step=100
            )
            
            silence_segments = []
            for start_ms, end_ms in silence_ranges:
                silence_segments.append(SilenceSegment(
                    start_time=start_ms / 1000.0,
                    end_time=end_ms / 1000.0,
                    duration=(end_ms - start_ms) / 1000.0
                ))
            
            logger.debug(f"Detected {len(silence_segments)} silence segments")
            return silence_segments
            
        except Exception as e:
            logger.warning(f"Silence detection failed: {e}")
            return []
    
    def get_audio_info(self, file_path: Path, audio_segment: Optional[AudioSegment] = None) -> AudioMetadata:
        """
        Extract metadata from audio file.
        
        Args:
            file_path: Path to audio file
            audio_segment: Optional pre-loaded audio segment
            
        Returns:
            Audio metadata
        """
        try:
            if audio_segment is None:
                audio_segment = AudioSegment.from_file(file_path)
            
            # Calculate average and peak volume
            samples = np.array(audio_segment.get_array_of_samples())
            
            # RMS calculation for average volume
            rms = np.sqrt(np.mean(samples**2))
            avg_volume = 20 * np.log10(rms / 32768.0) if rms > 0 else -100.0
            
            # Peak volume
            peak = np.max(np.abs(samples))
            peak_volume = 20 * np.log10(peak / 32768.0) if peak > 0 else -100.0
            
            # Silence ratio
            silence_segments = self.detect_silence(audio_segment)
            total_silence = sum(seg.duration for seg in silence_segments)
            silence_ratio = total_silence / audio_segment.duration_seconds if audio_segment.duration_seconds > 0 else 0.0
            
            # Get file info
            file_stats = os.stat(file_path)
            bitrate = int(file_stats.st_size * 8 / audio_segment.duration_seconds) if audio_segment.duration_seconds > 0 else 0
            
            return AudioMetadata(
                duration=audio_segment.duration_seconds,
                sample_rate=audio_segment.frame_rate,
                channels=audio_segment.channels,
                bitrate=bitrate,
                format=file_path.suffix[1:],  # Remove dot
                avg_volume=avg_volume,
                peak_volume=peak_volume,
                silence_ratio=silence_ratio
            )
            
        except Exception as e:
            logger.error(f"Failed to extract audio metadata: {e}")
            # Return minimal metadata
            return AudioMetadata(
                duration=0.0,
                sample_rate=0,
                channels=0,
                bitrate=0,
                format=file_path.suffix[1:],
                avg_volume=-100.0,
                peak_volume=-100.0,
                silence_ratio=0.0
            )
    
    def _load_audio(self, file_path: Path) -> AudioSegment:
        """Load audio file with error handling."""
        if not file_path.exists():
            raise AudioProcessingError(f"Audio file not found: {file_path}")
        
        file_format = file_path.suffix[1:].lower()
        if file_format not in self.SUPPORTED_FORMATS:
            raise AudioProcessingError(f"Unsupported audio format: {file_format}")
        
        try:
            return AudioSegment.from_file(file_path, format=file_format)
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio file: {e}")
    
    def _standardize_format(self, audio: AudioSegment) -> AudioSegment:
        """Convert audio to standard format for processing."""
        # Convert to mono if needed
        if audio.channels > 1 and self.channels == 1:
            audio = audio.set_channels(1)
            logger.debug("Converted to mono")
        
        # Resample if needed
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
            logger.debug(f"Resampled to {self.sample_rate}Hz")
        
        # Convert to 16-bit if needed
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
            logger.debug("Converted to 16-bit")
        
        return audio
    
    def _smart_split_on_silence(self, audio: AudioSegment, silence_segments: List[Tuple[int, int]], 
                               max_duration_ms: int) -> List[AudioSegment]:
        """Split audio intelligently at silence points."""
        chunks = []
        current_start = 0
        
        for i, (silence_start, silence_end) in enumerate(silence_segments):
            chunk_duration = silence_start - current_start
            
            # Check if we should split here
            if chunk_duration >= max_duration_ms * 0.8:  # 80% of max duration
                # Include half of the silence in the chunk
                chunk_end = silence_start + (silence_end - silence_start) // 2
                chunks.append(audio[current_start:chunk_end])
                current_start = chunk_end
            
            # Force split if we're way over the limit
            elif chunk_duration > max_duration_ms * 1.2:  # 120% of max duration
                chunks.append(audio[current_start:silence_start])
                current_start = silence_end
        
        # Add final chunk
        if current_start < len(audio):
            chunks.append(audio[current_start:])
        
        logger.info(f"Split audio into {len(chunks)} chunks based on silence")
        return chunks
    
    def _time_based_split(self, audio: AudioSegment, max_duration_ms: int) -> List[AudioSegment]:
        """Split audio based on time intervals."""
        chunks = []
        
        for i in range(0, len(audio), max_duration_ms):
            chunk = audio[i:i + max_duration_ms]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        logger.info(f"Split audio into {len(chunks)} chunks based on time")
        return chunks
    
    def _save_chunks(self, chunks: List[AudioSegment], base_name: str) -> List[AudioChunk]:
        """Save audio chunks to disk."""
        saved_chunks = []
        cumulative_time = 0.0
        
        for i, chunk in enumerate(chunks):
            chunk_name = f"{base_name}_chunk_{i:03d}"
            chunk_path = self._save_audio(chunk, chunk_name)
            
            chunk_duration = chunk.duration_seconds
            saved_chunks.append(AudioChunk(
                chunk_id=i,
                file_path=chunk_path,
                start_time=cumulative_time,
                end_time=cumulative_time + chunk_duration,
                duration=chunk_duration
            ))
            
            cumulative_time += chunk_duration
        
        return saved_chunks
    
    def _save_audio(self, audio: AudioSegment, name: str) -> Path:
        """Save audio segment to disk."""
        output_path = self.temp_dir / f"{name}.{self.output_format}"
        
        try:
            audio.export(
                output_path,
                format=self.output_format,
                parameters=["-acodec", "pcm_s16le"] if self.output_format == "wav" else []
            )
            logger.debug(f"Saved audio to: {output_path}")
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio: {e}")
    
    def _update_metadata(self, original: AudioMetadata, processed: AudioSegment) -> AudioMetadata:
        """Update metadata with processed audio information."""
        # Recalculate volume metrics for processed audio
        processed_info = self.get_audio_info(Path("dummy"), processed)
        
        return AudioMetadata(
            duration=processed.duration_seconds,
            sample_rate=processed.frame_rate,
            channels=processed.channels,
            bitrate=original.bitrate,  # Keep original bitrate info
            format=self.output_format,
            avg_volume=processed_info.avg_volume,
            peak_volume=processed_info.peak_volume,
            silence_ratio=processed_info.silence_ratio
        )
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()