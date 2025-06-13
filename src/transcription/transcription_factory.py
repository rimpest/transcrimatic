"""
Factory for creating transcription engines based on configuration.
"""

import logging
import sys
from src.config import ConfigManager
from .transcription_engine import TranscriptionEngine
from .transcription_engine_with_fallback import TranscriptionEngineWithFallback

logger = logging.getLogger(__name__)


def create_transcription_engine(config: ConfigManager):
    """
    Create appropriate transcription engine based on configuration.
    
    Args:
        config: Configuration manager
        
    Returns:
        Transcription engine instance
    """
    # Check which engine to use
    engine_type = config.get("processing.whisperx.engine", "faster-whisper")
    
    if engine_type == "whisperx":
        # Check if WhisperX is available
        try:
            import whisperx
            from .transcription_engine_whisperx import TranscriptionEngineWhisperX
            logger.info("Creating WhisperX transcription engine")
            return TranscriptionEngineWhisperX(config)
        except ImportError:
            logger.warning("WhisperX not available. Falling back to faster-whisper.")
            if sys.version_info >= (3, 13):
                logger.warning("WhisperX is not compatible with Python 3.13 yet.")
            logger.warning("Using faster-whisper engine instead.")
            return TranscriptionEngineWithFallback(config)
    else:
        logger.info("Creating faster-whisper transcription engine with fallback")
        return TranscriptionEngineWithFallback(config)