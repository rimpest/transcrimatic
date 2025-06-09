"""Batch transcription processor for handling multiple files efficiently."""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .transcription_engine_with_fallback import TranscriptionEngineWithFallback
from .data_classes import Transcription
from ..audio.data_classes import ProcessedAudio
from ..config.config_manager import ConfigManager


class BatchTranscriptionProcessor:
    """Process multiple audio files for transcription with parallel processing support."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize batch processor with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Create transcription engine
        self.engine = TranscriptionEngineWithFallback(config_manager)
        
        # Batch processing settings
        self.max_workers = self.config.get("processing.batch.max_workers", 1)
        self.save_intermediate = self.config.get("processing.batch.save_intermediate", True)
        self.output_dir = Path(self.config.get("output.transcriptions_dir", "output/transcriptions"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_batch(self, audio_files: List[ProcessedAudio]) -> Dict[str, Transcription]:
        """Process multiple audio files in batch."""
        results = {}
        failed_files = []
        
        self.logger.info(f"Starting batch transcription of {len(audio_files)} files")
        
        if self.max_workers > 1 and len(audio_files) > 1:
            # Parallel processing
            results, failed_files = self._process_parallel(audio_files)
        else:
            # Sequential processing
            results, failed_files = self._process_sequential(audio_files)
        
        # Log summary
        self.logger.info(f"Batch processing complete: {len(results)} successful, {len(failed_files)} failed")
        
        if failed_files:
            self.logger.error(f"Failed files: {failed_files}")
        
        return results
    
    def _process_sequential(self, audio_files: List[ProcessedAudio]) -> tuple[Dict[str, Transcription], List[str]]:
        """Process files sequentially."""
        results = {}
        failed_files = []
        
        for i, audio_file in enumerate(audio_files):
            filename = audio_file.original_file.filename
            self.logger.info(f"Processing {i+1}/{len(audio_files)}: {filename}")
            
            try:
                # Validate audio quality
                if not self.engine.validate_audio_quality(audio_file):
                    self.logger.warning(f"Skipping {filename} due to poor audio quality")
                    failed_files.append(filename)
                    continue
                
                # Transcribe with retry
                transcription = self.engine.transcribe_with_retry(audio_file)
                
                # Save intermediate result if configured
                if self.save_intermediate:
                    self._save_transcription(filename, transcription)
                
                results[filename] = transcription
                
            except Exception as e:
                self.logger.error(f"Failed to process {filename}: {e}")
                failed_files.append(filename)
        
        return results, failed_files
    
    def _process_parallel(self, audio_files: List[ProcessedAudio]) -> tuple[Dict[str, Transcription], List[str]]:
        """Process files in parallel using thread pool."""
        results = {}
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_audio = {
                executor.submit(self._process_single_file, audio): audio
                for audio in audio_files
            }
            
            # Process completed tasks
            for future in as_completed(future_to_audio):
                audio_file = future_to_audio[future]
                filename = audio_file.original_file.filename
                
                try:
                    transcription = future.result()
                    if transcription:
                        results[filename] = transcription
                    else:
                        failed_files.append(filename)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {filename}: {e}")
                    failed_files.append(filename)
        
        return results, failed_files
    
    def _process_single_file(self, audio_file: ProcessedAudio) -> Optional[Transcription]:
        """Process a single file (for parallel execution)."""
        filename = audio_file.original_file.filename
        
        try:
            # Validate audio quality
            if not self.engine.validate_audio_quality(audio_file):
                self.logger.warning(f"Skipping {filename} due to poor audio quality")
                return None
            
            # Transcribe with retry
            transcription = self.engine.transcribe_with_retry(audio_file)
            
            # Save intermediate result if configured
            if self.save_intermediate:
                self._save_transcription(filename, transcription)
            
            return transcription
            
        except Exception as e:
            self.logger.error(f"Failed to process {filename}: {e}")
            return None
    
    def _save_transcription(self, filename: str, transcription: Transcription):
        """Save transcription to file."""
        try:
            # Create output filename
            base_name = Path(filename).stem
            output_file = self.output_dir / f"{base_name}_transcription.json"
            
            # Convert to dictionary
            data = transcription.to_dict()
            
            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Saved transcription to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save transcription for {filename}: {e}")
    
    def load_transcription(self, filename: str) -> Optional[Dict]:
        """Load a saved transcription from file."""
        try:
            base_name = Path(filename).stem
            input_file = self.output_dir / f"{base_name}_transcription.json"
            
            if not input_file.exists():
                return None
            
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to load transcription for {filename}: {e}")
            return None
    
    def generate_batch_report(self, results: Dict[str, Transcription]) -> Dict:
        """Generate summary report for batch processing."""
        report = {
            'total_files': len(results),
            'total_duration': sum(t.duration for t in results.values()),
            'total_segments': sum(len(t.segments) for t in results.values()),
            'average_confidence': sum(t.confidence for t in results.values()) / len(results) if results else 0,
            'speaker_stats': {},
            'language_stats': {},
            'files': {}
        }
        
        # Collect per-file statistics
        for filename, transcription in results.items():
            report['files'][filename] = {
                'duration': transcription.duration,
                'segments': len(transcription.segments),
                'speakers': transcription.speaker_count,
                'confidence': transcription.confidence,
                'language': transcription.language
            }
            
            # Update language stats
            lang = transcription.language
            if lang not in report['language_stats']:
                report['language_stats'][lang] = 0
            report['language_stats'][lang] += 1
            
            # Update speaker stats
            speaker_count = str(transcription.speaker_count)
            if speaker_count not in report['speaker_stats']:
                report['speaker_stats'][speaker_count] = 0
            report['speaker_stats'][speaker_count] += 1
        
        return report