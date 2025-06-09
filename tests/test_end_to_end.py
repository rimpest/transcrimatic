#!/usr/bin/env python3
"""End-to-end test script for TranscriMatic transcription pipeline."""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
from src.config.config_manager import ConfigManager
from src.transcription.transcription_engine_with_fallback import TranscriptionEngineWithFallback
from tests.mock_audio_processor import MockAudioProcessor
from tests.mock_llm_analyzer import MockLLMAnalyzer, MockConversationSegmenter


class TranscriMaticPipelineTest:
    """End-to-end test of the TranscriMatic pipeline."""
    
    def __init__(self, config_path: str = "config.test.yaml"):
        """Initialize test pipeline with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.audio_processor = MockAudioProcessor(self.config)
        self.transcription_engine = TranscriptionEngineWithFallback(self.config)
        self.conversation_segmenter = MockConversationSegmenter(self.config)
        self.llm_analyzer = MockLLMAnalyzer(self.config)
        
        # Results storage
        self.results = {
            "audio_processing": {},
            "transcription": {},
            "segmentation": {},
            "analysis": {},
            "performance": {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the test."""
        log_level = self.config.get("logging.level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('test_pipeline.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_pipeline(self, audio_path: Path) -> Dict[str, Any]:
        """Run complete pipeline on an audio file."""
        self.logger.info(f"Starting pipeline test with: {audio_path}")
        pipeline_start = time.time()
        
        try:
            # Step 1: Audio Processing
            self.logger.info("Step 1: Processing audio file...")
            audio_start = time.time()
            
            if not self.audio_processor.validate_audio(audio_path):
                raise ValueError(f"Invalid audio file: {audio_path}")
            
            processed_audio = self.audio_processor.process_audio_file(audio_path)
            audio_time = time.time() - audio_start
            
            self.results["audio_processing"] = {
                "status": "success",
                "duration": audio_time,
                "chunks": len(processed_audio.chunks),
                "metadata": {
                    "duration": processed_audio.metadata.duration,
                    "sample_rate": processed_audio.metadata.sample_rate,
                    "format": processed_audio.metadata.format
                }
            }
            
            self.logger.info(f"✓ Audio processed in {audio_time:.2f}s - {len(processed_audio.chunks)} chunks")
            
            # Step 2: Transcription
            self.logger.info("Step 2: Transcribing audio...")
            transcription_start = time.time()
            
            transcription = self.transcription_engine.transcribe_with_retry(processed_audio)
            transcription_time = time.time() - transcription_start
            
            self.results["transcription"] = {
                "status": "success",
                "duration": transcription_time,
                "segments": len(transcription.segments),
                "speakers": transcription.speaker_count,
                "confidence": transcription.confidence,
                "language": transcription.language,
                "text_preview": transcription.full_text[:200] + "..." if len(transcription.full_text) > 200 else transcription.full_text
            }
            
            self.logger.info(f"✓ Transcribed in {transcription_time:.2f}s - {len(transcription.segments)} segments, {transcription.speaker_count} speakers")
            
            # Step 3: Conversation Segmentation
            self.logger.info("Step 3: Segmenting conversation...")
            segmentation_start = time.time()
            
            conversation_segments = self.conversation_segmenter.segment_conversation(transcription)
            segmentation_time = time.time() - segmentation_start
            
            self.results["segmentation"] = {
                "status": "success",
                "duration": segmentation_time,
                "segments": len(conversation_segments),
                "total_duration": sum(seg["duration"] for seg in conversation_segments)
            }
            
            self.logger.info(f"✓ Segmented in {segmentation_time:.2f}s - {len(conversation_segments)} conversation segments")
            
            # Step 4: LLM Analysis
            self.logger.info("Step 4: Analyzing conversations...")
            analysis_start = time.time()
            
            analyses = []
            for segment in conversation_segments:
                analysis = self.llm_analyzer.analyze_conversation(
                    segment["transcript"],
                    segment
                )
                analyses.append({
                    "segment_id": segment["id"],
                    "summary": analysis["summary"],
                    "key_points": analysis["key_points"],
                    "action_items": analysis["action_items"],
                    "sentiment": analysis["sentiment"]["overall"]
                })
            
            analysis_time = time.time() - analysis_start
            
            self.results["analysis"] = {
                "status": "success",
                "duration": analysis_time,
                "segments_analyzed": len(analyses),
                "analyses": analyses
            }
            
            self.logger.info(f"✓ Analysis completed in {analysis_time:.2f}s")
            
            # Overall performance
            pipeline_time = time.time() - pipeline_start
            self.results["performance"] = {
                "total_duration": pipeline_time,
                "audio_duration": processed_audio.metadata.duration,
                "real_time_factor": pipeline_time / processed_audio.metadata.duration if processed_audio.metadata.duration > 0 else 0,
                "breakdown": {
                    "audio_processing": audio_time,
                    "transcription": transcription_time,
                    "segmentation": segmentation_time,
                    "analysis": analysis_time
                }
            }
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Pipeline completed successfully in {pipeline_time:.2f}s")
            self.logger.info(f"Real-time factor: {self.results['performance']['real_time_factor']:.2f}x")
            self.logger.info(f"{'='*50}\n")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.results["error"] = str(e)
            return self.results
    
    def save_results(self, output_path: Path):
        """Save test results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self):
        """Print a summary of the test results."""
        print("\n" + "="*60)
        print("TRANSCRIMATIC PIPELINE TEST SUMMARY")
        print("="*60)
        
        # Audio Processing
        if "audio_processing" in self.results and self.results["audio_processing"].get("status") == "success":
            print(f"\n✓ Audio Processing: SUCCESS")
            print(f"  - Chunks created: {self.results['audio_processing']['chunks']}")
            print(f"  - Audio duration: {self.results['audio_processing']['metadata']['duration']:.2f}s")
            print(f"  - Processing time: {self.results['audio_processing']['duration']:.2f}s")
        
        # Transcription
        if "transcription" in self.results and self.results["transcription"].get("status") == "success":
            print(f"\n✓ Transcription: SUCCESS")
            print(f"  - Segments: {self.results['transcription']['segments']}")
            print(f"  - Speakers: {self.results['transcription']['speakers']}")
            print(f"  - Confidence: {self.results['transcription']['confidence']:.2%}")
            print(f"  - Language: {self.results['transcription']['language']}")
            print(f"  - Processing time: {self.results['transcription']['duration']:.2f}s")
            print(f"  - Text preview: {self.results['transcription']['text_preview']}")
        
        # Segmentation
        if "segmentation" in self.results and self.results["segmentation"].get("status") == "success":
            print(f"\n✓ Segmentation: SUCCESS")
            print(f"  - Conversation segments: {self.results['segmentation']['segments']}")
            print(f"  - Processing time: {self.results['segmentation']['duration']:.2f}s")
        
        # Analysis
        if "analysis" in self.results and self.results["analysis"].get("status") == "success":
            print(f"\n✓ LLM Analysis: SUCCESS")
            print(f"  - Segments analyzed: {self.results['analysis']['segments_analyzed']}")
            print(f"  - Processing time: {self.results['analysis']['duration']:.2f}s")
            
            if self.results["analysis"]["analyses"]:
                print(f"\n  Sample Analysis (Segment 0):")
                sample = self.results["analysis"]["analyses"][0]
                print(f"    Summary: {sample['summary']}")
                print(f"    Sentiment: {sample['sentiment']}")
                if sample['key_points']:
                    print(f"    Key Points: {len(sample['key_points'])} identified")
                if sample['action_items']:
                    print(f"    Action Items: {len(sample['action_items'])} found")
        
        # Performance
        if "performance" in self.results:
            print(f"\n📊 Performance Metrics:")
            print(f"  - Total pipeline time: {self.results['performance']['total_duration']:.2f}s")
            print(f"  - Real-time factor: {self.results['performance']['real_time_factor']:.2f}x")
            print(f"  - Breakdown:")
            for step, duration in self.results['performance']['breakdown'].items():
                percentage = (duration / self.results['performance']['total_duration']) * 100
                print(f"    • {step}: {duration:.2f}s ({percentage:.1f}%)")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="End-to-end test of TranscriMatic transcription pipeline"
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to audio file to test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.test.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_results.json"),
        help="Path to save test results"
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate test audio files first"
    )
    
    args = parser.parse_args()
    
    # Generate test audio if requested
    if args.generate_audio:
        print("Generating test audio files...")
        import subprocess
        result = subprocess.run([
            sys.executable,
            "tests/generate_test_audio.py",
            "--output-dir", "test_audio"
        ])
        if result.returncode != 0:
            print("Failed to generate test audio")
            return 1
        args.audio = Path("test_audio/test_spanish_greeting.wav")
    
    # Validate audio file
    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        print("\nYou can generate test audio files with:")
        print("  python test_end_to_end.py --generate-audio --audio test_audio/test_spanish_greeting.wav")
        return 1
    
    # Run pipeline test
    print(f"\nTranscriMatic End-to-End Pipeline Test")
    print(f"Audio file: {args.audio}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    try:
        # Create and run test pipeline
        pipeline = TranscriMaticPipelineTest(str(args.config))
        results = pipeline.run_pipeline(args.audio)
        
        # Save results
        pipeline.save_results(args.output)
        
        # Print summary
        pipeline.print_summary()
        
        # Return success if no errors
        return 0 if "error" not in results else 1
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        logging.exception("Pipeline test error")
        return 1


if __name__ == "__main__":
    sys.exit(main())