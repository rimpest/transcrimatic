"""
Main Controller Module for TranscriMatic Audio System

This module orchestrates the entire workflow from USB device detection to final output generation.
It manages the pipeline execution, handles errors, and ensures proper module coordination.
"""

import logging
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.config import ConfigManager
from src.device import DeviceMonitor, DeviceEvent, DeviceInfo
from src.transfer import FileTransfer
from src.audio import AudioProcessor, AudioFile
from src.transcription import TranscriptionEngine
from src.conversation import ConversationSegmenter
from src.llm import LLMAnalyzer
from src.output import OutputFormatter


@dataclass
class SystemStatus:
    """System status information"""
    is_running: bool
    device_monitoring: bool
    active_processes: int
    last_device_connected: Optional[datetime]
    files_processed_today: int
    errors_today: int
    uptime: timedelta


@dataclass
class ProcessingTask:
    """Processing task information"""
    task_id: str
    audio_file: AudioFile
    status: str  # "pending", "processing", "completed", "failed"
    start_time: datetime
    end_time: Optional[datetime]
    error: Optional[str]


class ConfigurationError(Exception):
    """Configuration validation error"""
    pass


class MainController:
    """
    Main controller that orchestrates the entire TranscriMatic system.
    Manages all modules and coordinates the processing pipeline.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the main controller with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.is_running = False
        self.active_tasks: List[ProcessingTask] = []
        self.daily_stats = {"files_processed": 0, "errors": 0}
        self.start_time = datetime.now()
        self.last_device_connected = None
        self._shutdown_event = threading.Event()
        
        # Initialize configuration first
        self.config_manager = ConfigManager(config_path)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules in dependency order
        self._initialize_modules()
        
        # Setup system handlers
        self._setup_signal_handlers()
        
        self.logger.info("TranscriMatic system initialized")
    
    def _setup_logging(self):
        """Configure centralized logging"""
        log_config = self.config_manager.get("logging", {})
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        log_file = log_config.get("file", "transcrimatic.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_size", 10485760),  # 10MB
            backupCount=log_config.get("backup_count", 5)
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_config.get("level", "INFO"))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_modules(self):
        """Initialize all modules with proper error handling"""
        try:
            self.device_monitor = DeviceMonitor(self.config_manager)
            self.file_transfer = FileTransfer(self.config_manager)
            self.audio_processor = AudioProcessor(self.config_manager)
            self.transcription_engine = TranscriptionEngine(self.config_manager)
            self.conversation_segmenter = ConversationSegmenter(self.config_manager)
            self.llm_analyzer = LLMAnalyzer(self.config_manager)
            self.output_formatter = OutputFormatter(self.config_manager)
        except Exception as e:
            self.logger.critical(f"Failed to initialize modules: {e}")
            raise SystemError(f"Module initialization failed: {e}")
    
    def start(self):
        """Start the automation system"""
        self.logger.info("Starting TranscriMatic Audio System")
        
        try:
            # Validate configuration
            if not self.config_manager.validate_paths():
                raise ConfigurationError("Invalid paths in configuration")
            
            # Start device monitoring
            self.device_monitor.start_monitoring(self.process_device)
            self.is_running = True
            
            # Start health monitoring
            self._start_health_monitor()
            
            # Main event loop
            self._run_event_loop()
            
        except Exception as e:
            self.logger.critical(f"System startup failed: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Graceful shutdown"""
        self.logger.info("Initiating system shutdown")
        self.is_running = False
        self._shutdown_event.set()
        
        # Stop accepting new tasks
        self.device_monitor.stop_monitoring()
        
        # Wait for active tasks
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks")
            self._wait_for_active_tasks()
        
        # Cleanup resources
        self._cleanup()
        
        self.logger.info("System shutdown complete")
    
    def process_device(self, device_event: DeviceEvent):
        """
        Handle device connection events.
        
        Args:
            device_event: Device event from monitor
        """
        if device_event.event_type == "connected":
            device_name = device_event.device_info.device_name if device_event.device_info else "Unknown"
            self.logger.info(f"Device connected: {device_name} at {device_event.mount_point}")
            self.last_device_connected = datetime.now()
            
            # Transfer audio files
            try:
                audio_files = self.file_transfer.transfer_from_device(device_event.mount_point)
                if audio_files:
                    self.logger.info(f"Transferred {len(audio_files)} audio files")
                    # Queue for processing
                    threading.Thread(
                        target=self.run_pipeline,
                        args=(audio_files,),
                        daemon=True
                    ).start()
                else:
                    self.logger.info("No audio files found on device")
            except Exception as e:
                self.logger.error(f"File transfer failed: {e}")
                self.handle_error(e, "file_transfer")
        
        elif device_event.event_type == "disconnected":
            device_name = device_event.device_info.device_name if device_event.device_info else "Unknown"
            self.logger.info(f"Device disconnected: {device_name}")
    
    def run_pipeline(self, audio_files: List[AudioFile]):
        """Execute complete processing pipeline with error handling"""
        self.logger.info(f"Starting pipeline for {len(audio_files)} files")
        
        all_analyses = []
        processing_errors = []
        
        for audio_file in audio_files:
            task = ProcessingTask(
                task_id=str(uuid.uuid4()),
                audio_file=audio_file,
                status="processing",
                start_time=datetime.now(),
                end_time=None,
                error=None
            )
            self.active_tasks.append(task)
            
            try:
                # 1. Audio Processing
                self.logger.debug(f"Processing audio: {audio_file.filename}")
                processed_audio = self.audio_processor.process_audio(audio_file)
                
                # 2. Transcription
                self.logger.debug("Running transcription with speaker diarization")
                transcription = self.transcription_engine.transcribe(processed_audio)
                
                # 3. Save raw transcription
                self.output_formatter.save_transcription(transcription)
                
                # 4. Conversation Segmentation
                self.logger.debug("Segmenting conversations")
                conversations = self.conversation_segmenter.segment_conversations(transcription)
                self.logger.info(f"Found {len(conversations)} conversations")
                
                # 5. Analyze Conversations
                for conversation in conversations:
                    self.logger.debug(f"Analyzing conversation {conversation.id}")
                    analysis = self.llm_analyzer.analyze_conversation(conversation)
                    all_analyses.append(analysis)
                    
                    # 6. Save Analysis
                    self.output_formatter.save_analysis(analysis, conversation)
                
                # Update task status
                task.status = "completed"
                task.end_time = datetime.now()
                self.daily_stats["files_processed"] += 1
                
            except Exception as e:
                self.logger.error(f"Pipeline error for {audio_file.filename}: {e}")
                task.status = "failed"
                task.error = str(e)
                task.end_time = datetime.now()
                processing_errors.append((audio_file, e))
                self.daily_stats["errors"] += 1
                self.handle_error(e, f"pipeline_{audio_file.filename}")
                
            finally:
                self.active_tasks.remove(task)
        
        # 7. Generate Daily Summary
        if all_analyses:
            try:
                daily_summary = self.llm_analyzer.generate_daily_summary(all_analyses)
                self.output_formatter.save_daily_summary(daily_summary)
            except Exception as e:
                self.logger.error(f"Failed to generate daily summary: {e}")
        
        # Report results
        self.logger.info(
            f"Pipeline completed: {len(audio_files) - len(processing_errors)} successful, "
            f"{len(processing_errors)} failed"
        )
    
    def handle_error(self, error: Exception, module: str):
        """Centralized error handling with recovery strategies"""
        self.logger.error(f"Error in {module}: {error}", exc_info=True)
        
        # Module-specific recovery strategies
        recovery_strategies = {
            "device_monitor": self._recover_device_monitor,
            "file_transfer": self._recover_file_transfer,
            "transcription_engine": self._recover_transcription,
            "llm_analyzer": self._recover_llm,
            "audio_processor": self._recover_audio_processor
        }
        
        # Extract module name from error context
        module_name = module.split("_")[0] if "_" in module else module
        
        if module_name in recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {module_name}")
                recovery_strategies[module_name]()
                self.logger.info(f"Recovery successful for {module_name}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {module_name}: {recovery_error}")
        
        # Send notification if configured
        if self.config_manager.get("system.notifications.enabled", False):
            self._send_error_notification(error, module)
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        return SystemStatus(
            is_running=self.is_running,
            device_monitoring=self.device_monitor.is_monitoring if hasattr(self, 'device_monitor') else False,
            active_processes=len(self.active_tasks),
            last_device_connected=self.last_device_connected,
            files_processed_today=self.daily_stats["files_processed"],
            errors_today=self.daily_stats["errors"],
            uptime=datetime.now() - self.start_time
        )
    
    def _run_event_loop(self):
        """Main event loop"""
        self.logger.info("Entering main event loop")
        while self.is_running and not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=1)  # Check every second
    
    def _start_health_monitor(self):
        """Start health monitoring thread"""
        def monitor_health():
            interval = self.config_manager.get("system.monitoring.health_check_interval", 60)
            while self.is_running:
                status = self.get_status()
                self.logger.debug(f"System status: {status}")
                
                # Write status to file if configured
                status_file = self.config_manager.get("system.monitoring.status_file")
                if status_file:
                    self._write_status_file(status, status_file)
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_health, daemon=True)
        thread.start()
    
    def _wait_for_active_tasks(self):
        """Wait for all active tasks to complete"""
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            self.logger.info(f"Waiting for {len(self.active_tasks)} tasks...")
            time.sleep(5)
        
        if self.active_tasks:
            self.logger.warning(f"Timeout: {len(self.active_tasks)} tasks still active")
    
    def _cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources")
        # Modules will clean up their own resources
    
    def _recover_device_monitor(self):
        """Restart device monitoring"""
        self.device_monitor.stop_monitoring()
        time.sleep(5)
        self.device_monitor.start_monitoring(self.process_device)
    
    def _recover_file_transfer(self):
        """Recovery for file transfer module"""
        # FileTransfer is stateless, just log
        self.logger.info("File transfer recovery: no action needed")
    
    def _recover_transcription(self):
        """Recovery for transcription engine"""
        # Attempt to reinitialize the transcription engine
        self.transcription_engine = TranscriptionEngine(self.config_manager)
    
    def _recover_llm(self):
        """Recovery for LLM analyzer"""
        # Attempt to reinitialize the LLM analyzer
        self.llm_analyzer = LLMAnalyzer(self.config_manager)
    
    def _recover_audio_processor(self):
        """Recovery for audio processor"""
        # AudioProcessor is stateless, just log
        self.logger.info("Audio processor recovery: no action needed")
    
    def _send_error_notification(self, error: Exception, module: str):
        """Send error notification (placeholder)"""
        # TODO: Implement actual notification logic
        self.logger.info(f"Would send notification for error in {module}: {error}")
    
    def _write_status_file(self, status: SystemStatus, file_path: str):
        """Write status to file"""
        try:
            import json
            status_dict = {
                "is_running": status.is_running,
                "device_monitoring": status.device_monitoring,
                "active_processes": status.active_processes,
                "last_device_connected": status.last_device_connected.isoformat() if status.last_device_connected else None,
                "files_processed_today": status.files_processed_today,
                "errors_today": status.errors_today,
                "uptime_seconds": status.uptime.total_seconds()
            }
            Path(file_path).write_text(json.dumps(status_dict, indent=2))
        except Exception as e:
            self.logger.error(f"Failed to write status file: {e}")