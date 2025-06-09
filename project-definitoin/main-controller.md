# Main Controller Module

## Purpose
Orchestrates the entire workflow from USB device detection to final output generation. Manages the pipeline execution, handles errors, and ensures proper module coordination.

## Dependencies
- **External**: `asyncio`, `logging`, `signal`, `threading`
- **Internal**: All other modules

## Interface

### Input
- Configuration file path
- System signals (for graceful shutdown)

### Output
- Processed transcriptions and analyses
- System logs and status reports

### Public Methods

```python
class MainController:
    def __init__(self, config_path: Path):
        """Initialize controller with configuration file"""
        
    def start(self):
        """Start the automation system"""
        
    def stop(self):
        """Stop the system gracefully"""
        
    def process_device(self, device_event: DeviceEvent):
        """Process newly connected device"""
        
    def run_pipeline(self, audio_files: List[AudioFile]):
        """Run complete processing pipeline"""
        
    def handle_error(self, error: Exception, module: str):
        """Central error handling"""
        
    def get_status(self) -> SystemStatus:
        """Get current system status"""
```

## System Architecture

```python
class MainController:
    def __init__(self, config_path: Path):
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize all modules
        self.device_monitor = DeviceMonitor(self.config_manager)
        self.file_transfer = FileTransfer(self.config_manager)
        self.audio_processor = AudioProcessor(self.config_manager)
        self.transcription_engine = TranscriptionEngine(self.config_manager)
        self.conversation_segmenter = ConversationSegmenter(self.config_manager)
        self.llm_analyzer = LLMAnalyzer(self.config_manager)
        self.output_formatter = OutputFormatter(self.config_manager)
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
```

## Main Workflow

```python
def start(self):
    """Start the automation system"""
    self.logger.info("Starting Audio Recorder Auto-Import System")
    
    try:
        # Validate configuration
        if not self.config_manager.validate_paths():
            raise ConfigurationError("Invalid configuration")
        
        # Start device monitoring
        self.device_monitor.start_monitoring(self.process_device)
        
        # Keep main thread alive
        self._run_event_loop()
        
    except Exception as e:
        self.logger.error(f"System startup failed: {e}")
        self.stop()

def process_device(self, device_event: DeviceEvent):
    """Handle device connection event"""
    if device_event.event_type != "connected":
        return
    
    self.logger.info(f"Processing device: {device_event.device_info.device_name}")
    
    try:
        # 1. Transfer files
        audio_files = self.file_transfer.transfer_new_files(
            Path(device_event.mount_point)
        )
        
        if not audio_files:
            self.logger.info("No new files to process")
            return
        
        # 2. Run processing pipeline
        self.run_pipeline(audio_files)
        
    except Exception as e:
        self.handle_error(e, "device_processing")
```

## Pipeline Execution

```python
def run_pipeline(self, audio_files: List[AudioFile]):
    """Execute complete processing pipeline"""
    self.logger.info(f"Processing {len(audio_files)} audio files")
    
    all_analyses = []
    
    for audio_file in audio_files:
        try:
            # 1. Process audio
            self.logger.debug(f"Processing audio: {audio_file.filename}")
            processed_audio = self.audio_processor.process_audio(audio_file)
            
            # 2. Transcribe
            self.logger.debug("Running transcription")
            transcription = self.transcription_engine.transcribe(processed_audio)
            
            # 3. Save raw transcription
            self.output_formatter.save_transcription(transcription)
            
            # 4. Segment conversations
            self.logger.debug("Segmenting conversations")
            conversations = self.conversation_segmenter.segment_conversations(transcription)
            
            # 5. Analyze each conversation
            for conversation in conversations:
                self.logger.debug(f"Analyzing conversation {conversation.id}")
                analysis = self.llm_analyzer.analyze_conversation(conversation)
                all_analyses.append(analysis)
                
                # 6. Save analysis
                self.output_formatter.save_analysis(analysis, conversation)
            
        except Exception as e:
            self.handle_error(e, f"pipeline_processing_{audio_file.filename}")
            continue
    
    # 7. Generate daily summary
    if all_analyses:
        daily_summary = self.llm_analyzer.generate_daily_summary(all_analyses)
        self.output_formatter.save_daily_summary(daily_summary)
    
    self.logger.info("Pipeline processing complete")
```

## Error Handling

```python
def handle_error(self, error: Exception, module: str):
    """Centralized error handling with recovery strategies"""
    self.logger.error(f"Error in {module}: {error}", exc_info=True)
    
    # Module-specific recovery
    recovery_strategies = {
        "device_monitor": self._recover_device_monitor,
        "file_transfer": self._recover_file_transfer,
        "transcription_engine": self._recover_transcription,
        "llm_analyzer": self._recover_llm
    }
    
    if module in recovery_strategies:
        try:
            recovery_strategies[module]()
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
    
    # Send notification if configured
    if self.config_manager.get("notifications.enabled", False):
        self._send_error_notification(error, module)

def _recover_device_monitor(self):
    """Attempt to restart device monitoring"""
    self.logger.info("Attempting to restart device monitor")
    self.device_monitor.stop_monitoring()
    time.sleep(5)
    self.device_monitor.start_monitoring(self.process_device)
```

## Logging Configuration

```python
def _setup_logging(self):
    """Configure centralized logging"""
    log_config = self.config_manager.get("logging", {})
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_config.get("file", "audio_system.log"),
        maxBytes=log_config.get("max_size", 10485760),
        backupCount=log_config.get("backup_count", 5)
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_config.get("level", "INFO"))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    self.logger = logging.getLogger(__name__)
```

## Signal Handling

```python
def _setup_signal_handlers(self):
    """Setup graceful shutdown handlers"""
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

def _signal_handler(self, signum, frame):
    """Handle shutdown signals"""
    self.logger.info(f"Received signal {signum}, initiating shutdown")
    self.stop()

def stop(self):
    """Graceful shutdown"""
    self.logger.info("Stopping Audio Recorder System")
    
    # Stop device monitoring
    self.device_monitor.stop_monitoring()
    
    # Wait for active processing to complete
    self._wait_for_active_tasks()
    
    # Cleanup resources
    self._cleanup()
    
    self.logger.info("System stopped successfully")
```

## Status Monitoring

```python
@dataclass
class SystemStatus:
    is_running: bool
    device_monitoring: bool
    active_processes: int
    last_device_connected: Optional[datetime]
    files_processed_today: int
    errors_today: int

def get_status(self) -> SystemStatus:
    """Get current system status"""
    return SystemStatus(
        is_running=self.is_running,
        device_monitoring=self.device_monitor.is_monitoring,
        active_processes=len(self.active_tasks),
        last_device_connected=self.last_device_time,
        files_processed_today=self.daily_stats["files_processed"],
        errors_today=self.daily_stats["errors"]
    )
```

## Configuration

```yaml
system:
  auto_start: true
  daemon_mode: false
  
  processing:
    max_concurrent: 2
    retry_attempts: 3
    retry_delay: 5
    
  monitoring:
    health_check_interval: 60
    status_file: "/tmp/audio_system_status.json"
    
  notifications:
    enabled: false
    email: "user@example.com"
    webhook: "https://example.com/webhook"
```

## Module Relationships
- Uses: All modules
- Coordinates: Complete pipeline flow
- Monitors: System health and status

## Testing Considerations

- Test complete pipeline integration
- Simulate device connections
- Test error recovery mechanisms
- Verify graceful shutdown
- Test with concurrent processing

## Entry Point

```python
# main.py
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Audio Recorder Auto-Import & Transcription System"
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("config.yaml"),
        help="Configuration file path"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in daemon mode"
    )
    
    args = parser.parse_args()
    
    controller = MainController(args.config)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        controller.stop()
```

## Future Enhancements

- Web dashboard for monitoring
- REST API for remote control
- Multi-device support
- Cloud sync capabilities
- Plugin system for extensibility