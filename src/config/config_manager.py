"""
Config Manager Module for TranscriMatic

This module handles all configuration management for the audio transcription system,
including loading, validation, and access to configuration parameters.
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy

import yaml


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class ConfigurationNotFoundError(ConfigurationError):
    """Raised when configuration file is not found"""
    pass


class ConfigManager:
    """
    Manages system configuration with support for YAML/JSON formats,
    environment variable expansion, validation, and thread-safe access.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the ConfigManager with a configuration file path.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._default_values = self._get_default_values()
        
        # Load configuration on initialization
        self.load_config()
    
    def _get_default_values(self) -> Dict[str, Any]:
        """Define default values for optional configuration parameters."""
        return {
            "processing.whisper_device": "auto",
            "processing.faster_whisper.beam_size": 5,
            "processing.faster_whisper.best_of": 5,
            "processing.faster_whisper.temperature": 0.0,
            "processing.faster_whisper.vad_filter": True,
            "processing.faster_whisper.vad_options.min_silence_duration_ms": 500,
            "processing.faster_whisper.vad_options.speech_pad_ms": 200,
            "processing.audio.output_format": "wav",
            "processing.audio.sample_rate": 16000,
            "processing.audio.channels": 1,
            "processing.audio.normalization.target_dBFS": -20,
            "processing.audio.normalization.enabled": True,
            "processing.audio.noise_reduction.enabled": True,
            "processing.audio.noise_reduction.noise_floor": -40,
            "processing.audio.chunking.max_duration": 300,
            "processing.audio.chunking.min_silence": 0.5,
            "processing.audio.chunking.silence_thresh": -40,
            "processing.diarization.enabled": True,
            "processing.diarization.min_speakers": 1,
            "processing.diarization.max_speakers": 10,
            "processing.diarization.min_duration": 0.5,
            "llm.fallback_order": ["ollama", "gemini", "openai"],
            "llm.ollama.host": "localhost",
            "llm.ollama.port": 11434,
            "llm.ollama.timeout": 120,
            "llm.ollama.temperature": 0.3,
            "llm.gemini.temperature": 0.3,
            "llm.gemini.max_tokens": 2000,
            "llm.openai.temperature": 0.3,
            "llm.openai.max_tokens": 2000,
            "segmentation.min_conversation_gap": 30,
            "segmentation.min_conversation_length": 60,
            "segmentation.topic_similarity_threshold": 0.3,
            "segmentation.boundaries.combine_threshold": 60,
            "segmentation.boundaries.confidence_threshold": 0.7,
            "transfer.min_file_size": 1024,
            "transfer.max_file_size": 2147483648,
            "transfer.cleanup_device": False,
            "transfer.verify_transfer": True,
            "history.retention_days": 365,
            "output.formatting.date_format": "%Y-%m-%d",
            "output.formatting.time_format": "%H:%M:%S",
            "output.markdown.include_metadata": True,
            "output.markdown.include_navigation": True,
            "output.markdown.include_speaker_labels": True,
            "system.auto_start": True,
            "system.daemon_mode": False,
            "system.processing.max_concurrent": 2,
            "system.processing.retry_attempts": 3,
            "system.processing.retry_delay": 5,
            "system.monitoring.health_check_interval": 60,
            "system.notifications.enabled": False,
            "logging.level": "INFO",
            "logging.max_size": 10485760,
            "logging.backup_count": 5
        }
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and parse the configuration file.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            ConfigurationNotFoundError: If the configuration file doesn't exist
            ConfigurationError: If the configuration file is invalid
        """
        if not self.config_path.exists():
            raise ConfigurationNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    raw_config = yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    raw_config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration format: {self.config_path.suffix}"
                    )
            
            # Process environment variables
            config = self._process_config(raw_config)
            
            # Validate configuration
            if not self.validate_config(config):
                raise ConfigurationError("Configuration validation failed")
            
            with self._lock:
                self._config = config
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            self._log_configuration()
            
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _process_config(self, config: Any) -> Any:
        """
        Process configuration recursively to expand environment variables
        and handle special values.
        
        Args:
            config: Configuration data to process
            
        Returns:
            Processed configuration
        """
        if isinstance(config, dict):
            return {k: self._process_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._process_config(item) for item in config]
        elif isinstance(config, str):
            return self.expand_env_vars(config)
        else:
            return config
    
    def expand_env_vars(self, value: str) -> str:
        """
        Expand environment variables in configuration values.
        Supports ${VAR_NAME} syntax.
        
        Args:
            value: String that may contain environment variables
            
        Returns:
            String with environment variables expanded
        """
        if not isinstance(value, str):
            return value
        
        # Expand ${VAR_NAME} syntax
        import re
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replacer(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not found")
                return match.group(0)  # Return original if not found
            return env_value
        
        expanded = pattern.sub(replacer, value)
        
        # Also expand ~ to home directory
        if expanded.startswith('~'):
            expanded = os.path.expanduser(expanded)
        
        return expanded
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration against required rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If validation fails with details
        """
        errors = []
        
        # Required fields
        if 'device' not in config or 'identifier' not in config.get('device', {}):
            errors.append("device.identifier is required")
        
        if 'paths' not in config:
            errors.append("paths section is required")
        else:
            required_paths = ['audio_destination', 'transcriptions', 'summaries']
            for path_key in required_paths:
                if path_key not in config['paths']:
                    errors.append(f"paths.{path_key} is required")
        
        if 'processing' not in config:
            errors.append("processing section is required")
        else:
            if 'whisper_model' not in config['processing']:
                errors.append("processing.whisper_model is required")
            else:
                valid_models = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
                if config['processing']['whisper_model'] not in valid_models:
                    errors.append(f"processing.whisper_model must be one of: {valid_models}")
            
            if 'language' not in config['processing']:
                errors.append("processing.language is required")
        
        # LLM configuration
        if 'llm' not in config:
            errors.append("llm section is required")
        else:
            if 'provider' not in config['llm']:
                errors.append("llm.provider is required")
            else:
                valid_providers = ['ollama', 'gemini', 'openai']
                if config['llm']['provider'] not in valid_providers:
                    errors.append(f"llm.provider must be one of: {valid_providers}")
            
            # Check API keys for enabled providers
            if config['llm'].get('gemini', {}).get('enabled', False):
                api_key = config['llm']['gemini'].get('api_key', '')
                if not api_key or api_key == '${GEMINI_API_KEY}':
                    errors.append("GEMINI_API_KEY environment variable is required when Gemini is enabled")
            
            if config['llm'].get('openai', {}).get('enabled', False):
                api_key = config['llm']['openai'].get('api_key', '')
                if not api_key or api_key == '${OPENAI_API_KEY}':
                    errors.append("OPENAI_API_KEY environment variable is required when OpenAI is enabled")
        
        # Validate numeric ranges
        if 'processing' in config and 'audio' in config['processing']:
            audio_config = config['processing']['audio']
            
            if 'sample_rate' in audio_config:
                rate = audio_config['sample_rate']
                if not isinstance(rate, int) or rate < 8000 or rate > 48000:
                    errors.append("processing.audio.sample_rate must be between 8000 and 48000")
            
            if 'channels' in audio_config:
                channels = audio_config['channels']
                if channels not in [1, 2]:
                    errors.append("processing.audio.channels must be 1 or 2")
        
        # Validate paths exist or can be created
        if not errors and 'paths' in config:
            try:
                self.validate_paths(config)
            except Exception as e:
                errors.append(f"Path validation failed: {e}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(error_msg)
        
        return True
    
    def validate_paths(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that all configured paths exist or can be created.
        
        Args:
            config: Configuration to validate (uses self._config if None)
            
        Returns:
            True if all paths are valid
            
        Raises:
            ConfigurationError: If paths cannot be created or accessed
        """
        if config is None:
            with self._lock:
                config = self._config
        
        paths_config = config.get('paths', {})
        
        for path_key, path_value in paths_config.items():
            if not isinstance(path_value, str):
                continue
            
            path = Path(self.expand_env_vars(path_value))
            
            try:
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                
                # Check write permissions
                test_file = path / '.write_test'
                test_file.touch()
                test_file.unlink()
                
            except Exception as e:
                raise ConfigurationError(
                    f"Cannot create or write to {path_key} path '{path}': {e}"
                )
        
        # Also check history database path
        history_path = config.get('history', {}).get('database_path')
        if history_path:
            db_path = Path(self.expand_env_vars(history_path)).parent
            try:
                db_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(
                    f"Cannot create history database directory '{db_path}': {e}"
                )
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'processing.whisper_model')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            # First check if we have a default value
            if default is None and key in self._default_values:
                default = self._default_values[key]
            
            # Navigate through the configuration
            value = self._config
            for part in key.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
    
    def reload(self):
        """
        Reload the configuration from the file.
        
        This allows for hot-reloading of configuration changes.
        """
        logger.info("Reloading configuration...")
        old_config = deepcopy(self._config)
        
        try:
            self.load_config()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            # Restore old configuration on failure
            with self._lock:
                self._config = old_config
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def _log_configuration(self):
        """Log the loaded configuration with sensitive data masked."""
        def mask_sensitive(config: Any, path: str = "") -> Any:
            """Recursively mask sensitive values in configuration."""
            if isinstance(config, dict):
                masked = {}
                for k, v in config.items():
                    current_path = f"{path}.{k}" if path else k
                    if any(sensitive in k.lower() for sensitive in ['api_key', 'password', 'secret', 'token']):
                        masked[k] = "***MASKED***" if v else None
                    else:
                        masked[k] = mask_sensitive(v, current_path)
                return masked
            elif isinstance(config, list):
                return [mask_sensitive(item, path) for item in config]
            else:
                return config
        
        with self._lock:
            masked_config = mask_sensitive(self._config)
        
        logger.debug(f"Loaded configuration: {json.dumps(masked_config, indent=2)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        
        Returns:
            Copy of the configuration dictionary
        """
        with self._lock:
            return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        sentinel = object()
        value = self.get(key, default=sentinel)
        return value is not sentinel