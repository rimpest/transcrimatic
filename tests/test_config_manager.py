"""
Unit tests for the ConfigManager module.
"""

import os
import json
import yaml
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.config_manager import ConfigManager, ConfigurationError, ConfigurationNotFoundError


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration dictionary."""
        return {
            "device": {
                "identifier": "Sony_ICD-TX660",
                "vendor_id": "054c",
                "product_id": "0xxx"
            },
            "paths": {
                "audio_destination": "~/test/audio",
                "transcriptions": "~/test/transcriptions",
                "summaries": "~/test/summaries"
            },
            "processing": {
                "whisper_model": "base",
                "language": "es"
            },
            "llm": {
                "provider": "ollama",
                "ollama": {
                    "enabled": True,
                    "model": "llama3:8b"
                }
            }
        }
    
    @pytest.fixture
    def config_with_env_vars(self):
        """Return a configuration with environment variables."""
        return {
            "device": {
                "identifier": "Sony_ICD-TX660"
            },
            "paths": {
                "audio_destination": "${TEST_AUDIO_PATH}",
                "transcriptions": "~/test/transcriptions",
                "summaries": "~/test/summaries"
            },
            "processing": {
                "whisper_model": "base",
                "language": "es"
            },
            "llm": {
                "provider": "gemini",
                "gemini": {
                    "enabled": True,
                    "api_key": "${GEMINI_API_KEY}",
                    "model": "gemini-1.5-pro"
                }
            }
        }
    
    def test_init_and_load_yaml(self, temp_config_dir, valid_config):
        """Test initialization and loading of YAML configuration."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        assert config_manager.config_path == config_path
        assert config_manager.get("device.identifier") == "Sony_ICD-TX660"
        assert config_manager.get("processing.whisper_model") == "base"
    
    def test_load_json(self, temp_config_dir, valid_config):
        """Test loading JSON configuration."""
        config_path = temp_config_dir / "config.json"
        
        with open(config_path, 'w') as f:
            json.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        assert config_manager.get("device.identifier") == "Sony_ICD-TX660"
        assert config_manager.get("processing.language") == "es"
    
    def test_config_not_found(self, temp_config_dir):
        """Test handling of missing configuration file."""
        config_path = temp_config_dir / "nonexistent.yaml"
        
        with pytest.raises(ConfigurationNotFoundError) as exc_info:
            ConfigManager(config_path)
        
        assert "Configuration file not found" in str(exc_info.value)
    
    def test_unsupported_format(self, temp_config_dir):
        """Test handling of unsupported configuration format."""
        config_path = temp_config_dir / "config.txt"
        config_path.write_text("invalid format")
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "Unsupported configuration format" in str(exc_info.value)
    
    def test_invalid_yaml(self, temp_config_dir):
        """Test handling of invalid YAML syntax."""
        config_path = temp_config_dir / "config.yaml"
        config_path.write_text("invalid: yaml: syntax:")
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "Failed to parse configuration file" in str(exc_info.value)
    
    def test_environment_variable_expansion(self, temp_config_dir, config_with_env_vars):
        """Test expansion of environment variables."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_env_vars, f)
        
        # Set environment variables using temp directory paths
        test_audio_path = str(temp_config_dir / "test_audio")
        with patch.dict(os.environ, {
            'TEST_AUDIO_PATH': test_audio_path,
            'GEMINI_API_KEY': 'test-api-key-123'
        }):
            config_manager = ConfigManager(config_path)
            
            assert config_manager.get("paths.audio_destination") == test_audio_path
            assert config_manager.get("llm.gemini.api_key") == "test-api-key-123"
    
    def test_missing_environment_variable(self, temp_config_dir, config_with_env_vars):
        """Test handling of missing environment variables."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_env_vars, f)
        
        # Don't set GEMINI_API_KEY, use temp directory path
        test_audio_path = str(temp_config_dir / "test_audio") 
        with patch.dict(os.environ, {'TEST_AUDIO_PATH': test_audio_path}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                ConfigManager(config_path)
            
            assert "GEMINI_API_KEY environment variable is required" in str(exc_info.value)
    
    def test_home_directory_expansion(self, temp_config_dir, valid_config):
        """Test expansion of ~ to home directory."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        audio_path = config_manager.get("paths.audio_destination")
        assert not audio_path.startswith("~")
        assert audio_path.startswith(os.path.expanduser("~"))
    
    def test_validation_missing_required_fields(self, temp_config_dir):
        """Test validation of missing required fields."""
        invalid_configs = [
            {},  # Empty config
            {"device": {}},  # Missing device.identifier
            {"device": {"identifier": "test"}},  # Missing paths
            {
                "device": {"identifier": "test"},
                "paths": {"audio_destination": "~/test"}  # Missing required paths
            },
            {
                "device": {"identifier": "test"},
                "paths": {
                    "audio_destination": "~/test",
                    "transcriptions": "~/test",
                    "summaries": "~/test"
                }
                # Missing processing section
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            config_path = temp_config_dir / f"invalid_{i}.yaml"
            
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            with pytest.raises(ConfigurationError):
                ConfigManager(config_path)
    
    def test_validation_invalid_values(self, temp_config_dir, valid_config):
        """Test validation of invalid configuration values."""
        # Invalid whisper model
        invalid_config = valid_config.copy()
        invalid_config["processing"]["whisper_model"] = "invalid-model"
        
        config_path = temp_config_dir / "invalid_model.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "whisper_model must be one of" in str(exc_info.value)
        
        # Invalid provider
        invalid_config = valid_config.copy()
        invalid_config["llm"]["provider"] = "invalid-provider"
        
        config_path = temp_config_dir / "invalid_provider.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "llm.provider must be one of" in str(exc_info.value)
    
    def test_validation_numeric_ranges(self, temp_config_dir, valid_config):
        """Test validation of numeric configuration values."""
        # Invalid sample rate
        invalid_config = valid_config.copy()
        invalid_config["processing"]["audio"] = {"sample_rate": 100}
        
        config_path = temp_config_dir / "invalid_sample_rate.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "sample_rate must be between" in str(exc_info.value)
        
        # Invalid channels
        invalid_config["processing"]["audio"] = {"sample_rate": 16000, "channels": 3}
        
        config_path = temp_config_dir / "invalid_channels.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ConfigurationError) as exc_info:
            ConfigManager(config_path)
        
        assert "channels must be 1 or 2" in str(exc_info.value)
    
    def test_get_method(self, temp_config_dir, valid_config):
        """Test the get method for accessing configuration values."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        # Test basic access
        assert config_manager.get("device.identifier") == "Sony_ICD-TX660"
        assert config_manager.get("device.vendor_id") == "054c"
        
        # Test nested access
        assert config_manager.get("llm.ollama.enabled") == True
        assert config_manager.get("llm.ollama.model") == "llama3:8b"
        
        # Test non-existent keys with default
        assert config_manager.get("non.existent.key", "default") == "default"
        assert config_manager.get("device.non_existent", None) is None
        
        # Test default values from _default_values
        assert config_manager.get("processing.whisper_device") == "auto"
        assert config_manager.get("processing.audio.sample_rate") == 16000
    
    def test_dictionary_access(self, temp_config_dir, valid_config):
        """Test dictionary-style access to configuration."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        # Test __getitem__
        assert config_manager["device.identifier"] == "Sony_ICD-TX660"
        
        # Test __contains__
        assert "device.identifier" in config_manager
        assert "non.existent.key" not in config_manager
    
    def test_reload(self, temp_config_dir, valid_config):
        """Test configuration reload functionality."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        # Verify initial value
        assert config_manager.get("processing.whisper_model") == "base"
        
        # Modify configuration file
        valid_config["processing"]["whisper_model"] = "large-v3"
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Reload configuration
        config_manager.reload()
        
        # Verify updated value
        assert config_manager.get("processing.whisper_model") == "large-v3"
    
    def test_reload_failure_rollback(self, temp_config_dir, valid_config):
        """Test that reload rolls back on failure."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        original_model = config_manager.get("processing.whisper_model")
        
        # Write invalid configuration
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: syntax:")
        
        # Attempt reload (should fail)
        with pytest.raises(ConfigurationError):
            config_manager.reload()
        
        # Verify configuration wasn't changed
        assert config_manager.get("processing.whisper_model") == original_model
    
    def test_to_dict(self, temp_config_dir, valid_config):
        """Test converting configuration to dictionary."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        config_dict = config_manager.to_dict()
        
        # Verify it's a copy (not reference)
        config_dict["device"]["identifier"] = "modified"
        assert config_manager.get("device.identifier") == "Sony_ICD-TX660"
        
        # Verify structure
        assert "device" in config_dict
        assert "paths" in config_dict
        assert "processing" in config_dict
        assert "llm" in config_dict
    
    def test_thread_safety(self, temp_config_dir, valid_config):
        """Test thread-safe access to configuration."""
        import threading
        import time
        
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        results = []
        
        def read_config():
            for _ in range(100):
                value = config_manager.get("device.identifier")
                results.append(value)
                time.sleep(0.001)
        
        def reload_config():
            for _ in range(10):
                try:
                    config_manager.reload()
                except:
                    pass
                time.sleep(0.01)
        
        # Create threads
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=read_config))
        threads.append(threading.Thread(target=reload_config))
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all reads were successful
        assert all(r == "Sony_ICD-TX660" for r in results)
    
    def test_path_validation_and_creation(self, temp_config_dir, valid_config):
        """Test path validation and directory creation."""
        config_path = temp_config_dir / "config.yaml"
        
        # Use temp directory for paths
        valid_config["paths"] = {
            "audio_destination": str(temp_config_dir / "audio"),
            "transcriptions": str(temp_config_dir / "transcriptions"),
            "summaries": str(temp_config_dir / "summaries")
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(config_path)
        
        # Verify directories were created
        assert (temp_config_dir / "audio").exists()
        assert (temp_config_dir / "transcriptions").exists()
        assert (temp_config_dir / "summaries").exists()
    
    def test_logging_masks_sensitive_data(self, temp_config_dir, config_with_env_vars, caplog):
        """Test that sensitive data is masked in logs."""
        config_path = temp_config_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_env_vars, f)
        
        # Use temp directory paths
        test_audio_path = str(temp_config_dir / "test_audio")
        with patch.dict(os.environ, {
            'TEST_AUDIO_PATH': test_audio_path,
            'GEMINI_API_KEY': 'secret-api-key-123'
        }):
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            
            config_manager = ConfigManager(config_path)
            
            # Check that API key is masked in logs
            log_output = caplog.text
            assert "secret-api-key-123" not in log_output
            assert "***MASKED***" in log_output or "api_key" not in log_output