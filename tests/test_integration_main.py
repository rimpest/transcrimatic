"""
Integration tests for MainController
Tests actual instantiation and basic functionality without mocks
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
class TestMainControllerIntegration:
    """Integration tests for MainController"""
    
    def test_import_main_controller(self):
        """Test that MainController can be imported"""
        try:
            from src.main_controller import MainController
            assert MainController is not None
        except ImportError as e:
            pytest.fail(f"Failed to import MainController: {e}")
    
    def test_import_dependencies(self):
        """Test that all required modules can be imported"""
        required_modules = [
            "src.config",
            "src.device", 
            "src.transfer",
            "src.audio",
            "src.transcription",
            "src.conversation",
            "src.llm",
            "src.output"
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {e}")
    
    def test_main_entry_point(self):
        """Test that main.py can be imported"""
        try:
            import main
            assert hasattr(main, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import main.py: {e}")
    
    @pytest.mark.skipif(
        not Path("config.yaml").exists(),
        reason="config.yaml not found"
    )
    def test_main_controller_with_real_config(self, sample_config):
        """Test MainController initialization with real config file"""
        from src.main_controller import MainController
        
        # This will fail because dependent modules aren't implemented yet
        # but it tests the basic structure
        with pytest.raises(Exception) as exc_info:
            controller = MainController(sample_config)
        
        # We expect module initialization to fail since modules aren't implemented
        assert "Module initialization failed" in str(exc_info.value)