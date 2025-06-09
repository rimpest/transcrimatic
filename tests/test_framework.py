"""Test to verify pytest framework is working correctly"""

import pytest


class TestFramework:
    """Basic tests to verify test framework"""
    
    def test_basic_assertion(self):
        """Test basic assertion works"""
        assert 1 + 1 == 2
    
    def test_pytest_features(self):
        """Test pytest features work"""
        with pytest.raises(ValueError):
            raise ValueError("Test error")
    
    def test_fixtures(self, temp_dir):
        """Test fixtures work"""
        assert temp_dir.exists()
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello")
        assert test_file.read_text() == "Hello"