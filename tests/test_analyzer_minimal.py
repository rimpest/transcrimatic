"""
Minimal test of LLM Analyzer to debug issues.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation
from datetime import datetime

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_minimal():
    """Minimal test with simple conversation."""
    
    # Mock config
    class MockConfig:
        def get(self, key, default=None):
            if key == "llm":
                return {
                    "provider": "ollama",
                    "fallback_order": ["ollama"],
                    "ollama": {
                        "enabled": True,
                        "host": "localhost",
                        "port": 11434,
                        "model": "llama3.2:latest",
                        "timeout": 60,  # Increased timeout
                        "temperature": 0.3
                    }
                }
            return default
    
    print("Creating analyzer...")
    analyzer = LLMAnalyzer(MockConfig())
    
    # Very simple conversation
    conversation = Conversation(
        id="test-1",
        transcript="Simple test",
        speaker_transcript="[Hablante 1] Necesito el reporte.\n[Hablante 2] Lo haré mañana.",
        speakers=["Hablante 1", "Hablante 2"],
        duration=30.0,
        timestamp=datetime.now()
    )
    
    print("\nAnalyzing conversation...")
    try:
        analysis = analyzer.analyze_conversation(conversation)
        print(f"\n✓ Success! Provider: {analysis.llm_provider}")
        print(f"Summary length: {len(analysis.summary)} chars")
        print(f"Tasks found: {len(analysis.tasks)}")
        
        if analysis.tasks:
            print(f"\nFirst task: {analysis.tasks[0].description}")
            
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_minimal()