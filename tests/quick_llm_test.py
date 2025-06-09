"""
Quick test script to verify Ollama is working with llama3.2 model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.providers import OllamaProvider
from src.llm import LLMAnalyzer, Conversation
from datetime import datetime


def test_ollama_connection():
    """Test if Ollama is accessible and llama3.2 is available."""
    print("Testing Ollama connection...")
    
    # Test with llama3.2
    provider = OllamaProvider({
        "host": "localhost",
        "port": 11434,
        "model": "llama3.2:latest",  # Using llama3.2:latest
        "timeout": 30
    })
    
    if provider.is_available():
        print("✓ Ollama is running and accessible")
        
        # Test generation
        try:
            response = provider.generate("Di 'Hola mundo' en español en una sola línea")
            print(f"✓ Generation test successful: {response.strip()}")
            return True
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return False
    else:
        print("✗ Ollama is not available. Make sure it's running with: ollama serve")
        return False


def test_simple_analysis():
    """Test a simple conversation analysis."""
    print("\nTesting simple conversation analysis...")
    
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
                        "model": "llama3.2:latest",  # Using llama3.2:latest
                        "timeout": 60,
                        "temperature": 0.3
                    }
                }
            return default
    
    # Create analyzer
    analyzer = LLMAnalyzer(MockConfig())
    
    # Create simple test conversation
    conversation = Conversation(
        id="test-simple-1",
        transcript="Conversación de prueba",
        speaker_transcript="""[Hablante 1] Hola equipo, necesitamos revisar el proyecto.
[Hablante 2] Claro, ¿qué necesitas que haga?
[Hablante 1] Por favor prepara el informe para mañana.
[Hablante 2] Entendido, lo tendré listo.""",
        speakers=["Hablante 1", "Hablante 2"],
        duration=120.0,
        timestamp=datetime.now()
    )
    
    try:
        print("Analyzing conversation...")
        analysis = analyzer.analyze_conversation(conversation)
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"Provider used: {analysis.llm_provider}")
        print(f"\nSummary preview: {analysis.summary[:200]}...")
        print(f"\nExtracted:")
        print(f"- {len(analysis.tasks)} tasks")
        print(f"- {len(analysis.todos)} todos")
        print(f"- {len(analysis.followups)} followups")
        print(f"- {len(analysis.key_points)} key points")
        
        if analysis.tasks:
            print("\nFirst task found:")
            task = analysis.tasks[0]
            print(f"- Description: {task.description}")
            print(f"- Assigned by: {task.assigned_by}")
            print(f"- Assigned to: {task.assignee}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_02():
    """Test with the actual sample_02.txt file."""
    print("\nTesting with sample_02.txt...")
    
    # Read sample file
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "personal_samples/sample_02.txt")
    
    if not os.path.exists(sample_path):
        print(f"✗ Sample file not found: {sample_path}")
        return False
    
    with open(sample_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract speakers and create speaker transcript
    # This is a simplified extraction - in production, the conversation segmenter would do this
    lines = content.split('\n')
    speaker_lines = []
    current_speaker = None
    
    for line in lines:
        if ': ' in line and not line.startswith(' '):
            parts = line.split(': ', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                if speaker in ['Rafael Martinez', 'ARMANDO GARZA']:
                    if speaker == 'Rafael Martinez':
                        current_speaker = 'Hablante 1'
                    else:
                        current_speaker = 'Hablante 2'
                    speaker_lines.append(f"[{current_speaker}] {text}")
    
    speaker_transcript = '\n'.join(speaker_lines[:50])  # Use first 50 lines for testing
    
    # Create conversation
    conversation = Conversation(
        id="sample-02",
        transcript=content,
        speaker_transcript=speaker_transcript,
        speakers=["Rafael Martinez", "ARMANDO GARZA"],
        duration=993,  # 16:33 from the file
        timestamp=datetime.now()
    )
    
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
                        "model": "llama3.2",
                        "timeout": 120,
                        "temperature": 0.3
                    }
                }
            return default
    
    # Analyze
    analyzer = LLMAnalyzer(MockConfig())
    
    try:
        print("Analyzing ERP audit conversation...")
        analysis = analyzer.analyze_conversation(conversation)
        
        print(f"\n✓ Analysis completed!")
        print(f"\nSummary preview: {analysis.summary[:300]}...")
        
        print(f"\nKey points found ({len(analysis.key_points)}):")
        for i, point in enumerate(analysis.key_points[:3], 1):
            print(f"{i}. {point}")
        
        if analysis.tasks:
            print(f"\nTasks found ({len(analysis.tasks)}):")
            for i, task in enumerate(analysis.tasks[:3], 1):
                print(f"{i}. {task.description}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("LLM ANALYZER QUICK TEST")
    print("="*60)
    
    # Test 1: Ollama connection
    if not test_ollama_connection():
        print("\nPlease ensure Ollama is running and llama3.2 is installed:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.2")
        exit(1)
    
    # Test 2: Simple analysis
    test_simple_analysis()
    
    # Test 3: Sample file analysis
    test_with_sample_02()
    
    print("\n" + "="*60)
    print("Quick tests completed!")
    print("For comprehensive testing, run:")
    print("python tests/llm_test_framework.py")
    print("="*60)