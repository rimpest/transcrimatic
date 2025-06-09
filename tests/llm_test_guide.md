# LLM Analyzer Testing Guide

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up LLM Providers

#### Option A: Ollama (Recommended for Testing)
1. Install Ollama:
   ```bash
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # macOS
   brew install ollama
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull a Spanish-capable model:
   ```bash
   ollama pull llama3:8b
   # or for better Spanish support:
   ollama pull mistral:7b-instruct
   ```

#### Option B: Google Gemini
1. Get API key from: https://makersuite.google.com/app/apikey
2. Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

#### Option C: OpenAI
1. Get API key from: https://platform.openai.com/api-keys
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Running Tests

### 1. Unit Tests
```bash
# Run all LLM tests
python -m pytest tests/test_llm_analyzer.py -v

# Run specific test classes
python -m pytest tests/test_llm_analyzer.py::TestLLMProviders -v
python -m pytest tests/test_llm_analyzer.py::TestDataModels -v
python -m pytest tests/test_llm_analyzer.py::TestLLMAnalyzer -v

# Run with coverage
python -m pytest tests/test_llm_analyzer.py --cov=src/llm --cov-report=html
```

### 2. Integration Tests

Create a test script to verify actual LLM integration:

```python
# tests/test_llm_integration.py
import os
from datetime import datetime
from src.llm import LLMAnalyzer, Conversation

# Mock config manager
class MockConfig:
    def get(self, key, default=None):
        if key == "llm":
            return {
                "provider": "ollama",
                "fallback_order": ["ollama", "gemini", "openai"],
                "ollama": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 11434,
                    "model": "llama3:8b",
                    "timeout": 120,
                    "temperature": 0.3
                },
                "gemini": {
                    "enabled": True,
                    "api_key": os.environ.get("GEMINI_API_KEY"),
                    "model": "gemini-1.5-pro",
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                "openai": {
                    "enabled": True,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "model": "gpt-4-turbo-preview",
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            }
        return default

# Test conversation
test_conversation = Conversation(
    id="test-1",
    transcript="Conversación de prueba",
    speaker_transcript="""[Hablante 1] Buenos días equipo. Necesitamos revisar el proyecto Alpha.
[Hablante 2] Claro, ¿cuáles son las prioridades?
[Hablante 1] Primero, necesito que completes el informe de progreso para mañana.
[Hablante 2] Entendido, lo tendré listo antes del mediodía.
[Hablante 1] Perfecto. También hay que hacer seguimiento con el cliente sobre los cambios.
[Hablante 2] Me encargo de llamarlos esta tarde.
[Hablante 1] Excelente. No olvides actualizar el dashboard del proyecto.
[Hablante 2] Anotado. ¿Algo más?
[Hablante 1] Por ahora es todo. Nos vemos en la reunión del viernes.""",
    speakers=["Hablante 1", "Hablante 2"],
    duration=180.0,
    timestamp=datetime.now()
)

# Run test
if __name__ == "__main__":
    config = MockConfig()
    analyzer = LLMAnalyzer(config)
    
    print("Testing LLM Analyzer...")
    print("-" * 50)
    
    try:
        # Test analysis
        analysis = analyzer.analyze_conversation(test_conversation)
        
        print(f"✓ Analysis completed using: {analysis.llm_provider}")
        print(f"\nSummary:\n{analysis.summary}")
        print(f"\nKey Points: {len(analysis.key_points)}")
        for point in analysis.key_points:
            print(f"  - {point}")
        
        print(f"\nTasks found: {len(analysis.tasks)}")
        for task in analysis.tasks:
            print(f"  - {task.description}")
            print(f"    Assigned by: {task.assigned_by}")
            print(f"    Assigned to: {task.assignee}")
            print(f"    Priority: {task.priority}")
        
        print(f"\nTodos found: {len(analysis.todos)}")
        for todo in analysis.todos:
            print(f"  - {todo.description} (mentioned by: {todo.mentioned_by})")
        
        print(f"\nFollowups found: {len(analysis.followups)}")
        for followup in analysis.followups:
            print(f"  - {followup.topic}: {followup.action_required}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
```

### 3. Provider-Specific Tests

```python
# tests/test_provider_availability.py
from src.llm.providers import OllamaProvider, GeminiProvider, OpenAIProvider
import os

def test_ollama():
    provider = OllamaProvider({"host": "localhost", "port": 11434})
    if provider.is_available():
        print("✓ Ollama is available")
        response = provider.generate("Di 'Hola mundo' en español")
        print(f"  Response: {response[:50]}...")
    else:
        print("✗ Ollama is not available")

def test_gemini():
    if not os.environ.get("GEMINI_API_KEY"):
        print("✗ Gemini API key not set")
        return
    
    try:
        provider = GeminiProvider({"api_key": os.environ.get("GEMINI_API_KEY")})
        if provider.is_available():
            print("✓ Gemini is available")
    except Exception as e:
        print(f"✗ Gemini error: {e}")

def test_openai():
    if not os.environ.get("OPENAI_API_KEY"):
        print("✗ OpenAI API key not set")
        return
    
    try:
        provider = OpenAIProvider({"api_key": os.environ.get("OPENAI_API_KEY")})
        if provider.is_available():
            print("✓ OpenAI is available")
    except Exception as e:
        print(f"✗ OpenAI error: {e}")

if __name__ == "__main__":
    print("Testing LLM Provider Availability...")
    print("-" * 50)
    test_ollama()
    test_gemini()
    test_openai()
```

## Configuration Testing

### 1. Create Test Configuration
```yaml
# config.test.yaml
llm:
  provider: "ollama"
  fallback_order: ["ollama", "gemini", "openai"]
  
  ollama:
    enabled: true
    host: "localhost"
    port: 11434
    model: "llama3:8b"
    timeout: 30
    temperature: 0.3
    
  gemini:
    enabled: false
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-1.5-pro"
    temperature: 0.3
    max_tokens: 1000
    
  openai:
    enabled: false
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-turbo-preview"
    temperature: 0.3
    max_tokens: 1000
```

### 2. Test Fallback Logic
```python
# tests/test_fallback.py
# This test simulates primary provider failure
from unittest.mock import Mock
from src.llm import LLMAnalyzer

config = Mock()
config.get.return_value = {
    "provider": "ollama",
    "fallback_order": ["ollama", "gemini"],
    # ... rest of config
}

# Test with Ollama stopped to trigger fallback
```

## Performance Testing

```python
# tests/test_performance.py
import time
from concurrent.futures import ThreadPoolExecutor

def test_batch_processing():
    # Create multiple conversations
    conversations = [create_test_conversation(i) for i in range(5)]
    
    start = time.time()
    results = analyzer.batch_analyze(conversations)
    duration = time.time() - start
    
    print(f"Analyzed {len(results)} conversations in {duration:.2f}s")
    print(f"Average: {duration/len(results):.2f}s per conversation")
```

## Debugging Tips

1. **Enable logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check provider status**:
   ```bash
   # For Ollama
   curl http://localhost:11434/api/tags
   ```

3. **Test prompts directly**:
   ```python
   from src.llm.prompts import SUMMARY_PROMPT_WITH_SPEAKERS
   print(SUMMARY_PROMPT_WITH_SPEAKERS.format(
       conversation_text="[Hablante 1] Test",
       duration="5",
       speaker_count=1
   ))
   ```

## Common Issues

1. **Ollama not running**: Start with `ollama serve`
2. **Model not found**: Pull with `ollama pull llama3:8b`
3. **API key issues**: Check environment variables
4. **Timeout errors**: Increase timeout in config
5. **Memory issues**: Use smaller models or reduce batch size