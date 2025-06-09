# LLM Analyzer Testing Framework

This directory contains a comprehensive testing framework for the TranscriMatic LLM Analyzer module, designed to work with local Ollama models.

## Prerequisites

1. **Ollama Installation**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull llama3.2 model (or any Spanish-capable model)
   ollama pull llama3.2
   ```

2. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Test

To verify everything is working:

```bash
python tests/quick_llm_test.py
```

This will:
- Test Ollama connection
- Run a simple conversation analysis
- Test with the sample_02.txt file

## Comprehensive Testing

### Run All Tests

```bash
python tests/llm_test_framework.py
```

This runs the complete test suite including:
- Individual conversation tests (business meeting, tech support, client call, team standup)
- Edge case tests (multiple speakers, unclear assignments, informal conversations)
- Batch processing tests
- Performance benchmarks
- Daily summary generation

### Run Specific Tests

```bash
# Test a single conversation type
python tests/llm_test_framework.py --test single --conversation business_meeting

# Test only edge cases
python tests/llm_test_framework.py --test edge

# Test batch processing
python tests/llm_test_framework.py --test batch

# Test performance with 10 conversations
python tests/llm_test_framework.py --test performance

# Test daily summary generation
python tests/llm_test_framework.py --test summary
```

### Options

- `--output`: JSON file for test results (default: test_results.json)
- `--report`: Markdown report file (default: test_report.md)
- `--verbose`: Enable detailed logging

## Test Data

### Spanish Conversations (`test_data/spanish_conversations.py`)

Contains realistic Spanish conversation samples:

1. **Business Meeting** - Planning meeting with task assignments
2. **Tech Support** - IT troubleshooting session
3. **Client Call** - Sales quotation conversation
4. **Team Standup** - Development team daily meeting

Each conversation includes:
- Raw transcript (as would come from Whisper)
- Speaker-labeled transcript
- Expected tasks/outcomes for validation

### Edge Cases

- Multiple speakers talking in same line
- Unclear task assignments
- Informal conversations with embedded tasks

## Test Metrics

The framework evaluates:

1. **Functionality**:
   - Summary generation quality
   - Task extraction accuracy
   - Speaker attribution
   - Key points identification

2. **Performance**:
   - Processing time per conversation
   - Batch processing efficiency
   - Memory usage (if verbose)

3. **Accuracy**:
   - Task recall rate
   - False positive rate
   - Speaker identification accuracy

## Output Files

After running tests:

1. **test_results.json** - Detailed test results with metrics
2. **test_report.md** - Human-readable markdown report
3. **Console output** - Real-time test progress and results

## Customizing Tests

### Add New Test Conversations

Edit `test_data/spanish_conversations.py`:

```python
CONVERSATIONS["new_type"] = {
    "id": "conv-005",
    "title": "Nueva Conversación",
    "raw_transcript": "...",
    "speaker_transcript": "[Hablante 1] ...",
    "duration": 300,
    "speakers": ["Speaker 1", "Speaker 2"],
    "expected_tasks": [...]
}
```

### Modify Ollama Settings

In `llm_test_framework.py`, update the configuration:

```python
def _get_default_config(self) -> Dict:
    return {
        "llm": {
            "ollama": {
                "model": "llama3.2",  # Change model here
                "temperature": 0.3,   # Adjust creativity
                "timeout": 120        # Increase for slower systems
            }
        }
    }
```

## Troubleshooting

### Ollama Not Found

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Model Not Available

```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2
```

### Timeout Errors

Increase timeout in test configuration or use a smaller model.

### Memory Issues

- Use smaller batch sizes
- Use a smaller model (e.g., llama3.2:3b instead of llama3.2:7b)

## Performance Tips

1. **Use GPU acceleration** if available
2. **Adjust batch size** based on your system
3. **Cache results** for repeated tests
4. **Use smaller models** for faster testing

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Start Ollama
  run: |
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama serve &
    sleep 5
    ollama pull llama3.2

- name: Run LLM Tests
  run: python tests/llm_test_framework.py --output ci_results.json

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: llm-test-results
    path: |
      ci_results.json
      test_report.md
```