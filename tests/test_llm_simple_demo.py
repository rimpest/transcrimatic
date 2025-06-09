"""
Simple demo of LLM Analyzer functionality with minimal data.
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation
from src.llm.providers import OllamaProvider


def test_provider_only():
    """Test just the Ollama provider with a simple prompt."""
    print("Testing Ollama provider directly...")
    
    provider = OllamaProvider({
        "host": "localhost",
        "port": 11434,
        "model": "llama3.2:latest",
        "timeout": 30
    })
    
    if not provider.is_available():
        print("✗ Ollama is not available!")
        return False
    
    print("✓ Ollama is available")
    
    # Test with a simple analysis prompt
    prompt = """Analiza esta conversación corta y extrae la tarea principal:
[Hablante 1] Necesito el reporte de ventas para mañana.
[Hablante 2] Lo tendré listo antes del mediodía.

Responde con: TAREA: [descripción] | RESPONSABLE: [quien] | FECHA: [cuando]"""
    
    try:
        print("\nGenerating response...")
        response = provider.generate(prompt, temperature=0.1)
        print("\nResponse:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_minimal_conversation():
    """Test with minimal conversation data."""
    print("\n\nTesting LLM Analyzer with minimal data...")
    
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
                        "timeout": 60,
                        "temperature": 0.3
                    }
                }
            return default
    
    # Create very simple conversation
    conversation = Conversation(
        id="demo-1",
        transcript="Conversación sobre reporte de ventas",
        speaker_transcript="""[Hablante 1] Buenos días. Necesito el reporte de ventas para mañana por la mañana.
[Hablante 2] Buenos días. Entendido, lo tendré listo antes del mediodía.
[Hablante 1] Perfecto. Incluye las cifras del último trimestre por favor.
[Hablante 2] Claro, incluiré todo el análisis trimestral.""",
        speakers=["Jefe", "Analista"],
        duration=60.0,
        timestamp=datetime.now()
    )
    
    print(f"\nConversation ID: {conversation.id}")
    print(f"Duration: {conversation.duration}s")
    print(f"Speakers: {', '.join(conversation.speakers)}")
    
    print("\nTranscript:")
    print("-" * 60)
    print(conversation.speaker_transcript)
    print("-" * 60)
    
    # Create analyzer
    analyzer = LLMAnalyzer(MockConfig())
    
    try:
        print("\nAnalyzing conversation...")
        
        # Test just summary generation first
        summary = analyzer.generate_summary(conversation)
        
        print("\n📝 SUMMARY:")
        print("-" * 60)
        print(summary)
        
        # Test task extraction
        print("\n\nExtracting tasks...")
        tasks = analyzer.extract_tasks(conversation)
        
        print(f"\n📋 TASKS FOUND: {len(tasks)}")
        print("-" * 60)
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task.description}")
            print(f"   Assignee: {task.assignee or 'Not specified'}")
            print(f"   Assigned by: {task.assigned_by or 'Not specified'}")
            print(f"   Priority: {task.priority}")
            if task.due_date:
                print(f"   Due: {task.due_date}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_and_preview_transcription():
    """Load and preview the transcription data structure."""
    print("\n\nPreviewing transcription data structure...")
    
    transcription_file = "transcription_result_sample_audio_long.json"
    
    if not os.path.exists(transcription_file):
        print(f"✗ File not found: {transcription_file}")
        return
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTranscription file: {transcription_file}")
    print(f"Audio duration: {data['audio_file']['duration']}s")
    print(f"Total segments: {len(data['segments'])}")
    
    # Show first 5 segments
    print("\nFirst 5 segments:")
    print("-" * 60)
    for i, seg in enumerate(data['segments'][:5]):
        print(f"\nSegment {i}:")
        print(f"  Time: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
        print(f"  Speaker: {seg['speaker_id']}")
        print(f"  Text: {seg['text']}")
    
    # Show speaker distribution
    speakers = {}
    for seg in data['segments']:
        speaker = seg['speaker_id']
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    print(f"\n\nSpeaker distribution:")
    print("-" * 60)
    for speaker, count in speakers.items():
        print(f"{speaker}: {count} segments")


if __name__ == "__main__":
    print("="*80)
    print("LLM ANALYZER SIMPLE DEMO")
    print("="*80)
    
    # Test 1: Direct provider test
    if test_provider_only():
        # Test 2: Minimal conversation
        test_minimal_conversation()
    
    # Test 3: Preview transcription structure
    load_and_preview_transcription()
    
    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80)