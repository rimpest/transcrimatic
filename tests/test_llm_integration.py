"""
Integration tests for LLM Analyzer with actual LLM providers.
Run this to test real LLM functionality.
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation


class MockConfig:
    """Mock configuration manager for testing."""
    
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
                },
                "cost_optimization": {
                    "cache_similar_prompts": True,
                    "batch_conversations": True,
                    "max_batch_size": 5
                }
            }
        return default


def create_test_conversation():
    """Create a realistic test conversation in Spanish."""
    return Conversation(
        id="test-1",
        transcript="Conversación de prueba sobre proyecto Alpha",
        speaker_transcript="""[Hablante 1] Buenos días equipo. Necesitamos revisar el proyecto Alpha.
[Hablante 2] Claro, ¿cuáles son las prioridades?
[Hablante 1] Primero, necesito que completes el informe de progreso para mañana. Es muy importante que incluya todos los hitos alcanzados.
[Hablante 2] Entendido, lo tendré listo antes del mediodía. ¿Necesitas algún formato específico?
[Hablante 1] Usa el formato estándar de la empresa. También hay que hacer seguimiento con el cliente sobre los cambios propuestos.
[Hablante 2] Me encargo de llamarlos esta tarde. ¿Hay algún punto específico que deba mencionar?
[Hablante 1] Sí, confirma que están de acuerdo con el nuevo cronograma y pregunta sobre el presupuesto adicional.
[Hablante 2] Perfecto, tomaré nota de sus comentarios.
[Hablante 1] Excelente. No olvides actualizar el dashboard del proyecto con los nuevos avances.
[Hablante 2] Anotado. ¿Algo más?
[Hablante 1] Hay un pendiente importante: necesitamos revisar la arquitectura del sistema antes del viernes.
[Hablante 2] De acuerdo, agendaré una sesión de revisión.
[Hablante 1] Por ahora es todo. Nos vemos en la reunión del viernes para revisar el progreso.""",
        speakers=["Hablante 1", "Hablante 2"],
        duration=240.0,  # 4 minutes
        timestamp=datetime.now()
    )


def test_single_conversation():
    """Test analyzing a single conversation."""
    print("\n" + "="*60)
    print("TESTING SINGLE CONVERSATION ANALYSIS")
    print("="*60 + "\n")
    
    config = MockConfig()
    analyzer = LLMAnalyzer(config)
    conversation = create_test_conversation()
    
    try:
        print("Analyzing conversation...")
        analysis = analyzer.analyze_conversation(conversation)
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"  Provider used: {analysis.llm_provider}")
        print(f"  Duration: {conversation.duration/60:.1f} minutes")
        print(f"  Participants: {', '.join(analysis.participants)}")
        
        print(f"\n📝 SUMMARY:")
        print("-" * 40)
        print(analysis.summary)
        
        print(f"\n🔑 KEY POINTS ({len(analysis.key_points)}):")
        print("-" * 40)
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        print(f"\n📋 TASKS ({len(analysis.tasks)}):")
        print("-" * 40)
        for i, task in enumerate(analysis.tasks, 1):
            print(f"{i}. {task.description}")
            print(f"   Assigned by: {task.assigned_by or 'N/A'}")
            print(f"   Assigned to: {task.assignee or 'Not specified'}")
            print(f"   Priority: {task.priority}")
            if task.due_date:
                print(f"   Due date: {task.due_date}")
            if task.context:
                print(f"   Context: {task.context}")
            print()
        
        print(f"\n📌 TO-DOS ({len(analysis.todos)}):")
        print("-" * 40)
        for i, todo in enumerate(analysis.todos, 1):
            print(f"{i}. {todo.description}")
            print(f"   Mentioned by: {todo.mentioned_by or 'N/A'}")
            print(f"   Category: {todo.category}")
            print(f"   Urgency: {todo.urgency}")
            print()
        
        print(f"\n🔄 FOLLOW-UPS ({len(analysis.followups)}):")
        print("-" * 40)
        for i, followup in enumerate(analysis.followups, 1):
            print(f"{i}. {followup.topic}")
            print(f"   Action: {followup.action_required}")
            print(f"   Responsible: {followup.responsible_party or 'Not specified'}")
            if followup.deadline:
                print(f"   Deadline: {followup.deadline}")
            print(f"   Mentioned by: {followup.mentioned_by or 'N/A'}")
            print()
        
        return analysis
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_processing():
    """Test batch processing of multiple conversations."""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60 + "\n")
    
    config = MockConfig()
    analyzer = LLMAnalyzer(config)
    
    # Create multiple test conversations
    conversations = []
    for i in range(3):
        conv = create_test_conversation()
        conv.id = f"test-batch-{i+1}"
        conversations.append(conv)
    
    try:
        print(f"Analyzing {len(conversations)} conversations in batch...")
        import time
        start_time = time.time()
        
        analyses = analyzer.batch_analyze(conversations)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Batch analysis completed!")
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Average per conversation: {elapsed_time/len(conversations):.2f} seconds")
        print(f"  Successfully analyzed: {len(analyses)}/{len(conversations)}")
        
        # Generate daily summary
        print("\n" + "="*60)
        print("GENERATING DAILY SUMMARY")
        print("="*60 + "\n")
        
        daily_summary = analyzer.generate_daily_summary(analyses)
        
        print(f"📊 DAILY SUMMARY for {daily_summary.date}")
        print("-" * 40)
        print(f"Total conversations: {daily_summary.total_conversations}")
        print(f"Total duration: {daily_summary.total_duration/60:.1f} minutes")
        print(f"Total tasks: {len(daily_summary.all_tasks)}")
        print(f"Total to-dos: {len(daily_summary.all_todos)}")
        print(f"Total follow-ups: {len(daily_summary.all_followups)}")
        
        print(f"\n🌟 HIGHLIGHTS:")
        for i, highlight in enumerate(daily_summary.highlights, 1):
            print(f"{i}. {highlight}")
        
        print(f"\n👥 SPEAKER PARTICIPATION:")
        for speaker, percentage in daily_summary.speaker_participation.items():
            print(f"  {speaker}: {percentage}%")
        
        return analyses
        
    except Exception as e:
        print(f"\n✗ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_provider_fallback():
    """Test fallback mechanism when primary provider fails."""
    print("\n" + "="*60)
    print("TESTING PROVIDER FALLBACK")
    print("="*60 + "\n")
    
    # Create config with non-existent primary provider
    config = MockConfig()
    
    # Modify config to use a non-existent Ollama instance
    llm_config = config.get("llm")
    llm_config["ollama"]["port"] = 99999  # Invalid port
    
    analyzer = LLMAnalyzer(config)
    conversation = create_test_conversation()
    
    try:
        print("Testing with invalid primary provider...")
        analysis = analyzer.analyze_conversation(conversation)
        
        if analysis.llm_provider != "ollama":
            print(f"✓ Fallback worked! Used provider: {analysis.llm_provider}")
        else:
            print("✗ Fallback did not trigger as expected")
            
    except Exception as e:
        print(f"✗ All providers failed: {e}")


def test_provider_availability():
    """Test which providers are available."""
    print("\n" + "="*60)
    print("TESTING PROVIDER AVAILABILITY")
    print("="*60 + "\n")
    
    from src.llm.providers import OllamaProvider, GeminiProvider, OpenAIProvider
    
    # Test Ollama
    print("1. Ollama Provider:")
    try:
        ollama = OllamaProvider({"host": "localhost", "port": 11434})
        if ollama.is_available():
            print("   ✓ Available")
            # Test generation
            response = ollama.generate("Di 'Hola' en una palabra")
            print(f"   Test response: {response.strip()[:50]}")
        else:
            print("   ✗ Not available (is Ollama running?)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test Gemini
    print("\n2. Gemini Provider:")
    if os.environ.get("GEMINI_API_KEY"):
        try:
            gemini = GeminiProvider({"api_key": os.environ.get("GEMINI_API_KEY")})
            if gemini.is_available():
                print("   ✓ Available")
            else:
                print("   ✗ Not available")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("   ✗ API key not set (GEMINI_API_KEY)")
    
    # Test OpenAI
    print("\n3. OpenAI Provider:")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            openai = OpenAIProvider({"api_key": os.environ.get("OPENAI_API_KEY")})
            if openai.is_available():
                print("   ✓ Available")
            else:
                print("   ✗ Not available")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    else:
        print("   ✗ API key not set (OPENAI_API_KEY)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LLM ANALYZER INTEGRATION TESTS")
    print("="*60)
    
    # Test provider availability first
    test_provider_availability()
    
    # Test single conversation
    analysis = test_single_conversation()
    
    # Test batch processing (only if single test succeeded)
    if analysis:
        test_batch_processing()
    
    # Test fallback mechanism
    test_provider_fallback()
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)