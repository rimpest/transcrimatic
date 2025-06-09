"""
Test LLM Analyzer with actual LLM calls to show real output.
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation
from src.llm.providers import OllamaProvider


class MockConfig:
    """Mock configuration for testing."""
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
                    "timeout": 90,
                    "temperature": 0.3
                },
                "cost_optimization": {
                    "cache_similar_prompts": True,
                    "batch_conversations": True,
                    "max_batch_size": 3
                }
            }
        return default


def test_real_conversation_analysis():
    """Test with a real business conversation."""
    
    print("="*80)
    print("TESTING LLM ANALYZER WITH REAL CALLS")
    print("="*80)
    
    # Test 1: Business conversation with clear tasks
    business_conversation = Conversation(
        id="business-meeting-001",
        transcript="Reunión sobre el proyecto Alpha",
        speaker_transcript="""[Hablante 1] Buenos días equipo. Necesitamos revisar el estado del proyecto Alpha.
[Hablante 2] Buenos días María. El proyecto va según lo planeado.
[Hablante 1] Excelente. Carlos, necesito que prepares el informe ejecutivo para el viernes.
[Hablante 2] Entendido. ¿Qué información específica necesitas?
[Hablante 1] Incluye el progreso actual, los hitos alcanzados y los riesgos identificados.
[Hablante 2] Perfecto. Lo tendré listo para el viernes por la mañana.
[Hablante 1] También necesitamos coordinar con el equipo de diseño. Ana, ¿puedes encargarte?
[Hablante 3] Claro, agendaré una reunión con ellos esta semana.
[Hablante 1] No olviden actualizar el tablero de tareas diariamente.
[Hablante 2] Anotado. ¿Hay algo más que debamos priorizar?
[Hablante 1] Por ahora eso es todo. Nos vemos el viernes para revisar el informe.""",
        speakers=["María García", "Carlos Rodríguez", "Ana López"],
        duration=300.0,
        timestamp=datetime.now()
    )
    
    # Initialize analyzer
    print("\nInitializing LLM Analyzer...")
    analyzer = LLMAnalyzer(MockConfig())
    
    # Analyze conversation
    print("\nAnalyzing business conversation...")
    print("(This will take 30-60 seconds for complete analysis)")
    
    start_time = time.time()
    
    try:
        # Full analysis
        analysis = analyzer.analyze_conversation(business_conversation)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Analysis completed in {elapsed:.1f} seconds")
        
        # Display results
        print("\n" + "="*80)
        print("ANALYSIS RESULTS - BUSINESS MEETING")
        print("="*80)
        
        print(f"\n📝 SUMMARY ({len(analysis.summary)} chars):")
        print("-" * 70)
        print(analysis.summary)
        
        print(f"\n🔑 KEY POINTS ({len(analysis.key_points)}):")
        print("-" * 70)
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        print(f"\n📋 TASKS EXTRACTED ({len(analysis.tasks)}):")
        print("-" * 70)
        if analysis.tasks:
            for i, task in enumerate(analysis.tasks, 1):
                print(f"\nTask {i}:")
                print(f"  📝 Description: {task.description}")
                print(f"  👤 Assigned to: {task.assignee or 'Not specified'}")
                print(f"  👤 Assigned by: {task.assigned_by or 'Not specified'}")
                print(f"  📅 Due date: {task.due_date or 'Not specified'}")
                print(f"  🔴 Priority: {task.priority}")
                if task.context:
                    print(f"  📎 Context: {task.context}")
        else:
            print("No tasks found")
        
        print(f"\n📌 TO-DOS ({len(analysis.todos)}):")
        print("-" * 70)
        if analysis.todos:
            for i, todo in enumerate(analysis.todos, 1):
                print(f"\nTo-do {i}:")
                print(f"  📝 {todo.description}")
                print(f"  👤 Mentioned by: {todo.mentioned_by or 'Not specified'}")
                print(f"  📁 Category: {todo.category}")
                print(f"  ⚡ Urgency: {todo.urgency}")
        else:
            print("No to-dos found")
        
        print(f"\n🔄 FOLLOW-UPS ({len(analysis.followups)}):")
        print("-" * 70)
        if analysis.followups:
            for i, followup in enumerate(analysis.followups, 1):
                print(f"\nFollow-up {i}:")
                print(f"  📌 Topic: {followup.topic}")
                print(f"  🎯 Action: {followup.action_required}")
                print(f"  👤 Responsible: {followup.responsible_party or 'Not specified'}")
                if followup.deadline:
                    print(f"  📅 Deadline: {followup.deadline}")
        else:
            print("No follow-ups found")
        
        # Save JSON output
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"real_analysis_{analysis.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"\n\n💾 Full JSON output saved to: {output_file}")
        
        # Show JSON structure
        print("\n📄 JSON OUTPUT STRUCTURE:")
        print("-" * 70)
        print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)[:1000] + "...")
        
        return analysis
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_podcast_sample():
    """Test with actual podcast transcription."""
    
    print("\n\n" + "="*80)
    print("TESTING WITH PODCAST TRANSCRIPTION")
    print("="*80)
    
    # Load transcription data
    transcription_file = "transcription_result_sample_audio_long.json"
    
    if not os.path.exists(transcription_file):
        print(f"⚠️  {transcription_file} not found, skipping podcast test")
        return
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use first 10 segments for faster processing
    segments = data['segments'][:10]
    
    # Build speaker transcript
    speaker_lines = []
    for seg in segments:
        speaker_num = int(seg['speaker_id'].split('_')[1]) + 1
        speaker_lines.append(f"[Hablante {speaker_num}] {seg['text']}")
    
    podcast_conversation = Conversation(
        id="podcast-sample",
        transcript=" ".join([seg['text'] for seg in segments]),
        speaker_transcript="\n".join(speaker_lines),
        speakers=["Presentador", "Santiago"],
        duration=30.0,  # First 30 seconds
        timestamp=datetime.now()
    )
    
    print(f"\nLoaded podcast with {len(segments)} segments")
    print("\nTranscript preview:")
    print("-" * 70)
    print(podcast_conversation.speaker_transcript[:400] + "...")
    
    # Analyze
    analyzer = LLMAnalyzer(MockConfig())
    
    print("\n\nAnalyzing podcast conversation...")
    start_time = time.time()
    
    try:
        analysis = analyzer.analyze_conversation(podcast_conversation)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Analysis completed in {elapsed:.1f} seconds")
        
        print("\n" + "="*80)
        print("ANALYSIS RESULTS - PODCAST")
        print("="*80)
        
        print(f"\n📝 SUMMARY:")
        print("-" * 70)
        print(analysis.summary)
        
        print(f"\n🔑 KEY POINTS ({len(analysis.key_points)}):")
        print("-" * 70)
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        # For podcasts, we expect fewer tasks
        print(f"\n📋 TASKS: {len(analysis.tasks)} (expected: 0 for podcast)")
        print(f"📌 TO-DOS: {len(analysis.todos)}")
        print(f"🔄 FOLLOW-UPS: {len(analysis.followups)}")
        
        return analysis
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_quick_summary_only():
    """Test just summary generation for quick results."""
    
    print("\n\n" + "="*80)
    print("QUICK TEST - SUMMARY ONLY")
    print("="*80)
    
    simple_conv = Conversation(
        id="quick-test",
        transcript="Quick test",
        speaker_transcript="""[Hablante 1] Necesitamos entregar el proyecto antes del viernes.
[Hablante 2] Entendido, me pongo en ello ahora mismo.
[Hablante 1] Perfecto, avísame cuando esté listo.""",
        speakers=["Jefe", "Empleado"],
        duration=30.0,
        timestamp=datetime.now()
    )
    
    analyzer = LLMAnalyzer(MockConfig())
    
    print("\nGenerating summary only...")
    start_time = time.time()
    
    try:
        summary = analyzer.generate_summary(simple_conv)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Summary generated in {elapsed:.1f} seconds")
        print("\nSUMMARY:")
        print("-" * 70)
        print(summary)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("LLM ANALYZER - REAL OUTPUT TEST")
    print("This will make actual calls to Ollama")
    print("Make sure Ollama is running with llama3.2:latest")
    print("-" * 80)
    
    # Test 1: Quick summary test
    test_quick_summary_only()
    
    # Test 2: Full business conversation analysis
    test_real_conversation_analysis()
    
    # Test 3: Podcast sample (optional)
    test_with_podcast_sample()
    
    print("\n\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("Check test_outputs/ directory for JSON files")
    print("="*80)