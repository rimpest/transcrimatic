"""
Test LLM Analyzer with real transcription data.
This script tests the LLM analyzer using actual transcription results
and produces human-readable output for validation.
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation
from src.transcription.data_classes import TranscriptionSegment, Transcription


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
                }
            }
        return default


def load_transcription_data(file_path: str) -> dict:
    """Load transcription data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_conversation(transcription_data: dict) -> Conversation:
    """Convert transcription data to Conversation object for LLM analysis."""
    
    # Extract segments
    segments = transcription_data.get('segments', [])
    
    # Build raw transcript (without speaker labels)
    raw_transcript = " ".join([seg['text'] for seg in segments])
    
    # Build speaker-aware transcript
    speaker_transcript_lines = []
    current_speaker = None
    current_text = []
    
    for segment in segments:
        speaker = segment.get('speaker_id', 'UNKNOWN')
        text = segment.get('text', '').strip()
        
        if speaker != current_speaker:
            # Save previous speaker's text
            if current_speaker and current_text:
                speaker_label = f"Hablante {int(current_speaker.split('_')[1]) + 1}" if 'SPEAKER_' in current_speaker else current_speaker
                speaker_transcript_lines.append(f"[{speaker_label}] {' '.join(current_text)}")
            
            # Start new speaker
            current_speaker = speaker
            current_text = [text] if text else []
        else:
            # Continue with same speaker
            if text:
                current_text.append(text)
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        speaker_label = f"Hablante {int(current_speaker.split('_')[1]) + 1}" if 'SPEAKER_' in current_speaker else current_speaker
        speaker_transcript_lines.append(f"[{speaker_label}] {' '.join(current_text)}")
    
    speaker_transcript = "\n".join(speaker_transcript_lines)
    
    # Get unique speakers
    speakers = list(set([seg.get('speaker_id', 'UNKNOWN') for seg in segments]))
    
    # Get duration
    duration = transcription_data.get('audio_file', {}).get('duration', 60.0)
    
    # Create conversation object
    return Conversation(
        id=f"podcast-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transcript=raw_transcript,
        speaker_transcript=speaker_transcript,
        speakers=speakers,
        duration=duration,
        timestamp=datetime.now(),
        segments=segments  # Include original segments for reference
    )


def print_analysis_results(analysis, conversation):
    """Print analysis results in a human-readable format."""
    
    print("\n" + "="*80)
    print("LLM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📄 CONVERSATION INFO:")
    print(f"   ID: {conversation.id}")
    print(f"   Duration: {conversation.duration:.1f} seconds")
    print(f"   Speakers: {len(conversation.speakers)} ({', '.join([f'Hablante {i+1}' for i in range(len(conversation.speakers))])})")
    print(f"   Provider Used: {analysis.llm_provider}")
    
    print(f"\n📝 SUMMARY:")
    print("-" * 60)
    print(analysis.summary)
    
    print(f"\n🔑 KEY POINTS ({len(analysis.key_points)}):")
    print("-" * 60)
    for i, point in enumerate(analysis.key_points, 1):
        print(f"{i}. {point}")
    
    print(f"\n📋 TASKS EXTRACTED ({len(analysis.tasks)}):")
    print("-" * 60)
    if analysis.tasks:
        for i, task in enumerate(analysis.tasks, 1):
            print(f"\n{i}. {task.description}")
            print(f"   - Assigned by: {task.assigned_by or 'Not specified'}")
            print(f"   - Assigned to: {task.assignee or 'Not specified'}")
            print(f"   - Priority: {task.priority}")
            if task.due_date:
                print(f"   - Due date: {task.due_date}")
            if task.context:
                print(f"   - Context: {task.context}")
    else:
        print("No specific tasks found in this conversation.")
    
    print(f"\n📌 TO-DOS ({len(analysis.todos)}):")
    print("-" * 60)
    if analysis.todos:
        for i, todo in enumerate(analysis.todos, 1):
            print(f"{i}. {todo.description}")
            print(f"   - Mentioned by: {todo.mentioned_by or 'Not specified'}")
            print(f"   - Category: {todo.category}")
            print(f"   - Urgency: {todo.urgency}")
    else:
        print("No to-dos found in this conversation.")
    
    print(f"\n🔄 FOLLOW-UPS ({len(analysis.followups)}):")
    print("-" * 60)
    if analysis.followups:
        for i, followup in enumerate(analysis.followups, 1):
            print(f"{i}. {followup.topic}")
            print(f"   - Action required: {followup.action_required}")
            print(f"   - Responsible: {followup.responsible_party or 'Not specified'}")
            if followup.deadline:
                print(f"   - Deadline: {followup.deadline}")
            print(f"   - Mentioned by: {followup.mentioned_by or 'Not specified'}")
    else:
        print("No follow-ups found in this conversation.")
    
    print("\n" + "="*80)
    
    # Save results to file
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"llm_analysis_{conversation.id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


def test_with_sample_transcription():
    """Test LLM analyzer with sample transcription data."""
    
    # Load transcription data
    print("Loading transcription data...")
    transcription_file = "transcription_result_sample_audio_long.json"
    
    if not os.path.exists(transcription_file):
        print(f"Error: {transcription_file} not found!")
        return
    
    transcription_data = load_transcription_data(transcription_file)
    print(f"✓ Loaded transcription with {len(transcription_data.get('segments', []))} segments")
    
    # Convert to conversation
    print("\nConverting to conversation format...")
    conversation = convert_to_conversation(transcription_data)
    
    # Preview speaker transcript (first 500 chars)
    print("\nSpeaker transcript preview:")
    print("-" * 60)
    print(conversation.speaker_transcript[:500] + "...")
    
    # Create analyzer
    print("\nInitializing LLM Analyzer...")
    analyzer = LLMAnalyzer(MockConfig())
    
    # Analyze conversation
    print("\nAnalyzing conversation (this may take 30-60 seconds)...")
    try:
        analysis = analyzer.analyze_conversation(conversation)
        print("✓ Analysis completed successfully!")
        
        # Print results
        print_analysis_results(analysis, conversation)
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def test_specific_segments():
    """Test with specific segments that should contain tasks."""
    
    print("\n" + "="*80)
    print("TESTING WITH TASK-FOCUSED CONVERSATION")
    print("="*80)
    
    # Create a conversation with clear tasks
    task_conversation = Conversation(
        id="task-test-1",
        transcript="Conversación sobre tareas del proyecto",
        speaker_transcript="""[Hablante 1] Hola equipo, necesitamos revisar las tareas pendientes del proyecto.
[Hablante 2] Claro, ¿qué necesitas que haga primero?
[Hablante 1] Por favor, prepara el informe de ventas para el viernes. Es urgente.
[Hablante 2] Entendido. ¿Algo más?
[Hablante 1] Sí, también necesito que Carlos revise el código del módulo de pagos antes del miércoles.
[Hablante 3] Yo me encargo de eso.
[Hablante 1] Perfecto. Ana, por favor coordina con el cliente la reunión de seguimiento.
[Hablante 4] De acuerdo, la agendaré para la próxima semana.
[Hablante 1] No olviden actualizar el dashboard del proyecto con el progreso diario.
[Hablante 2] Anotado. ¿Cuándo es la fecha límite del proyecto completo?
[Hablante 1] Tenemos hasta fin de mes, pero hay que entregar el primer hito el día 15.""",
        speakers=["Hablante 1", "Hablante 2", "Hablante 3", "Hablante 4"],
        duration=180.0,
        timestamp=datetime.now()
    )
    
    # Analyze
    analyzer = LLMAnalyzer(MockConfig())
    
    try:
        print("\nAnalyzing task-focused conversation...")
        analysis = analyzer.analyze_conversation(task_conversation)
        print("✓ Analysis completed!")
        
        # Print results
        print_analysis_results(analysis, task_conversation)
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*80)
    print("LLM ANALYZER TEST WITH REAL DATA")
    print("="*80)
    
    # Test 1: With actual transcription data
    test_with_sample_transcription()
    
    # Test 2: With task-focused conversation
    test_specific_segments()
    
    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print("="*80)