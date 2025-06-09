"""
Test LLM Analyzer with the provided transcription sample using Ollama.
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
                    "timeout": 120,
                    "temperature": 0.3
                }
            }
        return default


def test_with_transcription_sample():
    """Test LLM Analyzer with the provided transcription sample."""
    
    print("="*80)
    print("LLM ANALYZER TEST WITH TRANSCRIPTION SAMPLE")
    print("="*80)
    
    # First, verify Ollama is available
    print("\nChecking Ollama availability...")
    provider = OllamaProvider({
        "host": "localhost",
        "port": 11434,
        "model": "llama3.2:latest"
    })
    
    if not provider.is_available():
        print("❌ Ollama is not running! Please start it with: ollama serve")
        return
    
    print("✅ Ollama is available")
    
    # Load transcription data
    print("\nLoading transcription data...")
    transcription_file = "transcription_result_sample_audio_long.json"
    
    if not os.path.exists(transcription_file):
        print(f"❌ File not found: {transcription_file}")
        return
    
    with open(transcription_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded transcription with {len(data['segments'])} segments")
    print(f"   Duration: {data['audio_file']['duration']} seconds")
    
    # Convert first 20 segments to conversation format for manageable processing
    segments = data['segments'][:20]  # First 20 segments
    
    # Build speaker-aware transcript
    speaker_transcript_lines = []
    current_speaker = None
    current_text = []
    
    for seg in segments:
        speaker = seg['speaker_id']
        text = seg['text'].strip()
        
        if speaker != current_speaker:
            # Save previous speaker's text
            if current_speaker and current_text:
                speaker_num = int(current_speaker.split('_')[1]) + 1
                speaker_transcript_lines.append(f"[Hablante {speaker_num}] {' '.join(current_text)}")
            
            # Start new speaker
            current_speaker = speaker
            current_text = [text] if text else []
        else:
            # Continue with same speaker
            if text:
                current_text.append(text)
    
    # Don't forget the last speaker
    if current_speaker and current_text:
        speaker_num = int(current_speaker.split('_')[1]) + 1
        speaker_transcript_lines.append(f"[Hablante {speaker_num}] {' '.join(current_text)}")
    
    speaker_transcript = "\n".join(speaker_transcript_lines)
    
    # Create conversation object
    conversation = Conversation(
        id=f"podcast-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transcript=" ".join([seg['text'] for seg in segments]),
        speaker_transcript=speaker_transcript,
        speakers=list(set([seg['speaker_id'] for seg in segments])),
        duration=segments[-1]['end_time'] if segments else 60.0,
        timestamp=datetime.now()
    )
    
    print(f"\n📄 Conversation Details:")
    print(f"   ID: {conversation.id}")
    print(f"   Speakers: {len(conversation.speakers)}")
    print(f"   Duration: {conversation.duration:.1f} seconds")
    
    print("\n📝 Speaker Transcript Preview:")
    print("-" * 70)
    print(speaker_transcript[:500] + "..." if len(speaker_transcript) > 500 else speaker_transcript)
    print("-" * 70)
    
    # Create analyzer
    print("\n🤖 Initializing LLM Analyzer...")
    analyzer = LLMAnalyzer(MockConfig())
    
    # Test individual components first
    print("\n📊 Testing individual analysis components:")
    
    # 1. Test Summary Generation
    print("\n1️⃣ Generating Summary...")
    start_time = time.time()
    try:
        summary = analyzer.generate_summary(conversation)
        elapsed = time.time() - start_time
        print(f"   ✅ Summary generated in {elapsed:.1f}s")
        print(f"   Length: {len(summary)} characters")
        print("\n   Summary:")
        print("   " + "-" * 60)
        print("   " + summary.replace("\n", "\n   "))
        print("   " + "-" * 60)
    except Exception as e:
        print(f"   ❌ Summary generation failed: {e}")
        summary = None
    
    # 2. Test Task Extraction
    print("\n2️⃣ Extracting Tasks...")
    start_time = time.time()
    try:
        tasks = analyzer.extract_tasks(conversation)
        elapsed = time.time() - start_time
        print(f"   ✅ Task extraction completed in {elapsed:.1f}s")
        print(f"   Tasks found: {len(tasks)}")
        
        if tasks:
            for i, task in enumerate(tasks, 1):
                print(f"\n   Task {i}:")
                print(f"     Description: {task.description}")
                print(f"     Assigned to: {task.assignee or 'Not specified'}")
                print(f"     Assigned by: {task.assigned_by or 'Not specified'}")
                print(f"     Priority: {task.priority}")
                if task.due_date:
                    print(f"     Due date: {task.due_date}")
        else:
            print("   ℹ️  No tasks found (expected for podcast conversation)")
    except Exception as e:
        print(f"   ❌ Task extraction failed: {e}")
        tasks = []
    
    # 3. Full Analysis (if individual tests succeeded)
    if summary:
        print("\n3️⃣ Running Full Analysis...")
        print("   (This combines all extraction methods)")
        
        start_time = time.time()
        try:
            analysis = analyzer.analyze_conversation(conversation)
            elapsed = time.time() - start_time
            
            print(f"\n   ✅ Full analysis completed in {elapsed:.1f}s")
            
            # Display complete results
            print("\n" + "="*80)
            print("COMPLETE ANALYSIS RESULTS")
            print("="*80)
            
            print(f"\n📊 Analysis Metrics:")
            print(f"   - Summary length: {len(analysis.summary)} chars")
            print(f"   - Key points: {len(analysis.key_points)}")
            print(f"   - Tasks: {len(analysis.tasks)}")
            print(f"   - To-dos: {len(analysis.todos)}")
            print(f"   - Follow-ups: {len(analysis.followups)}")
            print(f"   - LLM Provider: {analysis.llm_provider}")
            
            print(f"\n🔑 Key Points:")
            print("-" * 70)
            for i, point in enumerate(analysis.key_points, 1):
                print(f"{i}. {point}")
            
            if analysis.todos:
                print(f"\n📌 To-dos:")
                print("-" * 70)
                for i, todo in enumerate(analysis.todos, 1):
                    print(f"{i}. {todo.description}")
                    print(f"   Category: {todo.category}, Urgency: {todo.urgency}")
            
            if analysis.followups:
                print(f"\n🔄 Follow-ups:")
                print("-" * 70)
                for i, followup in enumerate(analysis.followups, 1):
                    print(f"{i}. {followup.topic}")
                    print(f"   Action: {followup.action_required}")
            
            # Save results
            output_dir = Path("test_outputs")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"transcription_analysis_{analysis.conversation_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Show JSON structure
            print("\n📄 JSON Output Sample:")
            print("-" * 70)
            json_output = json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)
            print(json_output[:800] + "..." if len(json_output) > 800 else json_output)
            
        except Exception as e:
            print(f"   ❌ Full analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Make sure we're in the right directory
    if not os.path.exists("transcription_result_sample_audio_long.json"):
        print("❌ Please run this script from the TranscriMatic root directory")
        print("   Current directory:", os.getcwd())
    else:
        test_with_transcription_sample()