"""
Test LLM Analyzer with diarization_result_sample.json
This will show how the analyzer handles a different conversation topic
and provide insights on output length and structure.
"""
import sys
import os
import json
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation


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
                    "timeout": 120,
                    "temperature": 0.3
                }
            }
        return default


def analyze_diarization_sample():
    """Test LLM Analyzer with diarization result sample."""
    
    print("="*80)
    print("LLM ANALYZER TEST WITH DIARIZATION SAMPLE")
    print("="*80)
    
    # Load diarization data
    print("\nLoading diarization_result_sample.json...")
    diarization_file = "personal_samples/diarization_result_sample.json"
    
    if not os.path.exists(diarization_file):
        print(f"❌ File not found: {diarization_file}")
        return
    
    with open(diarization_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded diarization with {len(data['segments'])} segments")
    print(f"   Audio file: {data['audio_file']['original_filename']}")
    print(f"   Duration: {data['audio_file']['duration']} seconds")
    
    # Get unique speakers
    speakers = list(set([seg['speaker_id'] for seg in data['segments']]))
    print(f"   Speakers detected: {len(speakers)} ({', '.join(speakers)})")
    
    # Build speaker-aware transcript
    speaker_transcript_lines = []
    current_speaker = None
    current_text = []
    
    # Track speaker changes and segment distribution
    speaker_changes = 0
    speaker_segments = {speaker: 0 for speaker in speakers}
    
    for seg in data['segments']:
        speaker = seg['speaker_id']
        text = seg['text'].strip()
        speaker_segments[speaker] += 1
        
        if speaker != current_speaker:
            speaker_changes += 1
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
    
    print(f"\n📊 Conversation Statistics:")
    print(f"   Total segments: {len(data['segments'])}")
    print(f"   Speaker changes: {speaker_changes}")
    for speaker, count in speaker_segments.items():
        percentage = (count / len(data['segments'])) * 100
        print(f"   {speaker}: {count} segments ({percentage:.1f}%)")
    
    # Show transcript preview
    print(f"\n📝 Speaker Transcript Preview (first 600 chars):")
    print("-" * 70)
    print(speaker_transcript[:600] + "..." if len(speaker_transcript) > 600 else speaker_transcript)
    print("-" * 70)
    
    # Create conversation object
    conversation = Conversation(
        id=f"diarization-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        transcript=data.get('full_text', ' '.join([seg['text'] for seg in data['segments']])),
        speaker_transcript=speaker_transcript,
        speakers=speakers,
        duration=data['audio_file']['duration'],
        timestamp=datetime.now()
    )
    
    print(f"\n🎯 Conversation Object Created:")
    print(f"   ID: {conversation.id}")
    print(f"   Raw transcript length: {len(conversation.transcript)} chars")
    print(f"   Speaker transcript length: {len(conversation.speaker_transcript)} chars")
    print(f"   Duration: {conversation.duration} seconds")
    
    # Initialize analyzer
    print("\n🤖 Initializing LLM Analyzer...")
    analyzer = LLMAnalyzer(MockConfig())
    
    # Run analysis step by step to monitor progress
    print("\n🔄 Running Step-by-Step Analysis:")
    print("(Each step will be timed to show processing characteristics)")
    
    total_start = time.time()
    
    # Step 1: Summary
    print("\n1️⃣ Summary Generation")
    print("   Status: Processing...", end="", flush=True)
    start = time.time()
    try:
        summary = analyzer.generate_summary(conversation)
        elapsed = time.time() - start
        print(f"\r   Status: ✅ Completed in {elapsed:.1f}s")
        print(f"   Length: {len(summary)} characters")
        print(f"   Word count: ~{len(summary.split())} words")
        
        print("\n   Summary Content:")
        print("   " + "-" * 60)
        # Split summary into lines for better readability
        summary_lines = summary.replace('. ', '.\n   ').split('\n')
        for line in summary_lines[:5]:  # Show first 5 lines
            if line.strip():
                print(f"   {line.strip()}")
        if len(summary_lines) > 5:
            print("   ...")
        print("   " + "-" * 60)
        
    except Exception as e:
        print(f"\r   Status: ❌ Failed: {e}")
        summary = None
    
    # Step 2: Tasks
    print("\n2️⃣ Task Extraction")
    print("   Status: Processing...", end="", flush=True)
    start = time.time()
    try:
        tasks = analyzer.extract_tasks(conversation)
        elapsed = time.time() - start
        print(f"\r   Status: ✅ Completed in {elapsed:.1f}s")
        print(f"   Tasks found: {len(tasks)}")
        
        if tasks:
            print("\n   Tasks Extracted:")
            for i, task in enumerate(tasks[:3], 1):  # Show first 3 tasks
                print(f"   {i}. {task.description}")
                print(f"      Assigned to: {task.assignee or 'Not specified'}")
                print(f"      Priority: {task.priority}")
                if task.due_date:
                    print(f"      Due: {task.due_date}")
        else:
            print("   No specific tasks identified in this conversation")
            
    except Exception as e:
        print(f"\r   Status: ❌ Failed: {e}")
        tasks = []
    
    # Step 3: Full Analysis
    print("\n3️⃣ Complete Analysis (includes key points, todos, followups)")
    print("   Status: Processing...", end="", flush=True)
    start = time.time()
    try:
        analysis = analyzer.analyze_conversation(conversation)
        elapsed = time.time() - start
        total_elapsed = time.time() - total_start
        
        print(f"\r   Status: ✅ Completed in {elapsed:.1f}s")
        print(f"   Total analysis time: {total_elapsed:.1f}s")
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("COMPLETE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n📊 Analysis Metrics:")
        print(f"   Summary: {len(analysis.summary)} chars, ~{len(analysis.summary.split())} words")
        print(f"   Key points: {len(analysis.key_points)} items")
        print(f"   Tasks: {len(analysis.tasks)} items")
        print(f"   To-dos: {len(analysis.todos)} items")
        print(f"   Follow-ups: {len(analysis.followups)} items")
        print(f"   LLM Provider: {analysis.llm_provider}")
        
        print(f"\n🔑 Key Points:")
        print("-" * 70)
        for i, point in enumerate(analysis.key_points, 1):
            print(f"{i}. {point}")
        
        if analysis.tasks:
            print(f"\n📋 Tasks Extracted:")
            print("-" * 70)
            for i, task in enumerate(analysis.tasks, 1):
                print(f"{i}. {task.description}")
                print(f"   → Assigned to: {task.assignee or 'Not specified'}")
                print(f"   → Priority: {task.priority}")
                if task.context:
                    print(f"   → Context: {task.context}")
        
        if analysis.todos:
            print(f"\n📌 To-dos:")
            print("-" * 70)
            for i, todo in enumerate(analysis.todos, 1):
                print(f"{i}. {todo.description}")
                print(f"   → Category: {todo.category}")
                print(f"   → Urgency: {todo.urgency}")
                print(f"   → Mentioned by: {todo.mentioned_by or 'Not specified'}")
        
        if analysis.followups:
            print(f"\n🔄 Follow-ups:")
            print("-" * 70)
            for i, followup in enumerate(analysis.followups, 1):
                print(f"{i}. {followup.topic}")
                print(f"   → Action: {followup.action_required}")
                print(f"   → Responsible: {followup.responsible_party or 'Not specified'}")
                if followup.deadline:
                    print(f"   → Deadline: {followup.deadline}")
        
        # Character and length analysis
        print(f"\n📏 Output Length Analysis:")
        print("-" * 70)
        total_output_chars = len(analysis.summary) + sum(len(point) for point in analysis.key_points)
        if analysis.tasks:
            total_output_chars += sum(len(task.description) for task in analysis.tasks)
        if analysis.todos:
            total_output_chars += sum(len(todo.description) for todo in analysis.todos)
        if analysis.followups:
            total_output_chars += sum(len(followup.topic + followup.action_required) for followup in analysis.followups)
        
        print(f"   Input transcript: {len(conversation.transcript)} chars")
        print(f"   Speaker transcript: {len(conversation.speaker_transcript)} chars")
        print(f"   Total output: {total_output_chars} chars")
        print(f"   Compression ratio: {len(conversation.transcript) / total_output_chars:.1f}:1")
        
        # Save results
        output_data = {
            "analysis": analysis.to_dict(),
            "input_stats": {
                "segments": len(data['segments']),
                "speakers": len(speakers),
                "speaker_changes": speaker_changes,
                "duration": conversation.duration,
                "input_chars": len(conversation.transcript),
                "speaker_transcript_chars": len(conversation.speaker_transcript)
            },
            "output_stats": {
                "summary_chars": len(analysis.summary),
                "summary_words": len(analysis.summary.split()),
                "key_points": len(analysis.key_points),
                "tasks": len(analysis.tasks),
                "todos": len(analysis.todos),
                "followups": len(analysis.followups),
                "total_output_chars": total_output_chars,
                "compression_ratio": round(len(conversation.transcript) / total_output_chars, 1)
            },
            "processing_time": {
                "total_seconds": total_elapsed,
                "summary_seconds": elapsed,
                "chars_per_second": len(conversation.transcript) / total_elapsed
            },
            "metadata": {
                "source_file": diarization_file,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        output_file = f"test_outputs/diarization_analysis_{conversation.id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Complete results saved to: {output_file}")
        
        # Show JSON structure sample
        print(f"\n📄 JSON Output Structure Sample:")
        print("-" * 70)
        sample_output = {
            "conversation_id": analysis.conversation_id,
            "summary": analysis.summary[:100] + "...",
            "key_points": analysis.key_points[:2] + ["..."] if len(analysis.key_points) > 2 else analysis.key_points,
            "tasks_count": len(analysis.tasks),
            "provider": analysis.llm_provider
        }
        print(json.dumps(sample_output, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"\r   Status: ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIARIZATION ANALYSIS TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Check if file exists
    if not os.path.exists("personal_samples/diarization_result_sample.json"):
        print("❌ Please run this script from the TranscriMatic root directory")
        print("   Current directory:", os.getcwd())
        print("   Looking for: personal_samples/diarization_result_sample.json")
    else:
        analyze_diarization_sample()