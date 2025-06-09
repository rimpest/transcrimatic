#!/usr/bin/env python3
"""
Complete analysis test with transcription data.
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


# Load transcription
print("Loading transcription data...")
with open("transcription_result_sample_audio_long.json", 'r') as f:
    data = json.load(f)

# Use first 15 segments for complete analysis
segments = data['segments'][:15]

# Build speaker transcript with proper speaker changes
speaker_transcript_lines = []
current_speaker = None
current_text = []

for seg in segments:
    speaker = seg['speaker_id']
    text = seg['text'].strip()
    
    if speaker != current_speaker:
        if current_speaker and current_text:
            speaker_num = int(current_speaker.split('_')[1]) + 1
            speaker_transcript_lines.append(f"[Hablante {speaker_num}] {' '.join(current_text)}")
        current_speaker = speaker
        current_text = [text] if text else []
    else:
        if text:
            current_text.append(text)

if current_speaker and current_text:
    speaker_num = int(current_speaker.split('_')[1]) + 1
    speaker_transcript_lines.append(f"[Hablante {speaker_num}] {' '.join(current_text)}")

speaker_transcript = "\n".join(speaker_transcript_lines)

print(f"\nAnalyzing {len(segments)} segments from podcast...")
print("\nTranscript preview:")
print("="*70)
print(speaker_transcript[:400] + "...")
print("="*70)

# Create conversation
conversation = Conversation(
    id=f"podcast-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    transcript=" ".join([s['text'] for s in segments]),
    speaker_transcript=speaker_transcript,
    speakers=list(set([s['speaker_id'] for s in segments])),
    duration=segments[-1]['end_time'] if segments else 45.0,
    timestamp=datetime.now()
)

# Initialize analyzer
print("\nInitializing LLM Analyzer...")
analyzer = LLMAnalyzer(MockConfig())

# Run complete analysis
print("\nRunning complete analysis (this will take 30-60 seconds)...")
print("Progress:")

start_time = time.time()

try:
    # Step 1: Summary
    print("  1. Generating summary...", end="", flush=True)
    t1 = time.time()
    summary = analyzer.generate_summary(conversation)
    print(f" ✓ ({time.time()-t1:.1f}s)")
    
    # Step 2: Tasks
    print("  2. Extracting tasks...", end="", flush=True)
    t2 = time.time()
    tasks = analyzer.extract_tasks(conversation)
    print(f" ✓ ({time.time()-t2:.1f}s)")
    
    # Step 3: Todos
    print("  3. Extracting to-dos...", end="", flush=True)
    t3 = time.time()
    todos = analyzer.extract_todos(conversation)
    print(f" ✓ ({time.time()-t3:.1f}s)")
    
    # Step 4: Followups
    print("  4. Extracting follow-ups...", end="", flush=True)
    t4 = time.time()
    followups = analyzer.extract_followups(conversation)
    print(f" ✓ ({time.time()-t4:.1f}s)")
    
    # Step 5: Complete analysis (includes key points)
    print("  5. Finalizing analysis...", end="", flush=True)
    t5 = time.time()
    analysis = analyzer.analyze_conversation(conversation)
    print(f" ✓ ({time.time()-t5:.1f}s)")
    
    total_time = time.time() - start_time
    
    print(f"\nTotal analysis time: {total_time:.1f} seconds")
    
    # Display results
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📝 SUMMARY:")
    print("-" * 70)
    print(analysis.summary)
    
    print(f"\n🔑 KEY POINTS ({len(analysis.key_points)}):")
    print("-" * 70)
    for i, point in enumerate(analysis.key_points, 1):
        print(f"{i}. {point}")
    
    print(f"\n📋 TASKS ({len(analysis.tasks)}):")
    print("-" * 70)
    if analysis.tasks:
        for i, task in enumerate(analysis.tasks, 1):
            print(f"{i}. {task.description}")
            print(f"   Assigned to: {task.assignee or 'N/A'}")
            print(f"   Priority: {task.priority}")
    else:
        print("No specific tasks found (normal for podcast conversations)")
    
    print(f"\n📌 TO-DOS ({len(analysis.todos)}):")
    print("-" * 70)
    if analysis.todos:
        for i, todo in enumerate(analysis.todos, 1):
            print(f"{i}. {todo.description}")
            print(f"   Category: {todo.category}")
            print(f"   Urgency: {todo.urgency}")
    else:
        print("No to-dos extracted")
    
    print(f"\n🔄 FOLLOW-UPS ({len(analysis.followups)}):")
    print("-" * 70)
    if analysis.followups:
        for i, followup in enumerate(analysis.followups, 1):
            print(f"{i}. {followup.topic}")
            print(f"   Action: {followup.action_required}")
    else:
        print("No follow-ups identified")
    
    # Save complete results
    output_data = {
        "analysis": analysis.to_dict(),
        "processing_time": {
            "total_seconds": total_time,
            "summary": time.time()-t1,
            "tasks": time.time()-t2,
            "todos": time.time()-t3,
            "followups": time.time()-t4
        },
        "metadata": {
            "segments_analyzed": len(segments),
            "audio_duration": conversation.duration,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    output_file = f"test_outputs/complete_analysis_{conversation.id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Complete results saved to: {output_file}")
    
    # Show JSON preview
    print("\n📄 JSON Output Preview:")
    print("-" * 70)
    json_str = json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)
    print(json_str[:600] + "...")
    
except Exception as e:
    print(f"\n❌ Analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)