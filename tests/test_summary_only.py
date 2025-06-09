#!/usr/bin/env python3
"""
Test just summary generation with transcription data.
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation
from src.llm.providers import OllamaProvider


# Load transcription
with open("transcription_result_sample_audio_long.json", 'r') as f:
    data = json.load(f)

# Use first 10 segments
segments = data['segments'][:10]

# Build simple transcript
lines = []
for seg in segments:
    speaker_num = int(seg['speaker_id'].split('_')[1]) + 1
    lines.append(f"[Hablante {speaker_num}] {seg['text']}")

speaker_transcript = "\n".join(lines)

print("TRANSCRIPT TO ANALYZE:")
print("="*70)
print(speaker_transcript)
print("="*70)

# Create conversation
conv = Conversation(
    id="test-summary",
    transcript=" ".join([s['text'] for s in segments]),
    speaker_transcript=speaker_transcript,
    speakers=["SPEAKER_00"],
    duration=30.0,
    timestamp=datetime.now()
)

# Test direct with provider
print("\nTesting direct summary generation with Ollama...")
provider = OllamaProvider({
    "host": "localhost",
    "port": 11434,
    "model": "llama3.2:latest",
    "timeout": 60
})

prompt = f"""Analiza esta conversación de un podcast en español y proporciona un resumen conciso de 2-3 párrafos:

{speaker_transcript}

Responde en español."""

print("\nGenerating summary...")
try:
    import time
    start = time.time()
    response = provider.generate(prompt, temperature=0.3)
    elapsed = time.time() - start
    
    print(f"\n✅ Summary generated in {elapsed:.1f} seconds")
    print("\nSUMMARY:")
    print("="*70)
    print(response)
    print("="*70)
    
    # Save output
    output = {
        "timestamp": datetime.now().isoformat(),
        "segments_analyzed": len(segments),
        "transcript": speaker_transcript,
        "summary": response,
        "generation_time": elapsed
    }
    
    with open("test_outputs/summary_test_result.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to test_outputs/summary_test_result.json")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()