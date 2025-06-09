#!/usr/bin/env python3
"""Generate sample audio files for testing TranscriMatic."""

import os
import sys
import numpy as np
import wave
from pathlib import Path
from typing import List, Tuple
import argparse

# Try to import text-to-speech libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not installed. Install with: pip install gtts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Install with: pip install pyttsx3")


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def generate_silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sample_rate * duration))


def generate_white_noise(duration: float, amplitude: float = 0.1, sample_rate: int = 16000) -> np.ndarray:
    """Generate white noise."""
    return np.random.normal(0, amplitude, int(sample_rate * duration))


def save_wav(filename: str, audio_data: np.ndarray, sample_rate: int = 16000):
    """Save audio data to WAV file."""
    # Normalize to 16-bit range
    audio_data = np.clip(audio_data, -1, 1)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def generate_tone_test(output_dir: Path):
    """Generate test files with tones and silence."""
    print("Generating tone test files...")
    
    # Test 1: Simple tone
    audio = generate_sine_wave(440, 2.0)  # A4 note for 2 seconds
    save_wav(str(output_dir / "test_tone_440hz.wav"), audio)
    
    # Test 2: Multiple tones with silence
    audio_parts = []
    audio_parts.append(generate_sine_wave(440, 1.0))
    audio_parts.append(generate_silence(0.5))
    audio_parts.append(generate_sine_wave(880, 1.0))
    audio_parts.append(generate_silence(0.5))
    audio_parts.append(generate_sine_wave(220, 1.0))
    
    audio = np.concatenate(audio_parts)
    save_wav(str(output_dir / "test_tones_with_silence.wav"), audio)
    
    # Test 3: Tone with noise
    tone = generate_sine_wave(440, 3.0)
    noise = generate_white_noise(3.0, 0.05)
    audio = tone + noise
    save_wav(str(output_dir / "test_tone_with_noise.wav"), audio)
    
    print(f"✓ Generated 3 tone test files in {output_dir}")


def generate_spanish_speech_samples(output_dir: Path):
    """Generate Spanish speech samples using TTS."""
    
    # Spanish test phrases for different scenarios
    test_phrases = [
        # Single speaker samples
        {
            "filename": "test_spanish_greeting.wav",
            "text": "Hola, buenos días. ¿Cómo está usted? Espero que esté teniendo un día maravilloso.",
            "description": "Simple greeting"
        },
        {
            "filename": "test_spanish_numbers.wav",
            "text": "Uno, dos, tres, cuatro, cinco. El precio es veinte euros con cincuenta céntimos.",
            "description": "Numbers and prices"
        },
        {
            "filename": "test_spanish_conversation.wav",
            "text": "Disculpe, ¿dónde está la biblioteca? Está a dos cuadras de aquí, junto al parque.",
            "description": "Simple conversation"
        },
        {
            "filename": "test_spanish_technical.wav",
            "text": "El sistema de transcripción automática utiliza inteligencia artificial para convertir audio en texto.",
            "description": "Technical vocabulary"
        },
        {
            "filename": "test_spanish_long.wav",
            "text": """Buenos días a todos. Hoy vamos a hablar sobre el cambio climático y sus efectos en España. 
                      Como sabemos, las temperaturas han aumentado considerablemente en los últimos años. 
                      Esto ha causado sequías más frecuentes y olas de calor más intensas. 
                      Es importante que tomemos medidas para reducir nuestro impacto ambiental. 
                      Podemos empezar con pequeñas acciones como reciclar, usar transporte público y ahorrar energía.
                      Muchas gracias por su atención.""",
            "description": "Long speech about climate"
        }
    ]
    
    # Multi-speaker conversation script
    conversation_script = [
        ("speaker1", "Hola María, ¿cómo estás?"),
        ("speaker2", "Muy bien, Juan. ¿Y tú qué tal?"),
        ("speaker1", "Bastante bien, gracias. Oye, ¿has visto las noticias hoy?"),
        ("speaker2", "Sí, increíble lo del terremoto en Chile."),
        ("speaker1", "Es terrible. Espero que no haya muchas víctimas."),
        ("speaker2", "Yo también. Por cierto, ¿vienes a la reunión de mañana?"),
        ("speaker1", "Claro que sí. ¿A qué hora era?"),
        ("speaker2", "A las diez de la mañana en la sala de conferencias."),
        ("speaker1", "Perfecto, ahí estaré. Hasta mañana entonces."),
        ("speaker2", "Hasta mañana, Juan. Que descanses.")
    ]
    
    if GTTS_AVAILABLE:
        print("Generating Spanish speech samples using gTTS...")
        
        # Generate single speaker samples
        for sample in test_phrases:
            try:
                tts = gTTS(text=sample["text"], lang='es', slow=False)
                tts.save(str(output_dir / sample["filename"]))
                print(f"✓ Generated: {sample['filename']} - {sample['description']}")
            except Exception as e:
                print(f"✗ Failed to generate {sample['filename']}: {e}")
        
        # Generate multi-speaker conversation (simulated with speed variations)
        try:
            conversation_parts = []
            for speaker, text in conversation_script:
                # Vary speed to simulate different speakers
                slow = speaker == "speaker1"
                tts = gTTS(text=text, lang='es', slow=slow)
                temp_file = output_dir / f"temp_{speaker}.mp3"
                tts.save(str(temp_file))
                
                # Add silence between utterances
                # Note: For real multi-speaker, you'd need different TTS engines
                
            print("✓ Generated conversation sample (simulated multi-speaker)")
        except Exception as e:
            print(f"✗ Failed to generate conversation: {e}")
    
    elif PYTTSX3_AVAILABLE:
        print("Generating Spanish speech samples using pyttsx3...")
        engine = pyttsx3.init()
        
        # Try to set Spanish voice
        voices = engine.getProperty('voices')
        spanish_voice = None
        for voice in voices:
            if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                spanish_voice = voice.id
                break
        
        if spanish_voice:
            engine.setProperty('voice', spanish_voice)
        
        # Generate samples
        for sample in test_phrases[:3]:  # Generate fewer samples with pyttsx3
            try:
                engine.save_to_file(sample["text"], str(output_dir / sample["filename"]))
                engine.runAndWait()
                print(f"✓ Generated: {sample['filename']} - {sample['description']}")
            except Exception as e:
                print(f"✗ Failed to generate {sample['filename']}: {e}")
    
    else:
        print("✗ No TTS engine available. Creating placeholder with instructions...")
        
        # Create a text file with instructions
        instructions = """
# Spanish Audio Samples Needed for Testing

No TTS engine is installed. To generate Spanish speech samples, you can:

1. Install gTTS (recommended):
   pip install gtts

2. Install pyttsx3:
   pip install pyttsx3

3. Or manually record these Spanish phrases:

## Single Speaker Samples:

1. **test_spanish_greeting.wav**
   "Hola, buenos días. ¿Cómo está usted? Espero que esté teniendo un día maravilloso."

2. **test_spanish_numbers.wav**
   "Uno, dos, tres, cuatro, cinco. El precio es veinte euros con cincuenta céntimos."

3. **test_spanish_conversation.wav**
   "Disculpe, ¿dónde está la biblioteca? Está a dos cuadras de aquí, junto al parque."

## Multi-Speaker Conversation:

Record the following dialogue with two different speakers:

Speaker 1: "Hola María, ¿cómo estás?"
Speaker 2: "Muy bien, Juan. ¿Y tú qué tal?"
Speaker 1: "Bastante bien, gracias. Oye, ¿has visto las noticias hoy?"
Speaker 2: "Sí, increíble lo del terremoto en Chile."
...

Save all recordings as 16kHz, mono WAV files.
"""
        
        with open(output_dir / "RECORDING_INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        print(f"✓ Created recording instructions in {output_dir}/RECORDING_INSTRUCTIONS.txt")


def generate_edge_case_samples(output_dir: Path):
    """Generate edge case audio samples for robust testing."""
    print("Generating edge case test files...")
    
    # Very short audio
    audio = generate_sine_wave(440, 0.1)  # 100ms
    save_wav(str(output_dir / "test_very_short.wav"), audio)
    
    # Very quiet audio
    audio = generate_sine_wave(440, 2.0) * 0.01  # Very low amplitude
    save_wav(str(output_dir / "test_very_quiet.wav"), audio)
    
    # Mostly silence
    audio_parts = []
    audio_parts.append(generate_silence(5.0))
    audio_parts.append(generate_sine_wave(440, 0.5))
    audio_parts.append(generate_silence(5.0))
    audio = np.concatenate(audio_parts)
    save_wav(str(output_dir / "test_mostly_silence.wav"), audio)
    
    # Clipped audio (distorted)
    audio = generate_sine_wave(440, 2.0) * 2.0  # Will be clipped
    save_wav(str(output_dir / "test_clipped.wav"), audio)
    
    print(f"✓ Generated 4 edge case test files in {output_dir}")


def main():
    """Main function to generate all test audio files."""
    parser = argparse.ArgumentParser(description="Generate test audio files for TranscriMatic")
    parser.add_argument("--output-dir", type=Path, default=Path("test_audio"),
                        help="Output directory for test files")
    parser.add_argument("--types", nargs='+', 
                        choices=['tones', 'speech', 'edge', 'all'],
                        default=['all'],
                        help="Types of test files to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating test audio files in: {args.output_dir}")
    print("=" * 50)
    
    if 'all' in args.types or 'tones' in args.types:
        generate_tone_test(args.output_dir)
        print()
    
    if 'all' in args.types or 'speech' in args.types:
        generate_spanish_speech_samples(args.output_dir)
        print()
    
    if 'all' in args.types or 'edge' in args.types:
        generate_edge_case_samples(args.output_dir)
        print()
    
    print("=" * 50)
    print(f"Test audio generation complete!")
    print(f"Files saved in: {args.output_dir}")
    
    # List generated files
    wav_files = list(args.output_dir.glob("*.wav"))
    if wav_files:
        print(f"\nGenerated {len(wav_files)} WAV files:")
        for f in sorted(wav_files):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()