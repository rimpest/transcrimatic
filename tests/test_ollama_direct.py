"""
Direct test of Ollama API to verify it's working.
"""
import requests
import json

def test_ollama_direct():
    """Test Ollama directly without the LLM framework."""
    
    # 1. Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✓ Ollama is running")
            print(f"  Available models: {[m['name'] for m in models]}")
        else:
            print("✗ Ollama returned error:", response.status_code)
            return
    except Exception as e:
        print("✗ Cannot connect to Ollama:", e)
        return
    
    # 2. Test simple generation
    print("\nTesting generation with llama3.2:latest...")
    
    prompt = "Responde en español en una sola línea: ¿Qué es Python?"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Generation successful")
            print(f"  Response: {result.get('response', 'No response')[:100]}...")
            print(f"  Model: {result.get('model')}")
            print(f"  Duration: {result.get('total_duration', 0) / 1e9:.2f}s")
        else:
            print("✗ Generation failed:", response.status_code)
            print("  Response:", response.text)
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (30s)")
    except Exception as e:
        print("✗ Error during generation:", e)
    
    # 3. Test with conversation analysis prompt
    print("\nTesting with analysis prompt...")
    
    analysis_prompt = """Analiza esta breve conversación:
[Hablante 1] Necesito el reporte para mañana.
[Hablante 2] Lo tendré listo.

Responde en JSON con: {"tarea": "descripción", "responsable": "quien"}"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:latest",
                "prompt": analysis_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Analysis successful")
            print(f"  Response: {result.get('response', 'No response')}")
        else:
            print("✗ Analysis failed:", response.status_code)
            
    except Exception as e:
        print("✗ Error during analysis:", e)


if __name__ == "__main__":
    test_ollama_direct()