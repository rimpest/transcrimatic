"""
Test single LLM prompts to show actual output quickly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.providers import OllamaProvider
import json
import time


def test_single_prompts():
    """Test individual prompts to show LLM output."""
    
    provider = OllamaProvider({
        "host": "localhost",
        "port": 11434,
        "model": "llama3.2:latest",
        "timeout": 30,
        "temperature": 0.3
    })
    
    if not provider.is_available():
        print("❌ Ollama not available")
        return
    
    print("="*80)
    print("LLM ANALYZER - ACTUAL OUTPUT EXAMPLES")
    print("="*80)
    
    # Test 1: Summary Generation
    print("\n1. SUMMARY GENERATION TEST")
    print("-" * 70)
    
    summary_prompt = """Analiza la siguiente conversación en español donde cada hablante está claramente identificado.

IMPORTANTE: Los hablantes están marcados como [Hablante 1], [Hablante 2], etc.

Proporciona:
1. Un resumen conciso (2-3 párrafos) mencionando quién dijo qué puntos importantes
2. Los puntos clave discutidos por cada hablante
3. Las decisiones tomadas y quién las propuso
4. Identifica el rol probable de cada hablante si es posible (ej: jefe, empleado, cliente)

Conversación:
[Hablante 1] Buenos días equipo. Necesitamos revisar el proyecto Alpha.
[Hablante 2] Buenos días María. El proyecto va según lo planeado.
[Hablante 1] Excelente. Carlos, necesito que prepares el informe ejecutivo para el viernes.
[Hablante 2] Entendido. ¿Qué información específica necesitas?
[Hablante 1] Incluye el progreso actual, los hitos alcanzados y los riesgos identificados.
[Hablante 2] Perfecto. Lo tendré listo para el viernes por la mañana.

Duración: 2 minutos
Número de hablantes: 2

Responde en español y sé específico sobre qué hablante mencionó cada punto."""
    
    print("Prompt sent to LLM...")
    start = time.time()
    summary = provider.generate(summary_prompt)
    print(f"Response time: {time.time() - start:.1f}s")
    
    print("\nLLM RESPONSE:")
    print(summary)
    
    # Test 2: Task Extraction
    print("\n\n2. TASK EXTRACTION TEST")
    print("-" * 70)
    
    task_prompt = """Extrae todas las tareas y acciones mencionadas en esta conversación.
IMPORTANTE: Identifica quién asignó cada tarea y a quién se la asignó.

Para cada tarea, identifica:
- Descripción clara de la tarea
- Quién la mencionó/asignó (ej: "Hablante 1")
- A quién se le asignó (ej: "Hablante 2" o "no especificado")
- Fecha límite (si se menciona)
- Prioridad (alta/media/baja basado en el contexto)
- Contexto de la conversación

Conversación con hablantes identificados:
[Hablante 1] Carlos, necesito que prepares el informe ejecutivo para el viernes.
[Hablante 2] Entendido. ¿Qué información específica necesitas?
[Hablante 1] Incluye el progreso actual, los hitos alcanzados y los riesgos identificados.

Formato de respuesta:
TAREA: [descripción]
ASIGNADA_POR: [Hablante X]
RESPONSABLE: [Hablante Y o "no especificado"]
FECHA: [fecha o "no especificada"]
PRIORIDAD: [alta/media/baja]
CONTEXTO: [breve contexto de por qué se asignó]
---"""
    
    print("Prompt sent to LLM...")
    start = time.time()
    tasks = provider.generate(task_prompt)
    print(f"Response time: {time.time() - start:.1f}s")
    
    print("\nLLM RESPONSE:")
    print(tasks)
    
    # Test 3: Key Points Extraction
    print("\n\n3. KEY POINTS EXTRACTION TEST")
    print("-" * 70)
    
    keypoints_prompt = """Del siguiente resumen de conversación, extrae los puntos clave más importantes.
Cada punto debe ser conciso (máximo 2 líneas) y debe mencionar qué hablante lo discutió.

Resumen:
La conversación trata sobre la revisión del proyecto Alpha. María (Hablante 1) solicita a Carlos (Hablante 2) que complete un informe ejecutivo para el viernes. Carlos confirma que incluirá el progreso actual, hitos alcanzados y riesgos identificados.

Lista de puntos clave (máximo 8):"""
    
    print("Prompt sent to LLM...")
    start = time.time()
    keypoints = provider.generate(keypoints_prompt)
    print(f"Response time: {time.time() - start:.1f}s")
    
    print("\nLLM RESPONSE:")
    print(keypoints)
    
    # Show expected parsed output
    print("\n\n" + "="*80)
    print("EXPECTED PARSED OUTPUT IN ANALYSIS OBJECT")
    print("="*80)
    
    expected_output = {
        "conversation_id": "business-meeting-001",
        "summary": "La conversación trata sobre la revisión del proyecto Alpha...",
        "key_points": [
            "María (Hablante 1) solicita revisión del proyecto Alpha",
            "Carlos (Hablante 2) confirma que el proyecto va según lo planeado",
            "Se asigna tarea de preparar informe ejecutivo para el viernes",
            "El informe debe incluir progreso, hitos y riesgos"
        ],
        "tasks": [
            {
                "description": "Preparar el informe ejecutivo del proyecto Alpha",
                "assignee": "Hablante 2",
                "assigned_by": "Hablante 1",
                "due_date": "viernes",
                "priority": "alta",
                "context": "Necesario para la revisión del proyecto Alpha"
            }
        ],
        "todos": [],
        "followups": [],
        "participants": ["María", "Carlos"],
        "duration": 120.0,
        "llm_provider": "ollama"
    }
    
    print(json.dumps(expected_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_single_prompts()