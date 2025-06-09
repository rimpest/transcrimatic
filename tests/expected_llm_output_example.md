# Expected LLM Analyzer Output Example

This document shows what the LLM Analyzer module should produce when analyzing a conversation.

## Input: Transcription Data

```json
{
  "audio_file": {
    "original_filename": "sample_audio_long.mp3",
    "duration": 60.0
  },
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "text": "Hola a todos, ¿qué tal? Bienvenidos a un episodio más de su podcast..."
    },
    // ... more segments
  ]
}
```

## Expected Output: Analysis Object

### 1. Summary (Spanish)
```
En este episodio del podcast "En tu orilla", el presentador da la bienvenida a Santiago de Lil Jesus. 
La conversación gira en torno al lanzamiento de nueva música de la banda Lil Jesus. El presentador 
expresa su entusiasmo por tener a Santiago en el programa y menciona que tiene muchas preguntas 
que hacerle. Se nota un ambiente cordial y de anticipación por las novedades musicales que se avecinan.
```

### 2. Key Points
- Bienvenida a Santiago de Lil Jesus al podcast
- Anuncio de nueva música próxima de Lil Jesus
- El presentador expresa entusiasmo por la entrevista
- Ambiente cordial entre el presentador y el invitado

### 3. Tasks Extracted
For this podcast conversation, likely no specific tasks would be extracted as it's an interview format.

Example from a business conversation:
```json
{
  "tasks": [
    {
      "description": "Preparar el reporte de ventas",
      "assignee": "Hablante 2",
      "assigned_by": "Hablante 1",
      "due_date": "mañana por la mañana",
      "priority": "alta",
      "context": "Necesario para la reunión con el cliente"
    }
  ]
}
```

### 4. To-dos
```json
{
  "todos": [
    {
      "description": "Escuchar la nueva música de Lil Jesus cuando salga",
      "category": "entretenimiento",
      "urgency": "baja",
      "mentioned_by": "Hablante 1"
    }
  ]
}
```

### 5. Follow-ups
```json
{
  "followups": [
    {
      "topic": "Nueva música de Lil Jesus",
      "action_required": "Estar pendiente del lanzamiento",
      "responsible_party": "Audiencia del podcast",
      "mentioned_by": "Hablante 1"
    }
  ]
}
```

## Complete Analysis Object Structure

```python
Analysis(
    conversation_id="podcast-20250608-215500",
    summary="En este episodio del podcast...",
    key_points=[
        "Bienvenida a Santiago de Lil Jesus",
        "Anuncio de nueva música próxima",
        "Entusiasmo del presentador",
        "Ambiente cordial"
    ],
    tasks=[],  # Empty for this podcast
    todos=[
        Todo(
            description="Escuchar nueva música de Lil Jesus",
            category="entretenimiento",
            urgency="baja",
            mentioned_by="Hablante 1"
        )
    ],
    followups=[
        Followup(
            topic="Nueva música de Lil Jesus",
            action_required="Estar pendiente del lanzamiento",
            responsible_party=None,
            deadline=None,
            mentioned_by="Hablante 1"
        )
    ],
    participants=["SPEAKER_00", "SPEAKER_01"],
    duration=60.0,
    timestamp=datetime.now(),
    llm_provider="ollama"
)
```

## JSON Output Format

```json
{
  "conversation_id": "podcast-20250608-215500",
  "summary": "En este episodio del podcast 'En tu orilla', el presentador da la bienvenida...",
  "key_points": [
    "Bienvenida a Santiago de Lil Jesus al podcast",
    "Anuncio de nueva música próxima de Lil Jesus",
    "El presentador expresa entusiasmo por la entrevista",
    "Ambiente cordial entre el presentador y el invitado"
  ],
  "tasks": [],
  "todos": [
    {
      "description": "Escuchar la nueva música de Lil Jesus cuando salga",
      "category": "entretenimiento",
      "urgency": "baja",
      "mentioned_by": "Hablante 1"
    }
  ],
  "followups": [
    {
      "topic": "Nueva música de Lil Jesus",
      "action_required": "Estar pendiente del lanzamiento",
      "responsible_party": null,
      "deadline": null,
      "mentioned_by": "Hablante 1"
    }
  ],
  "participants": ["SPEAKER_00", "SPEAKER_01"],
  "duration": 60.0,
  "timestamp": "2025-06-08T21:55:00",
  "llm_provider": "ollama"
}
```

## Notes on Expected Behavior

1. **Language**: All output should be in Spanish since the input is in Spanish
2. **Speaker Attribution**: The module should convert SPEAKER_00, SPEAKER_01 to Hablante 1, Hablante 2
3. **Context Awareness**: The LLM should understand this is a podcast interview, not a business meeting
4. **Task Extraction**: Should be smart enough to know that podcast conversations rarely have actionable tasks
5. **Quality Metrics**:
   - Summary should be 100-300 words
   - Key points should be 3-8 items
   - Tasks/todos/followups depend on conversation content