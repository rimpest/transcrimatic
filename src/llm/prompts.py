"""
Spanish language prompt templates for conversation analysis.
"""

SUMMARY_PROMPT_WITH_SPEAKERS = """Analiza la siguiente conversación en español donde cada hablante está claramente identificado.

IMPORTANTE: Los hablantes están marcados como [Hablante 1], [Hablante 2], etc.

Proporciona:
1. Un resumen conciso (2-3 párrafos) mencionando quién dijo qué puntos importantes
2. Los puntos clave discutidos por cada hablante
3. Las decisiones tomadas y quién las propuso
4. Identifica el rol probable de cada hablante si es posible (ej: jefe, empleado, cliente)

Conversación:
{conversation_text}

Duración: {duration} minutos
Número de hablantes: {speaker_count}

Responde en español y sé específico sobre qué hablante mencionó cada punto."""


TASK_EXTRACTION_WITH_SPEAKERS_PROMPT = """Extrae todas las tareas y acciones mencionadas en esta conversación.
IMPORTANTE: Identifica quién asignó cada tarea y a quién se la asignó.

Para cada tarea, identifica:
- Descripción clara de la tarea
- Quién la mencionó/asignó (ej: "Hablante 1")
- A quién se le asignó (ej: "Hablante 2" o "no especificado")
- Fecha límite (si se menciona)
- Prioridad (alta/media/baja basado en el contexto)
- Contexto de la conversación

Conversación con hablantes identificados:
{conversation_text}

Formato de respuesta:
TAREA: [descripción]
ASIGNADA_POR: [Hablante X]
RESPONSABLE: [Hablante Y o "no especificado"]
FECHA: [fecha o "no especificada"]
PRIORIDAD: [alta/media/baja]
CONTEXTO: [breve contexto de por qué se asignó]
---"""


TODO_EXTRACTION_WITH_SPEAKERS_PROMPT = """Extrae todos los elementos pendientes o "to-dos" mencionados en esta conversación.
Estos son diferentes a las tareas asignadas - son cosas que alguien mencionó que hay que hacer pero sin asignación específica.

Busca frases como:
- "hay que recordar..."
- "no olvidar..."
- "pendiente..."
- "importante hacer..."
- "tenemos que..."

Para cada to-do:
- Descripción clara
- Quién lo mencionó
- Categoría (trabajo/personal/proyecto/otro)
- Urgencia (alta/normal/baja)

Conversación:
{conversation_text}

Formato de respuesta:
TODO: [descripción]
MENCIONADO_POR: [Hablante X]
CATEGORIA: [categoría]
URGENCIA: [alta/normal/baja]
---"""


FOLLOWUP_EXTRACTION_WITH_SPEAKERS_PROMPT = """Extrae todos los puntos de seguimiento mencionados en esta conversación.
Estos son temas que requieren revisión, confirmación o acción futura.

Busca frases como:
- "dar seguimiento a..."
- "revisar más tarde..."
- "volver a hablar de..."
- "confirmar con..."
- "próxima reunión..."

Para cada seguimiento:
- Tema principal
- Acción requerida
- Quién debe hacerlo (si se especifica)
- Fecha límite (si se menciona)
- Quién lo mencionó

Conversación:
{conversation_text}

Formato de respuesta:
SEGUIMIENTO: [tema]
ACCION: [qué hay que hacer]
RESPONSABLE: [Hablante X o "no especificado"]
FECHA: [fecha o "no especificada"]
MENCIONADO_POR: [Hablante Y]
---"""


KEY_POINTS_EXTRACTION_PROMPT = """Del siguiente resumen de conversación, extrae los puntos clave más importantes.
Cada punto debe ser conciso (máximo 2 líneas) y debe mencionar qué hablante lo discutió.

Resumen:
{summary}

Lista de puntos clave (máximo 8):"""


DAILY_HIGHLIGHTS_PROMPT = """Analiza los siguientes resúmenes de conversaciones del día y genera los puntos destacados más importantes.

Conversaciones analizadas: {conversation_count}
Duración total: {total_duration} minutos

Resúmenes:
{summaries}

Genera una lista de 5-7 puntos destacados del día que capturen:
- Las decisiones más importantes
- Los temas principales discutidos
- Logros o avances significativos
- Problemas o desafíos identificados
- Próximos pasos críticos

Formato: Lista numerada en español."""


SPEAKER_ROLE_IDENTIFICATION_PROMPT = """Basándote en el contenido y estilo de comunicación en esta conversación, intenta identificar el rol probable de cada hablante.

Conversación:
{conversation_text}

Para cada hablante, considera:
- Tono de comunicación (formal/informal)
- Tipo de decisiones que toma
- Cómo se dirigen a ellos otros hablantes
- Temas que discute

Formato de respuesta:
[Hablante 1]: [rol probable y justificación breve]
[Hablante 2]: [rol probable y justificación breve]
etc."""


CONVERSATION_CATEGORY_PROMPT = """Clasifica esta conversación en una de las siguientes categorías:
- Reunión de equipo
- Llamada con cliente
- Sesión de planificación
- Revisión de proyecto
- Discusión técnica
- Entrevista
- Capacitación
- Conversación informal
- Otro

Conversación:
{conversation_text}

Responde con:
CATEGORIA: [categoría]
CONFIANZA: [alta/media/baja]
RAZON: [breve justificación]"""


def format_conversation_for_prompt(conversation: 'Conversation') -> str:
    """Format conversation for use in prompts."""
    return conversation.speaker_transcript


def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string."""
    minutes = int(seconds / 60)
    if minutes < 1:
        return "menos de 1"
    return str(minutes)


def create_batch_prompt(conversations: list, prompt_template: str) -> str:
    """Create a batch prompt for multiple conversations."""
    batch_text = "\n\n=== CONVERSACIÓN {} ===\n{}\n"
    
    all_conversations = ""
    for i, conv in enumerate(conversations, 1):
        all_conversations += batch_text.format(
            i, 
            format_conversation_for_prompt(conv)
        )
    
    return prompt_template.format(
        conversations=all_conversations,
        count=len(conversations)
    )