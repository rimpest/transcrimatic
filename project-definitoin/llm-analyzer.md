# LLM Analyzer Module

## Purpose
Analyzes segmented conversations using Large Language Models to generate summaries, extract action items, identify to-dos, and note follow-ups. Supports multiple LLM providers: local (Ollama), Google Gemini, and OpenAI, with all processing in Spanish.

## Dependencies
- **External**: 
  - Local: `ollama`, `requests`
  - Gemini: `google-generativeai`
  - OpenAI: `openai`
  - Common: `langchain`, `jinja2`, `backoff`
- **Internal**: [[ config_manager ]], [[ conversation_segmenter ]]

## Interface

### Input
- Segmented conversations from [[ conversation_segmenter ]]
- LLM configuration from [[ config_manager ]]

### Output
- Analysis results with summaries and extracted items
- Structured data for each conversation

### Public Methods

```python
class LLMAnalyzer:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def analyze_conversation(self, conversation: Conversation) -> Analysis:
        """Analyze single conversation"""
        
    def batch_analyze(self, conversations: List[Conversation]) -> List[Analysis]:
        """Analyze multiple conversations efficiently"""
        
    def generate_summary(self, conversation: Conversation) -> str:
        """Generate concise summary"""
        
    def extract_tasks(self, conversation: Conversation) -> List[Task]:
        """Extract action items and tasks"""
        
    def extract_todos(self, conversation: Conversation) -> List[Todo]:
        """Extract to-do items"""
        
    def extract_followups(self, conversation: Conversation) -> List[Followup]:
        """Extract follow-up items"""
        
    def generate_daily_summary(self, analyses: List[Analysis]) -> DailySummary:
        """Generate summary of all conversations"""
        
    def set_provider(self, provider: str):
        """Switch LLM provider dynamically"""
```

## Data Structures

```python
@dataclass
class Analysis:
    conversation_id: str
    summary: str
    key_points: List[str]
    tasks: List[Task]
    todos: List[Todo]
    followups: List[Followup]
    participants: List[str]
    duration: float
    timestamp: datetime
    llm_provider: str  # Track which provider was used

@dataclass
class Task:
    description: str
    assignee: Optional[str]
    assigned_by: Optional[str]  # Who assigned the task
    due_date: Optional[str]
    priority: str  # "alta", "media", "baja"
    context: str

@dataclass
class Todo:
    description: str
    category: str
    urgency: str
    mentioned_by: Optional[str]  # Which speaker mentioned it

@dataclass
class Followup:
    topic: str
    action_required: str
    responsible_party: Optional[str]
    deadline: Optional[str]
    mentioned_by: Optional[str]  # Which speaker mentioned it

@dataclass
class DailySummary:
    date: date
    total_conversations: int
    total_duration: float
    all_tasks: List[Task]
    all_todos: List[Todo]
    all_followups: List[Followup]
    highlights: List[str]
    speaker_participation: Dict[str, float]  # Speaker time percentage
```

## LLM Provider Integration

### Provider Factory
```python
class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> 'LLMProvider':
        """Create appropriate LLM provider based on configuration"""
        providers = {
            'ollama': OllamaProvider,
            'gemini': GeminiProvider,
            'openai': OpenAIProvider
        }
        
        if provider_type not in providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        return providers[provider_type](config)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
```

### Ollama Provider (Local)
```python
class OllamaProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        self.base_url = f"http://{config.get('host', 'localhost')}:{config.get('port', 11434)}"
        self.model = config.get('model', 'llama3:8b')
        self.timeout = config.get('timeout', 120)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local Ollama"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": kwargs.get('temperature', 0.3),
                "stream": False
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()['response']
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
```

### Gemini Provider
```python
class GeminiProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        import google.generativeai as genai
        
        genai.configure(api_key=config.get('api_key'))
        self.model = genai.GenerativeModel(
            config.get('model', 'gemini-1.5-pro')
        )
        self.generation_config = genai.GenerationConfig(
            temperature=config.get('temperature', 0.3),
            max_output_tokens=config.get('max_tokens', 2000),
        )
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Google Gemini"""
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )
        return response.text
    
    def is_available(self) -> bool:
        """Check if Gemini API is accessible"""
        try:
            # Test with a simple prompt
            self.model.generate_content("Test", 
                generation_config=genai.GenerationConfig(max_output_tokens=5))
            return True
        except:
            return False
```

### OpenAI Provider
```python
class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=config.get('api_key'),
            organization=config.get('organization')
        )
        self.model = config.get('model', 'gpt-4-turbo-preview')
        self.temperature = config.get('temperature', 0.3)
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de conversaciones en español."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', 2000)
        )
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            self.client.models.list()
            return True
        except:
            return False
```

### Prompt Templates

```python
SUMMARY_PROMPT_WITH_SPEAKERS = """
Analiza la siguiente conversación en español donde cada hablante está claramente identificado.

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

Responde en español y sé específico sobre qué hablante mencionó cada punto.
"""

TASK_EXTRACTION_WITH_SPEAKERS_PROMPT = """
Extrae todas las tareas y acciones mencionadas en esta conversación.
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
---
"""
```

## Analysis Pipeline

```python
def __init__(self, config_manager: ConfigManager):
    self.config = config_manager
    self.provider_config = config_manager.get("llm")
    
    # Initialize LLM provider based on configuration
    provider_type = self.provider_config.get("provider", "ollama")
    self.provider = LLMProviderFactory.create_provider(
        provider_type,
        self.provider_config.get(provider_type, {})
    )
    
    # Fallback providers in order of preference
    self.fallback_order = self.provider_config.get(
        "fallback_order", 
        ["ollama", "gemini", "openai"]
    )

def analyze_conversation(self, conversation: Conversation) -> Analysis:
    # 1. Use speaker-aware transcript
    conv_text = conversation.speaker_transcript  # This now has [Hablante X] labels
    
    # 2. Generate summary with speaker awareness
    summary = self._call_llm_with_fallback(
        SUMMARY_PROMPT_WITH_SPEAKERS.format(
            conversation_text=conv_text,
            duration=conversation.duration / 60,
            speaker_count=len(conversation.speakers)
        )
    )
    
    # 3. Extract tasks with speaker attribution
    tasks_response = self._call_llm_with_fallback(
        TASK_EXTRACTION_WITH_SPEAKERS_PROMPT.format(
            conversation_text=conv_text
        )
    )
    tasks = self._parse_tasks_with_speakers(tasks_response)
    
    # 4. Extract to-dos and follow-ups
    todos = self._extract_todos_with_speakers(conv_text)
    followups = self._extract_followups_with_speakers(conv_text)
    
    # 5. Extract key points with speaker attribution
    key_points = self._extract_key_points_with_speakers(summary)
    
    return Analysis(
        conversation_id=conversation.id,
        summary=summary,
        key_points=key_points,
        tasks=tasks,
        todos=todos,
        followups=followups,
        participants=conversation.speakers,
        duration=conversation.duration,
        timestamp=datetime.now(),
        llm_provider=self.provider.__class__.__name__
    )

def _call_llm_with_fallback(self, prompt: str, **kwargs) -> str:
    """Call LLM with automatic fallback to other providers"""
    last_error = None
    
    # Try primary provider first
    try:
        if self.provider.is_available():
            return self.provider.generate(prompt, **kwargs)
    except Exception as e:
        self.logger.warning(f"Primary provider failed: {e}")
        last_error = e
    
    # Try fallback providers
    for provider_name in self.fallback_order:
        if provider_name == self.provider.__class__.__name__.lower():
            continue  # Skip primary provider
            
        try:
            fallback_config = self.provider_config.get(provider_name, {})
            if fallback_config.get('enabled', False):
                fallback_provider = LLMProviderFactory.create_provider(
                    provider_name, 
                    fallback_config
                )
                
                if fallback_provider.is_available():
                    self.logger.info(f"Using fallback provider: {provider_name}")
                    return fallback_provider.generate(prompt, **kwargs)
                    
        except Exception as e:
            self.logger.warning(f"Fallback provider {provider_name} failed: {e}")
            last_error = e
    
    raise LLMError(f"All LLM providers failed. Last error: {last_error}")
```

## Structured Output Parsing

```python
def _parse_tasks_with_speakers(self, llm_response: str) -> List[Task]:
    """Parse structured task output with speaker attribution"""
    tasks = []
    task_blocks = llm_response.split("---")
    
    for block in task_blocks:
        if "TAREA:" in block:
            task = self._parse_task_block_with_speakers(block)
            if task:
                tasks.append(task)
    
    return tasks

def _parse_task_block_with_speakers(self, block: str) -> Optional[Task]:
    """Parse individual task block with speaker information"""
    lines = block.strip().split("\n")
    task_data = {}
    
    for line in lines:
        if "TAREA:" in line:
            task_data["description"] = line.split("TAREA:")[1].strip()
        elif "ASIGNADA_POR:" in line:
            task_data["assigned_by"] = line.split("ASIGNADA_POR:")[1].strip()
        elif "RESPONSABLE:" in line:
            assignee = line.split("RESPONSABLE:")[1].strip()
            if assignee.lower() not in ["no especificado", "no especificada"]:
                task_data["assignee"] = assignee
        elif "FECHA:" in line:
            date = line.split("FECHA:")[1].strip()
            if date.lower() not in ["no especificada", "no especificado"]:
                task_data["due_date"] = date
        elif "PRIORIDAD:" in line:
            task_data["priority"] = line.split("PRIORIDAD:")[1].strip().lower()
        elif "CONTEXTO:" in line:
            task_data["context"] = line.split("CONTEXTO:")[1].strip()
    
    return Task(**task_data) if "description" in task_data else None
```

## Configuration

```yaml
llm:
  provider: "ollama"  # Primary provider: ollama, gemini, openai
  fallback_order: ["ollama", "gemini", "openai"]
  
  # Local Ollama configuration
  ollama:
    enabled: true
    host: "localhost"
    port: 11434
    model: "llama3:8b-instruct"  # Spanish-capable model
    timeout: 120
    temperature: 0.3
    
  # Google Gemini configuration
  gemini:
    enabled: false  # Set to true and add API key to enable
    api_key: "${GEMINI_API_KEY}"  # Use environment variable
    model: "gemini-1.5-pro"
    temperature: 0.3
    max_tokens: 2000
    
  # OpenAI configuration
  openai:
    enabled: false  # Set to true and add API key to enable
    api_key: "${OPENAI_API_KEY}"  # Use environment variable
    organization: "${OPENAI_ORG}"  # Optional
    model: "gpt-4-turbo-preview"
    temperature: 0.3
    max_tokens: 2000
    
  # Common settings
  prompts:
    system_prompt: "Eres un asistente experto en análisis de conversaciones en español. Siempre identificas quién dice qué."
    include_speaker_labels: true
    
  extraction:
    task_keywords: ["hacer", "necesito", "hay que", "debes", "tarea", "acción", "encárgate"]
    todo_keywords: ["pendiente", "recordar", "no olvidar", "importante", "revisar"]
    followup_keywords: ["seguimiento", "revisar", "volver a", "próxima vez", "confirmar"]
    
  # Cost optimization
  cost_optimization:
    cache_similar_prompts: true
    batch_conversations: true
    max_batch_size: 5
```

## Error Handling

### LLM Connection Issues
```python
def _call_llm(self, prompt: str, retries: int = 3) -> str:
    """Call LLM with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.config.get("temperature", 0.3)
                },
                timeout=self.config.get("timeout", 120)
            )
            return response.json()["response"]
        except Exception as e:
            if attempt == retries - 1:
                raise LLMError(f"Failed to get LLM response: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Receives from: [[ conversation_segmenter ]]
- Outputs to: [[ output_formatter ]]

## Performance Optimization

1. **Batch Processing**: Send multiple conversations in one prompt
2. **Caching**: Cache similar conversation analyses
3. **Parallel Processing**: Analyze multiple conversations concurrently
4. **Prompt Optimization**: Minimize token usage while maintaining quality

## Quality Metrics

- Summary coherence and completeness
- Task extraction accuracy
- Processing time per conversation
- Token usage efficiency
- False positive/negative rates for extractions

## Testing Considerations

- Test with various conversation types and speaker counts
- Verify Spanish language quality across all providers
- Test task extraction accuracy with speaker attribution
- Handle provider unavailability and fallback scenarios
- Validate structured output parsing
- Test cost optimization with batching
- Verify API key security (environment variables)
- Test provider switching during runtime

## Security Considerations

- Store API keys in environment variables or secure vault
- Never log API keys or sensitive prompts
- Implement rate limiting for external APIs
- Sanitize conversation content before sending to external APIs
- Use minimal data retention policies for external providers

## Future Enhancements

- Fine-tuned Spanish conversation models
- Custom local models for specific domains
- Real-time streaming analysis
- Multi-language support
- Voice tone and emotion analysis
- Meeting type classification
- Integration with calendar systems for task scheduling
- Cost tracking and optimization dashboard