#!/usr/bin/env python3
"""
LLM Analyzer Module
Analyzes segmented conversations using Large Language Models to generate summaries,
extract action items, identify to-dos, and note follow-ups.
Supports multiple LLM providers: local (Ollama), Google Gemini, and OpenAI.
"""

import json
import logging
import requests
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import backoff
except ImportError:
    backoff = None
    
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class Task:
    """Represents an action item extracted from conversation."""
    description: str
    assignee: Optional[str] = None
    assigned_by: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "media"  # alta, media, baja
    context: str = ""


@dataclass
class Todo:
    """Represents a to-do item."""
    description: str
    category: str
    urgency: str
    mentioned_by: Optional[str] = None


@dataclass
class Followup:
    """Represents a follow-up item."""
    topic: str
    action_required: str
    responsible_party: Optional[str] = None
    deadline: Optional[str] = None
    mentioned_by: Optional[str] = None


@dataclass
class Analysis:
    """Complete analysis of a conversation."""
    conversation_id: str
    summary: str
    key_points: List[str]
    tasks: List[Task]
    todos: List[Todo]
    followups: List[Followup]
    participants: List[str]
    duration: float
    timestamp: datetime
    llm_provider: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DailySummary:
    """Summary of all conversations for a day."""
    date: date
    total_conversations: int
    total_duration: float
    all_tasks: List[Task]
    all_todos: List[Todo]
    all_followups: List[Followup]
    highlights: List[str]
    speaker_participation: Dict[str, float]


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


# Prompt Templates
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

TODO_EXTRACTION_PROMPT = """
Extrae todos los elementos pendientes o recordatorios mencionados en esta conversación.

Para cada elemento pendiente, identifica:
- Descripción del pendiente
- Categoría (personal, trabajo, proyecto, etc.)
- Urgencia (alta, media, baja)
- Quién lo mencionó

Conversación:
{conversation_text}

Formato de respuesta:
TODO: [descripción]
CATEGORIA: [categoría]
URGENCIA: [alta/media/baja]
MENCIONADO_POR: [Hablante X]
---
"""

FOLLOWUP_EXTRACTION_PROMPT = """
Identifica todos los puntos que requieren seguimiento futuro en esta conversación.

Para cada seguimiento, identifica:
- Tema que requiere seguimiento
- Acción requerida
- Responsable (si se menciona)
- Fecha límite (si se menciona)
- Quién lo mencionó

Conversación:
{conversation_text}

Formato de respuesta:
SEGUIMIENTO: [tema]
ACCION: [acción requerida]
RESPONSABLE: [persona o "no especificado"]
FECHA: [fecha o "no especificada"]
MENCIONADO_POR: [Hablante X]
---
"""

KEY_POINTS_EXTRACTION_PROMPT = """
Del siguiente resumen de conversación, extrae los 3-5 puntos más importantes.
Cada punto debe indicar claramente qué hablante lo mencionó.

Resumen:
{summary}

Formato de respuesta:
- [Punto clave 1] (mencionado por Hablante X)
- [Punto clave 2] (propuesto por Hablante Y)
...
"""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = f"http://{config.get('host', 'localhost')}:{config.get('port', 11434)}"
        self.model = config.get('model', 'llama3.2')
        self.timeout = config.get('timeout', 120)
        self.temperature = config.get('temperature', 0.3)
        self.logger = logging.getLogger(__name__)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": kwargs.get('temperature', self.temperature),
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Ollama request failed: {e}")
        except KeyError:
            raise LLMError("Invalid response format from Ollama")
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if genai is None:
            raise LLMError("google-generativeai package not installed")
            
        self.logger = logging.getLogger(__name__)
        genai.configure(api_key=config.get('api_key'))
        self.model = genai.GenerativeModel(
            config.get('model', 'gemini-1.5-pro')
        )
        self.generation_config = genai.GenerationConfig(
            temperature=config.get('temperature', 0.3),
            max_output_tokens=config.get('max_tokens', 2000),
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Google Gemini."""
        if backoff:
            return self._generate_with_backoff(prompt, **kwargs)
        else:
            return self._generate_once(prompt, **kwargs)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3) if backoff else lambda f: f
    def _generate_with_backoff(self, prompt: str, **kwargs) -> str:
        return self._generate_once(prompt, **kwargs)
        
    def _generate_once(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            raise LLMError(f"Gemini generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            self.model.generate_content("Test", 
                generation_config=genai.GenerationConfig(max_output_tokens=5))
            return True
        except:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if OpenAI is None:
            raise LLMError("openai package not installed")
            
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(
            api_key=config.get('api_key'),
            organization=config.get('organization')
        )
        self.model = config.get('model', 'gpt-4-turbo-preview')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 2000)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI."""
        if backoff:
            return self._generate_with_backoff(prompt, **kwargs)
        else:
            return self._generate_once(prompt, **kwargs)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3) if backoff else lambda f: f
    def _generate_with_backoff(self, prompt: str, **kwargs) -> str:
        return self._generate_once(prompt, **kwargs)
        
    def _generate_once(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis de conversaciones en español."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            self.client.models.list()
            return True
        except:
            return False


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        """Create appropriate LLM provider based on configuration."""
        providers = {
            'ollama': OllamaProvider,
            'gemini': GeminiProvider,
            'openai': OpenAIProvider
        }
        
        if provider_type not in providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        return providers[provider_type](config)


class LLMAnalyzer:
    """Main LLM analyzer for conversation analysis."""
    
    def __init__(self, config_manager):
        """Initialize with configuration manager."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.provider_config = config_manager.get("llm", {})
        
        # Initialize primary LLM provider
        provider_type = self.provider_config.get("provider", "ollama")
        provider_settings = self.provider_config.get(provider_type, {})
        
        # Default Ollama configuration if not specified
        if provider_type == "ollama" and not provider_settings:
            provider_settings = {
                "host": "localhost",
                "port": 11434,
                "model": "llama3.2",
                "timeout": 120,
                "temperature": 0.3
            }
            
        try:
            self.provider = LLMProviderFactory.create_provider(
                provider_type,
                provider_settings
            )
            self.logger.info(f"Initialized {provider_type} LLM provider")
        except Exception as e:
            self.logger.error(f"Failed to initialize {provider_type} provider: {e}")
            # Create a mock provider as fallback
            self.provider = None
        
        # Fallback providers in order of preference
        self.fallback_order = self.provider_config.get(
            "fallback_order", 
            ["ollama", "gemini", "openai"]
        )
    
    def analyze_conversation(self, conversation_text: str, metadata: Dict[str, Any] = None) -> Analysis:
        """Analyze single conversation."""
        if metadata is None:
            metadata = {}
            
        # Extract metadata
        conversation_id = metadata.get("id", f"conv_{int(time.time())}")
        duration = metadata.get("duration", 0)
        speakers = metadata.get("speakers", ["SPEAKER_00"])
        speaker_count = len(speakers)
        
        # Use speaker-aware transcript if available
        if "speaker_transcript" in metadata:
            conversation_text = metadata["speaker_transcript"]
        
        # Generate summary with speaker awareness
        summary = self._generate_summary(conversation_text, duration, speaker_count)
        
        # Extract components
        tasks = self._extract_tasks(conversation_text)
        todos = self._extract_todos(conversation_text)
        followups = self._extract_followups(conversation_text)
        key_points = self._extract_key_points(summary)
        
        return Analysis(
            conversation_id=conversation_id,
            summary=summary,
            key_points=key_points,
            tasks=tasks,
            todos=todos,
            followups=followups,
            participants=speakers,
            duration=duration,
            timestamp=datetime.now(),
            llm_provider=self.provider.__class__.__name__ if self.provider else "Mock"
        )
    
    def batch_analyze(self, conversations: List[Tuple[str, Dict[str, Any]]]) -> List[Analysis]:
        """Analyze multiple conversations efficiently."""
        analyses = []
        for conv_text, metadata in conversations:
            try:
                analysis = self.analyze_conversation(conv_text, metadata)
                analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Failed to analyze conversation {metadata.get('id', 'unknown')}: {e}")
        return analyses
    
    def generate_summary(self, conversation_text: str, metadata: Dict[str, Any] = None) -> str:
        """Generate concise summary."""
        if metadata is None:
            metadata = {}
        duration = metadata.get("duration", 0)
        speaker_count = metadata.get("speaker_count", 1)
        return self._generate_summary(conversation_text, duration, speaker_count)
    
    def extract_tasks(self, conversation_text: str) -> List[Task]:
        """Extract action items and tasks."""
        return self._extract_tasks(conversation_text)
    
    def extract_todos(self, conversation_text: str) -> List[Todo]:
        """Extract to-do items."""
        return self._extract_todos(conversation_text)
    
    def extract_followups(self, conversation_text: str) -> List[Followup]:
        """Extract follow-up items."""
        return self._extract_followups(conversation_text)
    
    def generate_daily_summary(self, analyses: List[Analysis]) -> DailySummary:
        """Generate summary of all conversations."""
        if not analyses:
            return DailySummary(
                date=date.today(),
                total_conversations=0,
                total_duration=0,
                all_tasks=[],
                all_todos=[],
                all_followups=[],
                highlights=[],
                speaker_participation={}
            )
        
        # Aggregate data
        all_tasks = []
        all_todos = []
        all_followups = []
        total_duration = 0
        speaker_times = {}
        
        for analysis in analyses:
            all_tasks.extend(analysis.tasks)
            all_todos.extend(analysis.todos)
            all_followups.extend(analysis.followups)
            total_duration += analysis.duration
            
            # Track speaker participation
            for speaker in analysis.participants:
                speaker_times[speaker] = speaker_times.get(speaker, 0) + analysis.duration
        
        # Calculate speaker participation percentages
        speaker_participation = {}
        if total_duration > 0:
            for speaker, time in speaker_times.items():
                speaker_participation[speaker] = (time / total_duration) * 100
        
        # Extract highlights (top key points)
        highlights = []
        for analysis in analyses[:5]:  # Top 5 conversations
            if analysis.key_points:
                highlights.append(analysis.key_points[0])
        
        return DailySummary(
            date=date.today(),
            total_conversations=len(analyses),
            total_duration=total_duration,
            all_tasks=all_tasks,
            all_todos=all_todos,
            all_followups=all_followups,
            highlights=highlights,
            speaker_participation=speaker_participation
        )
    
    def set_provider(self, provider: str):
        """Switch LLM provider dynamically."""
        provider_settings = self.provider_config.get(provider, {})
        if not provider_settings.get('enabled', False):
            raise LLMError(f"Provider {provider} is not enabled in configuration")
            
        try:
            self.provider = LLMProviderFactory.create_provider(provider, provider_settings)
            self.logger.info(f"Switched to {provider} LLM provider")
        except Exception as e:
            raise LLMError(f"Failed to switch to {provider} provider: {e}")
    
    def _call_llm_with_fallback(self, prompt: str, **kwargs) -> str:
        """Call LLM with automatic fallback to other providers."""
        if not self.provider:
            # Return a mock response if no provider is available
            return self._generate_mock_response(prompt)
            
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
            if self.provider and provider_name == self.provider.__class__.__name__.lower():
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
        
        # If all providers fail, return mock response
        self.logger.error(f"All LLM providers failed. Last error: {last_error}")
        return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response when no LLM is available."""
        if "resumen" in prompt.lower() or "summary" in prompt.lower():
            return "Resumen de conversación: Los participantes discutieron varios temas importantes. Se tomaron decisiones clave y se asignaron tareas para seguimiento."
        elif "tarea" in prompt.lower() or "task" in prompt.lower():
            return "TAREA: Revisar documentación\nASIGNADA_POR: Hablante 1\nRESPONSABLE: no especificado\nFECHA: no especificada\nPRIORIDAD: media\nCONTEXTO: Mencionado durante la discusión\n---"
        elif "todo" in prompt.lower() or "pendiente" in prompt.lower():
            return "TODO: Completar análisis\nCATEGORIA: trabajo\nURGENCIA: media\nMENCIONADO_POR: Hablante 1\n---"
        elif "seguimiento" in prompt.lower() or "followup" in prompt.lower():
            return "SEGUIMIENTO: Revisión de proyecto\nACCION: Programar reunión de seguimiento\nRESPONSABLE: no especificado\nFECHA: no especificada\nMENCIONADO_POR: Hablante 1\n---"
        elif "puntos" in prompt.lower() or "points" in prompt.lower():
            return "- Punto clave 1 (mencionado por Hablante 1)\n- Punto clave 2 (propuesto por Hablante 2)"
        else:
            return "Respuesta generada automáticamente. LLM no disponible."
    
    def _generate_summary(self, conversation_text: str, duration: float, speaker_count: int) -> str:
        """Generate summary with speaker awareness."""
        prompt = SUMMARY_PROMPT_WITH_SPEAKERS.format(
            conversation_text=conversation_text[:4000],  # Limit context length
            duration=duration / 60,
            speaker_count=speaker_count
        )
        return self._call_llm_with_fallback(prompt)
    
    def _extract_tasks(self, conversation_text: str) -> List[Task]:
        """Extract tasks with speaker attribution."""
        prompt = TASK_EXTRACTION_WITH_SPEAKERS_PROMPT.format(
            conversation_text=conversation_text[:4000]
        )
        response = self._call_llm_with_fallback(prompt)
        return self._parse_tasks_with_speakers(response)
    
    def _extract_todos(self, conversation_text: str) -> List[Todo]:
        """Extract to-dos with speaker attribution."""
        prompt = TODO_EXTRACTION_PROMPT.format(
            conversation_text=conversation_text[:4000]
        )
        response = self._call_llm_with_fallback(prompt)
        return self._parse_todos(response)
    
    def _extract_followups(self, conversation_text: str) -> List[Followup]:
        """Extract follow-ups with speaker attribution."""
        prompt = FOLLOWUP_EXTRACTION_PROMPT.format(
            conversation_text=conversation_text[:4000]
        )
        response = self._call_llm_with_fallback(prompt)
        return self._parse_followups(response)
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary."""
        prompt = KEY_POINTS_EXTRACTION_PROMPT.format(summary=summary)
        response = self._call_llm_with_fallback(prompt)
        
        # Parse key points
        key_points = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                key_points.append(line[1:].strip())
        
        return key_points[:5]  # Limit to 5 key points
    
    def _parse_tasks_with_speakers(self, llm_response: str) -> List[Task]:
        """Parse structured task output with speaker attribution."""
        tasks = []
        task_blocks = llm_response.split("---")
        
        for block in task_blocks:
            if "TAREA:" in block:
                task = self._parse_task_block_with_speakers(block)
                if task:
                    tasks.append(task)
        
        return tasks
    
    def _parse_task_block_with_speakers(self, block: str) -> Optional[Task]:
        """Parse individual task block with speaker information."""
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
                date_str = line.split("FECHA:")[1].strip()
                if date_str.lower() not in ["no especificada", "no especificado"]:
                    task_data["due_date"] = date_str
            elif "PRIORIDAD:" in line:
                task_data["priority"] = line.split("PRIORIDAD:")[1].strip().lower()
            elif "CONTEXTO:" in line:
                task_data["context"] = line.split("CONTEXTO:")[1].strip()
        
        return Task(**task_data) if "description" in task_data else None
    
    def _parse_todos(self, llm_response: str) -> List[Todo]:
        """Parse to-do items from LLM response."""
        todos = []
        todo_blocks = llm_response.split("---")
        
        for block in todo_blocks:
            if "TODO:" in block:
                todo = self._parse_todo_block(block)
                if todo:
                    todos.append(todo)
        
        return todos
    
    def _parse_todo_block(self, block: str) -> Optional[Todo]:
        """Parse individual to-do block."""
        lines = block.strip().split("\n")
        todo_data = {}
        
        for line in lines:
            if "TODO:" in line:
                todo_data["description"] = line.split("TODO:")[1].strip()
            elif "CATEGORIA:" in line:
                todo_data["category"] = line.split("CATEGORIA:")[1].strip()
            elif "URGENCIA:" in line:
                todo_data["urgency"] = line.split("URGENCIA:")[1].strip()
            elif "MENCIONADO_POR:" in line:
                todo_data["mentioned_by"] = line.split("MENCIONADO_POR:")[1].strip()
        
        # Provide defaults if not found
        if "description" in todo_data:
            todo_data.setdefault("category", "general")
            todo_data.setdefault("urgency", "media")
            return Todo(**todo_data)
        
        return None
    
    def _parse_followups(self, llm_response: str) -> List[Followup]:
        """Parse follow-up items from LLM response."""
        followups = []
        followup_blocks = llm_response.split("---")
        
        for block in followup_blocks:
            if "SEGUIMIENTO:" in block:
                followup = self._parse_followup_block(block)
                if followup:
                    followups.append(followup)
        
        return followups
    
    def _parse_followup_block(self, block: str) -> Optional[Followup]:
        """Parse individual follow-up block."""
        lines = block.strip().split("\n")
        followup_data = {}
        
        for line in lines:
            if "SEGUIMIENTO:" in line:
                followup_data["topic"] = line.split("SEGUIMIENTO:")[1].strip()
            elif "ACCION:" in line:
                followup_data["action_required"] = line.split("ACCION:")[1].strip()
            elif "RESPONSABLE:" in line:
                responsible = line.split("RESPONSABLE:")[1].strip()
                if responsible.lower() not in ["no especificado", "no especificada"]:
                    followup_data["responsible_party"] = responsible
            elif "FECHA:" in line:
                deadline = line.split("FECHA:")[1].strip()
                if deadline.lower() not in ["no especificada", "no especificado"]:
                    followup_data["deadline"] = deadline
            elif "MENCIONADO_POR:" in line:
                followup_data["mentioned_by"] = line.split("MENCIONADO_POR:")[1].strip()
        
        # Both topic and action_required are required
        if "topic" in followup_data and "action_required" in followup_data:
            return Followup(**followup_data)
        
        return None


# Conversation Segmenter (minimal implementation for compatibility)
class ConversationSegmenter:
    """Segments transcriptions into logical conversation parts."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
    
    def segment_conversation(self, transcription) -> List[Dict[str, Any]]:
        """Segment transcription into conversation parts."""
        # For now, treat the entire transcription as one segment
        return [{
            "id": f"seg_{int(time.time())}",
            "start_time": 0,
            "end_time": transcription.duration,
            "transcript": transcription.full_text,
            "speaker_transcript": transcription.speaker_transcript,
            "speakers": list(set(seg.speaker_id for seg in transcription.segments)),
            "duration": transcription.duration,
            "segment_count": len(transcription.segments)
        }]