"""
Main LLM Analyzer module for conversation analysis.
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff

from .providers import LLMProviderFactory, LLMProvider, LLMError
from .models import (
    Conversation, Analysis, Task, Todo, Followup, DailySummary
)
from .prompts import (
    SUMMARY_PROMPT_WITH_SPEAKERS,
    TASK_EXTRACTION_WITH_SPEAKERS_PROMPT,
    TODO_EXTRACTION_WITH_SPEAKERS_PROMPT,
    FOLLOWUP_EXTRACTION_WITH_SPEAKERS_PROMPT,
    KEY_POINTS_EXTRACTION_PROMPT,
    DAILY_HIGHLIGHTS_PROMPT,
    SPEAKER_ROLE_IDENTIFICATION_PROMPT,
    format_conversation_for_prompt,
    format_duration
)

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Analyzes conversations using LLMs with fallback support."""
    
    def __init__(self, config_manager):
        """Initialize with configuration manager."""
        self.config = config_manager
        self.provider_config = config_manager.get("llm", {})
        
        # Cache for provider instances
        self._provider_cache = {}
        
        # Analysis cache to avoid redundant calls
        self._analysis_cache = {}
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Track last used provider
        self._last_used_provider = "unknown"
        
        # Setup fallback order
        self.fallback_order = self.provider_config.get(
            "fallback_order", 
            ["ollama", "gemini", "openai"]
        )
        
        # Initialize primary provider
        provider_type = self.provider_config.get("provider", "ollama")
        self.primary_provider_type = provider_type
        
        try:
            self.provider = self._create_provider(provider_type)
            logger.info(f"Initialized primary LLM provider: {provider_type}")
        except Exception as e:
            logger.error(f"Failed to initialize primary provider {provider_type}: {e}")
            self.provider = None
    
    def _create_provider(self, provider_type: str) -> Optional[LLMProvider]:
        """Create and cache a provider instance."""
        if provider_type in self._provider_cache:
            return self._provider_cache[provider_type]
        
        provider_config = self.provider_config.get(provider_type, {})
        if not provider_config.get('enabled', False):
            return None
        
        try:
            provider = LLMProviderFactory.create_provider(
                provider_type, 
                provider_config
            )
            self._provider_cache[provider_type] = provider
            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            return None
    
    def analyze_conversation(self, conversation: Conversation) -> Analysis:
        """Analyze single conversation."""
        logger.info(f"Analyzing conversation {conversation.id}")
        
        # Check cache
        if conversation.id in self._analysis_cache:
            logger.info(f"Returning cached analysis for {conversation.id}")
            return self._analysis_cache[conversation.id]
        
        try:
            # Use speaker-aware transcript
            conv_text = conversation.speaker_transcript
            
            # 1. Generate summary with speaker awareness
            summary = self._generate_summary(conversation)
            
            # 2. Extract tasks with speaker attribution
            tasks = self._extract_tasks(conversation)
            
            # 3. Extract to-dos
            todos = self._extract_todos(conversation)
            
            # 4. Extract follow-ups
            followups = self._extract_followups(conversation)
            
            # 5. Extract key points
            key_points = self._extract_key_points(summary)
            
            # Create analysis object
            analysis = Analysis(
                conversation_id=conversation.id,
                summary=summary,
                key_points=key_points,
                tasks=tasks,
                todos=todos,
                followups=followups,
                participants=conversation.speakers,
                duration=conversation.duration,
                timestamp=datetime.now(),
                llm_provider=self._get_last_used_provider()
            )
            
            # Cache the result
            self._analysis_cache[conversation.id] = analysis
            
            logger.info(f"Completed analysis for conversation {conversation.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation {conversation.id}: {e}")
            raise
    
    def batch_analyze(self, conversations: List[Conversation]) -> List[Analysis]:
        """Analyze multiple conversations efficiently."""
        logger.info(f"Batch analyzing {len(conversations)} conversations")
        
        # Check batch size limit
        batch_size = self.provider_config.get('cost_optimization', {}).get(
            'max_batch_size', 5
        )
        
        if len(conversations) > batch_size:
            # Process in batches
            results = []
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i + batch_size]
                results.extend(self._process_batch(batch))
            return results
        else:
            return self._process_batch(conversations)
    
    def _process_batch(self, conversations: List[Conversation]) -> List[Analysis]:
        """Process a batch of conversations."""
        futures = []
        
        for conv in conversations:
            future = self.executor.submit(self.analyze_conversation, conv)
            futures.append((future, conv))
        
        results = []
        for future, conv in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze conversation {conv.id}: {e}")
                # Create minimal analysis on failure
                results.append(self._create_fallback_analysis(conv))
        
        return results
    
    def generate_summary(self, conversation: Conversation) -> str:
        """Generate concise summary."""
        return self._generate_summary(conversation)
    
    def extract_tasks(self, conversation: Conversation) -> List[Task]:
        """Extract action items and tasks."""
        return self._extract_tasks(conversation)
    
    def extract_todos(self, conversation: Conversation) -> List[Todo]:
        """Extract to-do items."""
        return self._extract_todos(conversation)
    
    def extract_followups(self, conversation: Conversation) -> List[Followup]:
        """Extract follow-up items."""
        return self._extract_followups(conversation)
    
    def generate_daily_summary(self, analyses: List[Analysis]) -> DailySummary:
        """Generate summary of all conversations."""
        logger.info(f"Generating daily summary for {len(analyses)} analyses")
        
        # Aggregate all data
        all_tasks = []
        all_todos = []
        all_followups = []
        total_duration = 0
        
        for analysis in analyses:
            all_tasks.extend(analysis.tasks)
            all_todos.extend(analysis.todos)
            all_followups.extend(analysis.followups)
            total_duration += analysis.duration
        
        # Calculate speaker participation
        speaker_times = self._calculate_speaker_participation(analyses)
        
        # Generate highlights
        highlights = self._generate_daily_highlights(analyses)
        
        return DailySummary(
            date=date.today(),
            total_conversations=len(analyses),
            total_duration=total_duration,
            all_tasks=all_tasks,
            all_todos=all_todos,
            all_followups=all_followups,
            highlights=highlights,
            speaker_participation=speaker_times
        )
    
    def set_provider(self, provider: str):
        """Switch LLM provider dynamically."""
        logger.info(f"Switching to provider: {provider}")
        
        new_provider = self._create_provider(provider)
        if new_provider and new_provider.is_available():
            self.provider = new_provider
            self.primary_provider_type = provider
            logger.info(f"Successfully switched to {provider}")
        else:
            raise LLMError(f"Provider {provider} is not available")
    
    def _generate_summary(self, conversation: Conversation) -> str:
        """Generate summary with speaker attribution."""
        prompt = SUMMARY_PROMPT_WITH_SPEAKERS.format(
            conversation_text=conversation.speaker_transcript,
            duration=format_duration(conversation.duration),
            speaker_count=len(conversation.speakers)
        )
        
        return self._call_llm_with_fallback(prompt)
    
    def _extract_tasks(self, conversation: Conversation) -> List[Task]:
        """Extract tasks with speaker information."""
        prompt = TASK_EXTRACTION_WITH_SPEAKERS_PROMPT.format(
            conversation_text=conversation.speaker_transcript
        )
        
        response = self._call_llm_with_fallback(prompt)
        return self._parse_tasks_with_speakers(response)
    
    def _extract_todos(self, conversation: Conversation) -> List[Todo]:
        """Extract to-dos with speaker information."""
        prompt = TODO_EXTRACTION_WITH_SPEAKERS_PROMPT.format(
            conversation_text=conversation.speaker_transcript
        )
        
        response = self._call_llm_with_fallback(prompt)
        return self._parse_todos_with_speakers(response)
    
    def _extract_followups(self, conversation: Conversation) -> List[Followup]:
        """Extract follow-ups with speaker information."""
        prompt = FOLLOWUP_EXTRACTION_WITH_SPEAKERS_PROMPT.format(
            conversation_text=conversation.speaker_transcript
        )
        
        response = self._call_llm_with_fallback(prompt)
        return self._parse_followups_with_speakers(response)
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary."""
        prompt = KEY_POINTS_EXTRACTION_PROMPT.format(summary=summary)
        
        response = self._call_llm_with_fallback(prompt)
        
        # Parse key points from response
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering or bullets
                point = re.sub(r'^[\d\-•\.]+\s*', '', line)
                if point:
                    points.append(point)
        
        return points[:8]  # Maximum 8 points
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def _call_llm_with_fallback(self, prompt: str, **kwargs) -> str:
        """Call LLM with automatic fallback to other providers."""
        last_error = None
        used_provider = None
        
        # Try primary provider first
        if self.provider:
            try:
                if self.provider.is_available():
                    response = self.provider.generate(prompt, **kwargs)
                    self._last_used_provider = self.provider.get_name()
                    return response
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
                last_error = e
        
        # Try fallback providers
        for provider_name in self.fallback_order:
            if provider_name == self.primary_provider_type:
                continue  # Skip primary provider
            
            provider = self._create_provider(provider_name)
            if not provider:
                continue
            
            try:
                if provider.is_available():
                    logger.info(f"Using fallback provider: {provider_name}")
                    response = provider.generate(prompt, **kwargs)
                    self._last_used_provider = provider_name
                    return response
            except Exception as e:
                logger.warning(f"Fallback provider {provider_name} failed: {e}")
                last_error = e
        
        raise LLMError(f"All LLM providers failed. Last error: {last_error}")
    
    def _get_last_used_provider(self) -> str:
        """Get the name of the last successfully used provider."""
        return getattr(self, '_last_used_provider', 'unknown')
    
    def _parse_tasks_with_speakers(self, llm_response: str) -> List[Task]:
        """Parse structured task output with speaker attribution."""
        tasks = []
        task_blocks = llm_response.split("---")
        
        for block in task_blocks:
            if "TAREA:" in block:
                task = self._parse_task_block(block)
                if task:
                    tasks.append(task)
        
        return tasks
    
    def _parse_task_block(self, block: str) -> Optional[Task]:
        """Parse individual task block."""
        lines = block.strip().split("\n")
        task_data = {
            "priority": "media",  # default
            "context": ""  # default
        }
        
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
    
    def _parse_todos_with_speakers(self, llm_response: str) -> List[Todo]:
        """Parse to-dos with speaker information."""
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
        todo_data = {
            "category": "general",  # default
            "urgency": "normal"  # default
        }
        
        for line in lines:
            if "TODO:" in line:
                todo_data["description"] = line.split("TODO:")[1].strip()
            elif "MENCIONADO_POR:" in line:
                todo_data["mentioned_by"] = line.split("MENCIONADO_POR:")[1].strip()
            elif "CATEGORIA:" in line:
                todo_data["category"] = line.split("CATEGORIA:")[1].strip()
            elif "URGENCIA:" in line:
                todo_data["urgency"] = line.split("URGENCIA:")[1].strip()
        
        return Todo(**todo_data) if "description" in todo_data else None
    
    def _parse_followups_with_speakers(self, llm_response: str) -> List[Followup]:
        """Parse follow-ups with speaker information."""
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
                date_str = line.split("FECHA:")[1].strip()
                if date_str.lower() not in ["no especificada", "no especificado"]:
                    followup_data["deadline"] = date_str
            elif "MENCIONADO_POR:" in line:
                followup_data["mentioned_by"] = line.split("MENCIONADO_POR:")[1].strip()
        
        # Ensure required fields
        if "topic" in followup_data and "action_required" in followup_data:
            return Followup(**followup_data)
        return None
    
    def _calculate_speaker_participation(self, analyses: List[Analysis]) -> Dict[str, float]:
        """Calculate speaker participation percentages."""
        speaker_counts = {}
        total_speakers = 0
        
        for analysis in analyses:
            for speaker in analysis.participants:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                total_speakers += 1
        
        # Calculate percentages
        speaker_participation = {}
        for speaker, count in speaker_counts.items():
            speaker_participation[speaker] = round((count / total_speakers) * 100, 2)
        
        return speaker_participation
    
    def _generate_daily_highlights(self, analyses: List[Analysis]) -> List[str]:
        """Generate daily highlights using LLM."""
        # Prepare summaries for highlight generation
        summaries = []
        for i, analysis in enumerate(analyses, 1):
            summaries.append(f"Conversación {i}:\n{analysis.summary}")
        
        prompt = DAILY_HIGHLIGHTS_PROMPT.format(
            conversation_count=len(analyses),
            total_duration=format_duration(sum(a.duration for a in analyses)),
            summaries="\n\n".join(summaries)
        )
        
        response = self._call_llm_with_fallback(prompt)
        
        # Parse highlights
        highlights = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                highlight = re.sub(r'^[\d\-\.]+\s*', '', line)
                if highlight:
                    highlights.append(highlight)
        
        return highlights[:7]  # Maximum 7 highlights
    
    def _create_fallback_analysis(self, conversation: Conversation) -> Analysis:
        """Create a minimal analysis when LLM fails."""
        return Analysis(
            conversation_id=conversation.id,
            summary="Error: No se pudo analizar esta conversación",
            key_points=[],
            tasks=[],
            todos=[],
            followups=[],
            participants=conversation.speakers,
            duration=conversation.duration,
            timestamp=datetime.now(),
            llm_provider="fallback"
        )
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)