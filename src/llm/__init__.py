"""
LLM Analyzer module for TranscriMatic.

This module provides AI-powered analysis of transcribed conversations,
including summary generation, task extraction, and daily reporting.
"""

from .analyzer import LLMAnalyzer
from .models import (
    Conversation,
    Analysis,
    Task,
    Todo,
    Followup,
    DailySummary
)
from .providers import (
    LLMProvider,
    LLMProviderFactory,
    OllamaProvider,
    GeminiProvider,
    OpenAIProvider,
    LLMError
)

__all__ = [
    'LLMAnalyzer',
    'Conversation',
    'Analysis',
    'Task',
    'Todo',
    'Followup',
    'DailySummary',
    'LLMProvider',
    'LLMProviderFactory',
    'OllamaProvider',
    'GeminiProvider',
    'OpenAIProvider',
    'LLMError'
]

__version__ = '1.0.0'