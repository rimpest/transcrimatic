"""
LLM Provider classes for different AI services.
Supports Ollama (local), Google Gemini, and OpenAI.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import requests
import time
import os

logger = logging.getLogger(__name__)


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
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 11434)
        self.base_url = f"http://{self.host}:{self.port}"
        self.model = config.get('model', 'llama3.2:latest')
        self.timeout = config.get('timeout', 120)
        self.temperature = config.get('temperature', 0.3)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": kwargs.get('temperature', self.temperature),
                    "stream": False,
                    "options": {
                        "num_predict": kwargs.get('max_tokens', 2000)
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_name(self) -> str:
        return "ollama"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package not installed")
        
        api_key = config.get('api_key')
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key not provided")
        
        self.genai.configure(api_key=api_key)
        self.model = self.genai.GenerativeModel(
            config.get('model', 'gemini-1.5-pro')
        )
        self.generation_config = self.genai.GenerationConfig(
            temperature=config.get('temperature', 0.3),
            max_output_tokens=config.get('max_tokens', 2000),
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Google Gemini."""
        try:
            generation_config = self.genai.GenerationConfig(
                temperature=kwargs.get('temperature', self.generation_config.temperature),
                max_output_tokens=kwargs.get('max_tokens', self.generation_config.max_output_tokens)
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            test_config = self.genai.GenerationConfig(max_output_tokens=5)
            self.model.generate_content("Test", generation_config=test_config)
            return True
        except:
            return False
    
    def get_name(self) -> str:
        return "gemini"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError("openai package not installed")
        
        api_key = config.get('api_key')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = self.OpenAI(
            api_key=api_key,
            organization=config.get('organization') or os.environ.get('OPENAI_ORG')
        )
        self.model = config.get('model', 'gpt-4-turbo-preview')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 2000)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un asistente experto en análisis de conversaciones en español. Siempre identificas quién dice qué."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            self.client.models.list()
            return True
        except:
            return False
    
    def get_name(self) -> str:
        return "openai"


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        'ollama': OllamaProvider,
        'gemini': GeminiProvider,
        'openai': OpenAIProvider
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        """Create appropriate LLM provider based on configuration."""
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider type."""
        if not issubclass(provider_class, LLMProvider):
            raise TypeError("Provider must inherit from LLMProvider")
        cls._providers[name] = provider_class


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass