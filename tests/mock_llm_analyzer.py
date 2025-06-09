"""Mock LLM analyzer for testing."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import random

# Import from main project
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager


class MockLLMAnalyzer:
    """Mock LLM analyzer that generates simulated analysis results."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.provider = self.config.get("llm.provider", "ollama")
        
    def analyze_conversation(self, speaker_transcript: str, segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM analysis of a conversation segment."""
        self.logger.info(f"Mock analyzing conversation with {self.provider}")
        
        # Extract speaker count from transcript
        speaker_count = len(set(line.strip()[1:-2] for line in speaker_transcript.split('\n') 
                               if line.strip().startswith('[') and line.strip().endswith(']:')))
        
        # Generate mock analysis based on transcript length
        transcript_length = len(speaker_transcript)
        
        # Simulate different types of conversations
        if transcript_length < 500:
            conversation_type = "brief_interaction"
            topics = ["Greeting", "Quick question"]
        elif transcript_length < 2000:
            conversation_type = "standard_conversation"
            topics = ["Daily planning", "Work discussion", "Personal chat"]
        else:
            conversation_type = "extended_discussion"
            topics = ["Project planning", "Technical discussion", "Meeting notes"]
        
        # Generate mock summary
        summary = self._generate_mock_summary(speaker_transcript, conversation_type)
        
        # Generate mock key points
        key_points = self._generate_mock_key_points(transcript_length, topics)
        
        # Generate mock action items
        action_items = self._generate_mock_action_items(conversation_type)
        
        # Generate mock sentiment analysis
        sentiment = self._generate_mock_sentiment()
        
        return {
            "summary": summary,
            "key_points": key_points,
            "action_items": action_items,
            "topics": topics,
            "sentiment": sentiment,
            "speaker_analysis": {
                "speaker_count": speaker_count,
                "dominant_speaker": "Hablante 1" if speaker_count > 0 else None,
                "interaction_type": conversation_type
            },
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "llm_provider": self.provider,
                "model": self.config.get(f"llm.{self.provider}.model", "mock-model"),
                "confidence": random.uniform(0.85, 0.95)
            }
        }
    
    def _generate_mock_summary(self, transcript: str, conversation_type: str) -> str:
        """Generate a mock summary based on conversation type."""
        summaries = {
            "brief_interaction": "Breve intercambio de saludos y consulta rápida sobre horarios.",
            "standard_conversation": "Conversación sobre planes del día, incluyendo discusión de tareas pendientes y coordinación de actividades.",
            "extended_discussion": "Reunión extensa cubriendo múltiples temas del proyecto, asignación de responsabilidades y planificación de próximos pasos."
        }
        
        return summaries.get(conversation_type, "Conversación general entre participantes.")
    
    def _generate_mock_key_points(self, length: int, topics: List[str]) -> List[str]:
        """Generate mock key points based on transcript length."""
        base_points = [
            "Se discutieron los puntos principales de la agenda",
            "Se acordaron los próximos pasos a seguir",
            "Se establecieron fechas límite para las tareas"
        ]
        
        if length > 1000:
            base_points.extend([
                "Se identificaron posibles riesgos y soluciones",
                "Se asignaron responsabilidades específicas a cada participante"
            ])
        
        return base_points[:min(len(base_points), max(2, length // 500))]
    
    def _generate_mock_action_items(self, conversation_type: str) -> List[Dict[str, str]]:
        """Generate mock action items."""
        if conversation_type == "brief_interaction":
            return []
        
        action_items = [
            {
                "task": "Revisar documentación del proyecto",
                "assignee": "Hablante 1",
                "deadline": "Fin de semana"
            },
            {
                "task": "Preparar presentación para la reunión",
                "assignee": "Hablante 2",
                "deadline": "Próximo martes"
            }
        ]
        
        if conversation_type == "extended_discussion":
            action_items.append({
                "task": "Coordinar con el equipo técnico",
                "assignee": "Todos",
                "deadline": "Esta semana"
            })
        
        return action_items
    
    def _generate_mock_sentiment(self) -> Dict[str, float]:
        """Generate mock sentiment analysis."""
        # Generate random but realistic sentiment scores
        positive = random.uniform(0.3, 0.7)
        negative = random.uniform(0.1, 0.3)
        neutral = 1.0 - positive - negative
        
        return {
            "positive": round(positive, 3),
            "negative": round(negative, 3),
            "neutral": round(neutral, 3),
            "overall": "positive" if positive > 0.5 else "neutral"
        }
    
    def test_llm_connection(self) -> bool:
        """Test if LLM service is available (always True for mock)."""
        self.logger.info(f"Mock testing {self.provider} connection")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model": self.config.get(f"llm.{self.provider}.model", "mock-model"),
            "status": "available",
            "mock": True,
            "capabilities": {
                "summarization": True,
                "sentiment_analysis": True,
                "action_extraction": True,
                "topic_modeling": True
            }
        }


class MockConversationSegmenter:
    """Mock conversation segmenter for testing."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
    
    def segment_conversation(self, transcription) -> List[Dict[str, Any]]:
        """Create mock conversation segments from transcription."""
        # Simple segmentation based on duration
        total_duration = transcription.duration
        segment_duration = self.config.get("segmentation.min_conversation_length", 60)
        
        segments = []
        start_time = 0.0
        segment_id = 0
        
        while start_time < total_duration:
            end_time = min(start_time + segment_duration, total_duration)
            
            # Extract relevant transcript portion
            relevant_segments = [
                seg for seg in transcription.segments
                if seg.start_time >= start_time and seg.end_time <= end_time
            ]
            
            if relevant_segments:
                segment_transcript = self._format_segment_transcript(relevant_segments)
                
                segments.append({
                    "id": segment_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "transcript": segment_transcript,
                    "speaker_count": len(set(seg.speaker_id for seg in relevant_segments)),
                    "segment_count": len(relevant_segments)
                })
                
                segment_id += 1
            
            start_time = end_time
        
        return segments
    
    def _format_segment_transcript(self, segments: List) -> str:
        """Format transcript segments with speaker labels."""
        lines = []
        current_speaker = None
        
        for seg in segments:
            speaker_label = f"Hablante {int(seg.speaker_id.split('_')[-1]) + 1}"
            
            if speaker_label != current_speaker:
                current_speaker = speaker_label
                lines.append(f"\n[{speaker_label}]:")
            
            lines.append(seg.text)
        
        return "\n".join(lines)