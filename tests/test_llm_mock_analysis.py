"""
Mock-based testing of LLM Analyzer to demonstrate expected behavior.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation, Analysis, Task, Todo, Followup
from src.llm.providers import OllamaProvider


class TestLLMAnalyzerWithMocks(unittest.TestCase):
    """Test LLM Analyzer with mocked responses to show expected behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get.return_value = {
            'provider': 'ollama',
            'fallback_order': ['ollama'],
            'ollama': {
                'enabled': True,
                'model': 'llama3.2:latest'
            }
        }
        
        # Sample conversation
        self.conversation = Conversation(
            id="test-conv-1",
            transcript="Conversación de prueba",
            speaker_transcript="""[Hablante 1] Buenos días equipo. Necesitamos revisar el proyecto Alpha.
[Hablante 2] Claro, ¿cuáles son las prioridades?
[Hablante 1] Primero, necesito que completes el informe de progreso para mañana.
[Hablante 2] Entendido, lo tendré listo antes del mediodía.""",
            speakers=["María García", "Carlos Rodríguez"],
            duration=120.0,
            timestamp=datetime.now()
        )
    
    @patch('src.llm.analyzer.LLMProviderFactory.create_provider')
    def test_expected_analysis_output(self, mock_create_provider):
        """Test that shows expected analysis output format."""
        
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.get_name.return_value = "ollama"
        
        # Mock responses for different prompts
        mock_provider.generate.side_effect = [
            # Summary response
            """La conversación trata sobre la revisión del proyecto Alpha. María García (Hablante 1) 
            solicita a Carlos Rodríguez (Hablante 2) que complete un informe de progreso con 
            fecha límite para mañana antes del mediodía. Carlos confirma que cumplirá con la entrega.""",
            
            # Task extraction response
            """TAREA: Completar el informe de progreso del proyecto Alpha
ASIGNADA_POR: Hablante 1
RESPONSABLE: Hablante 2
FECHA: mañana antes del mediodía
PRIORIDAD: alta
CONTEXTO: Necesario para la revisión del proyecto Alpha
---""",
            
            # Todo extraction response
            """TODO: Revisar el proyecto Alpha
MENCIONADO_POR: Hablante 1
CATEGORIA: proyecto
URGENCIA: alta
---""",
            
            # Followup extraction response
            """SEGUIMIENTO: Estado del proyecto Alpha
ACCION: Revisar el progreso después de recibir el informe
RESPONSABLE: Hablante 1
FECHA: después de mañana
MENCIONADO_POR: Hablante 1
---""",
            
            # Key points response
            """1. Se necesita revisar el proyecto Alpha
2. Carlos debe completar el informe de progreso
3. La fecha límite es mañana antes del mediodía
4. María está liderando la revisión del proyecto"""
        ]
        
        mock_create_provider.return_value = mock_provider
        
        # Create analyzer and run analysis
        analyzer = LLMAnalyzer(self.mock_config)
        analysis = analyzer.analyze_conversation(self.conversation)
        
        # Print expected output format
        print("\n" + "="*80)
        print("EXPECTED LLM ANALYZER OUTPUT")
        print("="*80)
        
        print(f"\n📄 Analysis Object:")
        print(f"   Conversation ID: {analysis.conversation_id}")
        print(f"   Timestamp: {analysis.timestamp}")
        print(f"   Provider: {analysis.llm_provider}")
        print(f"   Duration: {analysis.duration}s")
        
        print(f"\n📝 Summary:")
        print("-" * 60)
        print(analysis.summary)
        
        print(f"\n🔑 Key Points ({len(analysis.key_points)}):")
        print("-" * 60)
        for point in analysis.key_points:
            print(f"- {point}")
        
        print(f"\n📋 Tasks ({len(analysis.tasks)}):")
        print("-" * 60)
        for task in analysis.tasks:
            print(f"Task: {task.description}")
            print(f"  Assigned by: {task.assigned_by}")
            print(f"  Assigned to: {task.assignee}")
            print(f"  Due: {task.due_date}")
            print(f"  Priority: {task.priority}")
            print(f"  Context: {task.context}")
        
        print(f"\n📌 To-dos ({len(analysis.todos)}):")
        print("-" * 60)
        for todo in analysis.todos:
            print(f"Todo: {todo.description}")
            print(f"  Mentioned by: {todo.mentioned_by}")
            print(f"  Category: {todo.category}")
            print(f"  Urgency: {todo.urgency}")
        
        print(f"\n🔄 Follow-ups ({len(analysis.followups)}):")
        print("-" * 60)
        for followup in analysis.followups:
            print(f"Topic: {followup.topic}")
            print(f"  Action: {followup.action_required}")
            print(f"  Responsible: {followup.responsible_party}")
            print(f"  Deadline: {followup.deadline}")
        
        # Save as JSON
        print(f"\n💾 JSON Output:")
        print("-" * 60)
        print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
        
        # Assertions
        self.assertEqual(analysis.conversation_id, "test-conv-1")
        self.assertEqual(len(analysis.tasks), 1)
        self.assertEqual(analysis.tasks[0].description, "Completar el informe de progreso del proyecto Alpha")
        self.assertEqual(len(analysis.key_points), 4)
        
    def test_podcast_conversation_expected_output(self):
        """Test expected output for podcast conversation."""
        
        # Load actual podcast data
        podcast_file = "transcription_result_sample_audio_long.json"
        if os.path.exists(podcast_file):
            with open(podcast_file, 'r') as f:
                data = json.load(f)
            
            # Create conversation from first few segments
            segments = data['segments'][:5]
            speaker_transcript = "\n".join([
                f"[Hablante 1] {seg['text']}" if seg['speaker_id'] == 'SPEAKER_00' else f"[Hablante 2] {seg['text']}"
                for seg in segments
            ])
            
            podcast_conv = Conversation(
                id="podcast-test",
                transcript=" ".join([seg['text'] for seg in segments]),
                speaker_transcript=speaker_transcript,
                speakers=["Presentador", "Santiago"],
                duration=60.0,
                timestamp=datetime.now()
            )
            
            print("\n" + "="*80)
            print("EXPECTED OUTPUT FOR PODCAST CONVERSATION")
            print("="*80)
            
            print("\n📻 Podcast Transcript Preview:")
            print("-" * 60)
            print(speaker_transcript[:300] + "...")
            
            print("\n📝 Expected Summary:")
            print("-" * 60)
            print("""En este episodio del podcast "En tu orilla", el presentador da la bienvenida 
a Santiago de Lil Jesus. La conversación se centra en el próximo lanzamiento 
de nueva música de la banda. El presentador expresa su entusiasmo por tener 
a Santiago en el programa y menciona que tiene muchas preguntas preparadas.""")
            
            print("\n🔑 Expected Key Points:")
            print("-" * 60)
            print("- Bienvenida a Santiago de Lil Jesus")
            print("- Anuncio de nueva música próxima")
            print("- Entusiasmo del presentador por la entrevista")
            print("- Anticipación por las novedades musicales")
            
            print("\n📋 Expected Tasks:")
            print("-" * 60)
            print("(None - podcast conversations typically don't have tasks)")
            
            print("\n📌 Expected To-dos:")
            print("-" * 60)
            print("- Estar pendiente del lanzamiento de nueva música de Lil Jesus")
            print("  Category: entretenimiento")
            print("  Urgency: baja")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)