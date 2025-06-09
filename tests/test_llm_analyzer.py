"""
Unit tests for the LLM Analyzer module.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date
import json

from src.llm import (
    LLMAnalyzer,
    Conversation,
    Analysis,
    Task,
    Todo,
    Followup,
    DailySummary,
    LLMProviderFactory,
    OllamaProvider,
    GeminiProvider,
    OpenAIProvider,
    LLMError
)


class TestLLMProviders(unittest.TestCase):
    """Test LLM provider implementations."""
    
    def test_ollama_provider_initialization(self):
        """Test Ollama provider initialization."""
        config = {
            'host': 'localhost',
            'port': 11434,
            'model': 'llama3:8b',
            'timeout': 120
        }
        provider = OllamaProvider(config)
        
        self.assertEqual(provider.host, 'localhost')
        self.assertEqual(provider.port, 11434)
        self.assertEqual(provider.model, 'llama3:8b')
        self.assertEqual(provider.get_name(), 'ollama')
    
    @patch('requests.post')
    def test_ollama_generate(self, mock_post):
        """Test Ollama generation."""
        mock_response = Mock()
        mock_response.json.return_value = {'response': 'Test response'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        provider = OllamaProvider({'model': 'llama3:8b'})
        result = provider.generate("Test prompt")
        
        self.assertEqual(result, 'Test response')
        mock_post.assert_called_once()
    
    @patch('requests.get')
    def test_ollama_is_available(self, mock_get):
        """Test Ollama availability check."""
        mock_get.return_value.status_code = 200
        
        provider = OllamaProvider({})
        self.assertTrue(provider.is_available())
    
    def test_provider_factory(self):
        """Test provider factory creation."""
        # Test Ollama creation
        provider = LLMProviderFactory.create_provider('ollama', {})
        self.assertIsInstance(provider, OllamaProvider)
        
        # Test invalid provider
        with self.assertRaises(ValueError):
            LLMProviderFactory.create_provider('invalid', {})


class TestDataModels(unittest.TestCase):
    """Test data model classes."""
    
    def test_task_creation(self):
        """Test Task model creation and serialization."""
        task = Task(
            description="Test task",
            assignee="Hablante 1",
            assigned_by="Hablante 2",
            priority="alta",
            context="Test context"
        )
        
        self.assertEqual(task.description, "Test task")
        self.assertEqual(task.priority, "alta")
        
        # Test serialization
        task_dict = task.to_dict()
        self.assertIn('description', task_dict)
        self.assertEqual(task_dict['assignee'], "Hablante 1")
    
    def test_analysis_creation(self):
        """Test Analysis model creation."""
        analysis = Analysis(
            conversation_id="test-123",
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            tasks=[],
            todos=[],
            followups=[],
            participants=["Hablante 1", "Hablante 2"],
            duration=300.0,
            timestamp=datetime.now(),
            llm_provider="ollama"
        )
        
        self.assertEqual(analysis.conversation_id, "test-123")
        self.assertEqual(len(analysis.key_points), 2)
        
        # Test serialization
        analysis_dict = analysis.to_dict()
        self.assertIn('summary', analysis_dict)
        self.assertIn('timestamp', analysis_dict)
    
    def test_daily_summary(self):
        """Test DailySummary model."""
        task1 = Task(description="Task 1", assignee="Person A")
        task2 = Task(description="Task 2", assignee="Person B")
        
        summary = DailySummary(
            date=date.today(),
            total_conversations=5,
            total_duration=3600.0,
            all_tasks=[task1, task2],
            all_todos=[],
            all_followups=[],
            highlights=["Highlight 1"],
            speaker_participation={"Hablante 1": 60.0, "Hablante 2": 40.0}
        )
        
        # Test task grouping
        by_assignee = summary.get_tasks_by_assignee()
        self.assertIn("Person A", by_assignee)
        self.assertEqual(len(by_assignee["Person A"]), 1)
        
        # Test serialization
        summary_dict = summary.to_dict()
        self.assertEqual(summary_dict['total_duration_minutes'], 60.0)
        self.assertIn('summary_stats', summary_dict)


class TestLLMAnalyzer(unittest.TestCase):
    """Test main LLM Analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.get.return_value = {
            'provider': 'ollama',
            'fallback_order': ['ollama', 'gemini'],
            'ollama': {
                'enabled': True,
                'model': 'llama3:8b'
            }
        }
        
        # Create test conversation
        self.test_conversation = Conversation(
            id="test-conv-1",
            transcript="Test transcript",
            speaker_transcript="[Hablante 1] Necesito que hagas la tarea X\n[Hablante 2] Claro, lo haré mañana",
            speakers=["Hablante 1", "Hablante 2"],
            duration=300.0,
            timestamp=datetime.now()
        )
    
    @patch('src.llm.analyzer.LLMProviderFactory.create_provider')
    def test_analyzer_initialization(self, mock_create_provider):
        """Test analyzer initialization."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider
        
        analyzer = LLMAnalyzer(self.mock_config)
        
        self.assertIsNotNone(analyzer.provider)
        mock_create_provider.assert_called()
    
    @patch('src.llm.analyzer.LLMProviderFactory.create_provider')
    def test_analyze_conversation(self, mock_create_provider):
        """Test conversation analysis."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.get_name.return_value = "ollama"
        mock_provider.generate.side_effect = [
            "Resumen: Hablante 1 asignó una tarea a Hablante 2",  # Summary
            "TAREA: Hacer tarea X\nASIGNADA_POR: Hablante 1\nRESPONSABLE: Hablante 2\nPRIORIDAD: alta\n---",  # Tasks
            "TODO: Revisar resultados\nMENCIONADO_POR: Hablante 1\n---",  # Todos
            "SEGUIMIENTO: Verificar progreso\nACCION: Revisar mañana\n---",  # Followups
            "1. Tarea asignada por Hablante 1\n2. Hablante 2 confirmó"  # Key points
        ]
        mock_create_provider.return_value = mock_provider
        
        analyzer = LLMAnalyzer(self.mock_config)
        analysis = analyzer.analyze_conversation(self.test_conversation)
        
        self.assertEqual(analysis.conversation_id, "test-conv-1")
        self.assertIn("Hablante 1 asignó", analysis.summary)
        self.assertEqual(len(analysis.tasks), 1)
        self.assertEqual(analysis.tasks[0].description, "Hacer tarea X")
        self.assertEqual(analysis.tasks[0].assigned_by, "Hablante 1")
    
    @patch('src.llm.analyzer.LLMProviderFactory.create_provider')
    def test_fallback_logic(self, mock_create_provider):
        """Test provider fallback logic."""
        # Primary provider fails
        mock_primary = Mock()
        mock_primary.is_available.return_value = True
        mock_primary.generate.side_effect = Exception("Primary failed")
        mock_primary.get_name.return_value = "ollama"
        
        # Fallback provider works
        mock_fallback = Mock()
        mock_fallback.is_available.return_value = True
        mock_fallback.generate.return_value = "Fallback response"
        mock_fallback.get_name.return_value = "gemini"
        
        # Configure factory to return appropriate providers
        def create_provider(provider_type, config):
            if provider_type == 'ollama':
                return mock_primary
            elif provider_type == 'gemini':
                return mock_fallback
            return None
        
        mock_create_provider.side_effect = create_provider
        
        analyzer = LLMAnalyzer(self.mock_config)
        result = analyzer._call_llm_with_fallback("Test prompt")
        
        self.assertEqual(result, "Fallback response")
    
    def test_task_parsing(self):
        """Test task parsing from LLM response."""
        analyzer = LLMAnalyzer(self.mock_config)
        
        llm_response = """
        TAREA: Completar informe
        ASIGNADA_POR: Hablante 1
        RESPONSABLE: Hablante 2
        FECHA: 2024-01-15
        PRIORIDAD: alta
        CONTEXTO: Para la reunión del lunes
        ---
        TAREA: Revisar código
        ASIGNADA_POR: Hablante 2
        RESPONSABLE: no especificado
        PRIORIDAD: media
        ---
        """
        
        tasks = analyzer._parse_tasks_with_speakers(llm_response)
        
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].description, "Completar informe")
        self.assertEqual(tasks[0].assignee, "Hablante 2")
        self.assertEqual(tasks[0].due_date, "2024-01-15")
        self.assertEqual(tasks[1].assignee, None)
    
    def test_todo_parsing(self):
        """Test to-do parsing from LLM response."""
        analyzer = LLMAnalyzer(self.mock_config)
        
        llm_response = """
        TODO: Comprar materiales
        MENCIONADO_POR: Hablante 1
        CATEGORIA: trabajo
        URGENCIA: alta
        ---
        TODO: Actualizar documentación
        MENCIONADO_POR: Hablante 2
        CATEGORIA: proyecto
        URGENCIA: normal
        ---
        """
        
        todos = analyzer._parse_todos_with_speakers(llm_response)
        
        self.assertEqual(len(todos), 2)
        self.assertEqual(todos[0].description, "Comprar materiales")
        self.assertEqual(todos[0].urgency, "alta")
        self.assertEqual(todos[1].category, "proyecto")
    
    def test_daily_summary_generation(self):
        """Test daily summary generation."""
        # Create mock analyses
        analysis1 = Analysis(
            conversation_id="conv-1",
            summary="Summary 1",
            key_points=["Point 1"],
            tasks=[Task(description="Task 1", priority="alta")],
            todos=[],
            followups=[],
            participants=["Hablante 1", "Hablante 2"],
            duration=600.0,
            timestamp=datetime.now(),
            llm_provider="ollama"
        )
        
        analysis2 = Analysis(
            conversation_id="conv-2",
            summary="Summary 2",
            key_points=["Point 2"],
            tasks=[Task(description="Task 2", assignee="Hablante 1")],
            todos=[Todo(description="Todo 1")],
            followups=[],
            participants=["Hablante 1", "Hablante 3"],
            duration=900.0,
            timestamp=datetime.now(),
            llm_provider="ollama"
        )
        
        # Mock provider for highlights
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = "1. Highlight 1\n2. Highlight 2"
        
        with patch.object(LLMAnalyzer, '_create_provider', return_value=mock_provider):
            analyzer = LLMAnalyzer(self.mock_config)
            analyzer.provider = mock_provider
            
            daily_summary = analyzer.generate_daily_summary([analysis1, analysis2])
        
        self.assertEqual(daily_summary.total_conversations, 2)
        self.assertEqual(daily_summary.total_duration, 1500.0)
        self.assertEqual(len(daily_summary.all_tasks), 2)
        self.assertEqual(len(daily_summary.all_todos), 1)
        self.assertIn("Hablante 1", daily_summary.speaker_participation)
    
    def test_conversation_speaker_lines_extraction(self):
        """Test extraction of speaker lines from conversation."""
        conv = Conversation(
            id="test",
            transcript="",
            speaker_transcript="[Hablante 1] Primera línea\nContinuación de línea 1\n[Hablante 2] Segunda línea\n[Hablante 1] Tercera línea",
            speakers=["Hablante 1", "Hablante 2"],
            duration=100.0,
            timestamp=datetime.now()
        )
        
        speaker_lines = conv.get_speaker_lines()
        
        self.assertIn("Hablante 1", speaker_lines)
        self.assertIn("Hablante 2", speaker_lines)
        self.assertEqual(len(speaker_lines["Hablante 1"]), 2)
        self.assertEqual(len(speaker_lines["Hablante 2"]), 1)


class TestPromptFormatting(unittest.TestCase):
    """Test prompt formatting functions."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        from src.llm.prompts import format_duration
        
        self.assertEqual(format_duration(30), "menos de 1")
        self.assertEqual(format_duration(60), "1")
        self.assertEqual(format_duration(150), "2")
        self.assertEqual(format_duration(3600), "60")


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    @patch('src.llm.analyzer.LLMProviderFactory.create_provider')
    def test_all_providers_fail(self, mock_create_provider):
        """Test when all providers fail."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = False
        mock_create_provider.return_value = mock_provider
        
        analyzer = LLMAnalyzer(Mock())
        
        with self.assertRaises(LLMError):
            analyzer._call_llm_with_fallback("Test prompt")
    
    def test_invalid_api_key_handling(self):
        """Test handling of missing API keys."""
        # Test Gemini without API key
        with self.assertRaises(ValueError):
            GeminiProvider({})
        
        # Test OpenAI without API key
        with self.assertRaises(ValueError):
            OpenAIProvider({})


if __name__ == '__main__':
    unittest.main()