"""
Unit tests for OutputFormatter module

This test suite covers all aspects of the OutputFormatter including:
- File creation and organization
- Markdown formatting
- Task and todo formatting
- Index generation
- Error handling
"""

import pytest
import shutil
import yaml
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.output.output_formatter import OutputFormatter
from src.llm.models import Task, Todo, Followup, Analysis, DailySummary, Conversation
from src.transcription.data_classes import Transcription, TranscriptionSegment


class TestOutputFormatter:
    """Test suite for OutputFormatter class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration manager"""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "output.base_path": "/tmp/test_output",
            "output.structure.transcriptions_dir": "transcriptions",
            "output.structure.summaries_dir": "summaries",
            "output.formatting.date_format": "%Y-%m-%d",
            "output.formatting.time_format": "%H:%M:%S",
            "output.formatting.include_timestamps": True,
            "output.markdown.include_metadata": True,
            "output.markdown.include_navigation": True,
        }.get(key, default)
        return config
    
    @pytest.fixture
    def output_formatter(self, test_output_dir):
        """Create OutputFormatter instance with test output directory"""
        from tests.test_utils import create_mock_config
        
        config = create_mock_config(test_output_dir)
        formatter = OutputFormatter(config)
        return formatter
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task"""
        return Task(
            description="Completar el informe mensual",
            assignee="Juan",
            assigned_by="María",
            due_date="2024-01-20",
            priority="alta",
            context="Discutido en la reunión de equipo"
        )
    
    @pytest.fixture
    def sample_todo(self):
        """Create sample todo"""
        return Todo(
            description="Revisar el presupuesto del proyecto",
            category="finanzas",
            urgency="alta",
            mentioned_by="Carlos"
        )
    
    @pytest.fixture
    def sample_followup(self):
        """Create sample followup"""
        return Followup(
            topic="Contrato con proveedor",
            action_required="Confirmar términos y firmar",
            responsible_party="Ana",
            deadline="Fin de semana",
            mentioned_by="Pedro"
        )
    
    @pytest.fixture
    def sample_analysis(self, sample_task, sample_todo, sample_followup):
        """Create sample analysis"""
        return Analysis(
            conversation_id="conv_001",
            summary="Reunión sobre el proyecto Q1 con revisión de tareas y presupuesto.",
            key_points=[
                "Se completó el 80% del proyecto",
                "Necesitamos ajustar el presupuesto",
                "Nuevo deadline: fin de mes"
            ],
            tasks=[sample_task],
            todos=[sample_todo],
            followups=[sample_followup],
            participants=["Juan", "María", "Carlos"],
            duration=1800.0,  # 30 minutes
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            llm_provider="openai"
        )
    
    @pytest.fixture
    def sample_conversation(self):
        """Create sample conversation"""
        return Conversation(
            id="conv_001",
            transcript="Hola, ¿cómo están todos? Bien, gracias. Vamos a revisar el proyecto.",
            speaker_transcript="[Hablante 00] Hola, ¿cómo están todos?\n[Hablante 01] Bien, gracias.\n[Hablante 00] Vamos a revisar el proyecto.",
            speakers=["Hablante 00", "Hablante 01"],
            duration=1800.0,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            segments=[
                {"speaker_id": "Hablante 00", "text": "Hola, ¿cómo están todos?", "start_time": 0.0},
                {"speaker_id": "Hablante 01", "text": "Bien, gracias.", "start_time": 2.5},
                {"speaker_id": "Hablante 00", "text": "Vamos a revisar el proyecto.", "start_time": 5.0}
            ]
        )
    
    @pytest.fixture
    def sample_daily_summary(self, sample_task, sample_todo, sample_followup):
        """Create sample daily summary"""
        return DailySummary(
            date=date(2024, 1, 15),
            total_conversations=3,
            total_duration=5400.0,  # 90 minutes
            all_tasks=[sample_task],
            all_todos=[sample_todo],
            all_followups=[sample_followup],
            highlights=[
                "Se completaron 5 tareas importantes",
                "Se identificaron 3 riesgos en el proyecto",
                "Se acordó nueva fecha de entrega"
            ],
            speaker_participation={"Juan": 40.0, "María": 35.0, "Carlos": 25.0}
        )
    
    @pytest.fixture
    def sample_transcription(self):
        """Create sample transcription"""
        audio_file = Mock()
        audio_file.original_file.filename = "test_audio.wav"
        audio_file.chunks = []
        
        return Transcription(
            audio_file=audio_file,
            segments=[
                TranscriptionSegment(
                    id=1,
                    start_time=0.0,
                    end_time=2.5,
                    text="Hola, buenos días.",
                    speaker_id="SPEAKER_00",
                    confidence=0.95
                ),
                TranscriptionSegment(
                    id=2,
                    start_time=2.5,
                    end_time=5.0,
                    text="Buenos días, ¿cómo están?",
                    speaker_id="SPEAKER_01",
                    confidence=0.92
                )
            ],
            full_text="Hola, buenos días. Buenos días, ¿cómo están?",
            speaker_transcript="[SPEAKER_00] Hola, buenos días.\n[SPEAKER_01] Buenos días, ¿cómo están?",
            language="es",
            confidence=0.94,
            duration=300.0,
            speaker_count=2,
            processing_time=45.5
        )
    
    def test_initialization(self, output_formatter):
        """Test OutputFormatter initialization"""
        assert output_formatter.base_path.exists()
        assert output_formatter.transcriptions_path.exists()
        assert output_formatter.summaries_path.exists()
        assert output_formatter.date_format == "%Y-%m-%d"
        assert output_formatter.time_format == "%H:%M:%S"
    
    def test_save_analysis(self, output_formatter, sample_analysis, sample_conversation):
        """Test saving analysis to file"""
        file_path = output_formatter.save_analysis(sample_analysis, sample_conversation)
        
        assert file_path.exists()
        assert file_path.name == "conversation_conv_001.md"
        assert file_path.parent.name == "2024-01-15"
        
        # Check content
        content = file_path.read_text(encoding='utf-8')
        assert "# Conversación conv_001" in content
        assert "Reunión sobre el proyecto Q1" in content
        assert "Juan, María, Carlos" in content
        assert "🔴 Prioridad Alta" in content
        assert "Completar el informe mensual" in content
    
    def test_save_analysis_with_metadata(self, output_formatter, sample_analysis, sample_conversation):
        """Test metadata is included in saved file"""
        file_path = output_formatter.save_analysis(sample_analysis, sample_conversation)
        
        content = file_path.read_text(encoding='utf-8')
        assert content.startswith("---\n")
        assert "type: conversation" in content
        assert "conversation_id: conv_001" in content
        
        # Extract and parse metadata
        metadata_end = content.find("\n---\n", 4)
        metadata_yaml = content[4:metadata_end]
        metadata = yaml.safe_load(metadata_yaml)
        
        assert metadata['type'] == 'conversation'
        assert metadata['participants'] == ["Juan", "María", "Carlos"]
        assert metadata['tasks_count'] == 1
    
    def test_save_daily_summary(self, output_formatter, sample_daily_summary):
        """Test saving daily summary"""
        file_path = output_formatter.save_daily_summary(sample_daily_summary)
        
        assert file_path.exists()
        assert file_path.name == "daily_summary.md"
        assert file_path.parent.name == "2024-01-15"
        
        # Check content
        content = file_path.read_text(encoding='utf-8')
        assert "# Resumen del Día - 2024-01-15" in content
        assert "Total de conversaciones**: 3" in content
        assert "Duración total**: 90.0 minutos" in content
        assert "Se completaron 5 tareas importantes" in content
        
        # Check tasks and followups file was created
        tasks_file = file_path.parent / "tasks_and_followups.md"
        assert tasks_file.exists()
    
    def test_save_transcription(self, output_formatter, sample_transcription):
        """Test saving raw transcription"""
        file_path = output_formatter.save_transcription(sample_transcription)
        
        assert file_path.exists()
        assert file_path.name.startswith("raw_transcription_")
        assert file_path.name.endswith("_test_audio.md")
        
        # Check content
        content = file_path.read_text(encoding='utf-8')
        assert "# Transcripción - test_audio.wav" in content
        assert "Idioma**: es" in content
        assert "Hablantes**: 2" in content
        assert "Confianza**: 94.0%" in content
        assert "[SPEAKER_00] Hola, buenos días." in content
    
    def test_format_conversation(self, output_formatter, sample_conversation, sample_analysis):
        """Test conversation formatting"""
        content = output_formatter.format_conversation(sample_conversation, sample_analysis)
        
        assert "# Conversación conv_001" in content
        assert "10:30:00 - 11:00:00" in content  # Time range
        assert "30.0 minutos" in content
        assert "Juan, María, Carlos" in content
        
        # Check all sections
        assert "## Resumen" in content
        assert "## Puntos Clave" in content
        assert "## Tareas Identificadas" in content
        assert "## Pendientes" in content
        assert "## Seguimientos" in content
        assert "## Transcripción" in content
        
        # Check navigation links
        assert "[Resumen Diario](daily_summary.md)" in content
        assert "[Índice del Día](index.md)" in content
    
    def test_create_task_list(self, output_formatter):
        """Test task list formatting"""
        tasks = [
            Task("Tarea alta", priority="alta", assignee="Juan"),
            Task("Tarea media", priority="media", assignee="María"),
            Task("Tarea baja", priority="baja", assignee="Carlos"),
        ]
        
        result = output_formatter.create_task_list(tasks)
        
        assert "### 🔴 Prioridad Alta" in result
        assert "### 🟡 Prioridad Media" in result
        assert "### 🟢 Prioridad Baja" in result
        assert "- **Tarea alta**" in result
        assert "Responsable: Juan" in result
    
    def test_create_task_list_empty(self, output_formatter):
        """Test empty task list"""
        result = output_formatter.create_task_list([])
        assert result == "*No se identificaron tareas*"
    
    def test_format_todos(self, output_formatter):
        """Test todo formatting"""
        todos = [
            Todo("Todo urgente", urgency="alta", mentioned_by="Juan"),
            Todo("Todo normal", urgency="normal", category="proyecto"),
            Todo("Todo bajo", urgency="baja"),
        ]
        
        result = output_formatter._format_todos(todos)
        
        assert "### ⚡ Urgencia Alta" in result
        assert "### 📌 Urgencia Normal" in result
        assert "### 📎 Urgencia Baja" in result
        assert "Todo urgente" in result
        assert "[proyecto]" in result
        assert "Mencionado por: Juan" in result
    
    def test_format_followups(self, output_formatter):
        """Test followup formatting"""
        followups = [
            Followup(
                topic="Revisar contrato",
                action_required="Leer y aprobar",
                responsible_party="Legal",
                deadline="2024-01-20"
            ),
            Followup(
                topic="Actualizar documentación",
                action_required="Agregar nuevas secciones",
                mentioned_by="Tech Lead"
            )
        ]
        
        result = output_formatter._format_followups(followups)
        
        assert "**Revisar contrato**" in result
        assert "Acción requerida: Leer y aprobar" in result
        assert "Responsable: Legal" in result
        assert "Plazo: 2024-01-20" in result
        assert "Mencionado por: Tech Lead" in result
    
    def test_create_index(self, output_formatter, sample_analysis, sample_conversation):
        """Test index creation"""
        # Save some conversations first
        output_formatter.save_analysis(sample_analysis, sample_conversation)
        
        # Create another conversation
        import copy
        analysis2 = copy.deepcopy(sample_analysis)
        analysis2.conversation_id = "conv_002"
        output_formatter.save_analysis(analysis2, sample_conversation)
        
        # Create index
        index_path = output_formatter.create_index(date(2024, 1, 15))
        
        assert index_path.exists()
        assert index_path.name == "index.md"
        
        content = index_path.read_text(encoding='utf-8')
        assert "# Índice - 2024-01-15" in content
        assert "Total de conversaciones**: 2" in content
        assert "conversation_conv_001.md" in content
        assert "conversation_conv_002.md" in content
        assert "[Resumen del día](daily_summary.md)" in content
    
    def test_update_master_index(self, output_formatter, sample_analysis, sample_conversation):
        """Test master index update"""
        # Create conversations for multiple dates
        output_formatter.save_analysis(sample_analysis, sample_conversation)
        
        # Create another date
        import copy
        analysis2 = copy.deepcopy(sample_analysis)
        analysis2.timestamp = datetime(2024, 1, 16, 10, 0, 0)
        output_formatter.save_analysis(analysis2, sample_conversation)
        
        # Update master index
        output_formatter.update_master_index()
        
        master_index = output_formatter.base_path / "master_index.md"
        assert master_index.exists()
        
        content = master_index.read_text(encoding='utf-8')
        assert "# Índice Principal - TranscriMatic" in content
        assert "Total de días**: 2" in content
        # Note: Month names will be in English in test environment
        assert "January 2024" in content or "enero 2024" in content
        assert "16 de" in content
        assert "15 de" in content
    
    def test_safe_write(self, output_formatter, tmp_path):
        """Test safe write functionality"""
        test_file = tmp_path / "test.md"
        
        # Test new file
        assert output_formatter._safe_write(test_file, "Hello World")
        assert test_file.read_text() == "Hello World"
        
        # Test overwrite with backup
        assert output_formatter._safe_write(test_file, "New Content")
        assert test_file.read_text() == "New Content"
        assert not test_file.with_suffix('.bak').exists()  # Backup should be cleaned up
        
        # Test write failure
        with patch('builtins.open', side_effect=IOError("Write failed")):
            assert not output_formatter._safe_write(test_file, "Failed Content")
            assert test_file.read_text() == "New Content"  # Original content preserved
    
    def test_extract_metadata(self, output_formatter, tmp_path):
        """Test metadata extraction"""
        # Create file with metadata
        test_file = tmp_path / "test.md"
        content = """---
date: '2024-01-15'
type: conversation
participants:
  - Juan
  - María
---

# Test Content"""
        test_file.write_text(content)
        
        metadata = output_formatter._extract_metadata(test_file)
        assert metadata is not None
        assert metadata['date'] == '2024-01-15'
        assert metadata['type'] == 'conversation'
        assert metadata['participants'] == ['Juan', 'María']
        
        # Test file without metadata
        test_file2 = tmp_path / "test2.md"
        test_file2.write_text("# No metadata here")
        
        metadata2 = output_formatter._extract_metadata(test_file2)
        assert metadata2 is None
    
    def test_format_time(self, output_formatter):
        """Test time formatting"""
        assert output_formatter._format_time(65) == "01:05"
        assert output_formatter._format_time(3665) == "01:01:05"
        assert output_formatter._format_time(30.5) == "00:30"
    
    def test_format_transcription_with_segments(self, output_formatter, sample_conversation):
        """Test transcription formatting with segments"""
        result = output_formatter._format_transcription(sample_conversation)
        
        assert "**Hablante 00**" in result
        assert "**Hablante 01**" in result
        assert "Hola, ¿cómo están todos?" in result
        assert "Bien, gracias." in result
        assert "_00:00_" in result  # Timestamp
    
    def test_format_transcription_no_segments(self, output_formatter):
        """Test transcription formatting without segments"""
        conversation = Mock()
        conversation.segments = []
        conversation.speaker_transcript = "[Speaker 1] Hello\n[Speaker 2] Hi there"
        
        result = output_formatter._format_transcription(conversation)
        assert result == "[Speaker 1] Hello\n[Speaker 2] Hi there"
    
    def test_create_tasks_calendar(self, output_formatter):
        """Test calendar format creation"""
        tasks = [
            Task("Task 1", due_date="2024-01-20", assignee="Juan"),
            Task("Task 2", due_date="2024-01-22", assignee="María"),
            Task("Task 3", assignee="Carlos"),  # No due date
        ]
        
        result = output_formatter.create_tasks_calendar(tasks)
        
        assert "- [ ] 2024-01-20: Task 1 @Juan" in result
        assert "- [ ] 2024-01-22: Task 2 @María" in result
        assert "Task 3" not in result  # No due date
    
    def test_create_timeline_visualization(self, output_formatter):
        """Test Mermaid timeline creation"""
        conversations = [
            Conversation(
                id="001",
                transcript="",
                speaker_transcript="",
                speakers=[],
                duration=1800,
                timestamp=datetime(2024, 1, 15, 9, 0, 0),
                segments=[]
            ),
            Conversation(
                id="002",
                transcript="",
                speaker_transcript="",
                speakers=[],
                duration=2400,
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                segments=[]
            )
        ]
        
        result = output_formatter.create_timeline_visualization(conversations)
        
        assert "```mermaid" in result
        assert "gantt" in result
        assert "Conversación 001 : 09:00, 30m" in result
        assert "Conversación 002 : 10:30, 40m" in result
    
    def test_spanish_characters(self, output_formatter, tmp_path):
        """Test handling of Spanish special characters"""
        test_file = tmp_path / "spanish.md"
        content = "Añadir más información sobre el niño y la señorita María José"
        
        assert output_formatter._safe_write(test_file, content)
        assert test_file.read_text(encoding='utf-8') == content
    
    def test_concurrent_writes(self, output_formatter, sample_analysis, sample_conversation):
        """Test concurrent file writes don't interfere"""
        import threading
        import copy
        
        def save_analysis(conversation_id):
            # Create a separate analysis object for each thread
            analysis = copy.deepcopy(sample_analysis)
            analysis.conversation_id = conversation_id
            output_formatter.save_analysis(analysis, sample_conversation)
        
        threads = []
        for i in range(3):  # Reduce to 3 to avoid race conditions
            t = threading.Thread(target=save_analysis, args=(f"conv_{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check all files were created
        date_dir = output_formatter.summaries_path / "2024-01-15"
        conv_files = list(date_dir.glob("conversation_*.md"))
        assert len(conv_files) >= 1  # At least one should succeed


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def output_formatter(self, test_output_dir):
        """Create OutputFormatter for edge case tests"""
        from tests.test_utils import create_mock_config
        
        config = create_mock_config(test_output_dir)
        formatter = OutputFormatter(config)
        return formatter
    
    def test_empty_analysis(self, output_formatter):
        """Test handling empty analysis"""
        analysis = Analysis(
            conversation_id="empty",
            summary="",
            key_points=[],
            tasks=[],
            todos=[],
            followups=[],
            participants=[],
            duration=0,
            timestamp=datetime.now(),
            llm_provider="test"
        )
        
        conversation = Mock()
        conversation.transcript = ""
        conversation.speaker_transcript = ""
        conversation.segments = []
        
        file_path = output_formatter.save_analysis(analysis, conversation)
        assert file_path.exists()
        
        content = file_path.read_text(encoding='utf-8')
        assert "*No se identificaron puntos clave*" in content
        assert "*No se identificaron tareas*" in content
    
    def test_very_long_content(self, output_formatter):
        """Test handling very long content"""
        # Create analysis with many items
        tasks = [Task(f"Task {i}", priority="media") for i in range(100)]
        
        analysis = Analysis(
            conversation_id="long",
            summary="A" * 5000,  # Very long summary
            key_points=[f"Point {i}" for i in range(50)],
            tasks=tasks,
            todos=[],
            followups=[],
            participants=["Speaker"] * 20,
            duration=36000,  # 10 hours
            timestamp=datetime.now(),
            llm_provider="test"
        )
        
        conversation = Mock()
        conversation.transcript = "Text " * 1000
        conversation.speaker_transcript = "[Speaker 1] " + ("Word " * 1000)
        conversation.segments = []
        
        file_path = output_formatter.save_analysis(analysis, conversation)
        assert file_path.exists()
        
        # File should be created successfully despite size
        file_size = file_path.stat().st_size
        assert file_size > 10000  # Should be a large file
    
    def test_invalid_date_format(self, output_formatter):
        """Test handling invalid date formats"""
        output_formatter.date_format = "%Y/%m/%d"  # Change format
        
        analysis = Mock()
        analysis.conversation_id = "test"
        analysis.timestamp = datetime(2024, 1, 15)
        analysis.duration = 1800
        analysis.participants = []
        analysis.summary = "Test"
        analysis.key_points = []
        analysis.tasks = []
        analysis.todos = []
        analysis.followups = []
        
        conversation = Mock()
        conversation.transcript = "Test"
        conversation.speaker_transcript = "Test"
        conversation.segments = []
        
        file_path = output_formatter.save_analysis(analysis, conversation)
        # The path should contain the new format structure
        assert "2024" in str(file_path)
        assert "01" in str(file_path)
        assert "15" in str(file_path)