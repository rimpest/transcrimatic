#!/usr/bin/env python3
"""
Demo script for OutputFormatter module

This script demonstrates how to use the OutputFormatter to create
structured Markdown outputs for conversation analyses.
"""

import os
import sys
from datetime import datetime, date
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.output.output_formatter import OutputFormatter
from src.llm.models import Task, Todo, Followup, Analysis, DailySummary, Conversation


def create_sample_config():
    """Create a mock config manager for demo"""
    class MockConfig:
        def __init__(self, base_path):
            self.base_path = base_path
        
        def get(self, key, default=None):
            config_map = {
                "output.base_path": str(self.base_path),
                "output.structure.transcriptions_dir": "transcriptions",
                "output.structure.summaries_dir": "summaries",
                "output.formatting.date_format": "%Y-%m-%d",
                "output.formatting.time_format": "%H:%M:%S",
                "output.formatting.include_timestamps": True,
                "output.markdown.include_metadata": True,
                "output.markdown.include_navigation": True,
            }
            return config_map.get(key, default)
    
    # Create demo output directory
    demo_dir = Path(__file__).parent / "output_demo"
    demo_dir.mkdir(exist_ok=True)
    
    return MockConfig(demo_dir)


def create_sample_data():
    """Create sample conversation and analysis data"""
    
    # Create sample tasks
    tasks = [
        Task(
            description="Revisar el informe mensual de ventas",
            assignee="Ana García",
            assigned_by="Carlos Rodríguez",
            due_date="2024-01-25",
            priority="alta",
            context="Necesario para la presentación del lunes"
        ),
        Task(
            description="Actualizar la documentación del proyecto",
            assignee="Luis Martín",
            priority="media",
            context="Incluir los cambios de la última versión"
        ),
        Task(
            description="Organizar reunión con el equipo de desarrollo",
            assignee="María López",
            assigned_by="Ana García",
            due_date="2024-01-22",
            priority="baja"
        )
    ]
    
    # Create sample todos
    todos = [
        Todo(
            description="Comprobar el estado de los servidores",
            category="infraestructura",
            urgency="alta",
            mentioned_by="Luis Martín"
        ),
        Todo(
            description="Preparar material para la capacitación",
            category="recursos humanos",
            urgency="normal",
            mentioned_by="María López"
        )
    ]
    
    # Create sample followups
    followups = [
        Followup(
            topic="Contrato con nuevo proveedor",
            action_required="Revisar términos y condiciones, enviar para firma",
            responsible_party="Departamento Legal",
            deadline="Fin de semana",
            mentioned_by="Carlos Rodríguez"
        ),
        Followup(
            topic="Implementación del nuevo sistema CRM",
            action_required="Definir cronograma de migración de datos",
            responsible_party="Equipo de TI",
            deadline="Próxima reunión",
            mentioned_by="Ana García"
        )
    ]
    
    # Create sample conversation
    conversation = Conversation(
        id="reunion_001",
        transcript="Buenos días a todos. Empezamos con la revisión del estado del proyecto. Ana, ¿puedes darnos un resumen? Claro, Carlos. Hemos avanzado bastante este mes...",
        speaker_transcript="[Ana García] Buenos días a todos. Empezamos con la revisión del estado del proyecto.\n[Carlos Rodríguez] Ana, ¿puedes darnos un resumen?\n[Ana García] Claro, Carlos. Hemos avanzado bastante este mes...",
        speakers=["Ana García", "Carlos Rodríguez", "Luis Martín", "María López"],
        duration=2700.0,  # 45 minutos
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        segments=[
            {"speaker_id": "Ana García", "text": "Buenos días a todos. Empezamos con la revisión del estado del proyecto.", "start_time": 0.0},
            {"speaker_id": "Carlos Rodríguez", "text": "Ana, ¿puedes darnos un resumen?", "start_time": 5.2},
            {"speaker_id": "Ana García", "text": "Claro, Carlos. Hemos avanzado bastante este mes.", "start_time": 8.1},
            {"speaker_id": "Luis Martín", "text": "Los servidores están funcionando bien, pero necesitamos hacer algunas actualizaciones.", "start_time": 15.4},
            {"speaker_id": "María López", "text": "El equipo está preparado para la próxima fase del proyecto.", "start_time": 22.8}
        ]
    )
    
    # Create sample analysis
    analysis = Analysis(
        conversation_id="reunion_001",
        summary="Reunión de seguimiento del proyecto donde se revisó el progreso mensual, se identificaron tareas pendientes y se planificaron los próximos pasos. El equipo mostró buen avance y se destacó la necesidad de actualizar la infraestructura.",
        key_points=[
            "El proyecto ha avanzado según lo planificado durante este mes",
            "Se requiere actualización de servidores para optimizar el rendimiento",
            "El equipo está preparado para la siguiente fase de implementación",
            "Necesidad de revisar y firmar el contrato con el nuevo proveedor",
            "La documentación del proyecto requiere actualización"
        ],
        tasks=tasks,
        todos=todos,
        followups=followups,
        participants=["Ana García", "Carlos Rodríguez", "Luis Martín", "María López"],
        duration=2700.0,
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        llm_provider="claude-3-sonnet"
    )
    
    return conversation, analysis


def create_sample_daily_summary(analysis):
    """Create a sample daily summary"""
    return DailySummary(
        date=date(2024, 1, 15),
        total_conversations=3,
        total_duration=7200.0,  # 2 horas
        all_tasks=analysis.tasks,
        all_todos=analysis.todos,
        all_followups=analysis.followups,
        highlights=[
            "Se completó la revisión mensual del proyecto con resultados positivos",
            "Se identificaron 3 tareas de alta prioridad para esta semana",
            "El equipo mostró buena colaboración y preparación",
            "Se planificó la implementación del nuevo sistema CRM",
            "Se priorizó la actualización de la infraestructura de servidores"
        ],
        speaker_participation={
            "Ana García": 35.0,
            "Carlos Rodríguez": 30.0, 
            "Luis Martín": 20.0,
            "María López": 15.0
        }
    )


def main():
    """Main demo function"""
    print("🎯 OutputFormatter Demo - TranscriMatic")
    print("=" * 50)
    
    # Setup
    config = create_sample_config()
    formatter = OutputFormatter(config)
    
    print(f"📁 Output directory: {formatter.base_path}")
    print()
    
    # Create sample data
    conversation, analysis = create_sample_data()
    daily_summary = create_sample_daily_summary(analysis)
    
    # Demo 1: Save conversation analysis
    print("1️⃣ Saving conversation analysis...")
    analysis_path = formatter.save_analysis(analysis, conversation)
    print(f"   ✅ Saved to: {analysis_path}")
    print(f"   📄 File size: {analysis_path.stat().st_size} bytes")
    
    # Demo 2: Save daily summary
    print("\n2️⃣ Saving daily summary...")
    summary_path = formatter.save_daily_summary(daily_summary)
    print(f"   ✅ Saved to: {summary_path}")
    print(f"   📄 File size: {summary_path.stat().st_size} bytes")
    
    # Demo 3: Create task calendar
    print("\n3️⃣ Creating task calendar...")
    calendar_content = formatter.create_tasks_calendar(analysis.tasks)
    print("   📅 Calendar format:")
    print("   " + "\n   ".join(calendar_content.split('\n')[:3]))
    
    # Demo 4: Create timeline visualization
    print("\n4️⃣ Creating timeline visualization...")
    timeline = formatter.create_timeline_visualization([conversation])
    print("   📊 Mermaid timeline generated")
    print(f"   📏 Length: {len(timeline)} characters")
    
    # Demo 5: Show directory structure
    print("\n5️⃣ Generated directory structure:")
    def show_tree(path, prefix="", level=0):
        if level > 2:  # Limit depth
            return
        
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"   {prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and level < 2:
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix, level + 1)
    
    show_tree(formatter.base_path)
    
    # Demo 6: Show sample content
    print("\n6️⃣ Sample content preview:")
    print("   📝 Analysis file preview (first 300 chars):")
    content = analysis_path.read_text(encoding='utf-8')
    preview = content[:300] + "..." if len(content) > 300 else content
    print("   " + "\n   ".join(preview.split('\n')[:8]))
    
    print(f"\n🎉 Demo complete! Check {formatter.base_path} for all generated files.")
    print(f"📖 Open {formatter.base_path}/master_index.md to start browsing.")


if __name__ == "__main__":
    main()