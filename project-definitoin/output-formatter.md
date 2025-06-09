# Output Formatter Module

## Purpose
Formats and saves all analysis outputs in organized Markdown files with proper structure, linking, and metadata. Creates a navigable documentation structure for all transcriptions and analyses.

## Dependencies
- **External**: `jinja2`, `markdown`, `pathlib`, `yaml`
- **Internal**: [[ config_manager ]], [[ llm_analyzer ]]

## Interface

### Input
- Analysis results from [[ llm_analyzer ]]
- Output configuration from [[ config_manager ]]

### Output
- Formatted Markdown files in organized directory structure
- Index files for navigation

### Public Methods

```python
class OutputFormatter:
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        
    def save_analysis(self, analysis: Analysis, conversation: Conversation) -> Path:
        """Save single conversation analysis"""
        
    def save_daily_summary(self, daily_summary: DailySummary) -> Path:
        """Save daily summary file"""
        
    def save_transcription(self, transcription: Transcription) -> Path:
        """Save raw transcription"""
        
    def create_index(self, date: date) -> Path:
        """Create index file for a specific date"""
        
    def update_master_index(self):
        """Update master index with all dates"""
        
    def format_conversation(self, conversation: Conversation, analysis: Analysis) -> str:
        """Format conversation with analysis"""
        
    def create_task_list(self, tasks: List[Task]) -> str:
        """Create formatted task list"""
```

## File Structure

```
output_root/
├── transcriptions/
│   └── 2024-01-15/
│       ├── raw_transcription_001.md
│       ├── raw_transcription_002.md
│       └── index.md
├── summaries/
│   └── 2024-01-15/
│       ├── conversation_001.md
│       ├── conversation_002.md
│       ├── daily_summary.md
│       ├── tasks_and_followups.md
│       └── index.md
└── master_index.md
```

## Templates

### Conversation Template
```python
CONVERSATION_TEMPLATE = """# Conversación {conversation_id}

**Fecha**: {date}  
**Hora**: {start_time} - {end_time}  
**Duración**: {duration} minutos  
**Participantes**: {participants}

## Resumen

{summary}

## Puntos Clave

{key_points}

## Tareas Identificadas

{tasks}

## Pendientes

{todos}

## Seguimientos

{followups}

## Transcripción

{transcription}

---

[[ daily_summary ]] | [[ index ]] | [[ master_index ]]
"""
```

### Daily Summary Template
```python
DAILY_SUMMARY_TEMPLATE = """# Resumen del Día - {date}

**Total de conversaciones**: {total_conversations}  
**Duración total**: {total_duration} minutos  
**Participantes únicos**: {unique_participants}

## Puntos Destacados

{highlights}

## Todas las Tareas del Día

{all_tasks}

## Todos los Pendientes

{all_todos}

## Seguimientos Requeridos

{all_followups}

## Conversaciones

{conversation_links}

---

[[ master_index ]]
"""
```

## Formatting Methods

### Task Formatting
```python
def create_task_list(self, tasks: List[Task]) -> str:
    """Create formatted task list with grouping"""
    if not tasks:
        return "*No se identificaron tareas*"
    
    # Group by priority
    high_priority = [t for t in tasks if t.priority == "alta"]
    medium_priority = [t for t in tasks if t.priority == "media"]
    low_priority = [t for t in tasks if t.priority == "baja"]
    
    output = []
    
    if high_priority:
        output.append("### 🔴 Prioridad Alta\n")
        for task in high_priority:
            output.append(self._format_task(task))
    
    if medium_priority:
        output.append("\n### 🟡 Prioridad Media\n")
        for task in medium_priority:
            output.append(self._format_task(task))
    
    if low_priority:
        output.append("\n### 🟢 Prioridad Baja\n")
        for task in low_priority:
            output.append(self._format_task(task))
    
    return "\n".join(output)

def _format_task(self, task: Task) -> str:
    """Format individual task"""
    parts = [f"- **{task.description}**"]
    
    if task.assignee:
        parts.append(f"  - Responsable: {task.assignee}")
    if task.due_date:
        parts.append(f"  - Fecha límite: {task.due_date}")
    if task.context:
        parts.append(f"  - Contexto: {task.context}")
    
    return "\n".join(parts) + "\n"
```

### Transcription Formatting
```python
def _format_transcription(self, conversation: Conversation) -> str:
    """Format transcription with speaker labels and timestamps"""
    output = []
    current_speaker = None
    
    for segment in conversation.segments:
        # Add speaker change header
        if segment.speaker_id != current_speaker:
            current_speaker = segment.speaker_id
            output.append(f"\n**{current_speaker}** _{self._format_time(segment.start_time)}_\n")
        
        # Add segment text
        output.append(segment.text)
    
    return "\n".join(output)

def _format_time(self, seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
```

## File Operations

### Save Analysis
```python
def save_analysis(self, analysis: Analysis, conversation: Conversation) -> Path:
    """Save conversation analysis to file"""
    # Create date directory
    date_dir = self.summaries_path / analysis.timestamp.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Format content
    content = self.format_conversation(conversation, analysis)
    
    # Save file
    filename = f"conversation_{conversation.id}.md"
    file_path = date_dir / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Update indices
    self._update_date_index(date_dir)
    self.update_master_index()
    
    return file_path
```

### Create Index Files
```python
def create_index(self, date: date) -> Path:
    """Create index file for specific date"""
    date_str = date.strftime("%Y-%m-%d")
    date_dir = self.summaries_path / date_str
    
    # Find all conversation files
    conversation_files = sorted(date_dir.glob("conversation_*.md"))
    
    # Create index content
    content = [
        f"# Índice - {date_str}",
        "",
        f"**Total de conversaciones**: {len(conversation_files)}",
        "",
        "## Conversaciones",
        ""
    ]
    
    for file in conversation_files:
        # Extract metadata from file
        metadata = self._extract_metadata(file)
        content.append(
            f"- [{metadata['time']}]([[ {file.stem} ]]) - "
            f"{metadata['participants']} ({metadata['duration']} min)"
        )
    
    content.extend([
        "",
        "## Enlaces Rápidos",
        "",
        "- [[ daily_summary ]] - Resumen del día",
        "- [[ tasks_and_followups ]] - Todas las tareas y seguimientos",
        "- [[ master_index ]] - Índice principal"
    ])
    
    # Save index
    index_path = date_dir / "index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))
    
    return index_path
```

## Metadata Handling

```python
def _add_metadata_header(self, content: str, metadata: Dict[str, Any]) -> str:
    """Add YAML front matter to markdown files"""
    yaml_header = [
        "---",
        f"date: {metadata['date']}",
        f"type: {metadata['type']}",
        f"participants: {metadata['participants']}",
        f"duration: {metadata['duration']}",
        f"tasks_count: {metadata['tasks_count']}",
        "---",
        ""
    ]
    
    return "\n".join(yaml_header) + content

def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
    """Extract metadata from markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML front matter if present
    if content.startswith("---"):
        yaml_end = content.find("---", 3)
        if yaml_end != -1:
            yaml_content = content[3:yaml_end]
            return yaml.safe_load(yaml_content)
    
    # Fall back to parsing content
    return self._parse_content_metadata(content)
```

## Configuration

```yaml
output:
  base_path: "/home/user/audio_analysis"
  
  structure:
    transcriptions_dir: "transcriptions"
    summaries_dir: "summaries"
    
  formatting:
    date_format: "%Y-%m-%d"
    time_format: "%H:%M:%S"
    
  markdown:
    include_metadata: true
    include_navigation: true
    syntax_highlighting: true
    
  indices:
    create_daily: true
    create_master: true
    update_frequency: "immediate"
```

## Special Formatting

### Task Calendar Integration
```python
def _format_tasks_calendar(self, tasks: List[Task]) -> str:
    """Format tasks in calendar-compatible format"""
    calendar_items = []
    
    for task in tasks:
        if task.due_date:
            calendar_items.append(
                f"- [ ] {task.due_date}: {task.description} "
                f"@{task.assignee or 'sin asignar'}"
            )
    
    return "\n".join(sorted(calendar_items))
```

### Conversation Graph
```python
def _create_conversation_graph(self, conversations: List[Conversation]) -> str:
    """Create visual timeline of conversations"""
    graph = ["```mermaid", "gantt", "    title Línea de Tiempo de Conversaciones", 
             "    dateFormat HH:mm", "    axisFormat %H:%M"]
    
    for conv in conversations:
        start = self._format_time_short(conv.start_time)
        duration = int(conv.duration / 60)  # minutes
        graph.append(
            f"    Conversación {conv.id} : {start}, {duration}m"
        )
    
    graph.append("```")
    return "\n".join(graph)
```

## Error Handling

### File System Errors
```python
def _safe_write(self, file_path: Path, content: str) -> bool:
    """Write file with error handling"""
    try:
        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix('.bak')
            shutil.copy2(file_path, backup_path)
        
        # Write new content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Remove backup on success
        if backup_path.exists():
            backup_path.unlink()
        
        return True
        
    except Exception as e:
        self.logger.error(f"Failed to write {file_path}: {e}")
        # Restore backup if exists
        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
        return False
```

## Module Relationships
- Uses: [[ config_manager ]]
- Called by: [[ main_controller ]]
- Receives from: [[ llm_analyzer ]]
- Outputs to: File system

## Output Quality

### Validation
- Ensure all internal links are valid
- Verify markdown syntax
- Check file encoding (UTF-8)
- Validate YAML front matter

### Accessibility
- Clear heading hierarchy
- Descriptive link text
- Proper list formatting
- Mobile-friendly tables

## Testing Considerations

- Test with special characters in Spanish
- Verify file creation permissions
- Test with very long conversations
- Handle concurrent writes
- Validate Obsidian compatibility

## Future Enhancements

- HTML export option
- PDF generation
- Search index creation
- Tag-based organization
- Export to task management tools