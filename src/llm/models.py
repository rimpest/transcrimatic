"""
Data models for LLM analysis results.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict


@dataclass
class Task:
    """Represents a task extracted from conversation."""
    description: str
    assignee: Optional[str] = None
    assigned_by: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "media"  # alta, media, baja
    context: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "assignee": self.assignee,
            "assigned_by": self.assigned_by,
            "due_date": self.due_date,
            "priority": self.priority,
            "context": self.context
        }


@dataclass
class Todo:
    """Represents a to-do item extracted from conversation."""
    description: str
    category: str = "general"
    urgency: str = "normal"  # alta, normal, baja
    mentioned_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "category": self.category,
            "urgency": self.urgency,
            "mentioned_by": self.mentioned_by
        }


@dataclass
class Followup:
    """Represents a follow-up item extracted from conversation."""
    topic: str
    action_required: str
    responsible_party: Optional[str] = None
    deadline: Optional[str] = None
    mentioned_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "action_required": self.action_required,
            "responsible_party": self.responsible_party,
            "deadline": self.deadline,
            "mentioned_by": self.mentioned_by
        }


@dataclass
class Analysis:
    """Complete analysis result for a conversation."""
    conversation_id: str
    summary: str
    key_points: List[str]
    tasks: List[Task]
    todos: List[Todo]
    followups: List[Followup]
    participants: List[str]
    duration: float  # in seconds
    timestamp: datetime
    llm_provider: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "key_points": self.key_points,
            "tasks": [t.to_dict() for t in self.tasks],
            "todos": [t.to_dict() for t in self.todos],
            "followups": [f.to_dict() for f in self.followups],
            "participants": self.participants,
            "duration": self.duration,
            "timestamp": self.timestamp.isoformat(),
            "llm_provider": self.llm_provider
        }
    
    def get_all_action_items(self) -> List[Dict]:
        """Get all actionable items (tasks, todos, followups) in one list."""
        items = []
        
        for task in self.tasks:
            item = task.to_dict()
            item['type'] = 'task'
            items.append(item)
        
        for todo in self.todos:
            item = todo.to_dict()
            item['type'] = 'todo'
            items.append(item)
        
        for followup in self.followups:
            item = followup.to_dict()
            item['type'] = 'followup'
            items.append(item)
        
        return items


@dataclass
class DailySummary:
    """Summary of all conversations for a day."""
    date: date
    total_conversations: int
    total_duration: float  # in seconds
    all_tasks: List[Task]
    all_todos: List[Todo]
    all_followups: List[Followup]
    highlights: List[str]
    speaker_participation: Dict[str, float]  # Speaker -> percentage
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date.isoformat(),
            "total_conversations": self.total_conversations,
            "total_duration": self.total_duration,
            "total_duration_minutes": round(self.total_duration / 60, 2),
            "all_tasks": [t.to_dict() for t in self.all_tasks],
            "all_todos": [t.to_dict() for t in self.all_todos],
            "all_followups": [f.to_dict() for f in self.all_followups],
            "highlights": self.highlights,
            "speaker_participation": self.speaker_participation,
            "summary_stats": {
                "total_tasks": len(self.all_tasks),
                "total_todos": len(self.all_todos),
                "total_followups": len(self.all_followups),
                "total_action_items": len(self.all_tasks) + len(self.all_todos) + len(self.all_followups)
            }
        }
    
    def get_tasks_by_assignee(self) -> Dict[str, List[Task]]:
        """Group tasks by assignee."""
        by_assignee = {}
        for task in self.all_tasks:
            assignee = task.assignee or "Sin asignar"
            if assignee not in by_assignee:
                by_assignee[assignee] = []
            by_assignee[assignee].append(task)
        return by_assignee
    
    def get_high_priority_items(self) -> Dict[str, List]:
        """Get all high priority items."""
        return {
            "tasks": [t for t in self.all_tasks if t.priority == "alta"],
            "todos": [t for t in self.all_todos if t.urgency == "alta"]
        }


@dataclass
class Conversation:
    """Input conversation data structure."""
    id: str
    transcript: str
    speaker_transcript: str  # Transcript with [Hablante X] labels
    speakers: List[str]
    duration: float  # in seconds
    timestamp: datetime
    segments: List[Dict] = field(default_factory=list)  # Optional segment data
    
    def get_speaker_lines(self) -> Dict[str, List[str]]:
        """Extract lines by speaker from speaker_transcript."""
        speaker_lines = {}
        current_speaker = None
        current_text = []
        
        for line in self.speaker_transcript.split('\n'):
            if line.strip().startswith('[Hablante'):
                # Save previous speaker's text
                if current_speaker and current_text:
                    if current_speaker not in speaker_lines:
                        speaker_lines[current_speaker] = []
                    speaker_lines[current_speaker].append(' '.join(current_text))
                
                # Extract new speaker
                try:
                    speaker_end = line.index(']')
                    current_speaker = line[1:speaker_end]
                    current_text = [line[speaker_end + 1:].strip()]
                except ValueError:
                    continue
            elif current_speaker and line.strip():
                current_text.append(line.strip())
        
        # Save last speaker's text
        if current_speaker and current_text:
            if current_speaker not in speaker_lines:
                speaker_lines[current_speaker] = []
            speaker_lines[current_speaker].append(' '.join(current_text))
        
        return speaker_lines