from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Single message in a conversational session."""
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., description="Message text")
    media_url: Optional[str] = Field(None, description="Optional image/audio/video reference")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ConversationSession(BaseModel):
    """
    State store model for multi-turn citizen dialogue.
    Maintains message history, extracted slots, and progress flags.
    Matches 24h TTL Conversation State Store in WardMitra architecture.
    """
    session_id: str = Field(..., description="Unique conversation session identifier")
    citizen_id: Optional[str] = Field(None, description="Identified citizen phone or user ID")
    channel: str = Field("web", description="'whatsapp' | 'web' | 'mobile'")
    history: List[ChatMessage] = Field(default_factory=list, description="Chronological chat history")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Collected grievance slots (description, location, media, etc.)")
    status: str = Field("collecting_info", description="'collecting_info' | 'ready_for_inference' | 'completed' | 'cancelled'")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
