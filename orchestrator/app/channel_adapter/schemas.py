from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class NormalizedMessage(BaseModel):
    """
    Standardized message schema across all communication channels.
    Matches 'Normalize text / image / video / audio into one message schema' in architecture.
    """
    channel: str = Field(..., description="'whatsapp' | 'web' | 'mobile'")
    session_id: str = Field(..., description="Unique conversation session identifier")
    citizen_id: str = Field("CITIZEN_ANON", description="Citizen identity or phone number")
    text: str = Field("", description="Extracted text content")
    media_type: Optional[str] = Field(None, description="'image' | 'video' | 'audio' | None")
    media_url: Optional[str] = Field(None, description="Media URL or reference ID")
    media_bytes: Optional[bytes] = Field(None, description="Raw media binary data if available")
    latitude: Optional[float] = Field(None, description="GPS latitude if shared")
    longitude: Optional[float] = Field(None, description="GPS longitude if shared")
    raw_payload: Dict[str, Any] = Field(default_factory=dict, description="Original channel webhook payload")

class WebhookVerificationResponse(BaseModel):
    hub_challenge: str
