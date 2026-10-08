from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.schemas.ai_inference import FullInferencePipelineOutput, StructuredComplaintJSON

class DialogueTurnOutput(BaseModel):
    """
    Standard output of a single conversational dialogue turn.
    Sent back to citizen through channel adapter (WhatsApp / Web / Mobile).
    """
    reply_text: str = Field(..., description="Conversational text message to display to citizen")
    session_id: str = Field(..., description="Active session ID")
    status: str = Field("in_progress", description="'in_progress' | 'completed' | 'rejected'")
    is_complete: bool = Field(False, description="Whether all information is collected and verified")
    missing_slots: List[str] = Field(default_factory=list, description="Slots still needed if incomplete")
    collected_slots: Dict[str, Any] = Field(default_factory=dict, description="Current slot values")
    structured_complaint: Optional[StructuredComplaintJSON] = None
    ai_inference_result: Optional[FullInferencePipelineOutput] = None
