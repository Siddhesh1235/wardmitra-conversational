from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SlotItem(BaseModel):
    name: str = Field(..., description="Slot identifier: 'description', 'location', 'media', 'category'")
    value: Optional[Any] = None
    is_filled: bool = False
    prompt_mr: str = Field(..., description="Marathi prompt asking for this slot")
    prompt_en: str = Field(..., description="English prompt asking for this slot")

class SlotCheckResult(BaseModel):
    """
    Result of deterministic slot evaluation for current conversational turn.
    Controls dialogue gating: Incomplete -> ask next question, Complete -> submitForClassification.
    """
    is_complete: bool = Field(..., description="True if all mandatory grievance slots are collected")
    missing_slots: List[str] = Field(default_factory=list, description="Names of unfilled mandatory slots")
    next_slot_to_ask: Optional[str] = Field(None, description="Immediate next slot needed")
    next_question_mr: Optional[str] = Field(None, description="Clarifying question in Marathi")
    next_question_en: Optional[str] = Field(None, description="Clarifying question in English")
    collected_slots: Dict[str, Any] = Field(default_factory=dict, description="Consolidated valid slot values")
