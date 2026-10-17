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

class CivicExtractedSlots(BaseModel):
    """
    Pydantic schema for multi-slot extraction from natural conversation turns.
    Extracts category, problem description, landmark/location, urgency, and self-corrections.
    """
    category: Optional[str] = Field(None, description="Civic category: pothole, garbage, drainage, streetlight, water_leak, tree_hazard, etc.")
    description: Optional[str] = Field(None, description="Concise description of the problem mentioned by citizen")
    landmark: Optional[str] = Field(None, description="Road, chowk, area, shop, society, or landmark mentioned")
    urgency: Optional[str] = Field("NORMAL", description="Severity: LOW, NORMAL, HIGH, or CRITICAL (e.g. spark, water in house, open manhole)")
    is_correction: bool = Field(False, description="True if the citizen is correcting or replacing previously stated information")
    correction_target: Optional[str] = Field(None, description="Slot to correct: 'location' | 'description' | 'category'")

