from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ContentModerationResult(BaseModel):
    is_safe: bool = Field(..., description="Whether the media passed NSFW/violence checks")
    nsfw_score: float = Field(0.0, description="NSFW probability score between 0.0 and 1.0")
    flagged_reasons: List[str] = Field(default_factory=list, description="Labels triggering safety violation")
    action: str = Field("accept", description="'accept' | 'flagged' | 'reject'")

class CivicPredictionItem(BaseModel):
    class_name: str
    confidence: float

class CivicClassificationResult(BaseModel):
    predicted_class: str = Field(..., description="Top-1 predicted civic category")
    confidence: float = Field(..., description="Top-1 confidence score (0.0 to 1.0)")
    top_predictions: List[CivicPredictionItem] = Field(default_factory=list)
    is_civic_related: bool = Field(True, description="False if classified as non-civic / unrelated")

class NLPAnalysisResult(BaseModel):
    intent: str = Field("register_complaint", description="register_complaint | status_check | inquiry | casual")
    detected_language: str = Field("mr", description="Language code: mr (Marathi), en (English), hi (Hindi), mr-latn (Marathlish)")
    extracted_category: str = Field(..., description="Civic category identified from text")
    urgency_level: str = Field("medium", description="low | medium | high | emergency")
    keywords: List[str] = Field(default_factory=list)
    location_mentions: List[str] = Field(default_factory=list)
    clean_summary: str = Field(..., description="Synthesized formal grievance description")

class ProfanityCheckResult(BaseModel):
    is_clean: bool = Field(True, description="True if no abusive words or hate speech detected")
    profanity_found: List[str] = Field(default_factory=list)
    toxicity_score: float = Field(0.0, description="0.0 to 1.0 toxicity metric")
    action: str = Field("pass", description="'pass' | 'warning' | 'block'")

class GeoLookupResult(BaseModel):
    latitude: float
    longitude: float
    ward_id: Optional[str] = None
    ward_name: Optional[str] = None
    is_within_boundary: bool = True
    councillor_id: Optional[str] = None

class DuplicateCheckResult(BaseModel):
    is_duplicate: bool = False
    similarity_score: float = Field(0.0, description="0 to 100 percentage score")
    parent_complaint_id: Optional[str] = None
    matched_complaints_count: int = 0
    distance_meters: Optional[float] = None

class ScoringResult(BaseModel):
    severity: str = Field("medium", description="low | medium | high | critical")
    priority: int = Field(2, description="1 (Highest) to 4 (Lowest)")
    sla_hours: int = Field(48, description="Target resolution hours")
    final_category: str
    confidence_aggregate: float

class StructuredComplaintJSON(BaseModel):
    citizen_id: str
    category: str
    priority: int
    severity: str
    title: str
    description: str
    ward_id: Optional[str]
    ward_name: Optional[str]
    latitude: float
    longitude: float
    media_urls: List[str] = Field(default_factory=list)
    ai_validation_summary: Dict[str, Any] = Field(default_factory=dict)
    is_duplicate: bool = False
    parent_complaint_id: Optional[str] = None

class FullInferencePipelineOutput(BaseModel):
    success: bool
    status: str = Field("complete", description="'complete' | 'flagged' | 'rejected'")
    moderation: Optional[ContentModerationResult] = None
    civic_classification: Optional[CivicClassificationResult] = None
    nlp: NLPAnalysisResult
    profanity: ProfanityCheckResult
    geo: GeoLookupResult
    deduplication: DuplicateCheckResult
    scoring: ScoringResult
    structured_complaint: Optional[StructuredComplaintJSON] = None
    rejection_reason: Optional[str] = None
