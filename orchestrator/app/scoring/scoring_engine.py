from typing import Optional, Dict, Any, List
from app.schemas.ai_inference import (
    CivicClassificationResult,
    NLPAnalysisResult,
    ProfanityCheckResult,
    GeoLookupResult,
    DuplicateCheckResult,
    ScoringResult,
    StructuredComplaintJSON
)

# SLA mappings in hours
CATEGORY_SLA_HOURS = {
    "Electricity": 12,
    "PipelineDefects": 24,
    "Drainage": 24,
    "PotHoles": 48,
    "Road_Incidents_Traffic": 12,
    "Health_Sanitation": 24,
    "Garbage": 24,
    "StreetLight": 48,
    "Trees": 48,
    "Encroachment": 72,
    "Banners_Flex": 72,
    "Noise_Pollution": 24
}

# Base Priority mapping (1 = highest urgency, 4 = lowest)
CATEGORY_BASE_PRIORITY = {
    "Electricity": 1,
    "PipelineDefects": 2,
    "Road_Incidents_Traffic": 1,
    "Drainage": 2,
    "PotHoles": 2,
    "Health_Sanitation": 2,
    "Garbage": 3,
    "StreetLight": 3,
    "Trees": 3,
    "Noise_Pollution": 3,
    "Encroachment": 4,
    "Banners_Flex": 4
}

class SeverityScoringEngine:
    """
    Aggregates all AI inference signals into actionable priority, severity, and target SLA.
    """
    def calculate_score(
        self,
        nlp: NLPAnalysisResult,
        civic: Optional[CivicClassificationResult] = None,
        duplicate: Optional[DuplicateCheckResult] = None,
        geo: Optional[GeoLookupResult] = None
    ) -> ScoringResult:
        # 1. Determine final category
        final_category = nlp.extracted_category
        confidence_agg = 0.85

        if civic and civic.is_civic_related:
            # If vision model has high confidence (>75%), align category
            if civic.confidence >= 0.75:
                final_category = civic.predicted_class
                confidence_agg = (civic.confidence + 0.90) / 2.0
            elif civic.predicted_class == nlp.extracted_category:
                confidence_agg = 0.95

        # 2. Determine base priority
        priority = CATEGORY_BASE_PRIORITY.get(final_category, 3)

        # Escalate priority if urgency mentioned
        if nlp.urgency_level in ("high", "emergency"):
            priority = max(1, priority - 1)
        elif nlp.urgency_level == "low":
            priority = min(4, priority + 1)

        # 3. Determine severity label
        if priority == 1:
            severity = "critical"
        elif priority == 2:
            severity = "high"
        elif priority == 3:
            severity = "medium"
        else:
            severity = "low"

        sla = CATEGORY_SLA_HOURS.get(final_category, 48)

        return ScoringResult(
            severity=severity,
            priority=priority,
            sla_hours=sla,
            final_category=final_category,
            confidence_aggregate=round(confidence_agg, 2)
        )

    def assemble_structured_complaint(
        self,
        citizen_id: str,
        nlp: NLPAnalysisResult,
        civic: Optional[CivicClassificationResult],
        scoring: ScoringResult,
        geo: GeoLookupResult,
        duplicate: DuplicateCheckResult,
        media_urls: Optional[List[str]] = None
    ) -> StructuredComplaintJSON:
        media = media_urls or []
        title = f"{scoring.final_category} issue at Ward {geo.ward_id or 'General'}"

        validation_summary = {
            "nlp_category": nlp.extracted_category,
            "vision_category": civic.predicted_class if civic else None,
            "vision_confidence": civic.confidence if civic else None,
            "duplicate_checked": True,
            "similarity_score": duplicate.similarity_score,
            "sla_hours": scoring.sla_hours
        }

        return StructuredComplaintJSON(
            citizen_id=citizen_id,
            category=scoring.final_category,
            priority=scoring.priority,
            severity=scoring.severity,
            title=title,
            description=nlp.clean_summary,
            ward_id=geo.ward_id,
            ward_name=geo.ward_name,
            latitude=geo.latitude,
            longitude=geo.longitude,
            media_urls=media,
            ai_validation_summary=validation_summary,
            is_duplicate=duplicate.is_duplicate,
            parent_complaint_id=duplicate.parent_complaint_id
        )
