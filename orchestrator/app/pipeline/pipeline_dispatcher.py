import logging
from typing import Optional, List, Dict, Any, Union
from PIL import Image

from app.schemas.ai_inference import FullInferencePipelineOutput
from app.pipeline.content_moderator import ContentModerator
from app.pipeline.civic_classifier import CivicImageClassifier
from app.pipeline.profanity_checker import ProfanityChecker
from app.pipeline.nlp_analyzer import NLPAnalyzer
from app.pipeline.geo_lookup import GeoLookupService
from app.pipeline.duplicate_detector import DuplicateDetector
from app.scoring.scoring_engine import SeverityScoringEngine

logger = logging.getLogger(__name__)

class AIInferenceDispatcher:
    """
    Central dispatcher orchestrating all 6 AI inference modules concurrently and synthesizing results.
    """
    def __init__(self, openai_api_key: Optional[str] = None):
        self.moderator = ContentModerator()
        self.civic_classifier = CivicImageClassifier(api_key=openai_api_key)
        self.profanity_checker = ProfanityChecker()
        self.nlp_analyzer = NLPAnalyzer(api_key=openai_api_key)
        self.geo_service = GeoLookupService()
        self.duplicate_detector = DuplicateDetector()
        self.scoring_engine = SeverityScoringEngine()

    def process(
        self,
        citizen_id: str,
        text: str,
        latitude: float,
        longitude: float,
        image_input: Optional[Union[str, bytes, Image.Image]] = None,
        existing_complaints: Optional[List[Dict[str, Any]]] = None,
        media_urls: Optional[List[str]] = None
    ) -> FullInferencePipelineOutput:
        logger.info(f"Dispatching AI Inference Pipeline for Citizen: {citizen_id}")

        # 1. Profanity & Toxicity Check
        profanity_res = self.profanity_checker.check_text(text)
        if not profanity_res.is_clean:
            logger.warning(f"Profanity detected from citizen {citizen_id}: {profanity_res.profanity_found}")
            # Flagged immediately
            nlp_res = self.nlp_analyzer.analyze(text)
            geo_res = self.geo_service.lookup_ward(latitude, longitude)
            scoring_res = self.scoring_engine.calculate_score(nlp_res, None, None, geo_res)
            return FullInferencePipelineOutput(
                success=False,
                status="rejected",
                moderation=self.moderator.moderate_image(image_input) if image_input else None,
                nlp=nlp_res,
                profanity=profanity_res,
                geo=geo_res,
                deduplication=self.duplicate_detector.evaluate_duplicate("", latitude, longitude, "", []),
                scoring=scoring_res,
                structured_complaint=None,
                rejection_reason="ABUSIVE_OR_INAPPROPRIATE_LANGUAGE"
            )

        # 2. Content Moderation (NSFW / Violence)
        moderation_res = None
        civic_res = None
        if image_input:
            moderation_res = self.moderator.moderate_image(image_input)
            if not moderation_res.is_safe:
                logger.warning(f"Media content moderation failed: {moderation_res.flagged_reasons}")
                nlp_res = self.nlp_analyzer.analyze(text)
                geo_res = self.geo_service.lookup_ward(latitude, longitude)
                scoring_res = self.scoring_engine.calculate_score(nlp_res, None, None, geo_res)
                return FullInferencePipelineOutput(
                    success=False,
                    status="flagged",
                    moderation=moderation_res,
                    nlp=nlp_res,
                    profanity=profanity_res,
                    geo=geo_res,
                    deduplication=self.duplicate_detector.evaluate_duplicate("", latitude, longitude, "", []),
                    scoring=scoring_res,
                    structured_complaint=None,
                    rejection_reason="IMAGE_SAFETY_VIOLATION"
                )

            # 3. Civic Classification with User's Trained YOLO Model
            civic_res = self.civic_classifier.classify_image(image_input)

        # 4. Multilingual NLP Analysis (Marathi + English)
        nlp_res = self.nlp_analyzer.analyze(text)

        # 5. Geo Reverse Lookup (GPS -> Ward ID)
        geo_res = self.geo_service.lookup_ward(latitude, longitude)

        # 6. Fraud & Duplicate Detection
        effective_category = civic_res.predicted_class if civic_res else nlp_res.extracted_category
        dup_res = self.duplicate_detector.evaluate_duplicate(
            new_category=effective_category,
            new_lat=latitude,
            new_lon=longitude,
            new_description=text,
            existing_complaints=existing_complaints or []
        )

        # 7. Severity & Priority Scoring
        scoring_res = self.scoring_engine.calculate_score(
            nlp=nlp_res,
            civic=civic_res,
            duplicate=dup_res,
            geo=geo_res
        )

        # 8. Assemble Final Structured Complaint JSON
        structured_complaint = self.scoring_engine.assemble_structured_complaint(
            citizen_id=citizen_id,
            nlp=nlp_res,
            civic=civic_res,
            scoring=scoring_res,
            geo=geo_res,
            duplicate=dup_res,
            media_urls=media_urls
        )

        return FullInferencePipelineOutput(
            success=True,
            status="complete",
            moderation=moderation_res,
            civic_classification=civic_res,
            nlp=nlp_res,
            profanity=profanity_res,
            geo=geo_res,
            deduplication=dup_res,
            scoring=scoring_res,
            structured_complaint=structured_complaint,
            rejection_reason=None
        )
