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

from concurrent.futures import ThreadPoolExecutor

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
        self.executor = ThreadPoolExecutor(max_workers=5)

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

        # Concurrent execution of independent validation and intelligence modules
        fut_prof = self.executor.submit(self.profanity_checker.check_text, text)
        fut_nlp = self.executor.submit(self.nlp_analyzer.analyze, text)
        fut_geo = self.executor.submit(self.geo_service.lookup_ward, latitude, longitude)

        fut_mod = self.executor.submit(self.moderator.moderate_image, image_input) if image_input else None
        fut_civic = self.executor.submit(self.civic_classifier.classify_image, image_input) if image_input else None

        # Collect results in parallel
        profanity_res = fut_prof.result()
        nlp_res = fut_nlp.result()
        geo_res = fut_geo.result()
        moderation_res = fut_mod.result() if fut_mod else None
        civic_res = fut_civic.result() if fut_civic else None

        # 1. Profanity check gate
        if not profanity_res.is_clean:
            logger.warning(f"Profanity detected from citizen {citizen_id}: {profanity_res.profanity_found}")
            scoring_res = self.scoring_engine.calculate_score(nlp_res, civic_res, None, geo_res)
            return FullInferencePipelineOutput(
                success=False,
                status="rejected",
                moderation=moderation_res,
                nlp=nlp_res,
                profanity=profanity_res,
                geo=geo_res,
                deduplication=self.duplicate_detector.evaluate_duplicate("", latitude, longitude, "", []),
                scoring=scoring_res,
                structured_complaint=None,
                rejection_reason="ABUSIVE_OR_INAPPROPRIATE_LANGUAGE"
            )

        # 2. Content Moderation gate
        if moderation_res and not moderation_res.is_safe:
            logger.warning(f"Media content moderation failed: {moderation_res.flagged_reasons}")
            scoring_res = self.scoring_engine.calculate_score(nlp_res, civic_res, None, geo_res)
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

        # 3. Fraud & Duplicate Detection
        effective_category = civic_res.predicted_class if civic_res else nlp_res.extracted_category
        dup_res = self.duplicate_detector.evaluate_duplicate(
            new_category=effective_category,
            new_lat=latitude,
            new_lon=longitude,
            new_description=text,
            existing_complaints=existing_complaints or []
        )

        # 4. Severity & Priority Scoring
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
