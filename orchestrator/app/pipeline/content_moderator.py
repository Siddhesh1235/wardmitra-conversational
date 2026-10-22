import io
import os
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
from PIL import Image

from app.schemas.ai_inference import ContentModerationResult

logger = logging.getLogger(__name__)

OPENAI_MODERATION_MODEL = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
# Image categories that block a photo (OpenAI's own per-category decision). The diagram asks for
# NSFW / violence; self-harm images are blocked too. Text-only categories (hate, harassment) do not apply to images.
BLOCKED_IMAGE_CATEGORIES = {
    "sexual": "NSFW_CONTENT_DETECTED",
    "sexual/minors": "NSFW_CONTENT_DETECTED",
    "violence": "VIOLENT_CONTENT_DETECTED",
    "violence/graphic": "GRAPHIC_VIOLENCE_DETECTED",
    "self-harm": "SELF_HARM_CONTENT_DETECTED",
    "self-harm/intent": "SELF_HARM_CONTENT_DETECTED",
    "self-harm/instructions": "SELF_HARM_CONTENT_DETECTED",
}
# If the moderation API is unreachable: false = accept the photo (logged), true = reject it
MODERATION_FAIL_CLOSED = os.getenv("MODERATION_FAIL_CLOSED", "false").lower() == "true"


class ContentModerator:
    """
    Image moderation, in order of preference:
    1. OpenAI moderation API (omni-moderation, free) — sexual, violence, graphic violence, self-harm
    2. Local Falconsai/nsfw_image_detection model, if `transformers` is installed
    3. A size-only heuristic, which does NOT detect NSFW content (last resort, logged)
    Also extracts keyframes from videos and scans each frame.
    """
    def __init__(
        self,
        model_name: str = "Falconsai/nsfw_image_detection",
        threshold: float = 0.70,
        api_key: Optional[str] = None,
        openai_client: Optional[Any] = None
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.classifier = None
        self.openai_client = openai_client
        if self.openai_client is None:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=key)
                except Exception as e:
                    logger.warning(f"OpenAI moderation client unavailable ({e}).")
        if self.openai_client is None:
            # Only load the heavy local model when the API path is not available
            self._init_model()

    @property
    def backend(self) -> str:
        """Which moderation engine is active: 'openai' | 'local_model' | 'size_heuristic'."""
        if self.openai_client is not None:
            return "openai"
        return "local_model" if self.classifier else "size_heuristic"

    @staticmethod
    def _to_data_uri(img: Image.Image) -> str:
        """JPEG data URI, resized to at most 1024 px (enough for moderation, fewer bytes sent)."""
        small = img.copy()
        small.thumbnail((1024, 1024))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _moderate_with_openai(self, img: Image.Image) -> Optional[ContentModerationResult]:
        """Returns None when the API call fails, so the caller can fall back."""
        try:
            response = self.openai_client.moderations.create(
                model=OPENAI_MODERATION_MODEL,
                input=[{"type": "image_url", "image_url": {"url": self._to_data_uri(img)}}],
            )
            result = response.results[0]
            categories: Dict[str, bool] = result.categories.model_dump(by_alias=True)
            scores: Dict[str, float] = result.category_scores.model_dump(by_alias=True)
        except Exception as e:
            logger.error(f"OpenAI moderation failed: {e}")
            return None

        reasons = sorted({reason for cat, reason in BLOCKED_IMAGE_CATEGORIES.items() if categories.get(cat)})
        nsfw_score = max(float(scores.get("sexual") or 0.0), float(scores.get("sexual/minors") or 0.0))
        if reasons:
            logger.warning(f"Photo blocked by moderation: {reasons}")
        return ContentModerationResult(
            is_safe=not reasons,
            nsfw_score=round(nsfw_score, 4),
            flagged_reasons=reasons,
            action="reject" if reasons else "accept",
        )

    def _init_model(self):
        try:
            from transformers import pipeline
            logger.info(f"Initializing HF NSFW pipeline for {self.model_name}...")
            self.classifier = pipeline("image-classification", model=self.model_name)
            logger.info("NSFW moderation model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load transformers NSFW model ({e}). Fallback moderation mode enabled.")
            self.classifier = None

    def moderate_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> ContentModerationResult:
        """
        Check image for NSFW / inappropriate content.
        """
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if self.openai_client is not None:
            result = self._moderate_with_openai(img)
            if result is not None:
                return result
            if MODERATION_FAIL_CLOSED:
                return ContentModerationResult(
                    is_safe=False, nsfw_score=0.0, flagged_reasons=["MODERATION_UNAVAILABLE"], action="reject"
                )
            logger.warning("Moderation API unavailable; photo accepted WITHOUT an NSFW check (MODERATION_FAIL_CLOSED=false).")

        if self.classifier:
            try:
                results = self.classifier(img)
                # results format: [{'label': 'nsfw', 'score': 0.02}, {'label': 'normal', 'score': 0.98}]
                nsfw_score = 0.0
                for item in results:
                    if item.get("label", "").lower() == "nsfw":
                        nsfw_score = float(item.get("score", 0.0))

                is_safe = nsfw_score < self.threshold
                action = "accept" if is_safe else "reject"
                flagged_reasons = ["NSFW_CONTENT_DETECTED"] if not is_safe else []

                return ContentModerationResult(
                    is_safe=is_safe,
                    nsfw_score=round(nsfw_score, 4),
                    flagged_reasons=flagged_reasons,
                    action=action
                )
            except Exception as e:
                logger.error(f"Error during transformer moderation: {e}")

        # Last-resort heuristic: only rejects tiny/corrupt images, it cannot detect NSFW content
        w, h = img.size
        is_safe = (w >= 50 and h >= 50)
        return ContentModerationResult(
            is_safe=is_safe,
            nsfw_score=0.05,
            flagged_reasons=[] if is_safe else ["CORRUPTED_OR_EMPTY_IMAGE"],
            action="accept" if is_safe else "reject"
        )

    def extract_keyframes_from_video(self, video_path: Union[str, Path], interval_seconds: int = 2) -> List[Image.Image]:
        """
        Sample keyframes from video (e.g. 1 frame every 2 seconds).
        """
        frames: List[Image.Image] = []
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(fps * interval_seconds)
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame % frame_interval == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))
                current_frame += 1

            cap.release()
        except ImportError:
            logger.warning("cv2 (OpenCV) not installed for video frame extraction.")
        except Exception as e:
            logger.error(f"Video frame sampling error: {e}")

        return frames

    def moderate_video(self, video_path: Union[str, Path]) -> ContentModerationResult:
        """
        Moderate video by extracting frames and verifying each frame.
        """
        frames = self.extract_keyframes_from_video(video_path)
        if not frames:
            # If no frames could be extracted, return safe by default or check file size
            return ContentModerationResult(
                is_safe=True,
                nsfw_score=0.0,
                flagged_reasons=[],
                action="accept"
            )

        highest_nsfw = 0.0
        for idx, frame in enumerate(frames):
            res = self.moderate_image(frame)
            if res.nsfw_score > highest_nsfw:
                highest_nsfw = res.nsfw_score
            if not res.is_safe:
                return ContentModerationResult(
                    is_safe=False,
                    nsfw_score=round(res.nsfw_score, 4),
                    flagged_reasons=[f"NSFW_DETECTED_AT_FRAME_{idx+1}"],
                    action="reject"
                )

        return ContentModerationResult(
            is_safe=True,
            nsfw_score=round(highest_nsfw, 4),
            flagged_reasons=[],
            action="accept"
        )
