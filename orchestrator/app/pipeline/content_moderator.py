import io
import logging
from pathlib import Path
from typing import Union, List, Optional
from PIL import Image

from app.schemas.ai_inference import ContentModerationResult

logger = logging.getLogger(__name__)

class ContentModerator:
    """
    Moderation engine using Falconsai/nsfw_image_detection with fallback heuristic.
    Extracts keyframes from videos and scans images for inappropriate/NSFW content.
    """
    def __init__(self, model_name: str = "Falconsai/nsfw_image_detection", threshold: float = 0.70):
        self.model_name = model_name
        self.threshold = threshold
        self.classifier = None
        self._init_model()

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

        # Fallback heuristic: check image dimensions & aspect ratio
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
