import os
import io
import json
import base64
import logging
from pathlib import Path
from typing import Union, List, Optional
from PIL import Image
from dotenv import load_dotenv

from app.schemas.ai_inference import CivicClassificationResult, CivicPredictionItem

load_dotenv()
logger = logging.getLogger(__name__)

CIVIC_CLASSES = [
    "Banners_Flex",
    "Drainage",
    "Electricity",
    "Encroachment",
    "Garbage",
    "Health_Sanitation",
    "Noise_Pollution",
    "PipelineDefects",
    "PotHoles",
    "Road_Incidents_Traffic",
    "StreetLight",
    "Trees",
    "Non_Civic"
]

class CivicImageClassifier:
    """
    Multimodal Civic Grievance Classifier using OpenAI GPT-4o-mini Vision API.
    Accurately identifies municipal issues (PotHoles, Garbage, StreetLight, etc.)
    and detects non-civic media (Notebooks, selfies, documents, indoor objects).
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Vision Client initialized for civic image classification.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

    def _image_to_base64(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """Convert input image into JPEG base64 string."""
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Resize large images to reduce latency & token usage (max 1024x1024)
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def classify_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> CivicClassificationResult:
        """
        Classify civic image using OpenAI Vision.
        """
        if not self.client:
            logger.warning("OpenAI client not available. Returning generic civic result.")
            return CivicClassificationResult(
                predicted_class="Non_Civic",
                confidence=0.5,
                top_predictions=[CivicPredictionItem(class_name="Non_Civic", confidence=0.5)],
                is_civic_related=False
            )

        try:
            b64_image = self._image_to_base64(image_input)
            system_prompt = f"""
You are an expert municipal AI classifier for Maharashtra Municipal Corporations (PCMC / KDMC / PMC).
Analyze the provided image and classify it into one of these categories:
{CIVIC_CLASSES}

CRITICAL RULES:
1. If the image is NOT an outdoor public municipal grievance (e.g. handwritten notebook, printed document, book, selfie, personal room, indoor furniture, random object, animal), classify strictly as "Non_Civic" and set "is_civic_related" to false.
2. If it is a real municipal issue (pothole on road, overflowing garbage, broken streetlight, open drainage, water leak, dangling electric wire, fallen tree), choose the exact matching civic category and set "is_civic_related" to true.

Output strictly a JSON object:
{{
  "predicted_class": "<Category from list>",
  "confidence": <float between 0.0 and 1.0>,
  "top_predictions": [
    {{"class_name": "<Category 1>", "confidence": <float>}},
    {{"class_name": "<Category 2>", "confidence": <float>}}
  ],
  "is_civic_related": <boolean>,
  "visual_description": "<1-sentence visual description of what is seen>"
}}
"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Classify this image for municipal grievance registration:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            data = json.loads(response.choices[0].message.content)
            pred_class = data.get("predicted_class", "Non_Civic")
            conf = float(data.get("confidence", 0.90))
            is_civic = bool(data.get("is_civic_related", pred_class != "Non_Civic"))

            raw_tops = data.get("top_predictions", [])
            top_items: List[CivicPredictionItem] = []
            if raw_tops:
                for item in raw_tops:
                    top_items.append(CivicPredictionItem(
                        class_name=item.get("class_name", pred_class),
                        confidence=float(item.get("confidence", conf))
                    ))
            else:
                top_items = [CivicPredictionItem(class_name=pred_class, confidence=conf)]

            return CivicClassificationResult(
                predicted_class=pred_class,
                confidence=round(conf, 4),
                top_predictions=top_items,
                is_civic_related=is_civic
            )

        except Exception as e:
            logger.error(f"Error during OpenAI Vision classification: {e}")
            return CivicClassificationResult(
                predicted_class="Non_Civic",
                confidence=0.5,
                top_predictions=[CivicPredictionItem(class_name="Non_Civic", confidence=0.5)],
                is_civic_related=False
            )

    def classify_video_frames(self, frames: List[Image.Image]) -> CivicClassificationResult:
        """
        Aggregate predictions across extracted video frames.
        Takes middle frame and highest confidence frame.
        """
        if not frames:
            raise ValueError("No video frames provided for classification.")

        # Classify sample frame (middle frame)
        mid_idx = len(frames) // 2
        return self.classify_image(frames[mid_idx])
