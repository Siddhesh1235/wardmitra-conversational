import logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.ai_inference import (
    FullInferencePipelineOutput,
    ContentModerationResult,
    CivicClassificationResult,
    NLPAnalysisResult
)
from app.pipeline.pipeline_dispatcher import AIInferenceDispatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wardmitra-ai-orchestrator")

app = FastAPI(
    title="WardMitra AI Inference & Orchestrator API",
    description="Municipal AI Inference Pipeline: NSFW Moderation, Civic YOLO11s Classification, Multilingual NLP, Geo Lookup & Deduplication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize central AI dispatcher
dispatcher = AIInferenceDispatcher()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "wardmitra-conversational-orchestrator",
        "vision_model": dispatcher.civic_classifier.model,
        "vision_backend": "OpenAI Multimodal Vision (GPT-4o-mini)"
    }

@app.post("/api/inference/process", response_model=FullInferencePipelineOutput)
async def process_complaint_inference(
    citizen_id: str = Form(...),
    text: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Main AI Inference Pipeline Endpoint.
    Executes all 6 validation modules concurrently and outputs structured complaint JSON.
    """
    try:
        image_bytes = await image.read() if image else None
        result = dispatcher.process(
            citizen_id=citizen_id,
            text=text,
            latitude=latitude,
            longitude=longitude,
            image_input=image_bytes
        )
        return result
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/classify-civic", response_model=CivicClassificationResult)
async def classify_civic_image(file: UploadFile = File(...)):
    """Classify image directly using user's fine-tuned YOLO11s civic model."""
    try:
        content = await file.read()
        return dispatcher.civic_classifier.classify_image(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/moderate-media", response_model=ContentModerationResult)
async def moderate_media_file(file: UploadFile = File(...)):
    """Check image for NSFW/violence violations."""
    try:
        content = await file.read()
        return dispatcher.moderator.moderate_image(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/nlp", response_model=NLPAnalysisResult)
def analyze_complaint_text(text: str = Form(...)):
    """Analyze Marathi/English complaint text for intent, urgency, and category."""
    try:
        return dispatcher.nlp_analyzer.analyze(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
