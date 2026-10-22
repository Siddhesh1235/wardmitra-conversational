import logging
from dotenv import load_dotenv

load_dotenv()

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.web_settings import configure_cors, register_test_ui

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

# Only the browser origins listed in CORS_ALLOWED_ORIGINS may call the API (see app/web_settings.py)
configure_cors(app)

# Initialize central AI dispatcher
dispatcher = AIInferenceDispatcher()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "wardmitra-conversational-orchestrator",
        "vision_model": dispatcher.civic_classifier.model,
        "vision_backend": "OpenAI Multimodal Vision (GPT-4o-mini)",
        # 'openai' expected; 'size_heuristic' means photos are NOT checked for NSFW content
        "moderation_backend": dispatcher.moderator.backend
    }
                    
from app.pipeline.geo_lookup import parse_numeric_location

@app.post("/api/inference/process", response_model=FullInferencePipelineOutput)
async def process_complaint_inference(
    citizen_id: str = Form(...),
    text: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    location: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Main AI Inference Pipeline Endpoint.
    Executes all 6 validation modules concurrently and outputs structured complaint JSON.
    Accepts coordinates via [latitude, longitude] in `location` or separate `latitude`/`longitude`.
    """
    try:
        # Parse numeric location from [lat, lon] or separate fields
        coords = None
        if location:
            coords = parse_numeric_location(location)
        if not coords and latitude is not None and longitude is not None:
            coords = parse_numeric_location([latitude, longitude])

        if not coords:
            # Fallback to central default if not provided
            lat_val, lon_val = 19.2435, 73.1310
        else:
            lat_val, lon_val = coords

        image_bytes = await image.read() if image else None
        # Run blocking OpenAI / Nominatim / pgvector calls in a worker thread, off the event loop
        result = await run_in_threadpool(
            dispatcher.process,
            citizen_id=citizen_id,
            text=text,
            latitude=lat_val,
            longitude=lon_val,
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
        return await run_in_threadpool(dispatcher.civic_classifier.classify_image, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/moderate-media", response_model=ContentModerationResult)
async def moderate_media_file(file: UploadFile = File(...)):
    """Check image for NSFW/violence violations."""
    try:
        content = await file.read()
        return await run_in_threadpool(dispatcher.moderator.moderate_image, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/nlp", response_model=NLPAnalysisResult)
def analyze_complaint_text(text: str = Form(...)):
    """Analyze Marathi/English complaint text for intent, urgency, and category."""
    try:
        return dispatcher.nlp_analyzer.analyze(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from app.core import ConversationalOrchestrator, DialogueTurnOutput

conversational_orchestrator = ConversationalOrchestrator(inference_dispatcher=dispatcher)

@app.post("/api/conversation/chat", response_model=DialogueTurnOutput)
async def chat_with_wardmitra(
    request: Request,
    session_id: Optional[str] = Form(None),
    citizen_id: Optional[str] = Form("CITIZEN_ANON"),
    text: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    location: Optional[str] = Form(None),
    channel: Optional[str] = Form("web"),
    image: Optional[UploadFile] = File(None)
):
    """
    Multi-turn conversational chatbot endpoint.
    Accepts BOTH multipart/form-data (with optional images) AND application/json.
    Supports location as [latitude, longitude] array/string or separate latitude/longitude.
    Conducts interactive dialogue in Marathi/English, collects missing slots,
    and automatically executes the AI inference pipeline upon completion.
    """
    content_type = request.headers.get("content-type", "")
    image_bytes = None
    # The logged-in citizen's CouncillorApp JWT; complaints are filed and read as this citizen
    auth_header = request.headers.get("authorization", "")
    auth_token = auth_header.split(" ", 1)[1].strip() if auth_header.lower().startswith("bearer ") else None
    loc_val = location

    if "application/json" in content_type:
        try:
            body = await request.json()
            session_id = body.get("session_id")
            citizen_id = body.get("citizen_id", "CITIZEN_ANON")
            text = body.get("text")
            latitude = body.get("latitude")
            longitude = body.get("longitude")
            loc_val = body.get("location") or location
            channel = body.get("channel", "web")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    else:
        if image:
            image_bytes = await image.read()

    # Parse numeric coordinates from location if provided
    parsed_coords = parse_numeric_location(loc_val) if loc_val else None
    if parsed_coords:
        latitude, longitude = parsed_coords

    if not session_id or not text:
        raise HTTPException(
            status_code=422,
            detail="Missing required fields: both 'session_id' and 'text' must be provided."
        )

    try:
        # Run the blocking dialogue turn in a worker thread, off the event loop
        return await run_in_threadpool(
            conversational_orchestrator.process_turn,
            session_id=session_id,
            citizen_id=citizen_id,
            text=text,
            latitude=latitude,
            longitude=longitude,
            image_bytes=image_bytes,
            channel=channel,
            location=loc_val,
            auth_token=auth_token
        )
    except Exception as e:
        logger.exception(f"Conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from app.schemas.ai_inference import GeoLookupResult

@app.get("/api/geo/ward-lookup", response_model=GeoLookupResult)
def lookup_ward_by_coords(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location: Optional[str] = None
):
    """
    Reverse geocoding & municipal ward lookup:
    Accepts GPS coordinates as [latitude, longitude] or separate lat/lon.
    Uses OpenStreetMap Nominatim for street address and Haversine formula
    to ward centroid for official municipal ward determination.
    """
    coords = None
    if location:
        coords = parse_numeric_location(location)
    if not coords and latitude is not None and longitude is not None:
        coords = parse_numeric_location([latitude, longitude])

    if not coords:
        raise HTTPException(
            status_code=400,
            detail="Please provide valid GPS coordinates in [latitude, longitude] format or as latitude and longitude query parameters."
        )

    lat, lon = coords
    return dispatcher.geo_service.lookup_ward(lat, lon, fetch_address=True)

@app.get("/api/geo/address-lookup", response_model=GeoLookupResult)
def lookup_ward_by_address(query: str):
    """
    Forward geocoding & municipal ward lookup:
    Finds GPS coordinates [latitude, longitude] from address/landmark text using
    OpenStreetMap Nominatim and maps to the official ward via Centroid Haversine distance.
    """
    res = dispatcher.geo_service.lookup_by_address_or_landmark(query)
    if not res:
        raise HTTPException(status_code=404, detail=f"Location or landmark '{query}' could not be resolved.")
    return res
from pathlib import Path

# Testing studio at /ui only when ENABLE_TEST_UI=true (off in production)
STATIC_DIR = Path(__file__).resolve().parent / "static"
register_test_ui(app, STATIC_DIR)



