import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Query, Response, HTTPException
from fastapi.responses import PlainTextResponse

from app.channel_adapter.normalizers import (
    WhatsAppNormalizer,
    WebChatNormalizer,
    MobileAppNormalizer
)
from app.core.dialogue_engine import ConversationalOrchestrator
from app.core.dialogue_models import DialogueTurnOutput

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/channels", tags=["Channel Adapter"])

# Meta Webhook verification token (configured in Meta Developer Portal)
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "wardmitra_verify_token_2026")

# Singleton orchestrator reference injected from main app
_orchestrator: Optional[ConversationalOrchestrator] = None

def set_orchestrator(orchestrator: ConversationalOrchestrator):
    global _orchestrator
    _orchestrator = orchestrator

def get_orchestrator() -> ConversationalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConversationalOrchestrator()
    return _orchestrator

@router.get("/whatsapp", response_class=PlainTextResponse)
def verify_whatsapp_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    Meta WhatsApp Cloud API Webhook Verification.
    Used by Meta to verify webhook authenticity during initial registration.
    """
    logger.info(f"Received WhatsApp verification request with mode: {hub_mode}")
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("WhatsApp webhook verified successfully!")
        return hub_challenge or ""
    raise HTTPException(status_code=403, detail="Verification token mismatch")

@router.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    """
    Receives incoming WhatsApp events (Text, GPS location pin, Images, Videos).
    Normalizes payload and routes to Conversational Orchestrator.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    normalized = WhatsAppNormalizer.normalize(body)
    if not normalized:
        # Status/receipt event (e.g. sent, delivered)
        return {"status": "ignored_status_update"}

    orchestrator = get_orchestrator()
    turn_output = orchestrator.process_turn(
        session_id=normalized.session_id,
        citizen_id=normalized.citizen_id,
        text=normalized.text,
        latitude=normalized.latitude,
        longitude=normalized.longitude,
        media_url=normalized.media_url,
        channel="whatsapp"
    )

    return {
        "status": "success",
        "channel": "whatsapp",
        "to": normalized.citizen_id,
        "reply": turn_output.reply_text,
        "dialogue_state": turn_output.status,
        "is_complete": turn_output.is_complete
    }

@router.post("/web", response_model=DialogueTurnOutput)
async def receive_web_chat_message(payload: Dict[str, Any]):
    """Receives Web Chat widget messages."""
    normalized = WebChatNormalizer.normalize(payload)
    orchestrator = get_orchestrator()
    return orchestrator.process_turn(
        session_id=normalized.session_id,
        citizen_id=normalized.citizen_id,
        text=normalized.text,
        latitude=normalized.latitude,
        longitude=normalized.longitude,
        media_url=normalized.media_url,
        channel="web"
    )

@router.post("/mobile", response_model=DialogueTurnOutput)
async def receive_mobile_message(payload: Dict[str, Any]):
    """Receives Mobile App (iOS / Android) logged-in citizen messages."""
    normalized = MobileAppNormalizer.normalize(payload)
    orchestrator = get_orchestrator()
    return orchestrator.process_turn(
        session_id=normalized.session_id,
        citizen_id=normalized.citizen_id,
        text=normalized.text,
        latitude=normalized.latitude,
        longitude=normalized.longitude,
        media_url=normalized.media_url,
        channel="mobile"
    )
