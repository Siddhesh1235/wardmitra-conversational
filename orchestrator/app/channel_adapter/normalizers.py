import logging
from typing import Dict, Any, Optional
from app.channel_adapter.schemas import NormalizedMessage

logger = logging.getLogger(__name__)

class WhatsAppNormalizer:
    """
    Normalizes Meta WhatsApp Cloud API / Webhook payloads into standard NormalizedMessage.
    Handles text, live GPS location pins, interactive button clicks, and media attachments.
    """
    @staticmethod
    def normalize(payload: Dict[str, Any]) -> Optional[NormalizedMessage]:
        try:
            entry = payload.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            messages = value.get("messages", [])

            if not messages:
                # Might be a status update event (read receipt, delivered)
                return None

            msg = messages[0]
            from_phone = str(msg.get("from", "UNKNOWN"))
            msg_type = msg.get("type", "text")
            session_id = f"wa_{from_phone}"
            citizen_id = f"+{from_phone}"

            extracted_text = ""
            lat = None
            lon = None
            media_type = None
            media_url = None

            if msg_type == "text":
                extracted_text = msg.get("text", {}).get("body", "")
            elif msg_type == "location":
                loc = msg.get("location", {})
                lat = float(loc.get("latitude", 0.0))
                lon = float(loc.get("longitude", 0.0))
                name = loc.get("name", "")
                address = loc.get("address", "")
                extracted_text = f"Location shared: {name} {address}".strip()
            elif msg_type == "image":
                media_type = "image"
                image_info = msg.get("image", {})
                media_url = image_info.get("id") or image_info.get("link")
                extracted_text = image_info.get("caption", "")
            elif msg_type == "video":
                media_type = "video"
                video_info = msg.get("video", {})
                media_url = video_info.get("id") or video_info.get("link")
                extracted_text = video_info.get("caption", "")
            elif msg_type == "interactive":
                inter = msg.get("interactive", {})
                extracted_text = (
                    inter.get("button_reply", {}).get("title") or
                    inter.get("list_reply", {}).get("title") or ""
                )

            return NormalizedMessage(
                channel="whatsapp",
                session_id=session_id,
                citizen_id=citizen_id,
                text=extracted_text,
                media_type=media_type,
                media_url=media_url,
                latitude=lat,
                longitude=lon,
                raw_payload=payload
            )
        except Exception as e:
            logger.error(f"Error normalizing WhatsApp payload: {e}")
            return None

class WebChatNormalizer:
    """Normalizes Web widget JSON payloads into standard NormalizedMessage."""
    @staticmethod
    def normalize(payload: Dict[str, Any]) -> NormalizedMessage:
        session_id = str(payload.get("session_id") or "web_guest")
        citizen_id = str(payload.get("citizen_id") or "CITIZEN_WEB")
        text = str(payload.get("text") or payload.get("message") or "")
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        media_url = payload.get("media_url") or payload.get("image_url")
        media_type = "image" if media_url else None

        return NormalizedMessage(
            channel="web",
            session_id=session_id,
            citizen_id=citizen_id,
            text=text,
            media_type=media_type,
            media_url=media_url,
            latitude=float(lat) if lat is not None else None,
            longitude=float(lon) if lon is not None else None,
            raw_payload=payload
        )

class MobileAppNormalizer:
    """Normalizes Mobile App (Android/iOS) API payloads with logged-in citizen credentials."""
    @staticmethod
    def normalize(payload: Dict[str, Any]) -> NormalizedMessage:
        citizen_id = str(payload.get("citizen_id") or payload.get("user_id") or "CITIZEN_MOBILE")
        session_id = str(payload.get("session_id") or f"mobile_{citizen_id}")
        text = str(payload.get("text") or payload.get("description") or "")
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        media_url = payload.get("media_url")
        media_type = payload.get("media_type") or ("image" if media_url else None)

        return NormalizedMessage(
            channel="mobile",
            session_id=session_id,
            citizen_id=citizen_id,
            text=text,
            media_type=media_type,
            media_url=media_url,
            latitude=float(lat) if lat is not None else None,
            longitude=float(lon) if lon is not None else None,
            raw_payload=payload
        )
