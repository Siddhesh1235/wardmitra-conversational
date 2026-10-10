import os
import uuid
import logging
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv

load_dotenv()

from app.schemas.ai_inference import StructuredComplaintJSON

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "https://wardmitra-api.spwhin.com/api").rstrip("/")

class WardMitraBackendClient:
    """
    HTTP Client & Tool Library integrating the AI Orchestrator with the existing Node.js production backend.
    Corresponds to 'Existing Node.js API — tool library for the orchestrator' in the architecture diagram:
    - registerComplaint (auto-assigns available worker via Node.js backend)
    - getComplaints (status follow-up)
    - listCategories, listPriorities
    """
    def __init__(self, base_url: Optional[str] = None, auth_token: Optional[str] = None, timeout_seconds: float = 6.0):
        self.base_url = (base_url or DEFAULT_BACKEND_URL).rstrip("/")
        self.auth_token = auth_token or os.getenv("BACKEND_AUTH_TOKEN")
        self.timeout = timeout_seconds

    def _prepare_complaint_payload(
        self,
        complaint: StructuredComplaintJSON,
        idempotency_key: Optional[str] = None
    ) -> tuple[str, dict, dict]:
        key = idempotency_key or f"{complaint.citizen_id}:{uuid.uuid4().hex[:12]}"
        url = f"{self.base_url}/complaints"
        payload = {
            "citizen_id": complaint.citizen_id,
            "category": complaint.category,
            "title": complaint.title,
            "description": complaint.description,
            "ward_id": complaint.ward_id,
            "ward_name": complaint.ward_name,
            "latitude": complaint.latitude,
            "longitude": complaint.longitude,
            "priority": complaint.priority,
            "severity": complaint.severity,
            "media_urls": complaint.media_urls,
            "ai_metadata": complaint.ai_validation_summary
        }
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": key
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return url, payload, headers

    def register_complaint_sync(
        self,
        complaint: StructuredComplaintJSON,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous version for direct call in dialogue engine."""
        url, payload, headers = self._prepare_complaint_payload(complaint, idempotency_key)
        logger.info(f"Submitting complaint to Node.js backend (sync): {url}")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload, headers=headers)
                if res.status_code in (200, 201):
                    data = res.json()
                    cid = None
                    if isinstance(data.get("data"), dict):
                        cid = data.get("data", {}).get("id") or data.get("data", {}).get("complaint_id")
                    elif isinstance(data.get("data"), (int, str)):
                        cid = data.get("data")
                    if not cid:
                        cid = data.get("id") or data.get("complaint_id") or f"CMP-{uuid.uuid4().hex[:6].upper()}"
                    return {"success": True, "status_code": res.status_code, "complaint_id": str(cid), "data": data}
                else:
                    logger.warning(f"Backend returned HTTP {res.status_code}: {res.text[:200]}")
                    fallback_id = f"CMP-{uuid.uuid4().hex[:6].upper()}"
                    return {"success": False, "status_code": res.status_code, "complaint_id": fallback_id, "raw": res.text}
        except Exception as e:
            logger.warning(f"Backend offline/unreachable ({e}). Assigned fallback tracking ticket.")
            fallback_id = f"CMP-{uuid.uuid4().hex[:6].upper()}"
            return {"success": False, "offline_queued": True, "complaint_id": fallback_id, "error": str(e)}

    async def register_complaint(
        self,
        complaint: StructuredComplaintJSON,
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Asynchronous version."""
        url, payload, headers = self._prepare_complaint_payload(complaint, idempotency_key)
        logger.info(f"Submitting complaint to Node.js backend (async): {url}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code in (200, 201):
                    data = res.json()
                    cid = data.get("data", {}).get("id") or f"CMP-{uuid.uuid4().hex[:6].upper()}"
                    return {"success": True, "status_code": res.status_code, "complaint_id": str(cid), "data": data}
                else:
                    fallback_id = f"CMP-{uuid.uuid4().hex[:6].upper()}"
                    return {"success": False, "status_code": res.status_code, "complaint_id": fallback_id, "raw": res.text}
        except Exception as e:
            logger.warning(f"Backend offline/unreachable ({e}). Assigned fallback tracking ticket.")
            fallback_id = f"CMP-{uuid.uuid4().hex[:6].upper()}"
            return {"success": False, "offline_queued": True, "complaint_id": fallback_id, "error": str(e)}

    def get_complaint_status_sync(self, complaint_id: str) -> Dict[str, Any]:
        """Check status synchronously."""
        url = f"{self.base_url}/complaints/{complaint_id}"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code == 200:
                    return {"success": True, "data": res.json()}
                return {"success": False, "status_code": res.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_recent_complaints_sync(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch recent verified complaints from live backend to feed into duplicate detector."""
        url = f"{self.base_url}/complaints"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(url, headers=headers)
                if res.status_code == 200:
                    raw_data = res.json().get("data", [])
                    formatted = []
                    for item in raw_data[:limit]:
                        loc = item.get("location")
                        lat, lon = None, None
                        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                            lat, lon = float(loc[0]), float(loc[1])
                        formatted.append({
                            "id": item.get("id"),
                            "category": str(item.get("category_id") or item.get("category", "")),
                            "description": item.get("description") or item.get("title", ""),
                            "latitude": lat or 19.228,
                            "longitude": lon or 73.070
                        })
                    logger.info(f"Fetched {len(formatted)} historical complaints from backend for deduplication.")
                    return formatted
        except Exception as e:
            logger.warning(f"Could not fetch recent complaints from backend ({e}). Fallback to local duplicate checking.")
        return []
