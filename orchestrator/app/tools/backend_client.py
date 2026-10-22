import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
import httpx
from dotenv import load_dotenv

load_dotenv()

from app.schemas.ai_inference import StructuredComplaintJSON

logger = logging.getLogger(__name__)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "https://wardmitra-api.spwhin.com/api").rstrip("/")
# The backend serves several cities; this orchestrator answers for one of them
DEFAULT_CITY_NAME = os.getenv("BACKEND_CITY_NAME", "Kalyan-Dombivli")
# Reference data (cities, wards, councillors, categories) changes rarely
REFERENCE_CACHE_TTL = int(os.getenv("BACKEND_REFERENCE_CACHE_TTL", "3600"))

# Backend (CouncillorApp) priority scale: 1=low, 2=medium, 3=high, 4=critical
SEVERITY_TO_BACKEND_PRIORITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class WardMitraBackendClient:
    """
    HTTP client for the CouncillorApp Node.js backend (the system of record).
    The backend itself uploads media to S3, auto-assigns the officer, applies SLA, and links duplicates;
    this client only forwards the conversation's result to it and reads data back.

    Auth: the backend files a complaint as the owner of the bearer token (`req.user`), so every
    citizen-facing call takes the citizen's own JWT (`auth_token`). BACKEND_AUTH_TOKEN is only a
    fallback for local testing.
    """
    def __init__(self, base_url: Optional[str] = None, auth_token: Optional[str] = None, timeout_seconds: float = 15.0):
        self.base_url = (base_url or DEFAULT_BACKEND_URL).rstrip("/")
        self.auth_token = auth_token or os.getenv("BACKEND_AUTH_TOKEN")
        self.timeout = timeout_seconds
        self._cache: Dict[str, tuple] = {}

    def _auth_headers(self, auth_token: Optional[str]) -> Dict[str, str]:
        token = auth_token or self.auth_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    @staticmethod
    def _error_message(res: httpx.Response) -> str:
        """Backend errors come as {message} or {error}; return a short human-readable reason."""
        try:
            body = res.json()
            return str(body.get("message") or body.get("error") or res.text)[:300]
        except Exception:
            return res.text[:300]

    # ------------------------------------------------------------------
    # Complaints (citizen JWT)
    # ------------------------------------------------------------------
    def register_complaint_sync(
        self,
        complaint: StructuredComplaintJSON,
        category_id: Optional[int] = None,
        landmark: Optional[str] = None,
        images: Optional[List[bytes]] = None,
        idempotency_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_coordinates: bool = True
    ) -> Dict[str, Any]:
        """
        POST /api/complaints as multipart/form-data (field `images`, max 4) so the backend stores photos in S3.
        `ward_id` is deliberately not sent: the backend uses the citizen's registered ward.
        `source_type=ai_chat` matches the backend's own AI chat, which skips the manual-form address checks.
        Without `include_coordinates` only the landmark is sent and the backend geocodes it.
        Returns {"success", "complaint_id", "error_message"}; never invents an ID.
        """
        url = f"{self.base_url}/complaints"
        form = {
            "title": complaint.title,
            "description": complaint.description,
            "priority": str(SEVERITY_TO_BACKEND_PRIORITY.get(complaint.severity, 2)),
            "source_type": "ai_chat",
            "metadata": json.dumps({"created_via": "wardmitra_orchestrator", **(metadata or {})}, ensure_ascii=False),
        }
        if include_coordinates:
            form["latitude"] = str(complaint.latitude)
            form["longitude"] = str(complaint.longitude)
        if category_id is not None:
            form["category_id"] = str(category_id)
        if landmark:
            form["landmark"] = landmark
        if complaint.address:
            form["address"] = complaint.address
        files = [("images", (f"photo_{i + 1}.jpg", img, "image/jpeg")) for i, img in enumerate((images or [])[:4])]
        headers = self._auth_headers(auth_token)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        logger.info(f"Submitting complaint to backend: {url} (images={len(files)}, category_id={category_id})")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, data=form, files=files or None, headers=headers)
            if res.status_code in (200, 201):
                data = res.json()
                inner = data.get("data")
                cid = (inner.get("id") if isinstance(inner, dict) else inner) or data.get("id")
                return {"success": True, "complaint_id": str(cid) if cid else None, "data": data}
            logger.warning(f"Backend rejected complaint: HTTP {res.status_code}: {res.text[:200]}")
            return {"success": False, "complaint_id": None, "status_code": res.status_code, "error_message": self._error_message(res)}
        except Exception as e:
            logger.warning(f"Backend offline/unreachable ({e}). Complaint was NOT registered.")
            return {"success": False, "complaint_id": None, "error_message": None, "error": str(e)}

    def get_complaint_status_sync(self, complaint_id: str, auth_token: Optional[str] = None) -> Dict[str, Any]:
        """
        GET /api/complaints/{id} -> {"success", "data": {...}} with the fields a citizen needs.
        The backend only allows the reporter (or ward staff) to read a complaint.
        """
        cid = str(complaint_id).upper().removeprefix("CMP-")
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/complaints/{cid}", headers=self._auth_headers(auth_token))
            if res.status_code != 200:
                return {"success": False, "status_code": res.status_code, "error_message": self._error_message(res)}
            c = res.json().get("complaint") or {}
            officer = c.get("officer") or {}
            return {"success": True, "data": {
                "id": c.get("id"),
                "title": c.get("title"),
                "status": c.get("status"),
                "category": c.get("category_name"),
                "ward": c.get("ward_name"),
                "assigned_officer": officer.get("name"),
                "officer_designation": officer.get("designation"),
                "councillor": c.get("councillor_name"),
                "created_at": c.get("created_at"),
                "sla_hours": round(c["sla_minutes"] / 60, 1) if isinstance(c.get("sla_minutes"), (int, float)) else None,
            }}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_my_complaints(self, auth_token: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """GET /api/complaints/my -> the citizen's latest complaints (newest first)."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/complaints/my", params={"limit": limit}, headers=self._auth_headers(auth_token))
            if res.status_code != 200:
                return {"success": False, "status_code": res.status_code, "error_message": self._error_message(res)}
            rows = res.json().get("data") or []
            return {"success": True, "data": [
                {"id": r.get("id"), "title": r.get("title"), "status": r.get("status"), "created_at": r.get("created_at")}
                for r in rows[:limit]
            ]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_categories(self, auth_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /api/user/categories (citizen-facing) -> [{category_id, category_name, phone_number}], cached."""
        cached = self._cache.get("/user/categories")
        if cached and time.time() - cached[0] < REFERENCE_CACHE_TTL:
            return cached[1]
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/user/categories", headers=self._auth_headers(auth_token))
            if res.status_code == 200:
                data = res.json().get("data") or []
                self._cache["/user/categories"] = (time.time(), data)
                return data
            logger.warning(f"Backend GET /user/categories returned HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"Backend GET /user/categories failed ({e}).")
        return cached[1] if cached else []

    # ------------------------------------------------------------------
    # Public reference data (no auth)
    # ------------------------------------------------------------------
    def _get_public_cached(self, path: str) -> Optional[Any]:
        """GET a public (no-auth) backend endpoint and return its `data`, cached for REFERENCE_CACHE_TTL seconds."""
        cached = self._cache.get(path)
        if cached and time.time() - cached[0] < REFERENCE_CACHE_TTL:
            return cached[1]
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}{path}")
                if res.status_code == 200:
                    data = res.json().get("data")
                    self._cache[path] = (time.time(), data)
                    return data
                logger.warning(f"Backend GET {path} returned HTTP {res.status_code}")
        except Exception as e:
            logger.warning(f"Backend GET {path} failed ({e}).")
        # Serve stale data rather than nothing if the backend is temporarily down
        return cached[1] if cached else None

    def list_cities(self) -> List[Dict[str, Any]]:
        """Cities served by the backend (id, name, total_wards)."""
        return self._get_public_cached("/cities") or []

    def list_wards(self, city_name: Optional[str] = DEFAULT_CITY_NAME) -> List[Dict[str, Any]]:
        """
        Wards with their councillors, from public GET /wards.
        The endpoint returns every city (it ignores query params), so we filter by `city_name` here.
        Each prabhag (number) has up to 4 seats, e.g. 'Ward 12A'..'Ward 12D', sharing one centroid.
        """
        wards = self._get_public_cached("/wards") or []
        if city_name:
            wards = [w for w in wards if w.get("city_name") == city_name]
        return wards

    def get_ward(self, ward_id: Any) -> Optional[Dict[str, Any]]:
        """Single ward record by backend ID, from public GET /wards/{id}."""
        return self._get_public_cached(f"/wards/{ward_id}")

    def get_recent_complaints_sync(self, limit: int = 25, auth_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recent complaints for local duplicate checks (rule-based fallback path only)."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self.base_url}/complaints", headers=self._auth_headers(auth_token))
                if res.status_code == 200:
                    formatted = []
                    for item in (res.json().get("data") or [])[:limit]:
                        loc = item.get("location")
                        if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
                            continue  # no coordinates -> cannot be compared by distance
                        formatted.append({
                            "id": item.get("id"),
                            "category": str(item.get("category_id") or item.get("category", "")),
                            "description": item.get("description") or item.get("title", ""),
                            "latitude": float(loc[0]),
                            "longitude": float(loc[1])
                        })
                    return formatted
        except Exception as e:
            logger.warning(f"Could not fetch recent complaints from backend ({e}).")
        return []
