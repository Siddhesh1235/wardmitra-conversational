"""
Tools the conversational agent (gpt-4o-mini, OpenAI function calling) can use in a turn.
Each tool wraps existing code: the municipal knowledge base, and the CouncillorApp backend
(which stores complaints, uploads photos, assigns officers and links duplicates itself).
"""
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.core.agent_router import extract_complaint_id

if TYPE_CHECKING:
    from app.core.main_agent import MainConversationalAgent
    from app.state.session_models import ConversationSession

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the verified municipal knowledge base: office timings, holidays, helplines and "
                "emergency numbers, complaint process and SLAs, departments, property tax, certificates, "
                "escalation, and each prabhag's/ward's councillors (नगरसेवक) with their phone numbers. "
                "Use for ANY factual question. Write the query as a short standalone question about the topic only "
                "(do not add the city or corporation name), e.g. 'office working hours', 'ward 5 councillor'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Standalone search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_complaints",
            "description": "List the citizen's own most recent complaints with ID and status. Use when they ask about their complaints without giving an ID.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_complaint_status",
            "description": "Get live status, category, ward, assigned officer and SLA of one complaint by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"complaint_id": {"type": "string", "description": "Complaint ID, e.g. '123' or 'CMP-123'"}},
                "required": ["complaint_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_photo_mismatch",
            "description": (
                "Use only when the draft shows PHOTO MISMATCH PENDING and the citizen's reply answers it. "
                "'use_photo' = the complaint is about what the photo shows; 'new_photo' = they will send another photo; "
                "'skip' = continue without a photo."
            ),
            "parameters": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": ["use_photo", "new_photo", "skip"]}},
                "required": ["choice"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_complaint",
            "description": (
                "Register the current complaint draft with the municipal corporation. Call ONLY after you showed "
                "the citizen a summary of the draft and they clearly confirmed (yes / हो / हाँ / ok / submit / नोंदवा)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class AgentToolbox:
    """Executes tool calls for one dialogue turn and records what happened (for the turn's output status)."""

    def __init__(
        self,
        agent: "MainConversationalAgent",
        session: "ConversationSession",
        language: str,
        auth_token: Optional[str] = None,
    ):
        self.agent = agent
        self.session = session
        self.language = language
        self.auth_token = auth_token
        self.tools_called: List[str] = []
        # Set by submit_complaint: {"submitted": bool, "complaint_id", "inference", "error_message"}
        self.submission: Optional[Dict[str, Any]] = None

    def run(self, name: str, arguments_json: str) -> Dict[str, Any]:
        self.tools_called.append(name)
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            args = {}
        handler = getattr(self, f"_tool_{name}", None)
        if not handler:
            return {"error": f"unknown tool '{name}'"}
        try:
            result = handler(**args)
            logger.info(f"Tool {name} args={args} -> {json.dumps(result, ensure_ascii=False, default=str)[:300]}")
            return result
        except Exception as e:
            logger.exception(f"Tool {name} failed: {e}")
            return {"error": "tool_failed"}

    # ------------------------------------------------------------------
    def _doc_content(self, doc: Dict[str, Any]) -> str:
        if self.language == "mr":
            return doc.get("content_mr", "")
        if self.language == "hi":
            return doc.get("content_hi") or doc.get("content_mr", "")
        return doc.get("content_en", "")

    def _tool_search_knowledge(self, query: str = "") -> Dict[str, Any]:
        res = self.agent.knowledge_base.search(query, top_k=3)
        if not res.get("found"):
            return {"found": False, "note": "Not in the verified knowledge base. Do not guess; suggest contacting the ward office."}
        return {
            "found": True,
            "results": [{"title": d.get("title", ""), "content": self._doc_content(d)} for d in res.get("results", [])],
        }

    def _auth_error(self, res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if res.get("status_code") == 401:
            return {"error": "citizen_not_logged_in", "note": "Ask the citizen to log in to the WardMitra app to see their complaints."}
        if res.get("status_code") in (403, 404):
            return {"error": "not_found_or_not_yours", "note": "This complaint ID was not found for this citizen."}
        return None

    def _tool_get_my_complaints(self) -> Dict[str, Any]:
        res = self.agent.backend_client.list_my_complaints(auth_token=self.auth_token)
        if res.get("success"):
            return {"complaints": res["data"], "count": len(res["data"])}
        return self._auth_error(res) or {"error": "backend_unavailable"}

    def _tool_get_complaint_status(self, complaint_id: str = "") -> Dict[str, Any]:
        cid = extract_complaint_id(complaint_id) or "".join(ch for ch in complaint_id if ch.isdigit())
        if not cid:
            return {"error": "invalid_complaint_id"}
        res = self.agent.backend_client.get_complaint_status_sync(cid, auth_token=self.auth_token)
        if res.get("success"):
            return res["data"]
        return self._auth_error(res) or {"error": "backend_unavailable"}

    def _tool_resolve_photo_mismatch(self, choice: str = "") -> Dict[str, Any]:
        if not self.session.slots.get("pending_photo_b64"):
            return {"error": "no_pending_photo_mismatch"}
        if choice not in ("use_photo", "new_photo", "skip"):
            return {"error": "invalid_choice", "allowed": ["use_photo", "new_photo", "skip"]}
        note = self.agent.resolve_photo_mismatch(self.session, choice, self.language)
        return {"resolved": choice, "note": note, "description": self.session.slots.get("description")}

    def _tool_submit_complaint(self) -> Dict[str, Any]:
        slots = self.session.slots
        if slots.get("pending_photo_b64"):
            return {"submitted": False, "error": "photo_mismatch_pending", "note": "Ask the citizen to choose 1, 2 or 3 for the photo first."}
        gate = self.agent.slot_checker.evaluate(slots, language=self.language)
        if not gate.is_complete:
            return {"submitted": False, "error": "draft_incomplete", "missing": gate.missing_slots}
        if not slots.get("awaiting_confirmation"):
            return {"submitted": False, "error": "not_confirmed", "note": "Show the draft summary and ask the citizen to confirm first."}
        if self.submission and self.submission.get("submitted"):
            return {"submitted": True, "complaint_id": self.submission.get("complaint_id"), "note": "Already submitted in this turn."}

        self.submission = self.agent.submit_complaint(self.session, self.language, self.auth_token)
        result = {k: v for k, v in self.submission.items() if k != "inference"}
        if not self.submission.get("submitted") and self.submission.get("status_code") == 401:
            result["note"] = "Citizen is not logged in; ask them to log in to the WardMitra app and try again."
        return result
