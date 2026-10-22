"""
Live evaluation of WardMitra conversations against golden_conversations.json.

Uses the real OpenAI model, the real knowledge base (static FAQs + live ward data from the public
/wards endpoint) and real geocoding. Citizen-specific backend calls (file / track / my complaints)
are served by FakeCitizenBackend, so no real complaint is ever created and no login token is needed.

Usage (from orchestrator/):
    python tests/eval/run_eval.py                 # all cases
    python tests/eval/run_eval.py -k complaint    # cases whose id contains 'complaint'
    python tests/eval/run_eval.py --min-pass 0.9  # exit code 1 below this pass rate
Requires OPENAI_API_KEY (from .env).
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["SESSION_STORE_TYPE"] = "memory"

from app.core.dialogue_engine import ConversationalOrchestrator  # noqa: E402
from app.schemas.ai_inference import CivicClassificationResult  # noqa: E402
from app.state.state_manager import ConversationStateManager  # noqa: E402
from app.tools.backend_client import WardMitraBackendClient  # noqa: E402

GOLDEN_FILE = Path(__file__).with_name("golden_conversations.json")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class FakeCitizenBackend(WardMitraBackendClient):
    """Real public endpoints (wards); fixed data for everything that needs a citizen login."""

    MY_COMPLAINTS = [
        {"id": 501, "title": "Pothole at Shivaji Chowk", "status": "assigned", "created_at": "2026-10-18T09:12:00Z"},
        {"id": 502, "title": "Streetlight not working", "status": "resolved", "created_at": "2026-10-10T18:40:00Z"},
    ]
    CATEGORIES = [
        {"category_id": 1, "category_name": "Water Supply"}, {"category_id": 2, "category_name": "Electricity"},
        {"category_id": 3, "category_name": "Waste Management"}, {"category_id": 4, "category_name": "Road & Infrastructure"},
        {"category_id": 8, "category_name": "Street Lighting"}, {"category_id": 9, "category_name": "Drainage"},
        {"category_id": 10, "category_name": "Other"},
    ]

    def __init__(self):
        super().__init__()
        self.registered: List[Dict[str, Any]] = []

    def register_complaint_sync(self, complaint, **kwargs):
        self.registered.append({"title": complaint.title, "severity": complaint.severity, **kwargs})
        return {"success": True, "complaint_id": "777", "data": {"success": True, "data": {"id": 777, "ward_id": 58}}}

    def get_complaint_status_sync(self, complaint_id, auth_token=None):
        if str(complaint_id).upper().removeprefix("CMP-") == "501":
            return {"success": True, "data": {
                "id": 501, "title": "Pothole at Shivaji Chowk", "status": "assigned", "category": "Road & Infrastructure",
                "ward": "Ward 19A", "assigned_officer": "R. Patil", "officer_designation": "Junior Engineer",
                "councillor": None, "created_at": "2026-10-18T09:12:00Z", "sla_hours": 72.0}}
        return {"success": False, "status_code": 404, "error_message": "Complaint not found"}

    def list_my_complaints(self, auth_token=None, limit=5):
        return {"success": True, "data": self.MY_COMPLAINTS[:limit]}

    def list_categories(self, auth_token=None):
        return self.CATEGORIES


def main_script(text: str) -> str:
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "devanagari" if dev >= lat else "latin"


def test_photo() -> bytes:
    """A small valid JPEG; what it 'shows' is set per turn via the stubbed vision result."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (160, 120), (110, 100, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def stub_vision(orch: ConversationalOrchestrator, photo: Dict[str, Any]) -> None:
    """Make the vision classifier report the given category, so the case tests the conversation, not the model."""
    result = CivicClassificationResult(
        predicted_class=photo["category"], confidence=0.95, is_civic_related=photo["category"] != "Non_Civic",
        visual_description=photo.get("visual"),
    )
    orch.dispatcher.civic_classifier.classify_image = lambda _image: result


def check_turn(reply: str, status: str, expect: Dict[str, Any], backend: Optional["FakeCitizenBackend"] = None) -> List[str]:
    failures = []
    if "registered_category_id" in expect:
        got = backend.registered[-1].get("category_id") if backend and backend.registered else None
        if got != expect["registered_category_id"]:
            failures.append(f"registered category_id {got}, expected {expect['registered_category_id']}")
    if "status_in" in expect and status not in expect["status_in"]:
        failures.append(f"status '{status}' not in {expect['status_in']}")
    if "contains_any" in expect and not any(s in reply for s in expect["contains_any"]):
        failures.append(f"reply has none of {expect['contains_any']}")
    for s in expect.get("not_contains", []):
        if s in reply:
            failures.append(f"reply must not contain '{s}'")
    if "digits_allowed" in expect:
        numbers = re.findall(r"\d+", reply.translate(DEVANAGARI_DIGITS).replace("-", ""))
        unexpected = sorted({n for n in numbers if n not in expect["digits_allowed"]})
        if unexpected:
            failures.append(f"unexpected numbers (possible invention): {unexpected}")
    if "script" in expect and main_script(reply) != expect["script"]:
        failures.append(f"reply script is {main_script(reply)}, expected {expect['script']}")
    return failures


def run_case(orch: ConversationalOrchestrator, case: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    turn_results = []
    for i, turn in enumerate(case["turns"], 1):
        started = time.time()
        image = None
        if turn.get("photo"):
            stub_vision(orch, turn["photo"])
            image = test_photo()
        out = orch.process_turn(session_id=session_id, citizen_id="EVAL", text=turn["user"],
                                image_bytes=image, auth_token="eval-citizen-token")
        failures = check_turn(out.reply_text, out.status, turn.get("expect", {}), orch.backend_client)
        turn_results.append({"turn": i, "user": turn["user"], "status": out.status, "reply": out.reply_text,
                             "seconds": round(time.time() - started, 1), "failures": failures})
        if verbose or failures:
            mark = "✅" if not failures else "❌"
            photo = f" 📷[{turn['photo']['category']}]" if turn.get("photo") else ""
            print(f"   {mark} T{i} 👤 {turn['user']}{photo}\n        🤖 [{out.status}] {out.reply_text[:300]}")
            for f in failures:
                print(f"        ⚠️  {f}")
    return {"id": case["id"], "passed": all(not t["failures"] for t in turn_results), "turns": turn_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", help="only run cases whose id contains this text")
    parser.add_argument("--min-pass", type=float, default=0.0, help="exit 1 if the pass rate is below this (0-1)")
    parser.add_argument("--report", help="write a JSON report to this path")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every turn, not only failures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    cases = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))["cases"]
    if args.k:
        cases = [c for c in cases if args.k in c["id"]]

    orch = ConversationalOrchestrator(
        state_manager=ConversationStateManager(backend_type="memory"),
        backend_client=FakeCitizenBackend(),
    )
    if not orch.client:
        print("OPENAI_API_KEY is not set; the live evaluation needs the LLM.")
        return 2

    results = []
    for case in cases:
        print(f"▶ {case['id']}")
        results.append(run_case(orch, case, args.verbose))

    passed = sum(r["passed"] for r in results)
    rate = passed / len(results) if results else 0.0
    turns = [t for r in results for t in r["turns"]]
    print(f"\n{'=' * 60}\nCases passed: {passed}/{len(results)} ({rate:.0%})"
          f"\nTurns passed: {sum(not t['failures'] for t in turns)}/{len(turns)}"
          f"\nAvg seconds per turn: {sum(t['seconds'] for t in turns) / max(len(turns), 1):.1f}")
    failed = [r["id"] for r in results if not r["passed"]]
    if failed:
        print("Failed: " + ", ".join(failed))
    if args.report:
        Path(args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if rate >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
