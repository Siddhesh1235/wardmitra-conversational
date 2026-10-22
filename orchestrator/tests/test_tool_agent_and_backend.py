import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs

import httpx
import pytest

from app.core.main_agent import MainConversationalAgent, MAX_PHOTO_ATTEMPTS
from app.schemas.ai_inference import StructuredComplaintJSON
from app.schemas.categories import resolve_backend_category
from app.slot_checker.deterministic_gate import SlotChecker
from app.state.session_models import ConversationSession
from app.tools import backend_client as backend_module
from app.tools.backend_client import WardMitraBackendClient


# ---------------------------------------------------------------- helpers

def _tool_call(name, args=None, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def _llm_response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedLLM:
    """Fake OpenAI client returning scripted responses in order and recording the requests."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _agent(llm=None):
    checker = SlotChecker()
    checker.extract_and_merge_slots = lambda existing_slots, **_: dict(existing_slots)  # no network
    return MainConversationalAgent(
        openai_client=llm,
        slot_checker=checker,
        inference_dispatcher=MagicMock(),
        backend_client=MagicMock(),
        knowledge_base=MagicMock(),
    )


def _session(**slots):
    return ConversationSession(session_id="s1", citizen_id="c1", slots=dict(slots))


COMPLETE_DRAFT = {"lang": "mr", "description": "रस्त्यावर मोठा खड्डा आहे", "location": "शिवाजी चौक"}


# ---------------------------------------------------------------- confirmation gate

def test_submit_refused_until_citizen_was_asked_to_confirm():
    llm = ScriptedLLM([
        _llm_response(tool_calls=[_tool_call("submit_complaint")]),
        _llm_response(content="खड्डा, शिवाजी चौक. तक्रार नोंदवू का?"),
    ])
    agent = _agent(llm)
    agent.submit_complaint = MagicMock()
    session = _session(**COMPLETE_DRAFT)

    out = agent.execute_turn(session, "शिवाजी चौकात खड्डा आहे")

    agent.submit_complaint.assert_not_called()
    tool_result = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert tool_result["error"] == "not_confirmed"
    assert session.slots["awaiting_confirmation"] is True
    assert out.status == "awaiting_confirmation" and not out.is_complete


def test_confirmed_draft_is_submitted_with_citizen_token():
    llm = ScriptedLLM([
        _llm_response(tool_calls=[_tool_call("submit_complaint")]),
        _llm_response(content="धन्यवाद! तक्रार क्र. 77 नोंदवली."),
    ])
    agent = _agent(llm)
    agent.submit_complaint = MagicMock(return_value={"submitted": True, "complaint_id": "77", "inference": None})
    session = _session(**COMPLETE_DRAFT, awaiting_confirmation=True)

    out = agent.execute_turn(session, "हो", auth_token="citizen-jwt")

    agent.submit_complaint.assert_called_once_with(session, "mr", "citizen-jwt")
    assert out.status == "completed" and out.is_complete and out.complaint_id == "77"
    assert session.slots["awaiting_confirmation"] is False


def test_failed_submission_never_reports_an_id():
    llm = ScriptedLLM([
        _llm_response(tool_calls=[_tool_call("submit_complaint")]),
        _llm_response(content="क्षमस्व, तक्रार नोंदवता आली नाही."),
    ])
    agent = _agent(llm)
    agent.submit_complaint = MagicMock(return_value={"submitted": False, "error_message": "Invalid or expired token", "status_code": 401})
    session = _session(**COMPLETE_DRAFT, awaiting_confirmation=True)

    out = agent.execute_turn(session, "हो")

    assert out.status == "registration_failed" and out.complaint_id is None and not out.is_complete
    tool_result = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert "log in" in tool_result["note"]


def test_question_uses_knowledge_tool_and_history_is_sent():
    llm = ScriptedLLM([
        _llm_response(tool_calls=[_tool_call("search_knowledge", {"query": "ward office timings"})]),
        _llm_response(content="कार्यालय सकाळी १० ते ५:४५ पर्यंत खुले असते."),
    ])
    agent = _agent(llm)
    agent.knowledge_base.search.return_value = {"found": True, "results": [{"title": "Timings", "content_mr": "१०:०० ते ५:४५"}]}
    session = _session(lang="mr")
    session.history = [SimpleNamespace(role="user", content="नमस्कार 9876543210"), SimpleNamespace(role="assistant", content="नमस्कार!")]

    out = agent.execute_turn(session, "ऑफिस किती वाजता उघडते?")

    assert out.status == "answering_faq"
    first_messages = llm.requests[0]["messages"]
    assert first_messages[1]["content"] == "नमस्कार [फोन नंबर]"  # phone masked before reaching the LLM
    assert json.loads(llm.requests[1]["messages"][-1]["content"])["results"][0]["content"] == "१०:०० ते ५:४५"


# ---------------------------------------------------------------- questions do not fill the complaint draft

@pytest.mark.parametrize("message_type,expect_description", [("question", False), ("confirmation", False), ("complaint_detail", True)])
def test_only_complaint_messages_fill_slots(message_type, expect_description):
    extraction = {"message_type": message_type, "description": "कचरा पडला आहे", "landmark": None, "category": "garbage"}
    llm = ScriptedLLM([_llm_response(content=json.dumps(extraction))])
    checker = SlotChecker(openai_client=llm)

    slots = checker.extract_and_merge_slots(existing_slots={}, incoming_text="प्रभाग 5 चे नगरसेवक कोण आहेत?")

    assert ("description" in slots) is expect_description


@pytest.mark.parametrize("message_type,expect_location", [("other", True), ("confirmation", True), ("question", False)])
def test_landmark_accepted_from_any_message_except_questions(message_type, expect_location):
    extraction = {"message_type": message_type, "description": None, "landmark": "Birla College", "category": None}
    checker = SlotChecker(openai_client=ScriptedLLM([_llm_response(content=json.dumps(extraction))]))
    checker.geo_service = MagicMock()
    checker.geo_service.lookup_by_address_or_landmark.return_value = None

    slots = checker.extract_and_merge_slots(existing_slots={"description": "garbage"}, incoming_text="Near Birla College")

    assert (slots.get("location") == "Birla College") is expect_location


# ---------------------------------------------------------------- photo retry limit

def test_unsuitable_photos_blocked_after_max_attempts():
    agent = _agent(llm=None)
    agent.dispatcher.moderator.moderate_image.return_value = SimpleNamespace(is_safe=True)
    agent.dispatcher.civic_classifier.classify_image.return_value = SimpleNamespace(is_civic_related=False, predicted_class="Non_Civic")
    session = _session(lang="en")

    for attempt in range(1, MAX_PHOTO_ATTEMPTS + 1):
        out = agent.execute_turn(session, "", image_bytes=b"img")
        assert out.status == "photo_rejected"
        assert session.slots["photo_rejections"] == attempt
    assert session.slots["photo_blocked"] is True
    assert "continue without a photo" in out.reply_text

    agent.dispatcher.civic_classifier.classify_image.reset_mock()
    agent.execute_turn(session, "", image_bytes=b"img")
    agent.dispatcher.civic_classifier.classify_image.assert_not_called()  # no more photo analysis once blocked


# ---------------------------------------------------------------- category mapping

BACKEND_CATEGORIES = [
    {"category_id": 1, "category_name": "Water Supply"},
    {"category_id": 3, "category_name": "Waste Management"},
    {"category_id": 4, "category_name": "Road & Infrastructure"},
    {"category_id": 8, "category_name": "Street Lighting"},
    {"category_id": 10, "category_name": "Other"},
]


@pytest.mark.parametrize("ai_category,expected_id", [
    ("PotHoles", 4), ("Garbage", 3), ("StreetLight", 8), ("PipelineDefects", 1), ("Trees", 10), ("Unknown", 10),
])
def test_resolve_backend_category(ai_category, expected_id):
    assert resolve_backend_category(ai_category, BACKEND_CATEGORIES)["category_id"] == expected_id


def test_resolve_backend_category_without_backend_list():
    assert resolve_backend_category("PotHoles", []) is None


# ---------------------------------------------------------------- backend wire format

@pytest.fixture
def captured_requests(monkeypatch):
    """Route the client's httpx calls to an in-memory handler and capture them."""
    captured = []

    def handler(request: httpx.Request):
        captured.append(request)
        if request.url.path.endswith("/complaints") and request.method == "POST":
            return httpx.Response(201, json={"success": True, "data": {"id": 501, "ward_id": 58}})
        if request.url.path.endswith("/complaints/501"):
            return httpx.Response(200, json={"complaint": {
                "id": 501, "title": "Pothole", "status": "assigned", "category_name": "Road & Infrastructure",
                "ward_name": "Ward 1A", "officer": {"name": "R. Patil", "designation": "JE"}, "sla_minutes": 2880}})
        return httpx.Response(404, json={"success": False, "message": "Not found"})

    real_client = httpx.Client
    monkeypatch.setattr(backend_module.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return captured


def _complaint(severity="critical"):
    return StructuredComplaintJSON(
        citizen_id="c1", category="PotHoles", priority=1, severity=severity, title="Pothole",
        description="Big pothole", ward_id="53", ward_name="Old ward 53", latitude=19.24, longitude=73.13,
    )


def test_register_sends_multipart_with_citizen_token(captured_requests):
    client = WardMitraBackendClient(base_url="http://backend.test/api", auth_token="fallback-token")
    res = client.register_complaint_sync(
        _complaint(), category_id=4, landmark="शिवाजी चौक", images=[b"\xff\xd8jpeg"], auth_token="citizen-jwt"
    )
    assert res == {"success": True, "complaint_id": "501", "data": {"success": True, "data": {"id": 501, "ward_id": 58}}}

    req = captured_requests[0]
    body = req.content.decode("utf-8", errors="ignore")
    assert req.headers["authorization"] == "Bearer citizen-jwt"
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert 'name="category_id"\r\n\r\n4' in body
    assert 'name="priority"\r\n\r\n4' in body          # critical -> backend 4 (1=low ... 4=critical)
    assert 'name="source_type"\r\n\r\nai_chat' in body
    assert 'name="images"; filename="photo_1.jpg"' in body
    assert 'name="ward_id"' not in body                # backend uses the citizen's registered ward


def test_register_without_coordinates_sends_only_landmark(captured_requests):
    client = WardMitraBackendClient(base_url="http://backend.test/api")
    client.register_complaint_sync(_complaint(), landmark="Birla College", include_coordinates=False)
    # Without photos httpx sends application/x-www-form-urlencoded (the backend parses both formats)
    form = parse_qs(captured_requests[0].content.decode("utf-8"))
    assert "latitude" not in form and form["landmark"] == ["Birla College"]


def test_complaint_status_is_parsed(captured_requests):
    client = WardMitraBackendClient(base_url="http://backend.test/api")
    res = client.get_complaint_status_sync("CMP-501", auth_token="citizen-jwt")
    assert captured_requests[0].url.path == "/api/complaints/501"
    assert res["data"]["status"] == "assigned"
    assert res["data"]["assigned_officer"] == "R. Patil"
    assert res["data"]["sla_hours"] == 48.0


def test_backend_error_message_is_returned(captured_requests):
    client = WardMitraBackendClient(base_url="http://backend.test/api")
    res = client.list_my_complaints(auth_token="x")  # handler answers 404 for this path
    assert res == {"success": False, "status_code": 404, "error_message": "Not found"}
