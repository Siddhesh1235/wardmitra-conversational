import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.core.conversation_guards import detect_photo_choice
from app.core.main_agent import MainConversationalAgent
from app.schemas.ai_inference import StructuredComplaintJSON
from app.schemas.categories import infer_draft_category, same_department
from app.slot_checker.deterministic_gate import SlotChecker
from app.state.session_models import ConversationSession


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (120, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _agent(photo_class="PipelineDefects", llm=None):
    checker = SlotChecker()
    checker.extract_and_merge_slots = lambda existing_slots, **_: dict(existing_slots)  # no network
    agent = MainConversationalAgent(
        openai_client=llm, slot_checker=checker, inference_dispatcher=MagicMock(),
        backend_client=MagicMock(), knowledge_base=MagicMock(),
    )
    agent.dispatcher.moderator.moderate_image.return_value = SimpleNamespace(is_safe=True)
    agent.dispatcher.civic_classifier.classify_image.return_value = SimpleNamespace(
        is_civic_related=True, predicted_class=photo_class, visual_description="Water spraying from a rusty pipe"
    )
    return agent


GARBAGE_DRAFT = {"lang": "mr", "description": "गल्लीत कचरा पडला आहे", "location": "शिवाजी चौक"}


# ---------------------------------------------------------------- category helpers

@pytest.mark.parametrize("a,b,same", [
    ("Garbage", "PipelineDefects", False),
    ("garbage", "Garbage", True),             # extractor value vs AI class
    ("water_leak", "PipelineDefects", True),
    ("StreetLight", "Electricity", True),     # both fit the Electricity department
    ("PotHoles", "Road_Incidents_Traffic", True),
    ("PotHoles", "Garbage", False),
    (None, "Garbage", True),                  # unknown never counts as a mismatch
])
def test_same_department(a, b, same):
    assert same_department(a, b) is same


@pytest.mark.parametrize("slots,expected", [
    ({"description": "गल्लीत कचरा पडला आहे"}, "Garbage"),
    ({"category": "water_leak", "description": "something"}, "PipelineDefects"),
    ({"description": "please help me"}, None),
    ({}, None),
])
def test_infer_draft_category(slots, expected):
    assert infer_draft_category(slots) == expected


@pytest.mark.parametrize("text,choice", [
    ("1", "use_photo"), ("१", "use_photo"), ("option 2", "new_photo"), ("3.", "skip"),
    ("दुसरा फोटो पाठवतो", "new_photo"), ("फोटोशिवाय पुढे जा", "skip"), ("बिना फोटो", "skip"),
    ("12 दिवसांपासून कचरा आहे", None), ("ऑफिस किती वाजता उघडते?", None),
])
def test_detect_photo_choice(text, choice):
    assert detect_photo_choice(text) == choice


# ---------------------------------------------------------------- the reported bug: garbage complaint + pipe-leak photo

def _mismatch_session(agent):
    session = ConversationSession(session_id="s1", slots=dict(GARBAGE_DRAFT))
    out = agent.execute_turn(session, "तक्रारीचा फोटो जोडला आहे", image_bytes=_jpeg())
    return session, out


def test_mismatching_photo_is_not_attached_and_citizen_is_asked():
    agent = _agent()
    session, out = _mismatch_session(agent)

    assert out.status == "photo_mismatch"
    assert "पाइपलाइन गळती" in out.reply_text and "कचरा" in out.reply_text
    assert "photo_b64" not in session.slots and session.slots.get("pending_photo_b64")
    assert "pending_photo_b64" not in out.collected_slots   # photo data is never returned to clients
    assert session.slots.get("photo_rejections") is None    # a mismatch is not a rejected attempt


def test_choice_1_turns_complaint_into_what_the_photo_shows():
    agent = _agent()
    session, _ = _mismatch_session(agent)
    out = agent.execute_turn(session, "1")

    assert session.slots["photo_b64"] and "pending_photo_b64" not in session.slots
    assert session.slots["category"] == "PipelineDefects"
    assert "पाइपलाइन गळती" in session.slots["description"]
    # The changed draft is shown again for a fresh confirmation; nothing is filed yet
    assert out.status == "awaiting_confirmation" and "पाइपलाइन गळती" in out.reply_text
    agent.backend_client.register_complaint_sync.assert_not_called()


def test_fallback_path_registers_only_after_yes():
    agent = _agent(photo_class="Garbage")
    agent.submit_complaint = MagicMock(return_value={"submitted": False, "error_message": "x"})
    session = ConversationSession(session_id="s1", slots=dict(GARBAGE_DRAFT))

    out = agent.execute_turn(session, "शिवाजी चौकात कचरा")
    assert out.status == "awaiting_confirmation"
    agent.submit_complaint.assert_not_called()

    agent.execute_turn(session, "नाही थांबा")
    agent.submit_complaint.assert_not_called()

    agent.execute_turn(session, "हो, नोंदवा")
    agent.submit_complaint.assert_called_once()


def test_choice_3_continues_without_photo():
    agent = _agent()
    session, _ = _mismatch_session(agent)
    agent.execute_turn(session, "3")

    assert "photo_b64" not in session.slots and "pending_photo_b64" not in session.slots
    assert session.slots["photo_declined"] is True
    assert session.slots["description"] == GARBAGE_DRAFT["description"]


def test_new_photo_replaces_pending_one():
    agent = _agent()
    session, _ = _mismatch_session(agent)
    agent.dispatcher.civic_classifier.classify_image.return_value = SimpleNamespace(
        is_civic_related=True, predicted_class="Garbage", visual_description="Pile of garbage on a lane"
    )
    out = agent.execute_turn(session, "", image_bytes=_jpeg())

    assert out.status != "photo_mismatch"
    assert session.slots["photo_category"] == "Garbage" and "pending_photo_b64" not in session.slots


def test_matching_photo_is_attached_without_question():
    agent = _agent(photo_class="Garbage")
    session = ConversationSession(session_id="s1", slots=dict(GARBAGE_DRAFT))
    out = agent.execute_turn(session, "हा फोटो", image_bytes=_jpeg())

    assert out.status != "photo_mismatch"
    assert session.slots["photo_b64"] and session.slots["photo_category"] == "Garbage"


def test_photo_only_report_gets_readable_description():
    agent = _agent()
    session = ConversationSession(session_id="s1", slots={"lang": "mr"})
    agent.execute_turn(session, "", image_bytes=_jpeg())

    assert session.slots["description"] == "पाण्याची पाइपलाइन गळती (Water spraying from a rusty pipe)"


# ---------------------------------------------------------------- LLM path

def _tool_call(name, args=None):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def _resp(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_free_text_answer_resolved_by_llm_tool_and_submit_blocked_while_pending():
    llm = ScriptedLLM([
        _resp(tool_calls=[_tool_call("submit_complaint")]),
        _resp(content="आधी फोटोबद्दल निवडा: 1, 2 किंवा 3."),
        _resp(tool_calls=[_tool_call("resolve_photo_mismatch", {"choice": "use_photo"})]),
        _resp(content="ठीक आहे, पाइपलाइन गळतीची तक्रार. नोंदवू का?"),
    ])
    agent = _agent(llm=llm)
    session, _ = _mismatch_session(agent)
    agent.submit_complaint = MagicMock()

    agent.execute_turn(session, "नोंदवा")
    assert json.loads(llm.requests[1]["messages"][-1]["content"])["error"] == "photo_mismatch_pending"
    agent.submit_complaint.assert_not_called()
    assert "PHOTO MISMATCH PENDING" in llm.requests[0]["messages"][0]["content"]

    agent.execute_turn(session, "हो, पाण्याचीच तक्रार आहे")
    assert session.slots["category"] == "PipelineDefects" and session.slots["photo_b64"]


# ---------------------------------------------------------------- explicit "yes" always registers

@pytest.mark.parametrize("text,forced", [
    ("ठीक आहे, तक्रार नोंदवा", True),
    ("हो", True),
    ("yes please register it", True),
    ("हो, पण ठिकाण बदला", False),        # qualified: must go back to the LLM, not straight to submit
    ("नाही", False),
    ("ऑफिस किती वाजता उघडते?", False),
])
def test_plain_yes_forces_submit_tool(text, forced):
    llm = ScriptedLLM([_resp(content="ok")])
    agent = _agent(llm=llm)
    session = ConversationSession(session_id="s1", slots={**GARBAGE_DRAFT, "awaiting_confirmation": True})

    agent.execute_turn(session, text)

    tool_choice = llm.requests[0]["tool_choice"]
    assert (tool_choice != "auto") is forced
    if forced:
        assert tool_choice["function"]["name"] == "submit_complaint"


def test_yes_without_prior_summary_is_not_forced():
    llm = ScriptedLLM([_resp(content="ok")])
    agent = _agent(llm=llm)
    session = ConversationSession(session_id="s1", slots=dict(GARBAGE_DRAFT))  # summary not shown yet
    agent.execute_turn(session, "हो")
    assert llm.requests[0]["tool_choice"] == "auto"


# ---------------------------------------------------------------- submission routes by the draft's category

def test_submission_uses_draft_category_not_vision_guess():
    agent = _agent()
    complaint = StructuredComplaintJSON(
        citizen_id="c1", category="PipelineDefects", priority=2, severity="medium", title="t", description="d",
        ward_id="1", ward_name="w", latitude=19.24, longitude=73.13,
    )
    agent.dispatcher.process.return_value = SimpleNamespace(
        success=True, structured_complaint=complaint,
        scoring=SimpleNamespace(final_category="PipelineDefects", severity="medium"),
    )
    agent.backend_client.list_categories.return_value = [
        {"category_id": 1, "category_name": "Water Supply"}, {"category_id": 3, "category_name": "Waste Management"},
    ]
    agent.backend_client.register_complaint_sync.return_value = {"success": True, "complaint_id": "9", "data": {}}
    session = ConversationSession(session_id="s1", slots=dict(GARBAGE_DRAFT))

    result = agent.submit_complaint(session, "mr", "jwt")

    assert agent.backend_client.register_complaint_sync.call_args.kwargs["category_id"] == 3
    assert result["category"] == "Waste Management"
