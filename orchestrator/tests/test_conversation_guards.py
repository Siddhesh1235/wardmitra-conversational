from unittest.mock import MagicMock

import pytest

from app.core.conversation_guards import detect_draft_command, detect_emergency
from app.core.main_agent import MainConversationalAgent
from app.slot_checker.deterministic_gate import SlotChecker
from app.state.session_models import ConversationSession


def _agent():
    checker = SlotChecker()
    checker.extract_and_merge_slots = lambda existing_slots, **_: dict(existing_slots)  # no network
    return MainConversationalAgent(
        openai_client=None, slot_checker=checker, inference_dispatcher=MagicMock(),
        backend_client=MagicMock(), knowledge_base=MagicMock(),
    )


# ---------------------------------------------------------------- cancel / restart detection

@pytest.mark.parametrize("text,expected", [
    ("रद्द करा", "cancel"),
    ("नको, राहू दे", "cancel"),
    ("तक्रार नोंदवू नका", "cancel"),
    ("रहने दो", "cancel"),
    ("cancel", "cancel"),
    ("never mind", "cancel"),
    ("नवीन तक्रार करायची आहे", "restart"),
    ("start over", "restart"),
    # Real complaints that merely contain similar words must not cancel anything
    ("bus stop जवळ कचरा पडला आहे", None),
    ("बस थांबा जवळ खड्डा आहे", None),
    ("यहाँ रहने वाले लोग परेशान हैं", None),
    ("कचरा इथे टाकायला नको म्हणून बोर्ड लावा आणि रोज साफ करा", None),
    ("हो", None),
])
def test_detect_draft_command(text, expected):
    assert detect_draft_command(text) == expected


def test_cancel_clears_draft_but_keeps_language():
    agent = _agent()
    session = ConversationSession(session_id="s1", slots={
        "lang": "mr", "description": "खड्डा", "location": "शिवाजी चौक", "photo_b64": "x",
        "awaiting_confirmation": True, "photo_rejections": 1,
    })
    out = agent.execute_turn(session, "रद्द करा")
    assert out.status == "cancelled"
    assert session.slots == {"lang": "mr"}
    assert "रद्द" in out.reply_text
    agent.dispatcher.process.assert_not_called()


def test_cancel_without_draft_is_handled_normally():
    agent = _agent()
    session = ConversationSession(session_id="s1", slots={"lang": "en"})
    out = agent.execute_turn(session, "cancel")
    assert out.status != "cancelled"


def test_restart_always_resets():
    agent = _agent()
    session = ConversationSession(session_id="s1", slots={"lang": "en", "description": "garbage"})
    out = agent.execute_turn(session, "start over")
    assert out.status == "restarted" and session.slots == {"lang": "en"}


# ---------------------------------------------------------------- emergencies

@pytest.mark.parametrize("text,kind", [
    ("रस्त्यावर विजेची वायर तुटून पडली आहे", "electric"),
    ("खांबावरून स्पार्क येत आहेत", "electric"),
    ("there is a live wire on the road", "electric"),
    ("शाळेसमोर उघडे मॅनहोल आहे", "manhole"),
    ("मार्केटमध्ये आग लागली आहे", "fire"),
    ("building mein gas leak ho raha hai", "gas"),
    ("जुनी इमारत कोसळली आहे", "collapse"),
    ("पावसामुळे घरात पाणी शिरले आहे", "flood"),
    # Routine complaints are not emergencies
    ("कचरा जाळल्यामुळे धूर येत आहे", None),
    ("स्ट्रीट लाईट बंद आहे", None),
    ("fire brigade office timing?", None),
])
def test_detect_emergency(text, kind):
    assert detect_emergency(text) == kind


def test_emergency_banner_is_prepended_and_marks_draft():
    agent = _agent()
    session = ConversationSession(session_id="s1", slots={"lang": "mr"})
    out = agent.execute_turn(session, "रस्त्यावर विजेची वायर तुटून पडली आहे")
    assert out.reply_text.startswith("⚠️")
    assert "1912" in out.reply_text and "112" in out.reply_text
    assert session.slots["emergency"] == "electric"
