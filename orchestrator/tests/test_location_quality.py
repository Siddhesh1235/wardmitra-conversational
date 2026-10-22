import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.pipeline.geo_lookup import GeoLookupService
from app.slot_checker.deterministic_gate import SlotChecker, is_vague_location, is_within_city


@pytest.mark.parametrize("location,vague", [
    ("आमच्या गल्लीत", True),
    ("माझ्या घराजवळ", True),
    ("इथे", True),
    ("हमारी गली में", True),
    ("near my house", True),
    ("our lane", True),
    ("", True),
    ("शिवाजी चौक", False),
    ("बिर्ला कॉलेज जवळ", False),
    ("साई कृपा सोसायटी समोर", False),
    ("Station Road, Kalyan West", False),
    ("रूम 12, गणेश बिल्डिंग", False),
])
def test_is_vague_location(location, vague):
    assert is_vague_location(location) is vague


def _checker(geo=None):
    checker = SlotChecker()
    checker.geo_service = geo or MagicMock()
    return checker


def _geo(within, dist=5000.0, address="Somewhere"):
    return SimpleNamespace(
        is_within_boundary=within, distance_to_centroid_meters=dist, ward_id="1", ward_name="Ward 1",
        centroid_lat=None, centroid_lon=None, address=address, display_name=address,
        latitude=19.0, longitude=73.0,
    )


def test_vague_location_is_asked_again_specifically():
    result = _checker().evaluate({"description": "कचरा पडला आहे", "location": "आमच्या गल्लीत"}, language="mr")
    assert not result.is_complete and result.missing_slots == ["location"]
    assert "लँडमार्क" in result.next_question_mr and "live location" in result.next_question_en


def test_specific_location_completes_draft():
    result = _checker().evaluate({"description": "कचरा पडला आहे", "location": "शिवाजी चौक"})
    assert result.is_complete


def _extraction_llm(**fields):
    payload = {"message_type": "complaint_detail", "category": None, "description": None, "landmark": None, **fields}
    message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
    create = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_vague_text_is_not_geocoded_and_specific_text_replaces_it():
    geo = MagicMock()
    geo.lookup_by_address_or_landmark.return_value = None
    checker = _checker(geo)
    checker.client = _extraction_llm(description="कचरा पडला आहे", landmark="आमच्या गल्लीत")

    slots = checker.extract_and_merge_slots({}, incoming_text="आमच्या गल्लीत कचरा पडला आहे")
    assert slots["location"] == "आमच्या गल्लीत"
    geo.lookup_by_address_or_landmark.assert_not_called()
    assert checker.evaluate(slots).missing_slots == ["location"]

    checker.client = _extraction_llm(landmark="बिर्ला कॉलेज")  # shorter than the vague text, still replaces it
    slots = checker.extract_and_merge_slots(slots, incoming_text="बिर्ला कॉलेज")
    assert slots["location"] == "बिर्ला कॉलेज"
    geo.lookup_by_address_or_landmark.assert_called_once_with("बिर्ला कॉलेज")
    assert checker.evaluate(slots).is_complete


def test_gps_outside_city_is_rejected_then_typed_landmark_wins():
    geo = MagicMock()
    geo.lookup_ward.return_value = _geo(within=False, dist=90_000, address="Pune Station")
    geo.lookup_by_address_or_landmark.return_value = None
    checker = _checker(geo)

    slots = checker.extract_and_merge_slots({"description": "खड्डा आहे"}, incoming_lat=18.53, incoming_lon=73.87)
    assert slots["outside_city"] is True
    result = checker.evaluate(slots, language="en")
    assert result.missing_slots == ["location"] and "outside" in result.next_question_en

    checker._set_text_location(slots, "शिवाजी चौक", "extracted_landmark")
    assert "has_gps" not in slots and "latitude" not in slots and "outside_city" not in slots
    assert checker.evaluate(slots).is_complete


def test_geocoded_match_outside_city_keeps_text_without_coordinates():
    geo = MagicMock()
    geo.lookup_by_address_or_landmark.return_value = _geo(within=False, dist=40_000)
    checker = _checker(geo)

    slots = checker.extract_and_merge_slots({"description": "पाणी गळती"}, incoming_text="गांधी मार्केट रोड")
    assert slots["location"] == "गांधी मार्केट रोड"
    assert "latitude" not in slots and "has_landmark_coords" not in slots


def test_real_city_boundary():
    geo = GeoLookupService()
    kalyan = geo.lookup_ward(19.2435, 73.1310, fetch_address=False)   # Shivaji Chowk, Kalyan
    pune = geo.lookup_ward(18.5286, 73.8743, fetch_address=False)     # Pune Station
    assert is_within_city(kalyan)
    assert not is_within_city(pune)
