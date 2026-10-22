import os
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple
from app.slot_checker.slot_models import SlotCheckResult, CivicExtractedSlots
from app.pipeline.geo_lookup import GeoLookupService, parse_numeric_location

logger = logging.getLogger(__name__)

# Mandatory civic slots required to register an actionable municipal grievance
MANDATORY_SLOTS = ["description", "location"]

SLOT_QUESTIONS = {
    "description": {
        "mr": "कृपया तुमच्या समस्येचे थोडे अधिक वर्णन करा (नेमकी काय अडचण आहे?).",
        "hi": "कृपया अपनी समस्या का थोड़ा और विवरण दें (असल में क्या दिक्कत है?).",
        "en": "Please describe your civic issue in a few words (what exactly is the problem?)."
    },
    "location": {
        "mr": "कृपया समस्येचे ठिकाण (परिसराचे नाव, लँडमार्क किंवा लोकेशन) सांगा.",
        "hi": "कृपया समस्या का स्थान (इलाके का नाम, लैंडमार्क या लोकेशन) बताइए.",
        "en": "Please specify the location, landmark, or area of the issue."
    },
    "media": {
        "mr": "शक्य असल्यास समस्येचा एक फोटो पाठवा, जेणेकरून त्वरित कारवाई करता येईल.",
        "hi": "संभव हो तो समस्या की एक फोटो भेजिए, ताकि जल्द कार्रवाई हो सके.",
        "en": "If possible, please upload a photo of the grievance for faster resolution."
    },
    # Asked instead of "location" when the given place cannot be found by a field officer
    "location_vague": {
        "mr": "कर्मचाऱ्यांना जागा नेमकी सापडावी म्हणून कृपया जवळचा लँडमार्क (दुकान, मंदिर, शाळा), सोसायटी/इमारतीचे नाव किंवा रस्त्याचे नाव सांगा, किंवा live location पाठवा.",
        "hi": "कर्मचारियों को जगह सही मिल सके, इसलिए कृपया नज़दीकी लैंडमार्क (दुकान, मंदिर, स्कूल), सोसाइटी/बिल्डिंग का नाम या सड़क का नाम बताएं, या live location भेजें।",
        "en": "So that the field staff can find the exact spot, please share a nearby landmark (shop, temple, school), the society/building name or street name, or send your live location."
    },
    "location_outside": {
        "mr": "हे ठिकाण आपल्या महानगरपालिकेच्या हद्दीबाहेर दिसत आहे. कृपया हद्दीतील अचूक ठिकाण किंवा लँडमार्क सांगा.",
        "hi": "यह स्थान आपकी नगर निगम सीमा के बाहर दिख रहा है। कृपया सीमा के भीतर का सही स्थान या लैंडमार्क बताएं।",
        "en": "This location appears to be outside the municipal corporation's limits. Please share the exact place or landmark within the city."
    }
}

# A location is "vague" when every word in it is one of these: it names no place a field officer could find
# ("आमच्या गल्लीत", "माझ्या घराजवळ", "near my house"). Any proper noun ("शिवाजी चौक") makes it specific.
GENERIC_LOCATION_WORDS = {
    # Marathi
    "आमच्या", "आमची", "आमचा", "आमचे", "माझ्या", "माझी", "माझा", "माझे", "आमच्याकडे", "इथे", "येथे", "इकडे",
    "तिथे", "तेथे", "जवळ", "जवळच", "समोर", "मागे", "बाजूला", "शेजारी", "गल्ली", "गल्लीत", "गल्लीमध्ये",
    "गल्लीतील", "रस्ता", "रस्त्यावर", "रस्त्यात", "रोड", "रोडवर", "घर", "घराजवळ", "घरासमोर", "घराच्या",
    "परिसर", "परिसरात", "भाग", "भागात", "एरिया", "एरियात", "कॉलनी", "कॉलनीत", "सोसायटी", "सोसायटीत",
    "बिल्डिंग", "बिल्डिंगमध्ये", "इमारत", "इमारतीत", "इमारतीजवळ", "चौक", "चौकात", "कोपऱ्यावर", "कोपऱ्यात",
    "मध्ये", "आहे", "या", "त्या", "ह्या",
    # Hindi
    "हमारी", "हमारे", "हमारा", "मेरी", "मेरे", "मेरा", "यहाँ", "यहां", "वहाँ", "वहां", "पास", "सामने", "गली",
    "में", "सड़क", "पर", "के", "की", "का", "इलाके", "इलाका", "इलाक़े", "मोहल्ला", "मोहल्ले", "कॉलोनी", "है",
    # English / romanized
    "my", "our", "here", "there", "near", "nearby", "opposite", "behind", "in", "at", "on", "the", "of", "to",
    "street", "lane", "road", "house", "home", "area", "locality", "colony", "society", "building", "chowk",
    "corner", "front", "beside", "next", "amchya", "majhya", "ithe", "jawal", "javal", "galli", "gallit",
    "rasta", "ghar", "samor", "hamari", "mere", "yaha", "yahan", "paas",
}

# Points this far (m) from the nearest ward centroid still count as inside the city, so a GPS fix just
# across a ward polygon edge is not rejected
CITY_EDGE_TOLERANCE_M = 1500.0


def is_vague_location(location: Optional[str]) -> bool:
    """True if the location text names no specific place (only generic words like 'our lane', 'near my house')."""
    words = re.findall(r"[\wऀ-ॿ]+", (location or "").lower())
    return not words or all(w in GENERIC_LOCATION_WORDS for w in words)


def is_within_city(geo_res: Any) -> bool:
    """Inside a ward polygon, or within CITY_EDGE_TOLERANCE_M of the nearest ward centroid."""
    if getattr(geo_res, "is_within_boundary", False):
        return True
    dist = getattr(geo_res, "distance_to_centroid_meters", None)
    return dist is not None and dist <= CITY_EDGE_TOLERANCE_M

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Language used when a message has no text to detect from (KDMC: Marathi)
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "mr")

# Words common in Marathi (Devanagari)
MR_MARKERS = {
    "आहे", "आहेत", "आणि", "मला", "माझ्या", "माझे", "माझा", "माझी", "तुमचे", "तुमची", "काय", "करा",
    "सांगा", "येथे", "इथे", "नाही", "मध्ये", "होते", "झाले", "झाला", "झाली", "पडले", "पडला", "पडली",
    "समोर", "जवळ", "आमच्या", "तुम्ही", "आम्ही",
}
MR_SUFFIXES = ("च्या", "ची", "चे", "चा")

# Words common in Hindi (Devanagari)
HI_MARKERS = {
    "है", "हैं", "और", "मुझे", "मेरे", "मेरा", "मेरी", "आपका", "आपकी", "क्या", "कीजिए", "बताइए",
    "नहीं", "में", "था", "थी", "रहा", "रही", "पड़ा", "पड़ी", "को", "से", "पास", "यहाँ", "यहां",
    "हमारे", "हमारी", "आप", "कृपा",
}

# Roman script (Hinglish / Marathlish)
HINGLISH = {
    "hai", "hain", "mera", "meri", "mere", "nahi", "nahin", "kya", "kripya", "paas", "raha", "rahi",
    "mujhe", "aap", "hamare", "ka", "ki", "ke", "se"
}
MARATHLISH = {
    "ahe", "ahet", "mala", "majha", "majhi", "majhya", "nahi", "kay", "kara", "jawal", "javal",
    "samor", "aamchya", "amchya", "tumhi", "zale", "zala", "pade", "padla", "padli", "ho"
}


def is_roman_script(text: str) -> bool:
    """True if the text has no Devanagari characters."""
    return not DEVANAGARI.search(text or "")


def detect_language(text: str, previous: Optional[str] = None) -> str:
    """Returns 'mr' | 'hi' | 'en'. Short/ambiguous messages keep the previous language."""
    if not text or not text.strip():
        # e.g. a photo sent without a caption at the start of a chat: use the city's main language
        return previous or DEFAULT_LANGUAGE

    words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    if not words:
        return previous or DEFAULT_LANGUAGE

    dev_chars = len(DEVANAGARI.findall(text))
    total_chars = len(re.findall(r"\w", text)) or 1

    # Devanagari script -> Hindi or Marathi?
    if dev_chars / total_chars > 0.4:
        mr = sum(w in MR_MARKERS for w in words)
        hi = sum(w in HI_MARKERS for w in words)
        mr += sum(1 for w in words if w.endswith(MR_SUFFIXES) and len(w) > 3) * 0.3
        if hi > mr:
            return "hi"
        if mr > hi:
            return "mr"
        # Tie / no markers: keep previous Hindi/Marathi, else default to Marathi
        return previous if previous in ("mr", "hi") else "mr"

    # Roman script
    mr_r = sum(w in MARATHLISH for w in words)
    hi_r = sum(w in HINGLISH for w in words)
    if mr_r > hi_r and mr_r > 0:
        return "mr"
    if hi_r > mr_r and hi_r > 0:
        return "hi"
    # Very short message ("ok", "yes") -> keep previous language
    if len(words) <= 2 and previous:
        return previous
    return "en"


class SlotChecker:
    """
    Intelligent Deterministic Gate controlling conversational flow.
    Features:
      - Pydantic structured multi-slot extraction via LLM with regex fallback
      - Self-correction and slot overwrite detection (e.g. 'not Rambaug, Shivaji Chowk')
      - Landmark Geo-Dictionary: auto-resolves ward and coordinates for recognized municipal landmarks
      - Deterministic gating: is_complete=True only when all mandatory slots are filled
    """
    def __init__(
        self,
        require_media: bool = False,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o-mini"
    ):
        self.require_media = require_media
        self.client = openai_client
        self.model = model
        self.geo_service = GeoLookupService()

    def extract_with_llm(self, text: str) -> Optional[CivicExtractedSlots]:
        """Extract multi-slot civic schema from citizen text using fast structured output."""
        if not self.client or not text or len(text.strip()) < 3:
            return None
        try:
            system_prompt = (
                "You are an AI civic slot extractor for municipal grievances in Marathi, Hindi, and English.\n"
                "Extract grievance details into JSON conforming to these fields:\n"
                "- category: 'pothole', 'garbage', 'drainage', 'streetlight', 'water_leak', 'tree_hazard', or null\n"
                "- description: clear civic problem description or null\n"
                "- landmark: specific street, road, chowk, society, building, or area or null\n"
                "- urgency: 'LOW', 'NORMAL', 'HIGH', or 'CRITICAL' (sparking wires, water in house, open manholes are CRITICAL)\n"
                "- is_correction: true if citizen is correcting/changing previous statement (e.g. 'nahi nahi', 'sorry not there', 'instead of'), else false\n"
                "- correction_target: 'location', 'description', or null\n"
                "- message_type: 'complaint_detail' if the message reports a civic problem or gives details of one "
                "(a bare place/landmark/address also counts), 'question' if the citizen is asking for information "
                "(timings, councillor, process, status), 'confirmation' for yes/no/ok replies, otherwise 'other'.\n"
                "For 'question', 'confirmation' and 'other' messages set category, description and landmark to null.\n"
                "Respond ONLY with valid JSON."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=180
            )
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            return CivicExtractedSlots(**parsed)
        except Exception as e:
            logger.warning(f"Structured slot extraction skipped or failed: {e}")
            return None

    def extract_and_merge_slots(
        self,
        existing_slots: Dict[str, Any],
        incoming_text: Optional[str] = None,
        incoming_lat: Optional[float] = None,
        incoming_lon: Optional[float] = None,
        incoming_media_url: Optional[str] = None,
        incoming_location: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract and consolidate slot values from citizen's current input and existing state.
        Handles:
          - Mobile app / GPS location in [latitude, longitude] format, dict, or string
          - OpenStreetMap Nominatim reverse geocoding for physical address
          - Centroid Haversine ward determination
          - Nominatim forward geocoding for textual landmarks
          - Self-correction overwrites and multi-slot extraction
        """
        slots = dict(existing_slots or {})

        # 1. Location extraction (GPS / mobile app [latitude, longitude] takes top priority)
        coords = None
        if incoming_location is not None:
            coords = parse_numeric_location(incoming_location)
        if not coords and incoming_lat is not None and incoming_lon is not None:
            coords = parse_numeric_location([incoming_lat, incoming_lon])

        if coords:
            lat, lon = coords
            slots["latitude"] = lat
            slots["longitude"] = lon
            slots["location_coords"] = [lat, lon]
            slots["has_gps"] = True

            # Resolve official municipal ward via Centroid Haversine / Polygon & address via Nominatim
            geo_res = self.geo_service.lookup_ward(lat, lon, fetch_address=True)
            slots["outside_city"] = not is_within_city(geo_res)
            slots["ward_id"] = geo_res.ward_id
            slots["ward_name"] = geo_res.ward_name
            if geo_res.distance_to_centroid_meters is not None:
                slots["distance_to_centroid_meters"] = geo_res.distance_to_centroid_meters
            if geo_res.centroid_lat is not None:
                slots["centroid_coords"] = [geo_res.centroid_lat, geo_res.centroid_lon]

            # Populate friendly physical address
            if geo_res.address:
                slots["location"] = geo_res.address
                slots["display_name"] = geo_res.display_name
            elif "location" not in slots or not slots.get("location"):
                slots["location"] = f"GPS: [{lat:.5f}, {lon:.5f}]"

        # 2. Media extraction
        if incoming_media_url:
            slots["media_url"] = incoming_media_url
            slots["has_media"] = True

        # 3. Dynamic Text Analysis with Self-Correction & Landmark Resolution
        if incoming_text and incoming_text.strip():
            text = incoming_text.strip()
            lower_text = text.lower()

            # Check if citizen supplied coordinates in text like "[19.2435, 73.1310]" or "19.2435, 73.1310"
            text_coords = parse_numeric_location(text)
            if text_coords and not slots.get("has_gps"):
                lat, lon = text_coords
                slots["latitude"] = lat
                slots["longitude"] = lon
                slots["location_coords"] = [lat, lon]
                slots["has_gps"] = True
                geo_res = self.geo_service.lookup_ward(lat, lon, fetch_address=True)
                slots["outside_city"] = not is_within_city(geo_res)
                slots["ward_id"] = geo_res.ward_id
                slots["ward_name"] = geo_res.ward_name
                if geo_res.distance_to_centroid_meters is not None:
                    slots["distance_to_centroid_meters"] = geo_res.distance_to_centroid_meters
                slots["location"] = geo_res.address or f"GPS: [{lat:.5f}, {lon:.5f}]"

            # Check if text is just an affirmative / refusal for media
            if any(phrase in lower_text for phrase in ["no photo", "nahi ahe", "नाही", "फोटो नाही", "no image"]):
                slots["media_skipped"] = True

            # Attempt LLM Structured Multi-Slot Extraction
            extracted = self.extract_with_llm(text)
            # Questions, yes/no replies and small talk must not become the complaint description
            is_complaint_text = extracted is None or extracted.message_type in (None, "complaint_detail")
            # A place name is still usable from a bare landmark ("Near Birla College" is often typed 'other')
            # or a confirmation ("yes, it's near Birla College") — but never from a question
            if (extracted and not is_complaint_text and extracted.message_type != "question"
                    and extracted.landmark and self._location_replaceable(slots, extracted.landmark)):
                self._set_text_location(slots, extracted.landmark, "extracted_landmark")

            if extracted and is_complaint_text:
                # Handle Self-Correction (Overwrite instead of append)
                if extracted.is_correction or any(c in lower_text for c in ["नाही नाही", "सॉरी", "sorry", "instead", "not there"]):
                    logger.info("Self-correction detected in citizen dialogue turn.")
                    if extracted.landmark:
                        self._set_text_location(slots, extracted.landmark, "self_corrected_landmark")
                        slots.pop("ward_id", None)
                        slots.pop("ward_name", None)
                        if not slots.get("has_gps"):
                            slots.pop("latitude", None)
                            slots.pop("longitude", None)
                            slots.pop("location_coords", None)
                    if extracted.description:
                        slots["description"] = extracted.description
                    if extracted.category:
                        slots["category"] = extracted.category
                else:
                    # Merge extracted fields cleanly
                    if extracted.landmark and self._location_replaceable(slots, extracted.landmark):
                        self._set_text_location(slots, extracted.landmark, "extracted_landmark")
                    if extracted.description and ("description" not in slots or len(extracted.description) > len(slots.get("description", ""))):
                        slots["description"] = extracted.description
                    if extracted.category and "category" not in slots:
                        slots["category"] = extracted.category
                    if extracted.urgency:
                        slots["urgency"] = extracted.urgency

            # Fallback heuristics for location & description if LLM didn't catch them
            if is_complaint_text and not slots.get("description") and len(text.split()) > 1:
                if not re.match(r"^(\+?\d+|hi|hello|namaste|नमस्कार)$", lower_text):
                    slots["description"] = text

            loc_keywords = [
                "near", "opposite", "road", "chowk", "chawk", "गल्ली", "रस्ता", "चौक", "समोर", "जवळ",
                "ward", "सोसायटी", "nagar", "nager", "नगर", "पश्चिम", "पूर्व", "west", "east", "market",
                "रोड", "मार्केट", "नाका", "स्टेशन", "station"
            ]
            # Keyword fallback: only when nothing was extracted, or (without an LLM) to replace a vague place
            if (is_complaint_text and any(kw in lower_text for kw in loc_keywords)
                    and (not slots.get("location") or (extracted is None and is_vague_location(slots["location"])))):
                self._set_text_location(slots, text, "heuristic_landmark")

            # 4. OpenStreetMap Nominatim Forward Geocoding + Centroid Ward Matching
            # If location is known textually, resolve to coordinates and ward.
            # Vague text is not geocoded: Nominatim would return the city centre and make it look precise.
            loc_str = slots.get("location")
            if loc_str and not slots.get("has_gps") and not is_vague_location(loc_str) and not slots.get("has_landmark_coords"):
                geo_match = self.geo_service.lookup_by_address_or_landmark(loc_str)
                if geo_match and not is_within_city(geo_match):
                    # The geocoder matched a same-named place elsewhere; keep the text, the backend geocodes it
                    logger.info(f"Geocoded '{loc_str}' outside the city; ignoring the coordinates.")
                    geo_match = None
                if geo_match:
                    slots["ward_id"] = geo_match.ward_id
                    slots["ward_name"] = geo_match.ward_name
                    slots["latitude"] = geo_match.latitude
                    slots["longitude"] = geo_match.longitude
                    slots["location_coords"] = [geo_match.latitude, geo_match.longitude]
                    slots["has_landmark_coords"] = True
                    if geo_match.distance_to_centroid_meters is not None:
                        slots["distance_to_centroid_meters"] = geo_match.distance_to_centroid_meters
                    if geo_match.address:
                        slots["location"] = geo_match.address
                        slots["display_name"] = geo_match.display_name

        return slots

    @staticmethod
    def _location_replaceable(slots: Dict[str, Any], new_location: str) -> bool:
        """A new landmark replaces a missing, vague or outside-city location, or a shorter one."""
        current = slots.get("location")
        if not current or is_vague_location(current) or slots.get("outside_city"):
            return True
        return not is_vague_location(new_location) and len(new_location) > len(current)

    @staticmethod
    def _set_text_location(slots: Dict[str, Any], location: str, location_type: str) -> None:
        """Set a typed location. A GPS fix that was outside the city is dropped in favour of the typed place."""
        if slots.get("outside_city"):
            for key in ("has_gps", "latitude", "longitude", "location_coords", "outside_city", "ward_id", "ward_name"):
                slots.pop(key, None)
        if slots.get("location") != location:
            # Coordinates geocoded from the previous text no longer apply
            if slots.pop("has_landmark_coords", None):
                for key in ("latitude", "longitude", "location_coords"):
                    slots.pop(key, None)
        slots["location"] = location
        slots["location_type"] = location_type

    def evaluate(
        self,
        current_slots: Dict[str, Any],
        language: str = "mr"
    ) -> SlotCheckResult:
        """
        Deterministic gatekeeper check.
        Evaluates filled vs missing slots and decides whether to continue dialogue or submit.
        """
        missing = []

        # Check description
        desc = current_slots.get("description")
        if not desc or len(str(desc).strip()) < 4:
            missing.append("description")

        # Check location: a GPS fix inside the city, or a specific (not vague) place name
        location_problem = None
        if current_slots.get("outside_city"):
            location_problem = "location_outside"
        elif not current_slots.get("has_gps"):
            text_loc = current_slots.get("location")
            if text_loc and is_vague_location(text_loc):
                location_problem = "location_vague"
            elif not text_loc and not ("latitude" in current_slots and "longitude" in current_slots):
                location_problem = "location"
        if location_problem:
            missing.append("location")

        # Check media if strictly required and not skipped
        if self.require_media and not current_slots.get("has_media") and not current_slots.get("media_skipped"):
            missing.append("media")

        is_complete = len(missing) == 0

        if is_complete:
            return SlotCheckResult(
                is_complete=True,
                missing_slots=[],
                next_slot_to_ask=None,
                next_question_mr=None,
                next_question_en=None,
                collected_slots=current_slots
            )

        # Incomplete: pick the first missing mandatory slot to ask next
        next_slot = missing[0]
        question_key = location_problem if next_slot == "location" else next_slot
        q_dict = SLOT_QUESTIONS.get(question_key, {})
        
        # Pick targeted question based on language
        if language == "hi":
            question_mr = q_dict.get("hi", q_dict.get("mr", "कृपया अधिक जानकारी दें."))
        else:
            question_mr = q_dict.get("mr", "कृपया अधिक माहिती द्या.")
            
        question_en = q_dict.get("en", "Please provide more details.")

        return SlotCheckResult(
            is_complete=False,
            missing_slots=missing,
            next_slot_to_ask=next_slot,
            next_question_mr=question_mr,
            next_question_en=question_en,
            collected_slots=current_slots
        )
