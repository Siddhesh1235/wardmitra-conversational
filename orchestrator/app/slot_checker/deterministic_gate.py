import re
import logging
from typing import Dict, Any, Optional, Tuple
from app.slot_checker.slot_models import SlotCheckResult

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
    }
}

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

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
        return previous or "en"

    words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    if not words:
        return previous or "en"

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
    Deterministic Gate controlling conversational flow.
    Every turn:
      - If required slots are missing -> returns 'is_complete=False' and generates next targeted question.
      - If all required slots are present -> returns 'is_complete=True', authorizing submission to AI Inference.
    """
    def __init__(self, require_media: bool = False):
        self.require_media = require_media

    def extract_and_merge_slots(
        self,
        existing_slots: Dict[str, Any],
        incoming_text: Optional[str] = None,
        incoming_lat: Optional[float] = None,
        incoming_lon: Optional[float] = None,
        incoming_media_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract and consolidate slot values from citizen's current input and existing state.
        """
        slots = dict(existing_slots or {})

        # 1. Location extraction (GPS coordinates take top priority)
        if incoming_lat is not None and incoming_lon is not None:
            # Validate non-zero coordinates
            if not (incoming_lat == 0.0 and incoming_lon == 0.0):
                slots["latitude"] = float(incoming_lat)
                slots["longitude"] = float(incoming_lon)
                slots["has_gps"] = True
                slots["location"] = f"GPS: {incoming_lat:.5f}, {incoming_lon:.5f}"

        # 2. Media extraction
        if incoming_media_url:
            slots["media_url"] = incoming_media_url
            slots["has_media"] = True

        # 3. Text analysis for description and textual location
        if incoming_text and incoming_text.strip():
            text = incoming_text.strip()
            
            # Check if text is just an affirmative / refusal for media
            lower_text = text.lower()
            if any(phrase in lower_text for phrase in ["no photo", "nahi ahe", "नाही", "फोटो नाही", "no image"]):
                slots["media_skipped"] = True

            # If description not yet established or incoming is a longer elaboration
            current_desc = slots.get("description", "")
            if not current_desc or len(text) > len(current_desc):
                # Ensure it's not just a location or greeting
                if len(text.split()) > 1 and not re.match(r"^(\+?\d+|hi|hello|namaste|नमस्कार)$", lower_text):
                    slots["description"] = text

            # Check if text explicitly mentions location keywords
            loc_keywords = [
                "near", "opposite", "road", "chowk", "chawk", "गल्ली", "रस्ता", "चौक", "समोर", "जवळ",
                "ward", "सोसायटी", "nagar", "nager", "नगर", "पश्चिम", "पूर्व", "west", "east", "market"
            ]
            if any(kw in lower_text for kw in loc_keywords) and "location" not in slots:
                slots["location"] = text
                slots["location_type"] = "landmark_mention"

        return slots

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

        # Check location (either GPS or landmark/address text must be present)
        has_gps = ("latitude" in current_slots and "longitude" in current_slots)
        has_text_loc = bool(current_slots.get("location"))
        if not (has_gps or has_text_loc):
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
        q_dict = SLOT_QUESTIONS.get(next_slot, {})
        
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
