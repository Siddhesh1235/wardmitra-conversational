"""
Deterministic conversation guards that run before the LLM agent:
- cancel / start-over commands for the complaint draft
- life-threatening emergencies, answered immediately with verified national helplines
These must work the same way every time, so they are rules, not prompts.
"""
import re
from typing import Any, Dict, Optional

# Slots that describe the conversation rather than one complaint; everything else is the complaint draft
PERSISTENT_SLOT_KEYS = {"lang"}

_DEVANAGARI_WORD = r"[\wऀ-ॿ]+"

# A short message containing one of these means "drop the current complaint"
# Deliberately not listed: "stop"/"थांबा" (bus stop), "रहने" ("यहाँ रहने वाले..."), which appear in real complaints
CANCEL_WORDS = {
    # Marathi
    "रद्द", "राहूदे", "नकोय", "थांबवा",
    # Hindi
    "छोड़ो", "छोडो", "कैंसल",
    # English / romanized
    "cancel", "nevermind", "radd", "cancle",
}
CANCEL_PHRASES = [
    "तक्रार नको", "नोंदवू नका", "नको आहे", "राहू दे", "रहने दो", "मत करो", "दर्ज मत",
    "never mind", "don't register", "dont register", "do not register", "no need",
]
RESTART_PHRASES = [
    "पुन्हा सुरू", "परत सुरू", "नवीन तक्रार", "दुसरी तक्रार", "फिर से शुरू", "नई शिकायत",
    "start over", "start again", "restart", "new complaint", "another complaint",
]
# Commands are short; longer messages are real content ("कचरा इथे नको" is a complaint, not a cancel)
MAX_COMMAND_WORDS = 6


def reset_complaint_draft(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the conversation-level slots, dropping the complaint draft."""
    return {k: v for k, v in slots.items() if k in PERSISTENT_SLOT_KEYS}


def detect_draft_command(text: str) -> Optional[str]:
    """Return 'cancel', 'restart' or None for a citizen message."""
    lower = (text or "").strip().lower()
    words = re.findall(_DEVANAGARI_WORD, lower)
    if not words or len(words) > MAX_COMMAND_WORDS:
        return None
    if any(p in lower for p in RESTART_PHRASES):
        return "restart"
    if any(p in lower for p in CANCEL_PHRASES) or any(w in CANCEL_WORDS for w in words):
        return "cancel"
    return None


DRAFT_COMMAND_REPLIES = {
    "cancel": {
        "mr": "ठीक आहे, ही तक्रार रद्द केली आहे. आणखी कोणती मदत हवी असल्यास नक्की सांगा.",
        "hi": "ठीक है, यह शिकायत रद्द कर दी गई है। किसी और सहायता की ज़रूरत हो तो बताइए।",
        "en": "Okay, this complaint has been cancelled. Let me know if you need help with anything else.",
    },
    "restart": {
        "mr": "ठीक आहे, नवीन तक्रार सुरू करूया. कृपया समस्या काय आहे आणि ती कुठे आहे ते सांगा.",
        "hi": "ठीक है, नई शिकायत शुरू करते हैं। कृपया बताइए कि समस्या क्या है और कहाँ है।",
        "en": "Sure, let's start a new complaint. Please tell me what the problem is and where it is.",
    },
}


AFFIRMATIVE_WORDS = {
    "हो", "होय", "हां", "हाँ", "जी", "ठीक", "बरोबर", "नोंदवा", "नोंदवून", "करा", "दर्ज", "करो", "कीजिए",
    "yes", "yeah", "yep", "ok", "okay", "sure", "submit", "register", "confirm", "proceed", "ho", "haan", "han",
}
NEGATIVE_WORDS = {"नाही", "नको", "नहीं", "मत", "no", "not", "dont", "don't", "cancel", "nahi"}


# "हो, पण ठिकाण बदला" / "yes but change the location" is not a plain confirmation
QUALIFIER_MARKERS = ["पण", "परंतु", "बदल", "थांब", "लेकिन", "मगर", "but", "change", "instead", "wait", "except"]


def is_affirmative(text: str) -> bool:
    """A short, unqualified 'yes / register it' reply."""
    lower = (text or "").strip().lower()
    words = re.findall(_DEVANAGARI_WORD + "|don't", lower)
    if not words or len(words) > MAX_COMMAND_WORDS or any(w in NEGATIVE_WORDS for w in words):
        return False
    if any(q in lower for q in QUALIFIER_MARKERS):
        return False
    return any(w in AFFIRMATIVE_WORDS for w in words)


CONFIRMATION_PROMPT = {
    "mr": "आपल्या तक्रारीचा सारांश:\n- समस्या: {description}\n- ठिकाण: {location}\n- फोटो: {photo}\n\nही तक्रार नोंदवू का? (हो / नाही)",
    "hi": "आपकी शिकायत का सारांश:\n- समस्या: {description}\n- स्थान: {location}\n- फोटो: {photo}\n\nक्या यह शिकायत दर्ज करें? (हाँ / नहीं)",
    "en": "Your complaint summary:\n- Problem: {description}\n- Location: {location}\n- Photo: {photo}\n\nShall I register this complaint? (yes / no)",
}


# ---------------------------------------------------------------------------
# Photo / complaint mismatch: the citizen picks one of three numbered options
# ---------------------------------------------------------------------------
PHOTO_CHOICES = ("use_photo", "new_photo", "skip")
PHOTO_CHOICE_PHRASES = {
    "use_photo": ["फोटो बरोबर", "हाच फोटो", "हीच तक्रार", "फोटोतील समस्या", "फोटो सही", "यही शिकायत",
                  "photo is correct", "photo is right", "same as photo", "as in the photo"],
    "new_photo": ["दुसरा फोटो", "नवीन फोटो", "वेगळा फोटो", "दूसरी फोटो", "नई फोटो", "दूसरा फोटो",
                  "new photo", "another photo", "different photo", "send again"],
    "skip": ["फोटोशिवाय", "फोटो नको", "फोटो शिवाय", "बिना फोटो", "फोटो नहीं", "फोटो के बिना",
             "without photo", "without the photo", "no photo", "skip photo", "skip"],
}
_CHOICE_BY_NUMBER = {"1": "use_photo", "2": "new_photo", "3": "skip"}


def detect_photo_choice(text: str) -> Optional[str]:
    """Parse a reply to the photo-mismatch question: '1'/'१'/'option 2'/'फोटोशिवाय' -> a PHOTO_CHOICES value."""
    lower = (text or "").strip().lower().translate(str.maketrans("१२३", "123"))
    words = re.findall(_DEVANAGARI_WORD, lower)
    if not words or len(words) > MAX_COMMAND_WORDS:
        return None
    number = re.fullmatch(r"\(?(?:option|पर्याय|विकल्प)?\s*([123])\s*[).]?(?:\s.*)?", lower)
    if number:
        return _CHOICE_BY_NUMBER[number.group(1)]
    for choice, phrases in PHOTO_CHOICE_PHRASES.items():
        if any(p in lower for p in phrases):
            return choice
    return None


PHOTO_MISMATCH_QUESTION = {
    "mr": ("फोटोमध्ये **{photo}** दिसत आहे, पण आपली तक्रार **{draft}** बद्दल आहे. कृपया निवडा:\n"
           "1. ही तक्रार **{photo}** ची आहे (फोटो बरोबर आहे)\n"
           "2. मी **{draft}** चा दुसरा फोटो पाठवतो\n"
           "3. फोटोशिवाय पुढे जा"),
    "hi": ("फोटो में **{photo}** दिख रहा है, लेकिन आपकी शिकायत **{draft}** के बारे में है। कृपया चुनें:\n"
           "1. यह शिकायत **{photo}** की है (फोटो सही है)\n"
           "2. मैं **{draft}** की दूसरी फोटो भेजता हूँ\n"
           "3. बिना फोटो के आगे बढ़ें"),
    "en": ("The photo shows **{photo}**, but your complaint is about **{draft}**. Please choose:\n"
           "1. The complaint is about the **{photo}** (the photo is right)\n"
           "2. I will send another photo of the **{draft}**\n"
           "3. Continue without a photo"),
}


# ---------------------------------------------------------------------------
# Emergencies
# ---------------------------------------------------------------------------
# kind -> trigger substrings (Marathi, Hindi, English/romanized)
EMERGENCY_TRIGGERS = {
    "electric": ["करंट", "शॉक", "वायर तुट", "तार तुट", "लोंबकळ", "स्पार्क", "ठिणग", "बिजली का तार", "तार गिर",
                 "live wire", "electric shock", "electrocut", "sparking", "wire fallen", "fallen wire", "current lag"],
    "manhole": ["उघडे मॅनहोल", "मॅनहोल उघड", "उघडा मॅनहोल", "खुला मैनहोल", "मैनहोल खुला", "open manhole", "manhole open", "manhole cover missing"],
    # Not "धूर"/smoke: burning garbage is a routine complaint, not a fire emergency
    "fire": ["आग लाग", "आग लगी", "जळत आहे", "on fire", "fire broke", "caught fire"],
    "gas": ["गॅस गळती", "गैस लीक", "gas leak", "gas smell", "गॅसचा वास"],
    "collapse": ["इमारत कोसळ", "भिंत कोसळ", "इमारत पड", "बिल्डिंग गिर", "दीवार गिर", "building collapse", "wall collapse"],
    "flood": ["घरात पाणी", "घरात पाणी शिरले", "घर में पानी", "पूर आला", "बाढ़", "water entered house", "flooding in house", "house flooded"],
}

# Verified national helplines (also in the knowledge base: emergency_helplines_national, emergency_safety_special)
EMERGENCY_BANNERS = {
    "electric": {
        "mr": "⚠️ **तातडीचे:** तुटलेल्या/लोंबकळणाऱ्या वायरपासून किमान १० मीटर दूर रहा. त्वरित **1912 (महावितरण)** आणि **112** वर कॉल करा.",
        "hi": "⚠️ **तुरंत:** टूटे/लटकते तार से कम से कम 10 मीटर दूर रहें। तुरंत **1912 (MSEDCL)** और **112** पर कॉल करें।",
        "en": "⚠️ **Urgent:** Stay at least 10 metres away from the wire. Call **1912 (MSEDCL)** and **112** immediately.",
    },
    "manhole": {
        "mr": "⚠️ **तातडीचे:** उघड्या मॅनहोलभोवती लोकांना जाऊ देऊ नका आणि खूण करा. त्वरित **112** वर कॉल करा.",
        "hi": "⚠️ **तुरंत:** खुले मैनहोल के पास किसी को जाने न दें और निशान लगाएं। तुरंत **112** पर कॉल करें।",
        "en": "⚠️ **Urgent:** Keep people away from the open manhole and mark it. Call **112** immediately.",
    },
    "fire": {
        "mr": "⚠️ **तातडीचे:** त्वरित **101 (अग्निशमन दल)** किंवा **112** वर कॉल करा आणि सुरक्षित ठिकाणी जा.",
        "hi": "⚠️ **तुरंत:** तुरंत **101 (फायर ब्रिगेड)** या **112** पर कॉल करें और सुरक्षित स्थान पर जाएं।",
        "en": "⚠️ **Urgent:** Call **101 (Fire Brigade)** or **112** immediately and move to a safe place.",
    },
    "gas": {
        "mr": "⚠️ **तातडीचे:** विजेची बटणे वापरू नका, परिसर रिकामा करा आणि त्वरित **101** वर कॉल करा.",
        "hi": "⚠️ **तुरंत:** बिजली के स्विच न छुएं, जगह खाली करें और तुरंत **101** पर कॉल करें।",
        "en": "⚠️ **Urgent:** Do not operate electrical switches, evacuate the area and call **101** immediately.",
    },
    "collapse": {
        "mr": "⚠️ **तातडीचे:** त्या भागापासून दूर रहा. त्वरित **112** आणि **1077 (आपत्ती व्यवस्थापन)** वर कॉल करा.",
        "hi": "⚠️ **तुरंत:** उस जगह से दूर रहें। तुरंत **112** और **1077 (आपदा प्रबंधन)** पर कॉल करें।",
        "en": "⚠️ **Urgent:** Stay away from the structure. Call **112** and **1077 (Disaster Management)** immediately.",
    },
    "flood": {
        "mr": "⚠️ **तातडीचे:** वीजपुरवठा बंद करा आणि सुरक्षित उंच ठिकाणी जा. मदतीसाठी **112** किंवा **1077** वर कॉल करा.",
        "hi": "⚠️ **तुरंत:** बिजली की आपूर्ति बंद करें और सुरक्षित ऊंचे स्थान पर जाएं। मदद के लिए **112** या **1077** पर कॉल करें।",
        "en": "⚠️ **Urgent:** Switch off the power supply and move to a safe, higher place. Call **112** or **1077** for help.",
    },
}


def detect_emergency(text: str) -> Optional[str]:
    """Return the emergency kind for a message describing danger to life, else None."""
    lower = (text or "").lower()
    for kind, triggers in EMERGENCY_TRIGGERS.items():
        if any(t in lower for t in triggers):
            return kind
    return None


def emergency_banner(kind: str, language: str) -> str:
    banners = EMERGENCY_BANNERS[kind]
    return banners.get(language, banners["en"])
