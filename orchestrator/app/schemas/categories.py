"""
Maps the orchestrator's AI categories (NLP + vision classes) to the backend's complaint categories.
The backend expects a numeric `category_id` (needed for officer auto-assignment); IDs are looked up
at runtime by name from GET /api/user/categories, so they are never hardcoded here.
Also used to check that a photo and the complaint text describe the same kind of problem.
"""
from typing import Any, Dict, List, Optional

# AI category -> backend category names to try, in order (backend seed: Water Supply, Electricity,
# Waste Management, Road & Infrastructure, Transport, Health & Sanitation, Education,
# Street Lighting, Drainage, Other)
AI_TO_BACKEND_CATEGORY_NAMES: Dict[str, List[str]] = {
    "PotHoles": ["Road & Infrastructure", "Roads"],
    "Road_Incidents_Traffic": ["Transport", "Road & Infrastructure"],
    "Encroachment": ["Road & Infrastructure"],
    "Banners_Flex": ["Road & Infrastructure"],
    "Garbage": ["Waste Management", "Garbage"],
    "Health_Sanitation": ["Health & Sanitation"],
    "Drainage": ["Drainage"],
    "PipelineDefects": ["Water Supply", "Water Issue"],
    "Electricity": ["Electricity"],
    "StreetLight": ["Street Lighting", "Electricity"],
    "Trees": ["Other"],
    "Noise_Pollution": ["Other"],
}
FALLBACK_CATEGORY_NAME = "Other"

# Human-readable names for the AI categories, so citizens and the LLM never see raw class names
CATEGORY_LABELS: Dict[str, Dict[str, str]] = {
    "PotHoles": {"mr": "रस्त्यावरील खड्डा", "hi": "सड़क का गड्ढा", "en": "pothole"},
    "Road_Incidents_Traffic": {"mr": "रस्ता/वाहतूक समस्या", "hi": "सड़क/यातायात समस्या", "en": "road or traffic problem"},
    "Encroachment": {"mr": "अतिक्रमण", "hi": "अतिक्रमण", "en": "encroachment"},
    "Banners_Flex": {"mr": "अनधिकृत बॅनर/होर्डिंग", "hi": "अवैध बैनर/होर्डिंग", "en": "illegal banner or hoarding"},
    "Garbage": {"mr": "कचरा", "hi": "कचरा", "en": "garbage"},
    "Health_Sanitation": {"mr": "आरोग्य व स्वच्छता समस्या", "hi": "स्वास्थ्य व स्वच्छता समस्या", "en": "health or sanitation problem"},
    "Drainage": {"mr": "ड्रेनेज/गटार समस्या", "hi": "नाली/सीवर समस्या", "en": "drainage problem"},
    "PipelineDefects": {"mr": "पाण्याची पाइपलाइन गळती", "hi": "पानी की पाइपलाइन लीकेज", "en": "water pipeline leak"},
    "Electricity": {"mr": "वीज समस्या", "hi": "बिजली समस्या", "en": "electricity problem"},
    "StreetLight": {"mr": "बंद पथदिवा", "hi": "बंद स्ट्रीट लाइट", "en": "streetlight problem"},
    "Trees": {"mr": "झाडाची समस्या", "hi": "पेड़ की समस्या", "en": "tree problem"},
    "Noise_Pollution": {"mr": "ध्वनी प्रदूषण", "hi": "ध्वनि प्रदूषण", "en": "noise pollution"},
}

# The slot extractor's own category values -> AI categories
EXTRACTOR_TO_AI_CATEGORY: Dict[str, str] = {
    "pothole": "PotHoles", "garbage": "Garbage", "drainage": "Drainage",
    "streetlight": "StreetLight", "water_leak": "PipelineDefects", "tree_hazard": "Trees",
}


def category_label(ai_category: Optional[str], language: str = "en") -> str:
    labels = CATEGORY_LABELS.get(ai_category or "")
    if not labels:
        return {"mr": "नागरी समस्या", "hi": "नागरिक समस्या"}.get(language, "civic issue")
    return labels.get(language, labels["en"])


def to_ai_category(value: Optional[str]) -> Optional[str]:
    """Normalise an extractor value ('water_leak') or AI class ('PipelineDefects') to an AI class."""
    if not value:
        return None
    if value in AI_TO_BACKEND_CATEGORY_NAMES:
        return value
    return EXTRACTOR_TO_AI_CATEGORY.get(str(value).strip().lower())


def same_department(category_a: Optional[str], category_b: Optional[str]) -> bool:
    """
    True if two AI categories go to a common backend category (e.g. StreetLight and Electricity both
    fit 'Electricity'). Unknown categories never count as a mismatch.
    """
    a, b = to_ai_category(category_a), to_ai_category(category_b)
    if not a or not b:
        return True
    return bool(set(AI_TO_BACKEND_CATEGORY_NAMES[a]) & set(AI_TO_BACKEND_CATEGORY_NAMES[b]))


def infer_draft_category(slots: Dict[str, Any]) -> Optional[str]:
    """
    The complaint draft's AI category: the extractor's category if set, else a keyword match on the
    description. None when nothing points to a category (no guessing).
    """
    explicit = to_ai_category(slots.get("category"))
    if explicit:
        return explicit
    from app.pipeline.nlp_analyzer import CATEGORY_KEYWORD_MAP
    description = str(slots.get("description") or "").lower()
    if not description:
        return None
    scores = {
        cat: sum(1 for kw in keywords if kw.lower() in description)
        for cat, keywords in CATEGORY_KEYWORD_MAP.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def resolve_backend_category(ai_category: Optional[str], backend_categories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Return {"category_id", "category_name"} for an AI category, using the backend's live category list.
    Falls back to 'Other'; returns None if the backend list is empty.
    """
    by_name = {
        str(c.get("category_name", "")).strip().lower(): c
        for c in backend_categories if c.get("category_id") is not None
    }
    if not by_name:
        return None
    candidates = AI_TO_BACKEND_CATEGORY_NAMES.get(ai_category or "", []) + [FALLBACK_CATEGORY_NAME]
    for name in candidates:
        match = by_name.get(name.lower())
        if match:
            return {"category_id": match["category_id"], "category_name": match["category_name"]}
    return None
