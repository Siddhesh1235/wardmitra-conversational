import re
import unicodedata
from typing import List, Set
from app.schemas.ai_inference import ProfanityCheckResult

# Curated Marathi, Hindi, and English profanity lexicon (Devanagari + Latin/Romanized)
MARATHI_HINDI_PROFANITIES: Set[str] = {
    # Devanagari
    "झावा", "झावण्या", "भाडखाऊ", "भाडव्या", "गांडू", "गांड", "लवडा", "लवड्या", "मादरचोद", 
    "बहेनचोद", "भोसडीच्या", "भोसडीका", "चुत्या", "चुतिया", "रानड्या", "हरामी", "कुत्रा", 
    "साला", "साले", "आयझव्या", "झवझव्या",
    # Romanized / Latin script
    "bhadwa", "bhadve", "bhadkhau", "gandu", "gaand", "lauda", "lavadya", "lavda", 
    "madarchod", "mc", "bc", "behenchod", "bhosdike", "bhosdichya", "bhosadike", 
    "chutiya", "chutya", "chootiya", "randi", "harami", "aaijhavya", "zhavanya", 
    "saala", "kutta", "kamina", "fucker", "bitch", "asshole", "bastard"
}

# Whitelist safe words that contain substrings of slang (prevent false positives)
SAFE_MARATHI_WHITELIST: Set[str] = {
    "फाटणे", "पाणी", "रस्ता", "झाड", "गाडी", "गल्ली", "लाईट", "पायवाट", "नाला", "कचरा"
}

class ProfanityChecker:
    """
    High-speed, zero-latency hybrid profanity & toxicity detection for Marathi, Hindi, and English.
    """
    def __init__(self, custom_words: List[str] = None):
        self.words = set(MARATHI_HINDI_PROFANITIES)
        if custom_words:
            self.words.update([w.lower().strip() for w in custom_words])

        # Compile regex with word boundaries
        escaped_words = [re.escape(w) for w in self.words]
        pattern = r"(?i)\b(" + "|".join(escaped_words) + r")\b"
        self.regex = re.compile(pattern)

    def _normalize_text(self, text: str) -> str:
        # Normalize unicode and repeated characters (e.g., 'chuuuutiiya' -> 'chutiya')
        text = unicodedata.normalize("NFKD", text)
        # Collapse >2 repeated characters into single char
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return text

    def check_text(self, text: str) -> ProfanityCheckResult:
        if not text or not text.strip():
            return ProfanityCheckResult(is_clean=True, profanity_found=[], toxicity_score=0.0, action="pass")

        normalized = self._normalize_text(text.lower())
        
        matches = self.regex.findall(normalized)
        unique_matches = list(set([m.strip() for m in matches if m.strip() not in SAFE_MARATHI_WHITELIST]))

        is_clean = len(unique_matches) == 0
        toxicity_score = 0.9 if not is_clean else 0.0
        action = "block" if not is_clean else "pass"

        return ProfanityCheckResult(
            is_clean=is_clean,
            profanity_found=unique_matches,
            toxicity_score=toxicity_score,
            action=action
        )
