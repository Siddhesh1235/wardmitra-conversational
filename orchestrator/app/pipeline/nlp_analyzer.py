import os
import json
import logging
from typing import Optional, Dict, Any
from app.schemas.ai_inference import NLPAnalysisResult
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

CIVIC_CATEGORIES = [
    "Banners_Flex",
    "Drainage",
    "Electricity",
    "Encroachment",
    "Garbage",
    "Health_Sanitation",
    "Noise_Pollution",
    "PipelineDefects",
    "PotHoles",
    "Road_Incidents_Traffic",
    "StreetLight",
    "Trees"
]

CATEGORY_KEYWORD_MAP = {
    "PotHoles": ["khadda", "khadde", "खड्डा", "खड्डे", "pothole", "potholes", "road bad", "rasta kharab"],
    "Garbage": ["kachra", "kachara", "कचरा", "garbage", "trash", "waste", "durgandhi", "दुर्गंधी", "dump"],
    "StreetLight": ["light", "streetlight", "दिवा", "लाईट", "pole", "खांब", "andhar", "अंधार"],
    "Drainage": ["drainage", "gutter", "gatar", "ड्रेनेज", "गटार", "nalaa", "नाला", "choke"],
    "PipelineDefects": ["pipe", "water leak", "pani", "पाणी", "गळती", "pipeline", "leakage"],
    "Electricity": ["current", "wire", "शॉक", "तार", "electricity", "power cut", "voltage"],
    "Trees": ["jhada", "tree", "झाड", "फांदी", "branch", "fallen tree"],
    "Encroachment": ["atikraman", "अतिक्रमण", "encroachment", "hawkers", "illegal shop"],
    "Noise_Pollution": ["speaker", "noise", "आवाज", "dhol", "ध्वनी"],
    "Banners_Flex": ["banner", "flex", "बॅनर", "होर्डिंग", "hoarding"],
    "Health_Sanitation": ["fogging", "dambhis", "डास", "sanitation", "swachhata", "रोगराई"],
    "Road_Incidents_Traffic": ["traffic", "jam", "accident", "अपघात", "ट्रॅफिक", "signal"]
}



class NLPAnalyzer:
    """
    Multilingual NLP Analyzer for Marathi (Devanagari & Latin) and English civic complaints.
    Uses GPT-4o-mini structured output with resilient local heuristic fallback.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

    def analyze(self, text: str) -> NLPAnalysisResult:
        if not text or not text.strip():
            return NLPAnalysisResult(
                intent="register_complaint",
                detected_language="mr",
                extracted_category="Garbage",
                urgency_level="medium",
                keywords=[],
                location_mentions=[],
                clean_summary="Unspecified civic complaint"
            )

        # If OpenAI client is configured, call GPT-4o-mini
        if self.client:
            try:
                return self._analyze_with_llm(text)
            except Exception as e:
                logger.error(f"OpenAI analysis failed ({e}). Falling back to local NLP heuristics.")

        return self._analyze_with_heuristics(text)

    def _analyze_with_llm(self, text: str) -> NLPAnalysisResult:
        system_prompt = f"""
You are an expert civic complaint classifier for Maharashtra Municipal Corporations (PCMC / KDMC / PMC).
Analyze the citizen message provided in Marathi (Devanagari), Romanized Marathi (Hinglish/Marathlish), or English.
Map the issue strictly to one of these 12 categories: {CIVIC_CATEGORIES}.

Output strictly a JSON object with:
- "intent": "register_complaint" | "inquiry" | "status_check"
- "detected_language": "mr" | "en" | "mr-latn"
- "extracted_category": one of the 12 categories
- "urgency_level": "low" | "medium" | "high" | "emergency"
- "keywords": list of 2-5 extracted keywords
- "location_mentions": list of any places, landmarks, or ward references mentioned
- "clean_summary": formal 1-sentence summary of the grievance in English or Marathi
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        return NLPAnalysisResult(
            intent=data.get("intent", "register_complaint"),
            detected_language=data.get("detected_language", "mr"),
            extracted_category=data.get("extracted_category", "Garbage"),
            urgency_level=data.get("urgency_level", "medium"),
            keywords=data.get("keywords", []),
            location_mentions=data.get("location_mentions", []),
            clean_summary=data.get("clean_summary", text)
        )

    def _analyze_with_heuristics(self, text: str) -> NLPAnalysisResult:
        lower_text = text.lower()
        matched_category = "Garbage"
        highest_matches = 0
        extracted_keywords = []

        for category, keywords in CATEGORY_KEYWORD_MAP.items():
            count = 0
            for kw in keywords:
                if kw.lower() in lower_text:
                    count += 1
                    extracted_keywords.append(kw)
            if count > highest_matches:
                highest_matches = count
                matched_category = category

        # Detect inquiry / guide / cancel intent
        intent = "register_complaint"
        if any(w in lower_text for w in ["guide", "help", "teach", "madat", "मदत", "kashi", "कशी", "माहिती", "info"]):
            intent = "inquiry"
            matched_category = "General_Inquiry"
        elif any(w in lower_text for w in ["cancel", "रद्द", "नको", "मागे"]):
            intent = "cancel"
            matched_category = "Cancel_Request"

        # Detect urgency keywords
        urgency = "medium"
        high_urgency_words = ["danger", "accident", "urgent", "धोका", "अपघात", "तातडीने", "emergency", "current"]
        if any(w in lower_text for w in high_urgency_words):
            urgency = "high"

        # Detect language
        has_devanagari = bool(any('\u0900' <= char <= '\u097F' for char in text))
        lang = "mr" if has_devanagari else "mr-latn"

        return NLPAnalysisResult(
            intent=intent,
            detected_language=lang,
            extracted_category=matched_category,
            urgency_level=urgency,
            keywords=list(set(extracted_keywords))[:5],
            location_mentions=[],
            clean_summary=text.strip()
        )
