"""
Agent Intent Classification and Router for WardMitra Conversational Orchestrator.
Determines "What should happen next?" based on citizen input, conversation history, and context.
"""
import re
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class AgentIntent(str, Enum):
    QUESTION = "question"                  # Inquiring about municipal policies, office hours, helpline, SLAs
    TRACK_COMPLAINT = "track_complaint"    # Checking status of existing complaint
    IMAGE_PROVIDED = "image_provided"      # Citizen provided an image for classification
    COMPLAINT = "complaint"                # Ready with full complaint details for AI inference pipeline
    INFORMATION_MISSING = "information_missing"  # Civic grievance with missing mandatory slots
    CHITCHAT = "chitchat"                  # Greetings, polite pleasantries, gratitude

INTENT_SYSTEM_PROMPT = """You are an Intent Classifier for 'WardMitra' Municipal AI Assistant (Kalyan-Dombivli, Maharashtra).
Classify the citizen's message into EXACTLY ONE of the following intents:

1. 'question': Citizen is asking for information (office timings, holiday, emergency helpline, municipal fees, property tax, SLAs, how-to).
2. 'track_complaint': Citizen is asking about the status of an existing complaint (e.g., 'माझी तक्रार #123 काय झाली', 'check status of CMP-45', 'status update').
3. 'complaint': Citizen is reporting a civic problem (potholes, garbage, sewage, streetlights, water leak).
4. 'chitchat': Greetings or pleasantries (hello, hi, नमस्कार, धन्यवाद, thank you, bye).

Respond in JSON format:
{
  "intent": "<question|track_complaint|complaint|chitchat>",
  "complaint_id": "<extracted_complaint_id_or_null>",
  "topic": "<brief_topic>"
}
"""

def extract_complaint_id(text: str) -> Optional[str]:
    """Extract complaint ticket ID patterns like CMP-12345, #123, 12345, etc."""
    # Matches patterns like CMP-1234, CMP-ABCDEF, #1234, ID 1234
    patterns = [
        r"(?:CMP|cmp|Cmp)[-_]?([A-Za-z0-9]+)",
        r"#\s*([0-9]{2,8})",
        r"(?:तक्रार\s*(?:क्र|क्रमांक|नं|नंबर)?\s*[:\s#]?\s*([0-9]{1,8}))",
        r"(?:ticket|complaint|token)\s*(?:no|id|#)?\s*[:\s#]?\s*([A-Za-z0-9\-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).replace("#", "").strip()
    return None

class IntentClassifier:
    """Classifies user intent using OpenAI LLM with deterministic fast-path fallbacks."""

    def __init__(self, openai_client: Optional[Any] = None, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    def classify(
        self,
        text: str,
        has_image: bool = False,
        slots: Optional[Dict[str, Any]] = None,
        is_slots_complete: bool = False
    ) -> Tuple[AgentIntent, Dict[str, Any]]:
        """
        Determine what should happen next.
        Returns (AgentIntent, metadata_dict)
        """
        metadata: Dict[str, Any] = {}
        cleaned = text.strip()

        # 1. Image priority if image is provided with minimal/no text
        if has_image and len(cleaned) < 5:
            return AgentIntent.IMAGE_PROVIDED, {"has_image": True}

        # 2. Fast-path check: FAQ / SLA Questions
        faq_keywords = ["किती दिवस", "किती वेळ", "वेळ लागेल", "कधी उघडते", "सुट्टी", "हेल्पलाइन", "office timing", "working hours", "sla", "निकालाची वेळ"]
        if any(kw in cleaned.lower() for kw in faq_keywords):
            return AgentIntent.QUESTION, metadata

        # 3. Fast-path check: Complaint Tracking
        extracted_cid = extract_complaint_id(cleaned)
        track_keywords = ["status", "track", "स्टेटस", "काय झालं", "निकाला", "झाली का", "update", "तक्रार क्रमांक"]
        if extracted_cid or any(kw in cleaned.lower() for kw in track_keywords):
            if any(kw in cleaned.lower() for kw in ["status", "track", "स्टेटस", "काय झालं", "झाली का", "update", "check"]):
                metadata["complaint_id"] = extracted_cid
                return AgentIntent.TRACK_COMPLAINT, metadata

        # 3. Fast-path check: Chitchat greetings
        greetings = ["hi", "hello", "hey", "नमस्कार", "राम राम", "सुप्रभात", "bye", "good morning", "thank you", "धन्यवाद", "आभार"]
        if cleaned.lower() in greetings or (len(cleaned.split()) <= 2 and any(g in cleaned.lower() for g in greetings)):
            return AgentIntent.CHITCHAT, metadata

        # 4. LLM-based intent recognition
        if self.client:
            try:
                import json
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": cleaned}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=80
                )
                data = json.loads(resp.choices[0].message.content)
                raw_intent = data.get("intent", "complaint").lower()
                metadata.update(data)

                if raw_intent == "question":
                    return AgentIntent.QUESTION, metadata
                elif raw_intent == "track_complaint":
                    cid = data.get("complaint_id") or extracted_cid
                    metadata["complaint_id"] = cid
                    return AgentIntent.TRACK_COMPLAINT, metadata
                elif raw_intent == "chitchat":
                    return AgentIntent.CHITCHAT, metadata
            except Exception as e:
                logger.warning(f"LLM intent classification fallback due to: {e}")

        # 5. Deterministic Question / FAQ fallback
        question_keywords = ["केव्हा", "कधी", "कुठे", "वेळ", "कार्यालय", "timing", "helpline", "ऑफिस", "फोन", "number", "tax", "कर", "sla", "दिवस", "सुट्टी"]
        if any(kw in cleaned.lower() for kw in question_keywords):
            return AgentIntent.QUESTION, metadata

        # 6. If an image is provided alongside complaint text
        if has_image:
            return AgentIntent.IMAGE_PROVIDED, {"has_image": True}

        # 7. Default to Complaint routing:
        # Check if mandatory slots are complete
        if is_slots_complete:
            return AgentIntent.COMPLAINT, metadata
        else:
            return AgentIntent.INFORMATION_MISSING, metadata
