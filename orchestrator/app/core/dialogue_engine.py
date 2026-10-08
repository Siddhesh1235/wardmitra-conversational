import os
import json
import logging
from typing import Optional, Dict, Any

from app.state.redis_manager import ConversationStateManager
from app.slot_checker.deterministic_gate import SlotChecker, detect_language
from app.pipeline.pipeline_dispatcher import AIInferenceDispatcher
from app.tools.backend_client import WardMitraBackendClient
from app.core.dialogue_models import DialogueTurnOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are 'WardMitra' (वार्डमित्र), an empathetic, helpful, and courteous municipal conversational AI assistant for citizens of Kalyan-Dombivli and Maharashtra Municipal Corporations.

YOUR CORE OBJECTIVES:
1. Help citizens easily register public civic grievances (garbage, potholes, streetlights, drainage, water leaks, trees, electricity).
2. Communicate warmly and strictly in the citizen's chosen language: Marathi (देवनागरी), Hindi (देवनागरी), or English.
3. Be concise and conversational (1-2 sentences max). Never sound robotic or bureaucratic.
4. If a piece of information is missing (such as location or problem description), acknowledge what the citizen said and politely ask for the missing detail.
5. NEVER assign workers or promise immediate completion. State that the complaint is forwarded to the ward office with the standard SLA time.
"""

class ConversationalOrchestrator:
    """
    Core dialogue engine driving conversational grievance collection.
    Coordinates between State Store, Slot-Checker gate, AI Inference Pipeline, and Node.js Backend API.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        state_manager: Optional[ConversationStateManager] = None,
        slot_checker: Optional[SlotChecker] = None,
        inference_dispatcher: Optional[AIInferenceDispatcher] = None,
        backend_client: Optional[WardMitraBackendClient] = None
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.state_mgr = state_manager or ConversationStateManager()
        self.slot_checker = slot_checker or SlotChecker()
        self.dispatcher = inference_dispatcher or AIInferenceDispatcher(openai_api_key=self.api_key)
        self.backend_client = backend_client or WardMitraBackendClient()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None
        self._init_llm()

    def _init_llm(self):
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("Conversational LLM Client (GPT-4o-mini) initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for dialogue: {e}")

    def _generate_conversational_reply(
        self,
        session_id: str,
        user_message: str,
        instruction: str,
        language: str = "mr"
    ) -> str:
        """
        Generate warm, conversational response using GPT-4o-mini with chat history context.
        Strictly follows the citizen's detected language (Marathi, Hindi, or English).
        """
        history = self.state_mgr.get_llm_messages(session_id, max_turns=6)

        if self.client:
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(history)
                lang_label = "Marathi (मराठी)" if language == "mr" else ("Hindi (हिंदी)" if language == "hi" else "English")
                messages.append({
                    "role": "system",
                    "content": f"Task instruction: {instruction}. Respond strictly in {lang_label}. Never mix other languages."
                })

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI dialogue generation error: {e}")

        # Fallback response
        return instruction

    def process_turn(
        self,
        session_id: str,
        citizen_id: str,
        text: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        image_bytes: Optional[bytes] = None,
        media_url: Optional[str] = None,
        channel: str = "web"
    ) -> DialogueTurnOutput:
        """
        Executes one full conversational turn for a citizen message.
        """
        logger.info(f"Processing dialogue turn for session: {session_id}, citizen: {citizen_id}")

        # 1. Fetch / Create Session State
        session = self.state_mgr.get_or_create_session(session_id, citizen_id=citizen_id, channel=channel)

        # 2. Add citizen message to history
        self.state_mgr.add_message(session_id, role="user", content=text, media_url=media_url)

        # 3. Intelligent Multi-lingual Detection (Marathi, Hindi, English, Hinglish, Marathlish)
        prev_lang = session.slots.get("lang")
        user_lang = detect_language(text, previous=prev_lang)
        session.slots["lang"] = user_lang

        # 4. Extract and update slots in session
        updated_slots = self.slot_checker.extract_and_merge_slots(
            existing_slots=session.slots,
            incoming_text=text,
            incoming_lat=latitude,
            incoming_lon=longitude,
            incoming_media_url=media_url
        )
        self.state_mgr.update_slots(session_id, updated_slots)

        # 5. Deterministic Slot-Checker Evaluation
        gate_result = self.slot_checker.evaluate(updated_slots, language=user_lang)

        # Case A: Incomplete information -> Ask next clarifying question
        if not gate_result.is_complete:
            missing_slot = gate_result.next_slot_to_ask
            base_question = gate_result.next_question_mr if user_lang in ("mr", "hi") else gate_result.next_question_en

            instruction = f"Acknowledge the citizen's input kindly, then ask: '{base_question}'"
            bot_reply = self._generate_conversational_reply(session_id, text, instruction, language=user_lang)

            # Persist bot reply
            self.state_mgr.add_message(session_id, role="assistant", content=bot_reply)

            return DialogueTurnOutput(
                reply_text=bot_reply,
                session_id=session_id,
                status="in_progress",
                is_complete=False,
                missing_slots=gate_result.missing_slots,
                collected_slots=updated_slots,
                structured_complaint=None,
                ai_inference_result=None
            )

        # Case B: All mandatory slots satisfied -> Execute AI Inference Pipeline
        logger.info(f"All slots collected for session {session_id}. Triggering AI Inference Pipeline.")
        self.state_mgr.set_status(session_id, "ready_for_inference")

        desc_to_analyze = updated_slots.get("description", text)
        lat_to_use = updated_slots.get("latitude", 19.228)
        lon_to_use = updated_slots.get("longitude", 73.070)

        # Fetch recent historical complaints from live backend for deduplication
        recent_complaints = self.backend_client.get_recent_complaints_sync(limit=25)

        # Execute 6 validation modules concurrently
        inference_result = self.dispatcher.process(
            citizen_id=citizen_id,
            text=desc_to_analyze,
            latitude=lat_to_use,
            longitude=lon_to_use,
            image_input=image_bytes,
            existing_complaints=recent_complaints
        )

        # Check if rejected by profanity or content moderation
        if not inference_result.success:
            reason = inference_result.rejection_reason or "SAFETY_VIOLATION"
            if reason == "ABUSIVE_OR_INAPPROPRIATE_LANGUAGE":
                rejection_msg = (
                    "क्षमस्व! तुमच्या संदेशात आक्षेपार्ह किंवा असभ्य भाषा आढळली आहे. कृपया सभ्य भाषेत तक्रार नोंदवा." if user_lang == "mr" else
                    ("क्षमा करें! आपके संदेश में अनुचित भाषा पाई गई है। कृपया विनम्र भाषा में शिकायत दर्ज करें।" if user_lang == "hi" else
                    "Sorry! Inappropriate language was detected. Please resubmit using polite language.")
                )
            else:
                rejection_msg = (
                    "क्षमस्व! पाठवलेला फोटो किंवा फाईल सुरक्षा नियमांत बसत नाही. कृपया समस्येचा खरा फोटो जोडा." if user_lang == "mr" else
                    ("क्षमा करें! अपलोड की गई फोटो सुरक्षा नियमों के अनुसार सही नहीं है। कृपया स्पष्ट फोटो अपलोड करें।" if user_lang == "hi" else
                    "Sorry! The uploaded media could not pass our safety check. Please upload a clear photo of the civic issue.")
                )

            self.state_mgr.add_message(session_id, role="assistant", content=rejection_msg)
            self.state_mgr.set_status(session_id, "rejected")

            return DialogueTurnOutput(
                reply_text=rejection_msg,
                session_id=session_id,
                status="rejected",
                is_complete=False,
                missing_slots=[],
                collected_slots=updated_slots,
                structured_complaint=None,
                ai_inference_result=inference_result
            )

        # Successful processing -> Register with Node.js Backend API
        complaint = inference_result.structured_complaint
        ward_info = complaint.ward_name if complaint else f"Ward {inference_result.geo.ward_id}"
        category_info = complaint.category if complaint else "Civic Grievance"
        sla_hours = inference_result.scoring.sla_hours if inference_result.scoring else 24

        ticket_id = "CMP-PENDING"
        if complaint:
            backend_res = self.backend_client.register_complaint_sync(complaint)
            ticket_id = backend_res.get("complaint_id", "CMP-1001")
            logger.info(f"Assigned official Ticket ID: {ticket_id}")

        instruction = (
            f"Inform citizen that their complaint about '{category_info}' has been verified and registered for {ward_info} "
            f"with official Complaint ID: {ticket_id}. "
            f"Mention resolution timeline of approximately {sla_hours} hours. Thank them for helping keep the city clean."
        )
        confirmation_reply = self._generate_conversational_reply(
            session_id, text, instruction, language=user_lang
        )

        self.state_mgr.add_message(session_id, role="assistant", content=confirmation_reply)
        self.state_mgr.set_status(session_id, "completed")

        return DialogueTurnOutput(
            reply_text=confirmation_reply,
            session_id=session_id,
            status="completed",
            is_complete=True,
            missing_slots=[],
            collected_slots=updated_slots,
            structured_complaint=complaint,
            ai_inference_result=inference_result
        )
