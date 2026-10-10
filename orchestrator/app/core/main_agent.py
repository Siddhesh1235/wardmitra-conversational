"""
Main Conversational Agent for WardMitra AI Orchestrator.
Coordinates the 5-branch routing architecture:
1. Information Missing -> Slot Checker -> Ask Follow-up
2. Image Provided -> Image Classification Tool (YOLO11s/Vision) -> Collect Results
3. Complaint -> Complaint Intelligence Tools (NLP, Geo, Duplicate) -> Collect Results
4. Question -> Knowledge Base / RAG -> Collect Results
5. Track Complaint -> Existing Node.js API -> Collect Results
Followed by Collect Results -> Generate Response -> Citizen.
"""
import logging
from typing import Optional, Dict, Any, List

from app.state.session_models import ConversationSession
from app.slot_checker.deterministic_gate import SlotChecker, detect_language
from app.pipeline.pipeline_dispatcher import AIInferenceDispatcher
from app.tools.backend_client import WardMitraBackendClient
from app.tools.rag_knowledge import MunicipalKnowledgeBase
from app.core.agent_router import IntentClassifier, AgentIntent, extract_complaint_id
from app.core.dialogue_models import DialogueTurnOutput

logger = logging.getLogger(__name__)

AGENT_RESPONSE_PROMPT = """You are 'WardMitra' (वार्डमित्र), an empathetic municipal conversational AI assistant for citizens of Kalyan-Dombivli and Maharashtra.
Formulate a warm, helpful, and concise response (1-2 sentences) based on the collected tool results.
Strictly respond in the citizen's detected language: {language}.

Context and Tool Output:
{tool_output}

Rules:
1. Speak with empathy and municipal courtesy.
2. If complaint is registered, mention the Complaint ID and standard SLA.
3. If information is missing, politely ask only for what is missing.
4. If answering a question, be factual and polite.
5. If tracking, provide the current status clearly.
"""

class MainConversationalAgent:
    """
    Supervising Agent coordinating all municipal AI tools and workflows.
    """
    def __init__(
        self,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        slot_checker: Optional[SlotChecker] = None,
        inference_dispatcher: Optional[AIInferenceDispatcher] = None,
        backend_client: Optional[WardMitraBackendClient] = None,
        knowledge_base: Optional[MunicipalKnowledgeBase] = None
    ):
        self.client = openai_client
        self.model = model
        self.slot_checker = slot_checker or SlotChecker()
        self.dispatcher = inference_dispatcher or AIInferenceDispatcher()
        self.backend_client = backend_client or WardMitraBackendClient()
        self.knowledge_base = knowledge_base or MunicipalKnowledgeBase()
        self.router = IntentClassifier(openai_client=self.client, model=self.model)

    def _generate_llm_response(self, language: str, tool_output: str, session_history: List[Any]) -> str:
        """Synthesize final empathetic response using LLM."""
        lang_name = "Marathi (मराठी)" if language == "mr" else ("Hindi (हिंदी)" if language == "hi" else "English")
        if self.client:
            try:
                system_msg = AGENT_RESPONSE_PROMPT.format(language=lang_name, tool_output=tool_output)
                messages = [{"role": "system", "content": system_msg}]
                
                # Append last 4 messages for context
                for msg in session_history[-4:]:
                    messages.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=160
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating LLM agent response: {e}")

        # Fallback to direct tool output
        return tool_output

    def execute_turn(
        self,
        session: ConversationSession,
        text: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        image_bytes: Optional[bytes] = None,
        media_url: Optional[str] = None
    ) -> DialogueTurnOutput:
        """
        Executes one turn through the 5-branch Agent Architecture.
        """
        # 1. Detect language
        prev_lang = session.slots.get("lang")
        user_lang = detect_language(text, previous=prev_lang)
        session.slots["lang"] = user_lang

        # 2. Extract and merge slots from incoming message
        has_image = bool(image_bytes or media_url)
        updated_slots = self.slot_checker.extract_and_merge_slots(
            existing_slots=session.slots,
            incoming_text=text,
            incoming_lat=latitude,
            incoming_lon=longitude,
            incoming_media_url=media_url
        )
        session.slots.update(updated_slots)

        gate_result = self.slot_checker.evaluate(session.slots, language=user_lang)

        # 3. Classify Intent ('What should happen next?')
        intent, metadata = self.router.classify(
            text=text,
            has_image=has_image,
            slots=session.slots,
            is_slots_complete=gate_result.is_complete
        )
        logger.info(f"MainAgent routing intent: {intent.value} for session: {session.session_id}")

        # ==========================================
        # Branch 1: QUESTION (Knowledge Base / RAG)
        # ==========================================
        if intent == AgentIntent.QUESTION:
            rag_res = self.knowledge_base.search(text)
            if rag_res.get("found"):
                top_doc = rag_res["results"][0]
                content = top_doc["content_mr"] if user_lang == "mr" else top_doc["content_en"]
                instruction = f"Provide official answer from municipal guidelines: {content}"
            else:
                instruction = rag_res.get("message", "माहिती उपलब्ध नाही. कृपया वॉर्ड कार्यालयात संपर्क साधा.")

            bot_reply = self._generate_llm_response(user_lang, instruction, session.history)
            return DialogueTurnOutput(
                reply_text=bot_reply,
                session_id=session.session_id,
                status="answering_faq",
                is_complete=False,
                missing_slots=gate_result.missing_slots,
                collected_slots=session.slots,
                structured_complaint=None,
                ai_inference_result=None
            )

        # ==========================================
        # Branch 2: TRACK COMPLAINT (Node.js API)
        # ==========================================
        if intent == AgentIntent.TRACK_COMPLAINT:
            cid = metadata.get("complaint_id") or extract_complaint_id(text)
            if cid and str(cid).lower() not in ("null", "none", ""):
                status_res = self.backend_client.get_complaint_status_sync(cid)
                if status_res.get("success"):
                    complaint_data = status_res.get("data", {})
                    c_status = complaint_data.get("status", "In Progress")
                    instruction = (
                        f"Complaint ID {cid} is currently '{c_status}'. Inform the citizen warmly."
                    )
                else:
                    instruction = (
                        f"Complaint ID {cid} was not found or is currently being verified. "
                        f"Ask citizen to check the number or contact Ward 19A office."
                    )
            else:
                instruction = (
                    "Citizen wants to track a complaint but did not provide the complaint ID. "
                    "Kindly ask for their Complaint Number (उदा. #123 किंवा CMP-456)."
                )

            bot_reply = self._generate_llm_response(user_lang, instruction, session.history)
            return DialogueTurnOutput(
                reply_text=bot_reply,
                session_id=session.session_id,
                status="tracking_complaint",
                is_complete=False,
                missing_slots=[],
                collected_slots=session.slots,
                structured_complaint=None,
                ai_inference_result=None
            )

        # ==========================================
        # Branch 3: CHITCHAT (Greetings / Thanks)
        # ==========================================
        if intent == AgentIntent.CHITCHAT:
            instruction = (
                "Greet the citizen warmly as WardMitra. Ask how you can assist them with municipal services or civic issues."
            )
            bot_reply = self._generate_llm_response(user_lang, instruction, session.history)
            return DialogueTurnOutput(
                reply_text=bot_reply,
                session_id=session.session_id,
                status="chitchat",
                is_complete=False,
                missing_slots=gate_result.missing_slots,
                collected_slots=session.slots,
                structured_complaint=None,
                ai_inference_result=None
            )

        # ==========================================
        # Branch 4: IMAGE PROVIDED (Civic Classifier)
        # ==========================================
        if intent == AgentIntent.IMAGE_PROVIDED and image_bytes:
            # Classify image using existing YOLO/Vision tool
            civic_output = self.dispatcher.civic_classifier.classify_image(image_bytes)
            detected_label = getattr(civic_output, "predicted_class", None) or getattr(civic_output, "category", None) or "Garbage / Civic issue"
            logger.info(f"Image classified as: {detected_label}")

            # Merge detected problem into session slots
            if not session.slots.get("description"):
                session.slots["description"] = f"फोटोवरून आढळलेली समस्या: {detected_label}"

            # Re-evaluate slots now that image contributed information
            gate_result = self.slot_checker.evaluate(session.slots, language=user_lang)

            if not gate_result.is_complete:
                missing_slot = gate_result.next_slot_to_ask
                base_question = gate_result.next_question_mr if user_lang in ("mr", "hi") else gate_result.next_question_en
                instruction = (
                    f"Acknowledge the uploaded photo shows '{detected_label}'. Then politely ask: '{base_question}'"
                )
                bot_reply = self._generate_llm_response(user_lang, instruction, session.history)
                return DialogueTurnOutput(
                    reply_text=bot_reply,
                    session_id=session.session_id,
                    status="in_progress",
                    is_complete=False,
                    missing_slots=gate_result.missing_slots,
                    collected_slots=session.slots,
                    structured_complaint=None,
                    ai_inference_result=None
                )

        # ==========================================
        # Branch 5: INFORMATION MISSING (Slot Checker)
        # ==========================================
        if intent == AgentIntent.INFORMATION_MISSING or not gate_result.is_complete:
            missing_slot = gate_result.next_slot_to_ask
            base_question = gate_result.next_question_mr if user_lang in ("mr", "hi") else gate_result.next_question_en
            instruction = f"Acknowledge the citizen's input kindly, then ask: '{base_question}'"
            bot_reply = self._generate_llm_response(user_lang, instruction, session.history)

            return DialogueTurnOutput(
                reply_text=bot_reply,
                session_id=session.session_id,
                status="in_progress",
                is_complete=False,
                missing_slots=gate_result.missing_slots,
                collected_slots=session.slots,
                structured_complaint=None,
                ai_inference_result=None
            )

        # =======================================================
        # Branch 6: COMPLAINT READY (Execute AI Pipeline & Submit)
        # =======================================================
        logger.info(f"Executing Complaint Intelligence Tools for session {session.session_id}")
        desc_to_analyze = session.slots.get("description", text)
        lat_to_use = session.slots.get("latitude", 19.228)
        lon_to_use = session.slots.get("longitude", 73.070)

        # Fetch recent complaints for pgvector deduplication
        recent_complaints = self.backend_client.get_recent_complaints_sync(limit=25)

        # Run AI validation modules concurrently
        inference_result = self.dispatcher.process(
            citizen_id=session.citizen_id or "CITIZEN_ANON",
            text=desc_to_analyze,
            latitude=lat_to_use,
            longitude=lon_to_use,
            image_input=image_bytes,
            existing_complaints=recent_complaints
        )

        # Handle moderation rejection
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

            return DialogueTurnOutput(
                reply_text=rejection_msg,
                session_id=session.session_id,
                status="rejected",
                is_complete=False,
                missing_slots=[],
                collected_slots=session.slots,
                structured_complaint=None,
                ai_inference_result=inference_result
            )

        # Check Duplicate
        dup = getattr(inference_result, "deduplication", None)
        if dup and getattr(dup, "is_duplicate", False):
            dup_id = getattr(dup, "parent_complaint_id", None) or "अगोदर नोंदवलेली"
            dup_msg = (
                f"याच समस्येची तक्रार (क्र. {dup_id}) आधीच नोंदवली गेलेली आहे आणि संबंधित विभागाकडे प्रलंबित आहे. तुमचे आभार!" if user_lang == "mr" else
                (f"इस समस्या की शिकायत (क्र. {dup_id}) पहले से दर्ज है और संबंधित विभाग में प्रक्रियाधीन है। धन्यवाद!" if user_lang == "hi" else
                f"A similar complaint (ID: {dup_id}) is already registered and under review. Thank you for notifying us!")
            )
            return DialogueTurnOutput(
                reply_text=dup_msg,
                session_id=session.session_id,
                status="completed",
                is_complete=True,
                missing_slots=[],
                collected_slots=session.slots,
                structured_complaint=inference_result.structured_complaint,
                ai_inference_result=inference_result
            )

        # Register Complaint via Node.js Backend API
        backend_resp = self.backend_client.register_complaint_sync(
            complaint=inference_result.structured_complaint
        )
        complaint_id = backend_resp.get("complaint_id", "CMP-REC")
        cat = inference_result.structured_complaint.category
        ward = inference_result.structured_complaint.ward_name or "वॉर्ड १९ए"

        confirm_msg = (
            f"धन्यवाद! तुमची तक्रार (क्र. {complaint_id}) '{cat}' अंतर्गत {ward} कार्यालयाकडे यशस्वीपणे नोंदवली आहे. संबंधित कर्मचारी लवकरच यावर कार्यवाही करेल." if user_lang == "mr" else
            (f"धन्यवाद! आपकी शिकायत (क्र. {complaint_id}) '{cat}' के तहत {ward} कार्यालय में दर्ज कर ली गई है। शीघ्र ही कार्रवाई की जाएगी।" if user_lang == "hi" else
            f"Thank you! Your complaint (ID: {complaint_id}) for '{cat}' has been successfully forwarded to {ward}. Our team will resolve it soon.")
        )

        return DialogueTurnOutput(
            reply_text=confirm_msg,
            session_id=session.session_id,
            status="completed",
            is_complete=True,
            missing_slots=[],
            collected_slots=session.slots,
            structured_complaint=inference_result.structured_complaint,
            ai_inference_result=inference_result
        )
