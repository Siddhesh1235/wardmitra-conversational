import os
import logging
from typing import Optional, Any, List, Dict

from app.state.state_manager import ConversationStateManager
from app.state.session_models import ChatMessage
from app.slot_checker.deterministic_gate import SlotChecker
from app.pipeline.pipeline_dispatcher import AIInferenceDispatcher
from app.tools.backend_client import WardMitraBackendClient
from app.tools.rag_knowledge import MunicipalKnowledgeBase, MUNICIPAL_KNOWLEDGE_DOCS
from app.tools.kb_sync import fetch_backend_knowledge_docs
from app.core.main_agent import MainConversationalAgent
from app.core.dialogue_models import DialogueTurnOutput
from app.core.conversation_guards import reset_complaint_draft

logger = logging.getLogger(__name__)

class ConversationalOrchestrator:
    """
    Core dialogue engine driving conversational grievance collection.
    Coordinates between State Store, Slot-Checker gate, AI Inference Pipeline,
    Knowledge Base RAG, Complaint Tracking, and Node.js Backend API via MainConversationalAgent.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        state_manager: Optional[ConversationStateManager] = None,
        slot_checker: Optional[SlotChecker] = None,
        inference_dispatcher: Optional[AIInferenceDispatcher] = None,
        backend_client: Optional[WardMitraBackendClient] = None,
        knowledge_base: Optional[MunicipalKnowledgeBase] = None
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.state_mgr = state_manager or ConversationStateManager()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None
        self._init_llm()
        self.slot_checker = slot_checker or SlotChecker(openai_client=self.client, model=self.model)
        self.dispatcher = inference_dispatcher or AIInferenceDispatcher(openai_api_key=self.api_key)
        self.backend_client = backend_client or WardMitraBackendClient()
        # Static municipal FAQs + live ward/councillor documents built from the Node.js backend
        self.knowledge_base = knowledge_base or MunicipalKnowledgeBase(
            docs=MUNICIPAL_KNOWLEDGE_DOCS + fetch_backend_knowledge_docs(self.backend_client),
            openai_api_key=self.api_key
        )

        # Initialize Main Conversational Agent (5-branch supervisor architecture)
        self.agent = MainConversationalAgent(
            openai_client=self.client,
            model=self.model,
            slot_checker=self.slot_checker,
            inference_dispatcher=self.dispatcher,
            backend_client=self.backend_client,
            knowledge_base=self.knowledge_base
        )

    def _init_llm(self):
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("Conversational LLM Client (GPT-4o-mini) initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for dialogue: {e}")

    def process_turn(
        self,
        session_id: str,
        citizen_id: str,
        text: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        image_bytes: Optional[bytes] = None,
        media_url: Optional[str] = None,
        channel: str = "web",
        location: Optional[Any] = None,
        auth_token: Optional[str] = None
    ) -> DialogueTurnOutput:
        """
        Executes one full conversational turn for a citizen message
        through the 5-branch Main Agent routing architecture.
        Supports location in [latitude, longitude] format, dict, or separate floats.
        `auth_token` is the citizen's backend JWT; it is passed through per turn and never persisted.
        """
        logger.info(f"Processing dialogue turn for session: {session_id}, citizen: {citizen_id}")

        # 1. Fetch / Create Session State (loaded once and saved once per turn,
        #    so history, slots and status are never overwritten by a stale copy)
        session = self.state_mgr.get_or_create_session(session_id, citizen_id=citizen_id, channel=channel)

        # 2. Execute Turn via Main Conversational Agent (history excludes the current message;
        #    the agent receives it separately as `text`)
        output = self.agent.execute_turn(
            session=session,
            text=text,
            latitude=latitude,
            longitude=longitude,
            image_bytes=image_bytes,
            media_url=media_url,
            location=location,
            auth_token=auth_token
        )

        # 3. Record both messages and the new status
        session.history.append(ChatMessage(role="user", content=text, media_url=media_url))
        session.history.append(ChatMessage(role="assistant", content=output.reply_text))
        session.status = output.status

        # 4. A finished complaint must not leak into the next one: keep only conversation-level slots
        if output.is_complete:
            session.slots = reset_complaint_draft(session.slots)

        self.state_mgr.save_session(session)
        return output
