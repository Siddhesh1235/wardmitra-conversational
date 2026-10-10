import os
import logging
from typing import Optional

from app.state.state_manager import ConversationStateManager
from app.slot_checker.deterministic_gate import SlotChecker
from app.pipeline.pipeline_dispatcher import AIInferenceDispatcher
from app.tools.backend_client import WardMitraBackendClient
from app.tools.rag_knowledge import MunicipalKnowledgeBase
from app.core.main_agent import MainConversationalAgent
from app.core.dialogue_models import DialogueTurnOutput

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
        self.slot_checker = slot_checker or SlotChecker()
        self.dispatcher = inference_dispatcher or AIInferenceDispatcher(openai_api_key=self.api_key)
        self.backend_client = backend_client or WardMitraBackendClient()
        self.knowledge_base = knowledge_base or MunicipalKnowledgeBase()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None
        self._init_llm()

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
        channel: str = "web"
    ) -> DialogueTurnOutput:
        """
        Executes one full conversational turn for a citizen message
        through the 5-branch Main Agent routing architecture.
        """
        logger.info(f"Processing dialogue turn for session: {session_id}, citizen: {citizen_id}")

        # 1. Fetch / Create Session State
        session = self.state_mgr.get_or_create_session(session_id, citizen_id=citizen_id, channel=channel)

        # 2. Add citizen message to history
        self.state_mgr.add_message(session_id, role="user", content=text, media_url=media_url)

        # 3. Execute Turn via Main Conversational Agent
        output = self.agent.execute_turn(
            session=session,
            text=text,
            latitude=latitude,
            longitude=longitude,
            image_bytes=image_bytes,
            media_url=media_url
        )

        # 4. Persist bot reply and session state
        self.state_mgr.add_message(session_id, role="assistant", content=output.reply_text)
        self.state_mgr.set_status(session_id, output.status)
        self.state_mgr.save_session(session)

        # 5. Save pgvector embedding if complaint was successfully registered
        if output.structured_complaint and self.dispatcher and hasattr(self.dispatcher, "duplicate_detector"):
            cid = output.structured_complaint.title or "CMP-AUTO"
            desc = output.structured_complaint.description or text
            try:
                self.dispatcher.duplicate_detector.save_complaint_embedding(cid, desc)
            except Exception as e:
                logger.debug(f"Could not save pgvector embedding: {e}")

        return output
