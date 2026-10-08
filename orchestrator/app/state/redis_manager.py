import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import redis

from app.state.session_models import ConversationSession, ChatMessage

logger = logging.getLogger(__name__)

# Default session TTL: 24 hours (86,400 seconds) as specified in architecture diagram
DEFAULT_SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class ConversationStateManager:
    """
    Manages conversational session states, chat histories, and collected slots.
    Persists to Redis with 24-hour TTL, with automatic In-Memory fallback for local development.
    """
    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = DEFAULT_SESSION_TTL):
        self.redis_url = redis_url or REDIS_URL
        self.ttl = ttl_seconds
        self.redis_client: Optional[redis.Redis] = None
        self._memory_store: Dict[str, str] = {}
        self._init_redis()

    def _init_redis(self):
        """Attempt to connect to Redis, graceful fallback to in-memory store if unavailable."""
        try:
            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5
            )
            client.ping()
            self.redis_client = client
            logger.info(f"Connected to Redis state store at {self.redis_url} (TTL: {self.ttl}s)")
        except Exception as e:
            logger.warning(
                f"Redis server not reachable at {self.redis_url} ({e}). "
                f"Operating in resilient In-Memory state store mode."
            )
            self.redis_client = None

    def _redis_key(self, session_id: str) -> str:
        return f"wardmitra:session:{session_id}"

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Fetch session by ID. Returns None if session does not exist."""
        key = self._redis_key(session_id)
        raw_json = None

        if self.redis_client:
            try:
                raw_json = self.redis_client.get(key)
            except Exception as e:
                logger.error(f"Redis get failed: {e}. Falling back to memory.")
                raw_json = self._memory_store.get(key)
        else:
            raw_json = self._memory_store.get(key)

        if not raw_json:
            return None

        try:
            data = json.loads(raw_json)
            return ConversationSession(**data)
        except Exception as e:
            logger.error(f"Failed to deserialize session {session_id}: {e}")
            return None

    def get_or_create_session(
        self,
        session_id: str,
        citizen_id: Optional[str] = None,
        channel: str = "web"
    ) -> ConversationSession:
        """Fetch existing session or initialize a fresh one."""
        session = self.get_session(session_id)
        if not session:
            session = ConversationSession(
                session_id=session_id,
                citizen_id=citizen_id,
                channel=channel,
                history=[],
                slots={},
                status="collecting_info"
            )
            self.save_session(session)
        else:
            # Update citizen_id if newly provided
            if citizen_id and not session.citizen_id:
                session.citizen_id = citizen_id
                self.save_session(session)
        return session

    def save_session(self, session: ConversationSession, ttl_seconds: Optional[int] = None) -> None:
        """Save session state to Redis (or in-memory) with 24h TTL."""
        session.updated_at = datetime.utcnow().isoformat()
        key = self._redis_key(session.session_id)
        raw_json = session.model_dump_json()
        expiry = ttl_seconds or self.ttl

        if self.redis_client:
            try:
                self.redis_client.setex(key, expiry, raw_json)
                return
            except Exception as e:
                logger.error(f"Redis save failed: {e}. Storing in memory fallback.")
                self._memory_store[key] = raw_json
        else:
            self._memory_store[key] = raw_json

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        media_url: Optional[str] = None
    ) -> ConversationSession:
        """Append message to session history and persist."""
        session = self.get_or_create_session(session_id)
        msg = ChatMessage(role=role, content=content, media_url=media_url)
        session.history.append(msg)
        self.save_session(session)
        return session

    def update_slots(self, session_id: str, new_slots: Dict[str, Any]) -> ConversationSession:
        """Update collected slot values in the session and persist."""
        session = self.get_or_create_session(session_id)
        session.slots.update(new_slots)
        self.save_session(session)
        return session

    def set_status(self, session_id: str, status: str) -> ConversationSession:
        """Update session workflow status."""
        session = self.get_or_create_session(session_id)
        session.status = status
        self.save_session(session)
        return session

    def clear_session(self, session_id: str) -> None:
        """Clear session data (e.g., when conversation finishes or cancels)."""
        key = self._redis_key(session_id)
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")
        self._memory_store.pop(key, None)

    def get_llm_messages(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Extract recent messages formatted for OpenAI Chat Completions API:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        session = self.get_session(session_id)
        if not session or not session.history:
            return []

        recent = session.history[-max_turns:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]
