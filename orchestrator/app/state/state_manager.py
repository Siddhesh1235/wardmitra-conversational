import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.state.session_models import ConversationSession, ChatMessage

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE_NAME", "wardmitra_sessions")
DEFAULT_STORE_TYPE = os.getenv("SESSION_STORE_TYPE", "auto").lower()
DEFAULT_KEY_PREFIX = os.getenv("SESSION_KEY_PREFIX", "wardmitra:session:")


class ConversationStateManager:
    """
    Dynamic State Manager for multi-turn citizen dialogue sessions.
    
    Dynamically supports and falls back across:
    1. AWS DynamoDB (Serverless Cloud State with configurable TTL)
    2. Redis (Cluster / Container caching with configurable TTL)
    3. In-Memory Resilient Store (Zero-dependency local/test mode)

    The backend can be selected explicitly via `backend_type` or automatically
    resolved based on available environment credentials ('auto' mode).
    """

    def __init__(
        self,
        backend_type: Optional[str] = None,
        redis_url: Optional[str] = None,
        dynamodb_table: Optional[str] = None,
        aws_region: Optional[str] = None,
        key_prefix: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ):
        self.requested_backend = (backend_type or DEFAULT_STORE_TYPE).lower()
        self.redis_url = redis_url or DEFAULT_REDIS_URL
        self.table_name = dynamodb_table or DEFAULT_DYNAMODB_TABLE
        self.aws_region = aws_region or os.getenv("AWS_REGION", "ap-south-1")
        self.key_prefix = key_prefix or DEFAULT_KEY_PREFIX
        self.ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_SESSION_TTL

        # Active backend handles
        self.redis_client = None
        self.dynamodb_table_resource = None
        self._memory_store: Dict[str, str] = {}
        self._active_backend: str = "memory"

        # Initialize the configured or auto-detected backend
        self._init_backend()

    @property
    def active_backend(self) -> str:
        """Returns the currently active storage backend ('dynamodb', 'redis', or 'memory')."""
        return self._active_backend

    def _init_backend(self):
        """Dynamically initialize chosen backend with graceful fallback chain."""
        mode = self.requested_backend

        # 1. DynamoDB mode or Auto mode
        if mode in ("dynamodb", "auto"):
            if self._try_init_dynamodb():
                self._active_backend = "dynamodb"
                return
            elif mode == "dynamodb":
                logger.warning(
                    "Explicit DynamoDB backend requested but connection failed. Falling back to Redis/Memory."
                )

        # 2. Redis mode or Auto fallback
        if mode in ("redis", "dynamodb", "auto"):
            if self._try_init_redis():
                self._active_backend = "redis"
                return
            elif mode == "redis":
                logger.warning(
                    "Explicit Redis backend requested but connection failed. Falling back to In-Memory."
                )

        # 3. In-Memory fallback
        self._active_backend = "memory"
        logger.info("Operating in resilient In-Memory state store mode.")

    def _try_init_dynamodb(self) -> bool:
        """Attempt to initialize AWS DynamoDB table resource."""
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not (aws_key and aws_secret):
            logger.debug("AWS credentials not found. Skipping DynamoDB init.")
            return False

        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=self.aws_region
            )
            dynamodb = session.resource("dynamodb")
            table = dynamodb.Table(self.table_name)
            # Verify table accessibility
            _ = table.table_status
            self.dynamodb_table_resource = table
            logger.info(
                f"Connected to AWS DynamoDB table '{self.table_name}' in region '{self.aws_region}' (TTL: {self.ttl}s)"
            )
            return True
        except Exception as e:
            logger.warning(f"Could not connect to AWS DynamoDB ({e}).")
            self.dynamodb_table_resource = None
            return False

    def _try_init_redis(self) -> bool:
        """Attempt to connect to Redis server."""
        try:
            import redis
            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=1.5
            )
            client.ping()
            self.redis_client = client
            logger.info(f"Connected to Redis state store at {self.redis_url} (TTL: {self.ttl}s)")
            return True
        except Exception as e:
            logger.debug(f"Redis not reachable at {self.redis_url} ({e}).")
            self.redis_client = None
            return False

    def _redis_key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Fetch session by ID from DynamoDB, Redis, or In-Memory store."""
        # 1. DynamoDB
        if self._active_backend == "dynamodb" and self.dynamodb_table_resource:
            try:
                res = self.dynamodb_table_resource.get_item(Key={"session_id": session_id})
                item = res.get("Item")
                if item:
                    item.pop("ttl", None)
                    return ConversationSession(**item)
                return None
            except Exception as e:
                logger.warning(f"DynamoDB get_session failed ({e}). Checking memory fallback.")

        # 2. Redis
        key = self._redis_key(session_id)
        if self._active_backend == "redis" and self.redis_client:
            try:
                raw_json = self.redis_client.get(key)
                if not raw_json:
                    return None
                data = json.loads(raw_json)
                return ConversationSession(**data)
            except Exception as e:
                logger.error(f"Redis get failed: {e}. Checking memory fallback.")

        # 3. In-Memory
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
        """Save session state to DynamoDB, Redis, or In-Memory store."""
        session.updated_at = datetime.utcnow().isoformat()
        expiry_seconds = ttl_seconds if ttl_seconds is not None else self.ttl
        epoch_ttl = int(time.time()) + expiry_seconds
        key = self._redis_key(session.session_id)

        # 1. DynamoDB
        if self._active_backend == "dynamodb" and self.dynamodb_table_resource:
            try:
                item = json.loads(session.model_dump_json())
                item["session_id"] = session.session_id
                item["ttl"] = epoch_ttl
                self.dynamodb_table_resource.put_item(Item=item)
                return
            except Exception as e:
                logger.warning(f"DynamoDB save_session failed ({e}). Falling back to memory.")

        # 2. Redis
        if self._active_backend == "redis" and self.redis_client:
            raw_json = session.model_dump_json()
            try:
                self.redis_client.setex(key, expiry_seconds, raw_json)
                return
            except Exception as e:
                logger.error(f"Redis save failed: {e}. Storing in memory fallback.")

        # 3. In-Memory fallback
        self._memory_store[key] = session.model_dump_json()

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
        """Clear session data across all backends."""
        if self._active_backend == "dynamodb" and self.dynamodb_table_resource:
            try:
                self.dynamodb_table_resource.delete_item(Key={"session_id": session_id})
            except Exception as e:
                logger.warning(f"DynamoDB delete failed: {e}")

        key = self._redis_key(session_id)
        if self._active_backend == "redis" and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")

        self._memory_store.pop(key, None)

    def get_llm_messages(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Extract recent messages formatted for LLM Chat API:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        session = self.get_session(session_id)
        if not session or not session.history:
            return []

        recent = session.history[-max_turns:]
        return [{"role": msg.role, "content": msg.content} for msg in recent]

    def health_check(self) -> Dict[str, Any]:
        """Return connectivity health check information."""
        return {
            "status": "healthy",
            "active_backend": self._active_backend,
            "requested_backend": self.requested_backend,
            "ttl_seconds": self.ttl,
            "redis_url": self.redis_url if self._active_backend == "redis" else None,
            "dynamodb_table": self.table_name if self._active_backend == "dynamodb" else None,
        }


# Generic alias
StateManager = ConversationStateManager
