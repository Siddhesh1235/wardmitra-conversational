"""
Backwards-compatibility proxy module.
Re-exports ConversationStateManager and StateManager from app.state.state_manager.
"""
from app.state.state_manager import ConversationStateManager, StateManager

__all__ = ["ConversationStateManager", "StateManager"]
