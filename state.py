"""LangGraph 工作流共享状态。"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class ConversationState(TypedDict, total=False):
    """一次对话轮次在各节点之间传递的数据。"""

    user_input: str
    history: list[BaseMessage]
    emotion: str
    memories: list[str]
    bond_value: float
    projected_bond_value: float
    bond_delta: float
    bond_event: str
    bond_tone: str
    mode: str
    current_mode: str
    risk_level: str
    safety_notice: str
    companion_alias: str
    skin_id: str
    story_progress: int
    story_context: str
    mode_label: str
    current_time: str
    last_interaction_at: str
    inactivity_notice: str
    memory_event: str
    interaction_event: str
    command: str
    simulation_requested: bool
    response: str
